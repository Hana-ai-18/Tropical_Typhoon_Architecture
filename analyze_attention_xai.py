"""
analyze_attention_xai.py
=========================
Chạy SAU KHI đã có checkpoint TC-FlowMatching đã train xong. Trích 2 nguồn
diễn giải riêng biệt từ VelocityTransformer:
  1. cross-attention weight (xem [XAI-ATTN] trong flow_matching_model.py)
  2. FiLM gamma/beta deviation (xem [XAI-FILM]) -- vì HORIZON-FILM tiêm
     context vào query side (x_emb) NGOÀI đường cross-attention tới memory,
     cross_attn một mình không còn kể hết câu chuyện horizon-conditioning
     nữa; cần cả 2 nguồn để tránh bỏ sót cơ chế nào đang thực sự "làm việc".
Phân nhóm storm theo mức độ turning (recurving vs straight-moving), rồi
tổng hợp + vẽ hình cho phần Results của bài báo.

USAGE
-----
python analyze_attention_xai.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --checkpoint   /kaggle/working/runs/fm_v5_seed2/best_model.pth \
    --output_dir   /kaggle/working/xai_analysis \
    --gpu_num 0

Kết quả:
  - attn_by_horizon.csv       : bảng số liệu thô (storm, horizon, attn_context, attn_time, group)
  - attn_summary.csv          : trung bình theo horizon x group -- dùng điền vào Table trong bài
  - attn_heatmap.png          : hình vẽ dùng trực tiếp cho Figure trong Results
  - film_deviation.csv        : gamma/beta deviation theo horizon -- KHÔNG phụ thuộc storm
                                 (film_gamma/beta là tham số MODEL, không phải per-sample),
                                 dùng cho 1 hình/bảng riêng minh họa horizon nào học "lệch"
                                 khỏi context gốc nhiều nhất
"""
from __future__ import annotations
import sys, os, argparse, math
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _forward_azimuth, _unwrap,
)


HORIZON_HOURS = [6 * (i + 1) for i in range(12)]   # [6, 12, ..., 72] for pred_len=12


def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


def load_fm(checkpoint_path: str, device):
    ck = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = ck.get("model_cfg") or {}
    model = TCFlowMatching(**model_cfg)
    state = ck.get("model_state", ck.get("model")) or ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [load_fm] missing keys (expected if resuming across the "
              f"reg_step_logits/log_b_horizon -> reg_dist_ema architecture "
              f"change): {missing}")
    if unexpected:
        print(f"  [load_fm] unexpected keys: {unexpected}")
    model.to(device).eval()
    return model


def classify_storm_turning(obs_traj: torch.Tensor, gt_traj: torch.Tensor,
                            turn_threshold_deg: float = 25.0) -> str:
    """
    Phân loại 1 storm là 'recurving' hay 'straight' dựa trên tổng độ đổi
    hướng (heading change) TRÊN TOÀN BỘ observation + ground-truth window,
    không chỉ observation -- vì ta muốn biết storm này CÓ THỰC SỰ recurve
    trong khung dự báo hay không, không chỉ đã recurve trước đó.

    turn_threshold_deg=25.0: một cutoff hợp lý cho "turning đáng kể" (một
    storm đi thẳng tây/tây bắc thường đổi hướng vài độ mỗi 6h; 25 độ tích
    lũy trên toàn trajectory là ngưỡng phân biệt rõ ràng loại rẽ ngoặt so
    với nhiễu đường đi tự nhiên). Đây LÀ một hằng số tay chọn -- nếu kết
    quả nhạy cảm với giá trị này, hãy thử quét vài ngưỡng (15/25/35) để
    xem pattern có ổn định không trước khi chốt số liệu cho bài báo.
    """
    full_traj = torch.cat([obs_traj[:, :2], gt_traj[:, :2]], dim=0)   # [T_obs+T_pred, 2]
    deg = _norm_to_deg(full_traj.unsqueeze(1)).squeeze(1)              # [T, 2]
    if deg.shape[0] < 3:
        return "unknown"
    bearings = [float(_forward_azimuth(deg[i], deg[i + 1]))
                for i in range(deg.shape[0] - 1)]
    total_turn = 0.0
    for i in range(len(bearings) - 1):
        d = bearings[i + 1] - bearings[i]
        d = math.degrees(math.atan2(math.sin(math.radians(d)), math.cos(math.radians(d))))
        total_turn += abs(d)
    return "recurving" if total_turn >= turn_threshold_deg else "straight"


@torch.no_grad()
def collect_attention_records(model, loader, device):
    """
    Chạy sample(..., return_attn=True) trên toàn bộ test set, trả về
    DataFrame dạng thô: 1 dòng / (storm, horizon).
    """
    records = []
    n_batches = 0
    for bi, batch in enumerate(loader):
        bl = move(list(batch), device)
        obs = bl[0]
        gt  = bl[1]
        try:
            tyid_list = bl[15]
        except IndexError:
            tyid_list = None

        try:
            _, _, _, xai = model.sample(bl, num_ensemble=20, return_xai=True,
                                         return_attn=True, use_curvature_score=True)
        except Exception as e:
            print(f"  batch {bi}: sample error, skipped ({e})")
            continue

        if "cross_attn" not in xai:
            print(f"  batch {bi}: no cross_attn in xai -- check return_attn wiring")
            continue

        # [num_layers, B, pred_len, 2] -> average over layers for a single
        # summary weight per (storm, horizon); per-layer detail is kept in
        # the raw tensor if you want to inspect individual layers later.
        attn = xai["cross_attn"].mean(dim=0)   # [B, pred_len, 2]
        B = attn.shape[0]

        for b in range(B):
            if tyid_list is not None and b < len(tyid_list) and \
               isinstance(tyid_list[b], dict) and "old" in tyid_list[b]:
                info = tyid_list[b]
                storm_key = f"{info['old'][1]}_{info['old'][0]}"
            else:
                storm_key = f"UNKNOWN_batch{bi}_{b}"

            group = classify_storm_turning(obs[:, b, :], gt[:, b, :])

            for h_idx, h_hours in enumerate(HORIZON_HOURS):
                if h_idx >= attn.shape[1]:
                    break
                records.append({
                    "storm": storm_key,
                    "horizon_h": h_hours,
                    "attn_time":    float(attn[b, h_idx, 0]),
                    "attn_context": float(attn[b, h_idx, 1]),
                    "group": group,
                })
        n_batches += 1

    print(f"  Processed {n_batches} batches, {len(records)} (storm, horizon) records.")
    return pd.DataFrame(records)


def make_summary_and_plot(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(os.path.join(output_dir, "attn_by_horizon.csv"), index=False)

    summary = (df.groupby(["group", "horizon_h"])["attn_context"]
                 .agg(["mean", "std", "count"])
                 .reset_index())
    summary.to_csv(os.path.join(output_dir, "attn_summary.csv"), index=False)
    print("\n  Summary (mean context-vector attention by group x horizon):")
    print(summary.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for group in sorted(df["group"].unique()):
            sub = summary[summary["group"] == group]
            ax.plot(sub["horizon_h"], sub["mean"], marker="o", label=group)
            ax.fill_between(sub["horizon_h"],
                             sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                             alpha=0.15)
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel("Mean cross-attention weight on context vector")
        ax.set_title("Attention to context vector vs. time embedding, by horizon and storm type")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig_path = os.path.join(output_dir, "attn_heatmap.png")
        fig.savefig(fig_path, dpi=200)
        print(f"\n  Figure saved: {fig_path}")
    except ImportError:
        print("\n  matplotlib not available -- skipped figure, CSVs still saved.")

    # Quick textual takeaway to help drafting the paper paragraph.
    straight_72 = summary[(summary.group == "straight") & (summary.horizon_h == 72)]
    recurv_72   = summary[(summary.group == "recurving") & (summary.horizon_h == 72)]
    straight_6  = summary[(summary.group == "straight") & (summary.horizon_h == 6)]
    recurv_6    = summary[(summary.group == "recurving") & (summary.horizon_h == 6)]
    print("\n  ── Quick read for drafting the paper paragraph ──────────────")
    if len(straight_6) and len(recurv_6):
        print(f"  At 6h : straight={straight_6['mean'].iloc[0]:.3f}  "
              f"recurving={recurv_6['mean'].iloc[0]:.3f}")
    if len(straight_72) and len(recurv_72):
        print(f"  At 72h: straight={straight_72['mean'].iloc[0]:.3f}  "
              f"recurving={recurv_72['mean'].iloc[0]:.3f}")
    print("  If these differ noticeably (e.g. >0.05 absolute) and the gap")
    print("  widens with horizon -> Scenario A (clear pattern) in the paper.")
    print("  If they stay close across all horizons -> Scenario B (report")
    print("  as a negative/limitation result, do not oversell in Introduction).")


def extract_film_deviation(model, output_dir: str):
    """
    [XAI-FILM] film_gamma/film_beta là nn.Embedding của VelocityTransformer
    -- tham số MODEL, không phụ thuộc batch/storm nào, nên chỉ cần đọc
    trực tiếp từ checkpoint đã load, không cần chạy qua test set.
    Deviation = khoảng cách L2 từ điểm khởi tạo zero-impact (gamma=1, beta=0).
    Horizon nào có deviation cao = model đã học "viết lại" context nhiều
    hơn cho horizon đó so với việc chỉ dùng context gốc không đổi.
    """
    vel = model.velocity if hasattr(model, "velocity") else model.module.velocity
    gamma_w = vel.film_gamma.weight.detach().cpu()   # [pred_len, d_model]
    beta_w  = vel.film_beta.weight.detach().cpu()

    gamma_dev = (gamma_w - 1.0).norm(dim=-1).numpy()
    beta_dev  = beta_w.norm(dim=-1).numpy()

    df_film = pd.DataFrame({
        "horizon_h": HORIZON_HOURS[:len(gamma_dev)],
        "gamma_deviation": gamma_dev,
        "beta_deviation": beta_dev,
    })
    df_film.to_csv(os.path.join(output_dir, "film_deviation.csv"), index=False)
    print("\n  FiLM gamma/beta deviation by horizon (0 = still at init, "
          "i.e. this horizon has NOT learned to modify the shared context):")
    print(df_film.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df_film["horizon_h"], df_film["gamma_deviation"], marker="o", label="gamma deviation (scale)")
        ax.plot(df_film["horizon_h"], df_film["beta_deviation"], marker="s", label="beta deviation (shift)")
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel("L2 deviation from zero-impact init")
        ax.set_title("How much each horizon has learned to modify the shared context (FiLM)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig_path = os.path.join(output_dir, "film_deviation.png")
        fig.savefig(fig_path, dpi=200)
        print(f"\n  Figure saved: {fig_path}")
    except ImportError:
        print("\n  matplotlib not available -- skipped figure, CSV still saved.")

    return df_film


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", default="xai_analysis")
    p.add_argument("--gpu_num", default="0")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12)
    # [CONFIRMED against loader_training.py / trajectoriesWithMe_unet_training.py]
    # Every field below is read via getattr(args, name, default) inside
    # data_loader / TrajectoryDataset.__init__, so all of them are optional
    # here too -- these --flags exist only so the values can be overridden
    # from the command line if a checkpoint was trained with non-default
    # settings; the defaults below match TrajectoryDataset's own defaults.
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--other_modal", default="gph")
    p.add_argument("--delim", default=" ")
    p.add_argument("--skip", type=int, default=1)
    p.add_argument("--min_ped", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.002)
    p.add_argument("--filter_region", action="store_true", default=False)
    p.add_argument("--min_pct_in_scs", type=float, default=15.0)
    p.add_argument("--turn_threshold_deg", type=float, default=25.0)
    p.add_argument("--skip_attention", action="store_true", default=False,
                    help="Skip the per-storm cross-attention analysis (slow, "
                         "requires running the full test set) and only "
                         "extract FiLM deviation (fast, model-only).")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = load_fm(args.checkpoint, device)
    print(f"  Loaded checkpoint: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)
    extract_film_deviation(model, args.output_dir)

    if args.skip_attention:
        print("\n  --skip_attention set -- skipping cross-attention analysis.")
        return

    # [FIX] data_loader's real signature in this project is
    # data_loader(args, {"root": ..., "type": "test"}, test=True) --
    # NOT the (dataset_root, obs_len=, pred_len=, ...) signature initially
    # assumed. Confirmed against train_flowmatching.py's own call sites
    # (trd, trl = data_loader(args, {...}, test=False); etc.) before
    # finalizing this script, rather than guessing.
    _, test_loader = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)

    df = collect_attention_records(model, test_loader, device)
    if df.empty:
        print("  No records collected -- check that return_attn is wired "
              "correctly in flow_matching_model.py and that the checkpoint "
              "loaded without errors.")
        return

    make_summary_and_plot(df, args.output_dir)


if __name__ == "__main__":
    main()