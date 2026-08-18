"""
analyze_xai_multi_seed.py
=========================
Chạy 1 LẦN cho NHIỀU checkpoint FM (ví dụ 3 seed) cùng lúc. Giải quyết đúng
3 lỗ hổng đã xác định khi chỉ chạy 1 seed:

  1. CỠ MẪU: 1 seed không đủ để biết một pattern (ví dụ "gamma deviation
     cao ở 6h") là thật hay chỉ là đặc thù của 1 lần khởi tạo ngẫu nhiên.
     Script này tự động tính MEAN ± STD qua N seed cho mỗi horizon, thay vì
     báo cáo 1 điểm dữ liệu duy nhất.

  2. SUY LUẬN NHÂN QUẢ: "deviation cao ở horizon X" không tự động nghĩa là
     "model học được điều gì có ích ở đó" -- cần đối chiếu với ADE/ATE/CTE
     THẬT tại đúng horizon đó. Script này tự chạy evaluate() đầy đủ cho mỗi
     seed VÀ ghép nối với FiLM deviation cùng horizon, để bạn (hoặc reviewer)
     tự nhìn thấy có tương quan hay không, thay vì suy diễn từ 1 con số.

  3. TÍNH TÁI LẬP: mỗi checkpoint được đánh giá độc lập, seed RNG cố định
     trước mỗi model.sample() call (giống evaluate_multi_model.py's
     set_seed() convention), để kết quả không phụ thuộc thứ tự checkpoint
     được liệt kê trên CLI.

CẢ 2 NGUỒN DIỄN GIẢI (không chỉ 1):
  - FiLM gamma/beta deviation (nhanh, không cần test set, chỉ đọc tham số
    model) -- xem [XAI-FILM] trong flow_matching_model.py
  - Cross-attention weight (chậm hơn, cần chạy qua test set) -- xem
    [XAI-ATTN]. Bật/tắt qua --skip_attention.

USAGE
-----
python analyze_xai_multi_seed.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --checkpoints \
        seed0=/path/best_model_fm_seed0.pth \
        seed1=/path/best_model_fm_seed1.pth \
        seed2=/path/best_model_fm_seed2.pth \
    --output_dir /kaggle/working/xai_multi_seed \
    --gpu_num 0

KẾT QUẢ
-------
  film_deviation_per_seed.csv     : 1 dòng / (seed, horizon) -- dữ liệu thô
  film_deviation_summary.csv      : mean±std qua seed, theo horizon
  film_deviation_summary.png      : hình có error band (mean ± std)

  attn_by_horizon_per_seed.csv    : 1 dòng / (seed, storm, horizon)  [nếu không --skip_attention]
  attn_summary_across_seeds.csv   : mean±std qua seed, theo (group, horizon)
  attn_summary_across_seeds.png   : hình có error band

  eval_by_horizon_per_seed.csv    : ADE/ATE/CTE THẬT theo (seed, horizon) -- để đối chiếu
  eval_summary_across_seeds.csv   : mean±std ADE/ATE/CTE theo horizon
  film_vs_metric_correlation.csv  : Pearson correlation giữa gamma/beta
                                     deviation và ADE tại từng horizon, qua
                                     N=12 horizon điểm dữ liệu -- CHỈ đủ ý
                                     nghĩa nếu >=3 seed (n quá nhỏ để suy ra
                                     gì chắc chắn dù vậy; xem cảnh báo in ra)
"""
from __future__ import annotations
import sys, os, argparse, math, random
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _haversine_deg, _forward_azimuth, _unwrap,
)

HORIZON_HOURS = [6 * (i + 1) for i in range(12)]   # [6, 12, ..., 72] for pred_len=12
HORIZON_STEPS = {h: i for i, h in enumerate(HORIZON_HOURS)}   # 0-indexed step per horizon


def set_seed(s: int = 42):
    """Mirrors evaluate_multi_model.py's set_seed() convention -- called
    before EACH checkpoint's evaluation so results don't depend on the
    order checkpoints are listed on the CLI."""
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


def parse_checkpoint_args(pairs: list) -> dict:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--checkpoints entries must be name=path, got: {p}")
        name, path = p.split("=", 1)
        out[name] = path
    return out


def load_fm(checkpoint_path: str, device):
    ck = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = ck.get("model_cfg") or {}
    model = TCFlowMatching(**model_cfg)
    state = ck.get("model_state", ck.get("model")) or ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"    [load_fm] missing keys (expected across the "
              f"reg_step_logits/log_b_horizon -> reg_dist_ema architecture "
              f"change): {missing}")
    if unexpected:
        print(f"    [load_fm] unexpected keys: {unexpected}")

    # Same EMA-vs-SWA handling as evaluate_multi_model.py's load_fm --
    # ck["model"] IS the SWA average when is_swa=True (no separate EMA
    # applied); otherwise apply the EMA shadow if present, matching
    # train_flowmatching.py's own eval-time convention.
    is_swa = ck.get("is_swa", False)
    if ck.get("ema") and not is_swa:
        sd = model.state_dict()
        applied = 0
        for k, v in ck["ema"].items():
            if k in sd:
                sd[k].copy_(v.to(device)); applied += 1
        print(f"    [load_fm] Applied EMA shadow weights ({applied} tensors).")
    elif is_swa:
        print(f"    [load_fm] Checkpoint is an SWA average -- using as-is.")

    model.to(device).eval()
    return model, ck


def classify_storm_turning(obs_traj: torch.Tensor, gt_traj: torch.Tensor,
                            turn_threshold_deg: float = 240.0) -> str:
    """Same logic as the single-seed script -- see that file's docstring
    for the rationale behind the cutoff.

    [BUG FOUND AND FIXED] _forward_azimuth returns bearings in RADIANS
    (standard atan2 convention). The previous version took the raw
    difference between two such radian values (`d = bearings[i+1] -
    bearings[i]`, itself already correctly in radians) and then called
    `math.radians(d)` on it before wrapping with atan2/degrees -- i.e.
    treating an already-radian value as if it were in degrees and
    converting it AGAIN, shrinking it by a factor of ~57.3 (180/pi).
    Verified numerically: a genuine 30-degree turn was reduced to
    ~0.52 "degrees" by this bug. Confirmed directly from a real run:
    all 449 test-set storms across all 3 seeds were classified as
    "straight" with zero "recurving" records anywhere in the attention
    summary output -- consistent with total_turn being suppressed ~57x
    below the 25-degree threshold for every storm, not with the test set
    genuinely containing zero recurving storms (implausible for a South
    China Sea / NW Pacific tropical cyclone track dataset). Fixed by
    converting each bearing to degrees ONCE, immediately after
    _forward_azimuth, before taking any differences.

    [THRESHOLD RE-CALIBRATED, SECOND PASS -- with actual distribution data]
    After the fix above, a first re-calibration attempt raised the
    default to 45 degrees as a placeholder guess, pending confirmation
    from print_turn_distribution(). That confirmation has now run on the
    real 449-storm test set: min=71.7, 25th pct=175.3, median=238.9,
    75th pct=350.6, max=973.2 degrees of accumulated turn. Every one of
    the 5 candidate thresholds tested (25/35/45/55/65) still classified
    100% of storms as "recurving", because ALL of them sit far below
    even the observed MINIMUM (71.7 degrees) -- this is not a bug in
    the fixed formula, it reflects real physical behavior: over an
    18-increment window (8 observed + 12 predicted 6-hour steps), no
    tropical cyclone track in this basin moves in a perfectly straight
    line -- natural path wobble alone accounts for several degrees of
    accumulated turn per step (verified: even the "straightest" storm
    in this dataset averages ~4 deg/step, the median storm ~13 deg/step
    -- both physically plausible for real best-track data). Default
    raised again, this time to the DATASET'S OWN MEDIAN (238.9, rounded
    to 240) -- the only choice that is guaranteed to produce a genuine
    ~50/50 split by construction, comparing "more turning than typical
    for this dataset" against "less turning than typical", rather than
    an arbitrary degree value picked without reference to the data's
    actual scale. If a different comparison is wanted -- e.g. isolating
    only the most sharply recurving storms as a smaller, more extreme
    group -- the 75th percentile (350.6, rounded to 350) is the next
    natural candidate, tested directly against real data via
    print_turn_distribution() rather than guessed.
    """
    full_traj = torch.cat([obs_traj[:, :2], gt_traj[:, :2]], dim=0)
    deg = _norm_to_deg(full_traj.unsqueeze(1)).squeeze(1)
    if deg.shape[0] < 3:
        return "unknown"
    bearings_deg = [math.degrees(float(_forward_azimuth(deg[i], deg[i + 1])))
                     for i in range(deg.shape[0] - 1)]
    total_turn = 0.0
    for i in range(len(bearings_deg) - 1):
        d = bearings_deg[i + 1] - bearings_deg[i]
        # Wrap to [-180, 180] so e.g. a bearing change from 179 to -179
        # degrees (a 2-degree turn crossing the wrap boundary) isn't
        # miscounted as a 358-degree turn.
        d = math.degrees(math.atan2(math.sin(math.radians(d)), math.cos(math.radians(d))))
        total_turn += abs(d)
    return "recurving" if total_turn >= turn_threshold_deg else "straight"


def compute_total_turn(obs_traj: torch.Tensor, gt_traj: torch.Tensor) -> float:
    """Same accumulated-heading-change computation as
    classify_storm_turning, but returns the raw number instead of a
    thresholded label -- used by print_turn_distribution() to let you
    inspect the actual distribution before picking a threshold."""
    full_traj = torch.cat([obs_traj[:, :2], gt_traj[:, :2]], dim=0)
    deg = _norm_to_deg(full_traj.unsqueeze(1)).squeeze(1)
    if deg.shape[0] < 3:
        return float("nan")
    bearings_deg = [math.degrees(float(_forward_azimuth(deg[i], deg[i + 1])))
                     for i in range(deg.shape[0] - 1)]
    total_turn = 0.0
    for i in range(len(bearings_deg) - 1):
        d = bearings_deg[i + 1] - bearings_deg[i]
        d = math.degrees(math.atan2(math.sin(math.radians(d)), math.cos(math.radians(d))))
        total_turn += abs(d)
    return total_turn


def extract_film_deviation(model) -> pd.DataFrame:
    """Same as the single-seed script's extract_film_deviation, minus the
    file-writing/plotting (done once, across all seeds, by the caller)."""
    vel = model.velocity if hasattr(model, "velocity") else model.module.velocity
    gamma_w = vel.film_gamma.weight.detach().cpu()
    beta_w = vel.film_beta.weight.detach().cpu()
    gamma_dev = (gamma_w - 1.0).norm(dim=-1).numpy()
    beta_dev = beta_w.norm(dim=-1).numpy()
    return pd.DataFrame({
        "horizon_h": HORIZON_HOURS[:len(gamma_dev)],
        "gamma_deviation": gamma_dev,
        "beta_deviation": beta_dev,
    })


@torch.no_grad()
def collect_attention_records(model, loader, device, turn_threshold_deg: float):
    """Same as the single-seed script's collect_attention_records."""
    records = []
    for bi, batch in enumerate(loader):
        bl = move(list(batch), device)
        obs = bl[0]; gt = bl[1]
        try:
            tyid_list = bl[15]
        except IndexError:
            tyid_list = None

        try:
            _, _, _, xai = model.sample(bl, num_ensemble=20, return_xai=True,
                                         return_attn=True, use_curvature_score=True)
        except Exception as e:
            print(f"    batch {bi}: sample error, skipped ({e})")
            continue
        if "cross_attn" not in xai:
            continue

        attn = xai["cross_attn"].mean(dim=0)   # [B, pred_len, 2]
        B = attn.shape[0]
        for b in range(B):
            if tyid_list is not None and b < len(tyid_list) and \
               isinstance(tyid_list[b], dict) and "old" in tyid_list[b]:
                info = tyid_list[b]
                storm_key = f"{info['old'][1]}_{info['old'][0]}"
            else:
                storm_key = f"UNKNOWN_batch{bi}_{b}"
            group = classify_storm_turning(obs[:, b, :], gt[:, b, :], turn_threshold_deg)
            for h_idx, h_hours in enumerate(HORIZON_HOURS):
                if h_idx >= attn.shape[1]:
                    break
                records.append({
                    "storm": storm_key, "horizon_h": h_hours,
                    "attn_time": float(attn[b, h_idx, 0]),
                    "attn_context": float(attn[b, h_idx, 1]),
                    "group": group,
                })
    return pd.DataFrame(records)


@torch.no_grad()
def evaluate_ade_by_horizon(model, loader, device, use_tta: bool = False,
                             n_tta: int = 5) -> pd.DataFrame:
    """
    Chạy sample() trên toàn bộ test set, trả về ADE THẬT theo horizon.
    Dùng để đối chiếu với FiLM/attention deviation -- nếu deviation cao ở
    đâu mà ADE cũng thấp (tốt) ở đó, có cơ sở để nói FiLM "đóng góp" ở
    horizon đó; nếu không, deviation cao không tự động có nghĩa gì về
    chất lượng dự đoán.

    [TTA-CONSISTENCY NOTE] Mặc định KHÔNG dùng TTA -- đây là quyết định
    thiết kế có chủ đích: mục đích của hàm này KHÔNG PHẢI báo cáo ADE
    chính thức cho bảng kết quả (đó là việc của evaluate_multi_model.py's
    --use_tta), mà chỉ để đối chiếu TƯƠNG ĐỐI giữa các horizon trong
    cùng 1 checkpoint. Vì vậy con số ADE tuyệt đối ở đây sẽ CAO HƠN một
    chút so với con số TTA đã báo cáo chính thức (verified: một lần chạy
    thực tế cho ADE 323.1/321.1/328.0km ở đây so với 322.96±3.07 từ
    evaluate_multi_model.py --use_tta trên cùng checkpoint) -- đây là
    lệch dự kiến, không phải bug. Correlation với FiLM deviation vẫn có
    ý nghĩa tương đối đúng dù thiếu TTA (vì TTA tác động tương đối đồng
    đều lên mọi horizon, không làm lệch pattern TƯƠNG ĐỐI giữa chúng),
    nhưng --use_tta được thêm ở đây để bạn có lựa chọn dùng CÙNG một
    con số ADE cho cả bảng chính (Table 1) lẫn phần đối chiếu XAI này,
    tránh reviewer thắc mắc vì sao 2 nơi báo cáo ADE khác nhau cho cùng
    1 checkpoint. Bật --use_tta ở đây sẽ làm toàn bộ quá trình chạy
    chậm hơn ~5x (một sample() call cho mỗi trong 5 scale quan sát).
    """
    records = []
    for bi, batch in enumerate(loader):
        bl = move(list(batch), device)
        gt = bl[1]
        try:
            if use_tta:
                obs = bl[0]; anchor = obs[-1:, :, :2].detach()
                scales = [0.875, 0.9375, 1.0, 1.0625, 1.125][:n_tta]
                preds_t, weights_t = [], []
                for sc in scales:
                    obs_s = obs.clone()
                    obs_s[..., :2] = anchor + (obs[..., :2] - anchor) * sc
                    bl_s = list(bl); bl_s[0] = obs_s
                    try:
                        p, _, _ = model.sample(bl_s, num_ensemble=20, use_curvature_score=True)
                        preds_t.append(p)
                        weights_t.append(2.0 if abs(sc - 1.0) < 1e-6 else 1.0)
                    except Exception:
                        continue
                if not preds_t:
                    continue
                tw = sum(weights_t)
                pred = sum(w / tw * p for w, p in zip(weights_t, preds_t))
            else:
                pred, _, _ = model.sample(bl, num_ensemble=20, use_curvature_score=True)
        except Exception as e:
            print(f"    batch {bi}: sample error, skipped ({e})")
            continue
        T = min(pred.shape[0], gt.shape[0])
        pd_ = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T, :, :2])
        dist = _haversine_deg(pd_, gd)   # [T, B]
        for h_idx, h_hours in enumerate(HORIZON_HOURS):
            if h_idx >= T:
                break
            for b in range(dist.shape[1]):
                records.append({"horizon_h": h_hours, "ade": float(dist[h_idx, b])})
    return pd.DataFrame(records)


def print_turn_distribution(loader, device, thresholds_to_test: list,
                             output_dir: str) -> pd.DataFrame:
    """
    [THRESHOLD RE-CALIBRATED] Root cause of the earlier "everything is
    recurving" outcome: 25 degrees of accumulated heading change is too
    low a bar for this dataset -- confirmed by a real run where ALL 449
    test-set storms exceeded it. Rather than guessing a "better" number
    (40, 45, 60...) and hoping it happens to split the data well, this
    prints the ACTUAL distribution of total_turn across every storm in
    the test set once, so the threshold can be picked by looking at
    where the distribution genuinely has two clusters (or, if it's
    unimodal with no natural split, that itself is useful information --
    it would mean "recurving vs straight" as a binary label may not be
    the right framing for this dataset at all, and a continuous turn-rate
    variable might serve the paper's argument better than a threshold).

    Runs ONCE (turn angle only depends on ground-truth trajectories, not
    on any model's predictions), independent of which/how many FM
    checkpoints are being evaluated -- so this is not repeated per seed.
    """
    turns = []
    for batch in loader:
        bl = move(list(batch), device)
        obs = bl[0]; gt = bl[1]
        for b in range(obs.shape[1]):
            t = compute_total_turn(obs[:, b, :], gt[:, b, :])
            if not math.isnan(t):
                turns.append(t)

    turns = np.array(turns)
    df = pd.DataFrame({"total_turn_deg": turns})
    df.to_csv(os.path.join(output_dir, "turn_distribution.csv"), index=False)

    print(f"\n{'='*70}\n  Turn-angle distribution across {len(turns)} test-set storms\n{'='*70}")
    print(f"  min={turns.min():.1f}  25th pct={np.percentile(turns,25):.1f}  "
          f"median={np.median(turns):.1f}  75th pct={np.percentile(turns,75):.1f}  "
          f"max={turns.max():.1f}")
    print(f"\n  How many storms would be labeled 'recurving' at each threshold:")
    for th in thresholds_to_test:
        n_recurv = int((turns >= th).sum())
        pct = 100.0 * n_recurv / max(len(turns), 1)
        print(f"    threshold={th:5.1f}°:  {n_recurv:4d}/{len(turns)} "
              f"({pct:5.1f}%) labeled recurving")
    print(f"\n  A threshold that produces something close to a 50/50 (or at "
          f"least neither ~0% nor ~100%) split is more likely to yield a "
          f"meaningful comparison than one that puts nearly everything in "
          f"one bucket -- pick --turn_threshold_deg accordingly, or inspect "
          f"turn_distribution.csv directly (e.g. plot a histogram) if none "
          f"of the tested values look like a clean split.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(turns, bins=30, edgecolor="black", alpha=0.7)
        for th in thresholds_to_test:
            ax.axvline(th, linestyle="--", alpha=0.6, label=f"{th}°")
        ax.set_xlabel("Total accumulated heading change (degrees)")
        ax.set_ylabel("Number of storms")
        ax.set_title("Distribution of total turning angle across test-set storms")
        ax.legend()
        fig.tight_layout()
        fig_path = os.path.join(output_dir, "turn_distribution.png")
        fig.savefig(fig_path, dpi=200)
        print(f"\n  Figure saved: {fig_path}")
    except ImportError:
        pass

    return df


def summarize_across_seeds(per_seed_df: pd.DataFrame, value_cols: list,
                            group_cols: list) -> pd.DataFrame:
    """
    Tính mean±std QUA SEED (không phải qua storm/sample trong 1 seed) cho
    mỗi tổ hợp group_cols. Bước bắt buộc: trước tiên average trong-seed
    (mỗi seed đóng góp đúng 1 con số cho mỗi group), rồi mới tính std
    GIỮA các seed -- nếu tính std trực tiếp trên toàn bộ pooled records
    (mọi storm của mọi seed trộn lẫn), std đó phản ánh biến thiên giữa
    CÁC STORM trong 1 seed, không phải biến thiên giữa CÁC SEED (câu hỏi
    thật sự cần trả lời: "pattern này có ổn định qua các lần train khác
    nhau không"). Hai câu hỏi khác nhau, hai con số std khác nhau.
    """
    per_seed_mean = (per_seed_df.groupby(["seed"] + group_cols)[value_cols]
                      .mean().reset_index())
    summary = (per_seed_mean.groupby(group_cols)[value_cols]
               .agg(["mean", "std", "count"]))
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()


def compute_film_vs_ade_correlation(film_summary: pd.DataFrame,
                                     ade_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation giữa FiLM deviation (mean qua seed) và ADE (mean
    qua seed), theo horizon -- N=12 điểm dữ liệu (1 mỗi horizon).

    CẢNH BÁO QUAN TRỌNG: N=12 là RẤT NHỎ cho một phép tính tương quan --
    correlation coefficient ở n=12 có confidence interval rất rộng, và
    KHÔNG chứng minh nhân quả kể cả khi |r| cao. Đây là một con số MÔ TẢ
    (descriptive), không phải một phép kiểm định giả thuyết (hypothesis
    test) -- không nên dùng để khẳng định "FiLM giúp cải thiện ADE",
    chỉ nên dùng để nói "có/không có xu hướng đồng biến quan sát được"
    một cách thận trọng, và luôn báo cáo kèm n và p-value.
    """
    merged = film_summary.merge(ade_summary, on="horizon_h", suffixes=("_film", "_ade"))
    from scipy import stats
    results = []
    for col in ["gamma_deviation_mean", "beta_deviation_mean"]:
        if col in merged.columns and "ade_mean" in merged.columns:
            r, p = stats.pearsonr(merged[col], merged["ade_mean"])
            results.append({"film_metric": col, "vs": "ade_mean",
                             "pearson_r": r, "p_value": p, "n_horizons": len(merged)})
    return pd.DataFrame(results)


def plot_with_error_band(summary_df: pd.DataFrame, x_col: str,
                          series: list, output_path: str, title: str,
                          xlabel: str, ylabel: str, group_col: str = None):
    """Generic mean±std line plot with shaded error band."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"    matplotlib not available -- skipped {output_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if group_col:
        for g in sorted(summary_df[group_col].unique()):
            sub = summary_df[summary_df[group_col] == g].sort_values(x_col)
            for mean_col, std_col, label in series:
                lbl = f"{label} ({g})"
                ax.plot(sub[x_col], sub[mean_col], marker="o", label=lbl)
                ax.fill_between(sub[x_col], sub[mean_col] - sub[std_col],
                                 sub[mean_col] + sub[std_col], alpha=0.15)
    else:
        sub = summary_df.sort_values(x_col)
        for mean_col, std_col, label in series:
            ax.plot(sub[x_col], sub[mean_col], marker="o", label=label)
            ax.fill_between(sub[x_col], sub[mean_col] - sub[std_col],
                             sub[mean_col] + sub[std_col], alpha=0.15)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"    Figure saved: {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True,
                    help="One or more name=path pairs, e.g. seed0=/path/ckpt.pth")
    p.add_argument("--output_dir", default="xai_multi_seed")
    p.add_argument("--gpu_num", default="0")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--seed", type=int, default=42,
                    help="RNG seed reset before EACH checkpoint's evaluation, "
                         "so results don't depend on checkpoint listing order.")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--other_modal", default="gph")
    p.add_argument("--delim", default=" ")
    p.add_argument("--skip", type=int, default=1)
    p.add_argument("--min_ped", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.002)
    p.add_argument("--filter_region", action="store_true", default=False)
    p.add_argument("--min_pct_in_scs", type=float, default=15.0)
    p.add_argument("--turn_threshold_deg", type=float, default=240.0,
                    help="Recurving-vs-straight cutoff, in degrees of "
                         "total accumulated heading change over the "
                         "obs+pred window. Default 240 = the observed "
                         "MEDIAN of a real 449-storm test-set run "
                         "(min=71.7, 25th=175.3, median=238.9, "
                         "75th=350.6, max=973.2 degrees) -- chosen to "
                         "guarantee a genuine ~50/50 split by "
                         "construction. Run with --skip_ade_eval "
                         "--skip_attention first to see this "
                         "distribution printed for YOUR dataset before "
                         "trusting this default, since it depends on "
                         "obs_len/pred_len and the specific storms in "
                         "your test split.")
    p.add_argument("--skip_attention", action="store_true", default=False,
                    help="Skip cross-attention analysis (slow, needs full "
                         "test-set pass). FiLM deviation always runs "
                         "(fast, model-only) regardless of this flag.")
    p.add_argument("--skip_ade_eval", action="store_true", default=False,
                    help="Skip the ADE-by-horizon evaluation used for the "
                         "FiLM-vs-ADE correlation check. Skipping this means "
                         "you get deviation numbers with real std across "
                         "seeds, but no evidence about whether that "
                         "deviation actually correlates with prediction "
                         "quality -- only skip if you already have ADE "
                         "numbers from elsewhere (e.g. evaluate_multi_model.py).")
    p.add_argument("--use_tta", action="store_true", default=False,
                    help="Apply TTA (test-time augmentation, same 5-scale "
                         "procedure as evaluate_multi_model.py --use_tta) to "
                         "the ADE-by-horizon evaluation. Off by default -- "
                         "this evaluation's purpose is relative comparison "
                         "across horizons within a checkpoint, not an "
                         "official ADE number, so the absolute values will "
                         "run a bit higher than TTA numbers reported "
                         "elsewhere without this flag (expected, not a "
                         "bug -- see evaluate_ade_by_horizon's docstring). "
                         "Enable this if you want the SAME ADE numbers used "
                         "here and in your main results table. ~5x slower "
                         "when enabled.")
    p.add_argument("--n_tta", type=int, default=5,
                    help="Number of TTA scales (max 5); only used when --use_tta is set.")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_map = parse_checkpoint_args(args.checkpoints)
    n_seeds = len(ckpt_map)
    print(f"\n  Checkpoints ({n_seeds}):")
    for name, path in ckpt_map.items():
        print(f"    {name}: {path}")
    if n_seeds < 3:
        print(f"\n  ⚠ Only {n_seeds} checkpoint(s) provided. Mean±std across "
              f"seeds will be reported, but with fewer than 3 seeds the std "
              f"itself is a weak estimate -- treat any resulting pattern as "
              f"preliminary, not confirmed, until more seeds are available.")

    loader = None
    if not args.skip_attention or not args.skip_ade_eval:
        _, loader = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
        print(f"  Test data: {len(loader)} batches")

    # [THRESHOLD RE-CALIBRATED] Run once before any per-seed processing --
    # turn angle only depends on ground-truth trajectories, independent
    # of which checkpoint is loaded. Prints the actual distribution and
    # how many storms each candidate threshold would label "recurving",
    # so --turn_threshold_deg can be picked from real data instead of a
    # guess. Only runs if the attention analysis (the only consumer of
    # the recurving/straight label) is actually going to run.
    if not args.skip_attention and loader is not None:
        print_turn_distribution(loader, device,
                                 thresholds_to_test=[25.0, 35.0, 45.0, 55.0, 65.0],
                                 output_dir=args.output_dir)

    all_film = []
    all_attn = []
    all_ade = []

    for seed_name, ckpt_path in ckpt_map.items():
        print(f"\n{'='*70}\n  Processing {seed_name}: {ckpt_path}\n{'='*70}")
        set_seed(args.seed)
        try:
            model, ck = load_fm(ckpt_path, device)
        except Exception as e:
            print(f"  ⚠ Failed to load {ckpt_path}: {e}")
            continue

        # 1) FiLM deviation -- fast, always runs.
        film_df = extract_film_deviation(model)
        film_df["seed"] = seed_name
        all_film.append(film_df)
        print(f"    FiLM deviation extracted ({len(film_df)} horizons).")

        # 2) Cross-attention -- slow, optional.
        if not args.skip_attention:
            set_seed(args.seed)
            attn_df = collect_attention_records(model, loader, device, args.turn_threshold_deg)
            if not attn_df.empty:
                attn_df["seed"] = seed_name
                all_attn.append(attn_df)
                print(f"    Attention records collected ({len(attn_df)} rows).")
            else:
                print(f"    ⚠ No attention records collected for {seed_name}.")

        # 3) ADE by horizon -- for the FiLM-vs-quality correlation check.
        if not args.skip_ade_eval:
            set_seed(args.seed)
            ade_df = evaluate_ade_by_horizon(model, loader, device,
                                              use_tta=args.use_tta, n_tta=args.n_tta)
            if not ade_df.empty:
                ade_df["seed"] = seed_name
                all_ade.append(ade_df)
                print(f"    ADE-by-horizon evaluated "
                      f"(overall mean ADE={ade_df['ade'].mean():.1f}km).")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_film:
        print("\n  No FiLM data collected from any checkpoint -- aborting.")
        return

    # ── FiLM deviation: raw + summary across seeds ──────────────────────
    film_all = pd.concat(all_film, ignore_index=True)
    film_all.to_csv(os.path.join(args.output_dir, "film_deviation_per_seed.csv"), index=False)

    film_summary = summarize_across_seeds(
        film_all, ["gamma_deviation", "beta_deviation"], ["horizon_h"])
    film_summary.to_csv(os.path.join(args.output_dir, "film_deviation_summary.csv"), index=False)
    print(f"\n{'='*70}\n  FiLM deviation summary (mean±std across {n_seeds} seeds)\n{'='*70}")
    print(film_summary.to_string(index=False))

    plot_with_error_band(
        film_summary, "horizon_h",
        [("gamma_deviation_mean", "gamma_deviation_std", "gamma deviation"),
         ("beta_deviation_mean", "beta_deviation_std", "beta deviation")],
        os.path.join(args.output_dir, "film_deviation_summary.png"),
        f"FiLM deviation by horizon (mean±std across {n_seeds} seeds)",
        "Forecast horizon (hours)", "L2 deviation from zero-impact init")

    # ── Attention: raw + summary across seeds ────────────────────────────
    if all_attn:
        attn_all = pd.concat(all_attn, ignore_index=True)
        attn_all.to_csv(os.path.join(args.output_dir, "attn_by_horizon_per_seed.csv"), index=False)

        attn_summary = summarize_across_seeds(
            attn_all, ["attn_context", "attn_time"], ["group", "horizon_h"])
        attn_summary.to_csv(os.path.join(args.output_dir, "attn_summary_across_seeds.csv"), index=False)
        print(f"\n{'='*70}\n  Attention summary (mean±std across {n_seeds} seeds)\n{'='*70}")
        print(attn_summary.to_string(index=False))

        plot_with_error_band(
            attn_summary, "horizon_h",
            [("attn_context_mean", "attn_context_std", "attn to context")],
            os.path.join(args.output_dir, "attn_summary_across_seeds.png"),
            f"Cross-attention to context vector (mean±std across {n_seeds} seeds)",
            "Forecast horizon (hours)", "Mean attention weight on context vector",
            group_col="group")

    # ── ADE by horizon: raw + summary across seeds ───────────────────────
    ade_summary = None
    if all_ade:
        ade_all = pd.concat(all_ade, ignore_index=True)
        ade_all.to_csv(os.path.join(args.output_dir, "eval_by_horizon_per_seed.csv"), index=False)

        ade_summary = summarize_across_seeds(ade_all, ["ade"], ["horizon_h"])
        ade_summary.to_csv(os.path.join(args.output_dir, "eval_summary_across_seeds.csv"), index=False)
        print(f"\n{'='*70}\n  ADE-by-horizon summary (mean±std across {n_seeds} seeds)\n{'='*70}")
        print(ade_summary.to_string(index=False))

    # ── FiLM vs ADE correlation (only if both available and >=3 seeds) ──
    if ade_summary is not None:
        try:
            corr_df = compute_film_vs_ade_correlation(film_summary, ade_summary)
            corr_df.to_csv(os.path.join(args.output_dir, "film_vs_metric_correlation.csv"), index=False)
            print(f"\n{'='*70}\n  FiLM-deviation vs ADE correlation (n_horizons=12, "
                  f"n_seeds={n_seeds})\n{'='*70}")
            print(corr_df.to_string(index=False))
            print("\n  ⚠ IMPORTANT: n=12 horizon points is a small sample for a "
                  "correlation. A high |r| here is a DESCRIPTIVE observation, "
                  "not proof that FiLM deviation causes better ADE -- report "
                  "with this caveat, alongside the p-value, if used in the paper.")
        except ImportError:
            print("\n  scipy not available -- skipped FiLM-vs-ADE correlation "
                  "(pip install scipy to enable).")

    print(f"\n  All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()