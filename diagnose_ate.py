"""
diagnose_ate_deep.py
=====================
Phân tích sâu bias tốc độ của velocity network — mở rộng diagnose_ate.py.

Trả lời:
  Q3: Bias có khác nhau giữa storm CHẬM / VỪA / NHANH không?
      (Nếu network bị kéo về "tốc độ trung bình dataset" thay vì tốc độ
      riêng từng storm, storm chậm sẽ bị over-predict mạnh hơn storm nhanh)
  Q4: Bias có tăng dần theo horizon không? (per-step, không chỉ trung bình)
  Q5: hard_score có tương quan với độ lớn của bias không?
      (Nếu storm "khó" theo hard_score cũng là storm bias mạnh nhất,
      đó là target tốt cho per-storm correction)

USAGE (giống diagnose_ate.py, cùng thư mục):

!python /kaggle/working/Tropical_Typhoon_Architecture/diagnose_ate_deep.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --split test \
    --gpu 0 \
    --n_ensemble 20 \
    --fm_checkpoint /kaggle/input/datasets/gmnguynhng/new-checkpoint/best_model_fm_seed0.pth \
    --n_batches 8
"""
from __future__ import annotations
import sys, os, argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _step_speeds_kmh, hard_score_from_obs,
)


def load_fm(checkpoint: str, device):
    ck = torch.load(checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg") or {}
    model = TCFlowMatching(**model_cfg).to(device)
    state = ck.get("model", ck.get("model_state"))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


@torch.no_grad()
def diagnose_deep(model, loader, device, n_batches: int = 8, K: int = 20):
    model.eval()

    per_step_ratio = []   # list of [T, B] tensors across batches x K draws

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        bl = move(list(batch), device)
        obs_traj = bl[0]
        T_obs, B, _ = obs_traj.shape

        h_score = hard_score_from_obs(obs_traj[:, :, :2],
                                       weight_logits=model.hard_score_weight_logits)
        obs_norm = obs_traj[:, :, :2]
        last_obs = obs_traj[-1, :, :2]
        t0 = torch.zeros(B, device=device)
        cond = model.encoder(bl, hard_score=h_score)

        obs_deg = _norm_to_deg(obs_norm)
        obs_spd_mu = _step_speeds_kmh(obs_deg).mean(0)  # [B]

        for _ in range(K):
            x_rel = torch.randn(B, model.pred_len, 2, device=device) * model.sigma_inference
            v = model.velocity(x_rel, t0, cond)
            x_rel = x_rel + v
            x_abs = model._from_relative(x_rel, last_obs)      # [B, T, 2]
            pred_deg = _norm_to_deg(x_abs.permute(1, 0, 2))    # [T, B, 2]

            last_deg = obs_deg[-1]
            pts = torch.cat([last_deg.unsqueeze(0), pred_deg], 0)  # [T+1, B, 2]
            pred_step_spd = _step_speeds_kmh(pts)                   # [T, B]

            ratio_tb = pred_step_spd / obs_spd_mu.clamp(min=1.0).unsqueeze(0)
            per_step_ratio.append(ratio_tb.cpu())

        print(f"  batch {i+1}/{n_batches} done (B={B})")

    all_ratio_tb = torch.cat(per_step_ratio, dim=1)  # [T, total_candidates]

    print()
    print("=" * 70)
    print("Q4 — Speed ratio TRUNG BÌNH theo từng HORIZON (không chỉ tổng)")
    print("=" * 70)
    T = all_ratio_tb.shape[0]
    for t in range(T):
        r = all_ratio_tb[t]
        print(f"  step {t:2d} ({(t+1)*6:3d}h): mean={r.mean():.4f}  median={r.median():.4f}  "
              f"std={r.std():.4f}  %>1.15={100*(r>1.15).float().mean():.1f}%  "
              f"%<0.85={100*(r<0.85).float().mean():.1f}%")

    # ── Q3 & Q5: per-storm aggregation, second pass over loader ──
    print()
    print("=" * 70)
    print("Q3/Q5 — Bias trung bình mỗi storm (K candidates) vs obs_speed / hard_score")
    print("=" * 70)

    storm_obs_speed = []
    storm_hard_score = []
    storm_mean_ratio = []

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        bl = move(list(batch), device)
        obs_traj = bl[0]
        T_obs, B, _ = obs_traj.shape

        h_score = hard_score_from_obs(obs_traj[:, :, :2],
                                       weight_logits=model.hard_score_weight_logits)
        obs_norm = obs_traj[:, :, :2]
        last_obs = obs_traj[-1, :, :2]
        t0 = torch.zeros(B, device=device)
        cond = model.encoder(bl, hard_score=h_score)

        obs_deg = _norm_to_deg(obs_norm)
        obs_spd_mu = _step_speeds_kmh(obs_deg).mean(0)  # [B]

        ratios_this_batch = []   # [K, B]
        for _ in range(K):
            x_rel = torch.randn(B, model.pred_len, 2, device=device) * model.sigma_inference
            v = model.velocity(x_rel, t0, cond)
            x_rel = x_rel + v
            x_abs = model._from_relative(x_rel, last_obs)
            pred_deg = _norm_to_deg(x_abs.permute(1, 0, 2))

            last_deg = obs_deg[-1]
            pts = torch.cat([last_deg.unsqueeze(0), pred_deg], 0)
            pred_spd_mu = _step_speeds_kmh(pts).mean(0)  # [B], avg over T

            ratio_b = pred_spd_mu / obs_spd_mu.clamp(min=1.0)
            ratios_this_batch.append(ratio_b.cpu())

        ratios_this_batch = torch.stack(ratios_this_batch, 0)  # [K, B]
        storm_mean_ratio.append(ratios_this_batch.mean(0))     # [B] mean over K
        storm_obs_speed.append(obs_spd_mu.cpu())
        storm_hard_score.append(h_score.cpu())

    storm_obs_speed  = torch.cat(storm_obs_speed).numpy()
    storm_hard_score = torch.cat(storm_hard_score).numpy()
    storm_mean_ratio = torch.cat(storm_mean_ratio).numpy()

    slow = storm_obs_speed < 8.0
    med  = (storm_obs_speed >= 8.0) & (storm_obs_speed < 15.0)
    fast = storm_obs_speed >= 15.0

    def _stat(mask, name):
        if mask.sum() == 0:
            print(f"  {name}: n=0")
            return
        vals = storm_mean_ratio[mask]
        print(f"  {name:12s}: n={mask.sum():3d}  mean_ratio={vals.mean():.4f}  "
              f"median={np.median(vals):.4f}  std={vals.std():.4f}")

    print("\n  -- Theo storm speed category (XAI-9 thresholds) --")
    _stat(slow, "SLOW(<8kmh)")
    _stat(med,  "MED(8-15)")
    _stat(fast, "FAST(>=15)")

    corr_speed = np.corrcoef(storm_obs_speed, storm_mean_ratio)[0, 1]
    corr_hard  = np.corrcoef(storm_hard_score, storm_mean_ratio)[0, 1]
    print(f"\n  corr(obs_speed, ratio)  = {corr_speed:+.4f}")
    print(f"  corr(hard_score, ratio) = {corr_hard:+.4f}")

    print()
    print("Diễn giải:")
    print("  - Nếu corr(obs_speed, ratio) ÂM MẠNH (vd < -0.3): network kéo mọi storm")
    print("    về TỐC ĐỘ TRUNG BÌNH dataset — storm chậm bị over-predict NHIỀU HƠN")
    print("    storm nhanh. Đây là 'regression to the mean' kinh điển.")
    print("    => Fix: thêm obs_speed như 1 tín hiệu MẠNH HƠN trong conditioning,")
    print("    hoặc calibration theo obs_speed thay vì chỉ theo horizon.")
    print("  - Nếu corr(hard_score, ratio) dương rõ: storm khó cũng là storm bias")
    print("    mạnh nhất => hard_score đã sẵn là tín hiệu tốt để làm per-storm scale.")
    print("  - Nếu Q4 cho thấy %>1.15 TĂNG DẦN theo horizon: bias tích lũy theo")
    print("    autoregressive-like error propagation trong chính velocity field,")
    print("    không phải lỗi 1-shot tại t=0 riêng lẻ.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_ensemble", type=int, default=20)
    p.add_argument("--fm_checkpoint", required=True)
    p.add_argument("--n_batches", type=int, default=8)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Loading test data...")
    import argparse as _ap
    _loader_args = _ap.Namespace(
        dataset_root=args.dataset_root, obs_len=8, pred_len=12,
        batch_size=64, num_workers=2, test_year=None, skip=1,
        min_ped=1, threshold=0.002,
    )
    _, loader = data_loader(_loader_args,
                             {"root": args.dataset_root, "type": args.split},
                             test=(args.split != "train"))
    print(f"Data: {len(loader)} batches")

    print(f"\nLoading FM: {args.fm_checkpoint}")
    model = load_fm(args.fm_checkpoint, device)

    diagnose_deep(model, loader, device, n_batches=args.n_batches, K=args.n_ensemble)


if __name__ == "__main__":
    main()