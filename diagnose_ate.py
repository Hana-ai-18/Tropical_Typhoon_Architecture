"""
diagnose_ate.py
================
Chẩn đoán ATE cho FM — dùng ĐÚNG load_fm() và data_loader() như
evaluate_multi_model.py, để đảm bảo model/dataset được load giống hệt
lúc eval thật.

Trả lời 2 câu hỏi:
  Q1: speed_correction_logits có bị kẹt gần trần sigmoid*2.0 không?
  Q2: candidate pool (K candidates, TRƯỚC re-rank) có bias tốc độ hệ
      thống không?

USAGE (chạy trực tiếp trên Kaggle, đặt cạnh evaluate_multi_model.py
trong /kaggle/working/Tropical_Typhoon_Architecture/):

!python /kaggle/working/Tropical_Typhoon_Architecture/diagnose_ate.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --split test \
    --gpu 0 \
    --n_ensemble 5 \
    --fm_checkpoint /kaggle/input/datasets/gmnguynhng/new-checkpoint/best_model_fm_seed0.pth \
    --n_batches 5
"""
from __future__ import annotations
import sys, os, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _step_speeds_kmh, hard_score_from_obs,
)


def load_fm(checkpoint: str, device):
    """Same as evaluate_multi_model.py's load_fm."""
    ck = torch.load(checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg") or {}
    if not model_cfg:
        print(f"  ⚠ FM checkpoint has no model_cfg — using constructor defaults.")
    model = TCFlowMatching(**model_cfg).to(device)
    state = ck.get("model", ck.get("model_state"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ FM load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.eval()
    return model


def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


@torch.no_grad()
def diagnose(model, loader, device, n_batches: int = 5):
    model.eval()

    # ── Q1: speed_correction_logits có kẹt trần không? ──────────────────
    print("=" * 70)
    print("Q1 — speed_correction per horizon (sigmoid*2.0, trần=2.0)")
    print("=" * 70)
    sc = (torch.sigmoid(model.speed_correction_logits) * 2.0).tolist()
    for t, v in enumerate(sc):
        flag = ("  <-- GẦN TRẦN (>1.85)" if v > 1.85
                else "  <-- GẦN SÀN (<0.15)" if v < 0.15 else "")
        print(f"  step {t:2d} ({(t + 1) * 6:3d}h): correction = {v:.4f}{flag}")

    # ── Q2: candidate pool trước re-rank có bias tốc độ không? ───────────
    print()
    print("=" * 70)
    print("Q2 — Speed ratio (pred/obs) TRƯỚC re-rank, toàn bộ K candidates")
    print("=" * 70)

    all_ratios = []
    K = model.n_ensemble

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
            x_abs = model._from_relative(x_rel, last_obs)   # [B, T, 2]
            pred_deg = _norm_to_deg(x_abs.permute(1, 0, 2))  # [T, B, 2]

            last_deg = obs_deg[-1]
            pts = torch.cat([last_deg.unsqueeze(0), pred_deg], 0)
            pred_spd_mu = _step_speeds_kmh(pts).mean(0)  # [B]

            ratio = (pred_spd_mu / obs_spd_mu.clamp(min=1.0)).cpu()
            all_ratios.append(ratio)

        print(f"  batch {i+1}/{n_batches} done (B={B}, K={K})")

    all_ratios = torch.cat(all_ratios)
    print()
    print(f"  n candidates total = {all_ratios.shape[0]}")
    print(f"  mean ratio (pred_speed/obs_speed) = {all_ratios.mean():.4f}")
    print(f"  median ratio                      = {all_ratios.median():.4f}")
    print(f"  std                                = {all_ratios.std():.4f}")
    print(f"  % candidates ratio < 0.85 (CHẬM hơn) = {(all_ratios < 0.85).float().mean()*100:.1f}%")
    print(f"  % candidates ratio > 1.15 (NHANH hơn) = {(all_ratios > 1.15).float().mean()*100:.1f}%")
    print()
    print("Diễn giải:")
    print("  - mean/median lệch rõ khỏi 1.0 + %lệch 1 phía cao (>60%)")
    print("    => BIAS HỆ THỐNG ở candidate pool. Re-rank (nhóm A) sẽ KHÔNG sửa được.")
    print("  - ratio phân tán đều quanh 1.0 (std lớn, không lệch 1 phía)")
    print("    => Network CÓ candidate đúng, re-rank (nhóm A) vẫn có thể giúp.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_ensemble", type=int, default=20)
    p.add_argument("--fm_checkpoint", required=True)
    p.add_argument("--n_batches", type=int, default=5)
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
    model.n_ensemble = args.n_ensemble

    diagnose(model, loader, device, n_batches=args.n_batches)


if __name__ == "__main__":
    main()