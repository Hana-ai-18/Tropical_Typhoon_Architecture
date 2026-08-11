"""
diagnose_nondeterminism.py
============================
Run this on Kaggle, in the SAME session/process as your evaluate_multi_model.py,
to isolate exactly which component is non-deterministic.

Usage:
    python diagnose_nondeterminism.py \
        --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
        --gru_checkpoint /kaggle/input/datasets/gmnguynhng/new-checkpoint/best_model_gru_seed0.pth \
        --gpu 0

This loads ONE GRU checkpoint, runs evaluate_one_model TWICE in a row
(same process, same seed reset before each), and reports whether ADE
differs between the two runs. If it does, it prints per-batch ADE to help
narrow down WHERE the divergence starts (encoder output vs GRU head vs
metric computation).
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from Model.data.loader_training import data_loader
from Model.paper_baseline_model import PaperBaseline, _norm_to_deg, haversine_km


def set_seed(s=42):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


def load_gru(checkpoint, device):
    ck = torch.load(checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg")
    if model_cfg:
        model = PaperBaseline(**model_cfg).to(device)
    else:
        model = PaperBaseline(model_type="gru", hidden_dim=256, n_layers=3).to(device)
    state = ck.get("model_state", ck.get("model"))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def run_once(model, loader, device, tag):
    set_seed(42)
    per_batch_ade = []
    for bi, batch in enumerate(loader):
        bl = move(list(batch), device)
        gt = bl[1]
        pred, _, _ = model.sample(bl, num_ensemble=1)
        T = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T])
        gd = _norm_to_deg(gt[:T, :, :2])
        dist = haversine_km(pd, gd)
        ade_b = float(dist.mean().item())
        per_batch_ade.append(ade_b)
        print(f"  [{tag}] batch {bi}: ADE={ade_b:.6f}")
    overall = float(np.mean(per_batch_ade))
    print(f"  [{tag}] OVERALL ADE = {overall:.6f}")
    return per_batch_ade, overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--gru_checkpoint", required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import argparse as _ap
    loader_args = _ap.Namespace(
        dataset_root=args.dataset_root, obs_len=8, pred_len=12,
        batch_size=64, num_workers=2, test_year=None, skip=1,
        min_ped=1, threshold=0.002,
    )
    _, loader = data_loader(loader_args, {"root": args.dataset_root, "type": "test"}, test=True)
    print(f"Data: {len(loader)} batches\n")

    print("=" * 70)
    print("RUN 1")
    print("=" * 70)
    model = load_gru(args.gru_checkpoint, device)
    ade1, overall1 = run_once(model, loader, device, "RUN1")

    print("\n" + "=" * 70)
    print("RUN 2 (same process, same model object, re-seeded)")
    print("=" * 70)
    ade2, overall2 = run_once(model, loader, device, "RUN2")

    print("\n" + "=" * 70)
    print("RUN 3 (RELOAD model from checkpoint fresh, re-seeded)")
    print("=" * 70)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model2 = load_gru(args.gru_checkpoint, device)
    ade3, overall3 = run_once(model2, loader, device, "RUN3")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"RUN1 overall ADE: {overall1:.6f}")
    print(f"RUN2 overall ADE: {overall2:.6f}  (diff from RUN1: {overall2-overall1:+.6f})")
    print(f"RUN3 overall ADE: {overall3:.6f}  (diff from RUN1: {overall3-overall1:+.6f})")

    if abs(overall2 - overall1) > 1e-4:
        print("\n⚠ RUN1 vs RUN2 differ (same model object, same process) —")
        print("  this points to a genuinely non-deterministic FORWARD PASS")
        print("  (e.g. a CUDA kernel in FNO3DEncoder/DataEncoder1D_Mamba/Env_net")
        print("  that isn't covered by cudnn.deterministic=True).")
    else:
        print("\n✓ RUN1 == RUN2 (same process) — forward pass itself is deterministic.")

    if abs(overall3 - overall1) > 1e-4:
        print("⚠ RUN1 vs RUN3 differ (fresh reload) — could be reload-time")
        print("  randomness (e.g. missing keys falling back to fresh random init)")
        print("  or confirms the same non-determinism as RUN2 if RUN2 also differed.")
    else:
        print("✓ RUN1 == RUN3 (fresh reload) — loading is deterministic too.")

    # Per-batch divergence detail if RUN1 vs RUN2 differ
    if abs(overall2 - overall1) > 1e-4:
        print("\nPer-batch ADE differences (RUN1 vs RUN2):")
        for i, (a1, a2) in enumerate(zip(ade1, ade2)):
            if abs(a1 - a2) > 1e-5:
                print(f"  batch {i}: RUN1={a1:.6f}  RUN2={a2:.6f}  diff={a2-a1:+.6f}")


if __name__ == "__main__":
    main()