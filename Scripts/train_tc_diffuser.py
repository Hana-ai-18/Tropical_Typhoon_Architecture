from __future__ import annotations

"""
Train TC-Diffuser baseline (velocity-space DDPM, Trajectron++ lineage) with
the SAME multi-modal encoder (PaperEncoder: FNO3D + Mamba + Env_net) used by
ST-Trans / paper LSTM-GRU-RNN / MMSTN / Phys-Diff baselines, for a fair
comparison.

Epoch loop / early-stopping / metrics-CSV / checkpoint format are IDENTICAL
in structure to the other three train scripts. The diffusion mechanics
themselves (VarianceSchedule, velocity-space DDPM loss with the original
per-step Wt weighting, ConcatSquashLinear-conditioned Transformer denoiser,
SingleIntegrator-style velocity integration) are kept faithful to
TC-Diffuser's own models/diffusion.py -- see Model/tc_diffuser_model.py
docstring for the full list of what was kept verbatim vs. adapted.

IMPORTANT EVALUATION NOTE: TC-Diffuser's evaluate()/sample() uses the
ORIGINAL repo's best-of-6 sampling (stochastic multi-sample, keep the
closest to ground truth), unlike the other three baselines which report a
single-sample ADE. This is a defining property of TC-Diffuser's design
(diffusion models are naturally multi-modal/stochastic), not an incidental
training detail, so it is kept faithful rather than flattened to 1 sample.
The metrics CSV includes a `sampling` column (`best_of_6`) on every row so
this is never silently conflated with the other baselines' numbers when
comparing results.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import random
import csv
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from Model.data.loader_training import data_loader
from Model.tc_diffuser_model import TCDiffuser
from Model.paper_baseline_model import (
    haversine_km, _norm_to_deg, _ate_cte_tensors,
    HORIZON_STEPS,
)


def move(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in x.items()}
    return out


def make_subset_loader(dataset, subset_size, batch_size, collate_fn, seed=42):
    n   = len(dataset)
    rng = random.Random(seed)
    idx = rng.sample(range(n), min(subset_size, n))
    return DataLoader(Subset(dataset, idx),
                      batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, num_workers=0, drop_last=False)


def save_metrics_csv(row: dict, csv_path: str):
    write_hdr = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_hdr:
            w.writeheader()
        w.writerow(row)


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, float) and not np.isnan(v) else "nan"


@torch.no_grad()
def evaluate(model: TCDiffuser, loader, device) -> dict:
    """ADE/FDE/ATE/CTE per horizon, SAME metric definitions as the other
    three baselines. Uses best-of-`model.best_k` sampling -- see module
    docstring; this is logged via the `sampling` column in the CSV, not
    hidden."""
    model.eval()

    all_ade, all_fde = [], []
    ade_buf = {h: [] for h in HORIZON_STEPS}
    ate_buf = {h: [] for h in HORIZON_STEPS}
    cte_buf = {h: [] for h in HORIZON_STEPS}
    all_ate_abs, all_cte_abs = [], []

    for batch in loader:
        bl         = move(list(batch), device)
        pred, _, _ = model.sample(bl)      # best-of-k internally
        gt         = bl[1]
        T          = min(pred.shape[0], gt.shape[0])

        pred_d = _norm_to_deg(pred[:T])
        gt_d   = _norm_to_deg(gt[:T])
        dist   = haversine_km(pred_d, gt_d)

        ate, cte = _ate_cte_tensors(pred[:T], gt[:T])

        all_ade.extend(dist.mean(0).tolist())
        all_fde.extend(dist[-1].tolist())
        all_ate_abs.extend(ate.abs().mean(0).tolist())
        all_cte_abs.extend(cte.abs().mean(0).tolist())

        for h, s in HORIZON_STEPS.items():
            if s < T:
                ade_buf[h].extend(dist[s].tolist())
                ate_buf[h].extend(ate[s].abs().tolist())
                cte_buf[h].extend(cte[s].abs().tolist())

    def _mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    result = dict(
        ADE     = _mean(all_ade),
        FDE     = _mean(all_fde),
        ATE_abs = _mean(all_ate_abs),
        CTE_abs = _mean(all_cte_abs),
    )
    for h in HORIZON_STEPS:
        result[f"{h}h"]         = _mean(ade_buf[h])
        result[f"ATE_abs_{h}h"] = _mean(ate_buf[h])
        result[f"CTE_abs_{h}h"] = _mean(cte_buf[h])

    return result


def run_test_evaluation(model, ckpt_path: str, args, device,
                        collate_fn, csv_path: str):
    print("\n" + "=" * 70)
    print("  TEST SET EVALUATION  (TC-Diffuser, best-of-{} sampling)".format(args.best_k))
    print("=" * 70)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}"
          f"  (best val ADE = {ckpt.get('best_ade', float('nan')):.1f} km)")

    test_dataset, test_loader = data_loader(
        args, {"root": args.dataset_root, "type": "test"}, test=True)
    print(f"  test : {len(test_dataset)} sequences  ({len(test_loader)} batches)")

    metrics = evaluate(model, test_loader, device)

    print(f"\n  {'Metric':<20} {'Value (km)':>12}")
    print(f"  {'-'*34}")
    for key, val in metrics.items():
        print(f"  {key:<20} {_fmt(val):>12}")
    print(f"  NOTE: best-of-{args.best_k} sampling (see module docstring) --")
    print(f"        not directly comparable to single-sample ADE of other baselines.")

    row = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "split": "test", "model_type": "TCDiffuser",
           "sampling": f"best_of_{args.best_k}"}
    row.update({k: _fmt(v) for k, v in metrics.items()})
    save_metrics_csv(row, csv_path)
    print(f"\n  Test metrics saved → {csv_path}")
    print("=" * 70)
    return metrics


# ══════════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train TC-Diffuser baseline (velocity-space DDPM) with shared multi-modal encoder")

    p.add_argument("--dataset_root", default="TCND_vn",  type=str)
    p.add_argument("--obs_len",      default=8,          type=int)
    p.add_argument("--pred_len",     default=12,         type=int)

    # TC-Diffuser architecture (defaults match the original repo's
    # models/autoencoder.py AutoEncoder.__init__: context_dim=256,
    # tf_layer=config.tf_layer, num_steps=100, beta_T=5e-2, mode='linear')
    p.add_argument("--unet_in_ch",  default=13,   type=int)
    p.add_argument("--context_dim", default=256,  type=int)
    p.add_argument("--tf_layer",    default=3,    type=int)
    p.add_argument("--num_steps",   default=100,  type=int,
                   help="DDPM num diffusion steps, matches original default")
    p.add_argument("--beta_1",      default=1e-4, type=float)
    p.add_argument("--beta_T",      default=5e-2, type=float)
    p.add_argument("--var_mode",    default="linear", type=str,
                   choices=["linear", "cosine"])
    p.add_argument("--dt",          default=1.0,  type=float,
                   help="Integration timestep for velocity->position "
                        "(SingleIntegrator convention); 1.0 = velocity is "
                        "already in per-step normalized-coordinate units.")
    p.add_argument("--best_k",      default=6,    type=int,
                   help="Best-of-k sampling for evaluation, matches "
                        "original repo's tc_diffuser.py train() which uses "
                        "sample=6, bestof=True.")
    p.add_argument("--sample_steps_stride", default=5, type=int,
                   help="Stride for the DDPM reverse loop (step= in the "
                        "original sample()); higher = faster but coarser "
                        "sampling. 1 = full num_steps reverse steps.")

    # Training infra (epoch framework matching the other three baselines)
    p.add_argument("--num_epochs",   default=1200,       type=int)
    p.add_argument("--batch_size",   default=90,         type=int,
                   help="Original TC-Diffuser repo uses batch_size=256, but "
                        "that was with a much lighter Trajectron++ encoder. "
                        "With the shared FNO3D+Mamba+Env_net encoder (heavier "
                        "than the original), 90 matches the convention used "
                        "by train_st_trans.py/train_paper_baseline.py for a "
                        "consistent, Kaggle-safe comparison across all baselines.")
    p.add_argument("--lr",           default=1e-4,       type=float)
    p.add_argument("--weight_decay", default=0.0,        type=float,
                   help="Original repo's optimizer uses no weight decay "
                        "(plain Adam), kept as default here.")
    p.add_argument("--grad_clip",    default=1.0,        type=float)
    p.add_argument("--lr_gamma",     default=0.98,        type=float,
                   help="Original repo uses ExponentialLR(gamma=0.98) "
                        "stepped once per epoch -- kept faithful.")
    p.add_argument("--patience",     default=100,        type=int)
    p.add_argument("--min_epochs",   default=50,         type=int)
    p.add_argument("--val_freq",     default=5,          type=int)
    p.add_argument("--val_subset",   default=300,        type=int,
                   help="Smaller than the non-diffusion baselines' default "
                        "(600) since best-of-6 DDPM sampling is heavier "
                        "per-sample; raise if your GPU/time budget allows.")
    p.add_argument("--num_workers",  default=2,          type=int)

    p.add_argument("--test_at_end",  action="store_true",
                   help="Đánh giá trên tập test sau khi training xong")

    p.add_argument("--output_dir",   default="runs/tc_diffuser", type=str)
    p.add_argument("--metrics_csv",  default="metrics.csv",      type=str)
    p.add_argument("--gpu_num",      default="0",                type=str)
    p.add_argument("--seed",         default=42,  type=int,
                   help="Random seed. Run 3-5 seeds for ESWA mean±std reporting, "
                        "same convention as train_flowmatching.py.")

    # DataLoader compat (same as the other train scripts)
    p.add_argument("--delim",        default=" ")
    p.add_argument("--skip",         default=1,   type=int)
    p.add_argument("--min_ped",      default=1,   type=int)
    p.add_argument("--threshold",    default=0.002, type=float)
    p.add_argument("--filter_region",  action="store_true", default=False)
    p.add_argument("--min_pct_in_scs", default=15.0, type=float)
    p.add_argument("--other_modal",  default="gph")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main(args):
    if args.seed != 42:
        args.output_dir = f"{args.output_dir}_seed{args.seed}"

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"(CUDA {torch.version.cuda}, {torch.cuda.device_count()} visible)")
    else:
        print(f"  ⚠ No GPU detected — training on CPU will be MUCH slower.")
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
    best_ckpt   = os.path.join(args.output_dir, "best_model.pth")

    print("=" * 70)
    print(f"  TC-DIFFUSER BASELINE  (velocity-space DDPM, shared multi-modal encoder)")
    print(f"  Encoder: PaperEncoder (FNO3D + Mamba + Env_net)  ← same as ST-Trans/LSTM/MMSTN/PhysDiff")
    print(f"  context_dim={args.context_dim}  tf_layer={args.tf_layer}"
          f"  num_steps={args.num_steps}  var_mode={args.var_mode}")
    print(f"  best_k(eval)={args.best_k}  sample_stride={args.sample_steps_stride}")
    print(f"  ⚠ NOTE: evaluate() uses best-of-{args.best_k} sampling (original repo's "
          f"own design), unlike the other three baselines' single-sample ADE.")
    print(f"  Metrics: ADE / ATE / CTE @ 12h / 24h / 48h / 72h")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────────
    train_dataset, train_loader = data_loader(
        args, {"root": args.dataset_root, "type": "train"}, test=False)
    val_dataset, val_loader = data_loader(
        args, {"root": args.dataset_root, "type": "val"}, test=True)

    from Model.data.trajectoriesWithMe_unet_training import seq_collate
    val_sub_loader = make_subset_loader(
        val_dataset, args.val_subset, args.batch_size, seq_collate)

    print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
    print(f"  val   : {len(val_dataset)} seq")

    # ── Model ─────────────────────────────────────────────────────────────
    model = TCDiffuser(
        obs_len=args.obs_len, pred_len=args.pred_len, unet_in_ch=args.unet_in_ch,
        context_dim=args.context_dim, tf_layer=args.tf_layer,
        num_steps=args.num_steps, beta_T=args.beta_T, beta_1=args.beta_1,
        var_mode=args.var_mode, dt=args.dt,
        best_k=args.best_k, sample_steps_stride=args.sample_steps_stride,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params : {n_params:,}")

    # ── Optimizer + Scheduler (faithful to original: plain Adam +
    # ExponentialLR(gamma=0.98) stepped once per epoch) ────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)

    best_ade     = float("inf")
    patience_cnt = 0
    train_start  = time.perf_counter()

    print("=" * 70)
    print(f"  TRAINING  ({len(train_loader)} steps/epoch)")
    print("=" * 70)

    for epoch in range(args.num_epochs):
        model.train()
        sum_loss = 0.0
        t0 = time.perf_counter()

        for i, batch in enumerate(train_loader):
            bl   = move(list(batch), device)
            bd   = model.get_loss_breakdown(bl)
            loss = bd["total"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            sum_loss += loss.item()

            if i % 30 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
                      f"  diff_loss={bd.get('diffusion_loss', 0):.4f}"
                      f"  quickADE~{bd.get('ADE', float('nan')):.1f}km"
                      f"  lr={lr:.2e}")

        avg_train = sum_loss / len(train_loader)
        scheduler.step()

        # ── Val loss (diffusion loss only, cheap — no sampling) ───────────
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                bl_v = move(list(batch), device)
                bd_v = model.get_loss_breakdown(bl_v)
                val_loss += bd_v["total"].item()
                n_val += 1
        avg_val = val_loss / max(n_val, 1)

        ep_t = time.perf_counter() - t0
        print(f"  Epoch {epoch:>4}  train_loss={avg_train:.4f}"
              f"  val_loss={avg_val:.4f}"
              f"  lr={optimizer.param_groups[0]['lr']:.2e}  t={ep_t:.0f}s")

        # ── ADE + ATE + CTE evaluation (best-of-k DDPM sampling) ───────────
        if epoch % args.val_freq == 0:
            r = evaluate(model, val_sub_loader, device)

            ade12 = r.get("12h",         float("nan"))
            ade24 = r.get("24h",         float("nan"))
            ade48 = r.get("48h",         float("nan"))
            ade72 = r.get("72h",         float("nan"))
            ade   = r.get("ADE",         float("nan"))
            ate12 = r.get("ATE_abs_12h", float("nan"))
            cte12 = r.get("CTE_abs_12h", float("nan"))
            ate72 = r.get("ATE_abs_72h", float("nan"))
            cte72 = r.get("CTE_abs_72h", float("nan"))

            t12 = "🎯" if ade12 < 50  else "❌"
            t24 = "🎯" if ade24 < 100 else "❌"
            t48 = "🎯" if ade48 < 200 else "❌"
            t72 = "🎯" if ade72 < 300 else "❌"

            print(f"  [VAL ep{epoch}] (best-of-{args.best_k})"
                  f"  ADE={ade:.1f}"
                  f"  12h={ade12:.0f}{t12}"
                  f"  24h={ade24:.0f}{t24}"
                  f"  48h={ade48:.0f}{t48}"
                  f"  72h={ade72:.0f}{t72} km")
            print(f"           "
                  f"  ATE@12h={ate12:.1f}  CTE@12h={cte12:.1f}"
                  f"  ATE@72h={ate72:.1f}  CTE@72h={cte72:.1f} km")

            save_metrics_csv({
                "timestamp"      : datetime.now().strftime("%Y%m%d_%H%M%S"),
                "split"          : "val",
                "epoch"          : epoch,
                "model_type"     : "TCDiffuser",
                "sampling"       : f"best_of_{args.best_k}",
                "train_loss"     : _fmt(avg_train),
                "val_loss"       : _fmt(avg_val),
                "ADE_km"         : _fmt(ade),
                "FDE_km"         : _fmt(r.get("FDE", float("nan"))),
                "12h_km"         : _fmt(ade12),
                "24h_km"         : _fmt(ade24),
                "48h_km"         : _fmt(ade48),
                "72h_km"         : _fmt(ade72),
                "ATE_abs_km"     : _fmt(r.get("ATE_abs", float("nan"))),
                "CTE_abs_km"     : _fmt(r.get("CTE_abs", float("nan"))),
                "ATE_abs_12h_km" : _fmt(ate12),
                "CTE_abs_12h_km" : _fmt(cte12),
                "ATE_abs_24h_km" : _fmt(r.get("ATE_abs_24h", float("nan"))),
                "CTE_abs_24h_km" : _fmt(r.get("CTE_abs_24h", float("nan"))),
                "ATE_abs_48h_km" : _fmt(r.get("ATE_abs_48h", float("nan"))),
                "CTE_abs_48h_km" : _fmt(r.get("CTE_abs_48h", float("nan"))),
                "ATE_abs_72h_km" : _fmt(ate72),
                "CTE_abs_72h_km" : _fmt(cte72),
            }, metrics_csv)

            if ade < best_ade:
                best_ade     = ade
                patience_cnt = 0
                torch.save({
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "best_ade"   : best_ade,
                    "model_type" : "TCDiffuser",
                    "paper"      : "TC-Diffuser (velocity-space DDPM, Trajectron++ lineage)",
                    "sampling"   : f"best_of_{args.best_k}",
                    "seed"       : args.seed,
                    "model_cfg"  : {
                        "obs_len": args.obs_len, "pred_len": args.pred_len,
                        "unet_in_ch": args.unet_in_ch,
                        "context_dim": args.context_dim, "tf_layer": args.tf_layer,
                        "num_steps": args.num_steps, "beta_1": args.beta_1,
                        "beta_T": args.beta_T, "var_mode": args.var_mode,
                        "dt": args.dt, "best_k": args.best_k,
                        "sample_steps_stride": args.sample_steps_stride,
                    },
                }, best_ckpt)
                print(f"  ✅ Best ADE {best_ade:.1f} km  (epoch {epoch})  [best-of-{args.best_k}]")
            else:
                patience_cnt += args.val_freq
                print(f"  No improvement {patience_cnt}/{args.patience}"
                      f"  (best={best_ade:.1f} km)")

            if epoch >= args.min_epochs and patience_cnt >= args.patience:
                print(f"  ⛔ Early stop @ epoch {epoch}")
                break

        if epoch % 100 == 0:
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "train_loss" : avg_train,
                "val_loss"   : avg_val,
                "seed"       : args.seed,
            }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

    total_h = (time.perf_counter() - train_start) / 3600
    print("=" * 70)
    print(f"  Model   : TC-Diffuser")
    print(f"  Best ADE: {best_ade:.1f} km  (best-of-{args.best_k} sampling)")
    print(f"  Total   : {total_h:.2f}h")
    print(f"  Metrics : {metrics_csv}")
    print("=" * 70)

    if args.test_at_end and os.path.exists(best_ckpt):
        from Model.data.trajectoriesWithMe_unet_training import seq_collate
        run_test_evaluation(model, best_ckpt, args, device,
                            seq_collate, metrics_csv)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    main(args)