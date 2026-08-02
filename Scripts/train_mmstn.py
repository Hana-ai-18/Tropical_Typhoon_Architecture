from __future__ import annotations

"""
Train MMSTN baseline (Social-GAN-style, Faiaz-lineage) with the SAME
multi-modal encoder (PaperEncoder: FNO3D + Mamba + Env_net) used by
ST-Trans / paper LSTM-GRU-RNN baselines, for a fair comparison.

Epoch loop / early-stopping / metrics-CSV / checkpoint format are IDENTICAL
in structure to train_st_trans.py and train_paper_baseline.py so all
baselines are directly comparable. The GAN mechanics themselves (separate
G/D optimizers, d_steps:g_steps alternation, best_k variety loss, noise
injection, label-smoothed BCE) are kept faithful to MMSTN's own train.py --
see Model/mmstn_model.py docstring for a full list of what was kept
verbatim vs. what was necessarily adapted.
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
from Model.Mmstn_model import MMSTN
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
def evaluate(model: MMSTN, loader, device) -> dict:
    """Identical metric definitions (ADE/FDE/ATE/CTE per horizon) to
    train_st_trans.py / train_paper_baseline.py, so numbers are directly
    comparable across all four baselines."""
    model.eval()

    all_ade, all_fde = [], []
    ade_buf = {h: [] for h in HORIZON_STEPS}
    ate_buf = {h: [] for h in HORIZON_STEPS}
    cte_buf = {h: [] for h in HORIZON_STEPS}
    all_ate_abs, all_cte_abs = [], []

    for batch in loader:
        bl         = move(list(batch), device)
        pred, _, _ = model.sample(bl)
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
    print("  TEST SET EVALUATION  (MMSTN)")
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

    row = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "split": "test", "model_type": "MMSTN"}
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
        description="Train MMSTN baseline (Social-GAN-style) with shared multi-modal encoder")

    p.add_argument("--dataset_root", default="TCND_vn",  type=str)
    p.add_argument("--obs_len",      default=8,          type=int)
    p.add_argument("--pred_len",     default=12,         type=int)

    # MMSTN architecture (defaults match mmstn/train.py where meaningful;
    # embedding_dim/encoder_h_dim/decoder_h_dim are MMSTN's own defaults,
    # unet_in_ch is this project's Data3d channel count)
    p.add_argument("--unet_in_ch",       default=13,   type=int)
    p.add_argument("--embedding_dim",    default=32,   type=int)
    p.add_argument("--encoder_h_dim_g",  default=64,   type=int)
    p.add_argument("--decoder_h_dim_g",  default=64,   type=int)
    p.add_argument("--encoder_h_dim_d",  default=128,  type=int,
                   help="MMSTN's own train.py default is 128 (note: this "
                        "differs from encoder_h_dim_g=64 in the original repo "
                        "-- D and G use different hidden sizes there too).")
    p.add_argument("--mlp_dim",          default=128,  type=int)
    p.add_argument("--num_layers",       default=1,    type=int)
    p.add_argument("--noise_dim",        default=16,   type=int,
                   help="MMSTN noise_dim=(16,) by default")
    p.add_argument("--noise_type",       default="gaussian", type=str,
                   choices=["gaussian", "uniform"])
    p.add_argument("--dropout",          default=0.0,  type=float)
    p.add_argument("--best_k",           default=6,    type=int,
                   help="MMSTN's own default best_k=6 (variety-loss samples)")
    p.add_argument("--l2_loss_weight",   default=1.0,  type=float)

    # GAN training cadence -- MMSTN's own defaults (d_steps=2, g_steps=1)
    p.add_argument("--d_steps", default=2, type=int)
    p.add_argument("--g_steps", default=1, type=int)
    p.add_argument("--g_learning_rate", default=1e-4, type=float)
    p.add_argument("--d_learning_rate", default=1e-4, type=float)
    p.add_argument("--clipping_threshold_g", default=0.0, type=float)
    p.add_argument("--clipping_threshold_d", default=2.0, type=float)

    # Training infra (epoch framework matching the other two baselines)
    p.add_argument("--num_epochs",   default=1200,       type=int)
    p.add_argument("--batch_size",   default=90,         type=int)
    p.add_argument("--weight_decay", default=0.0,        type=float)
    p.add_argument("--patience",     default=100,        type=int)
    p.add_argument("--min_epochs",   default=50,         type=int)
    p.add_argument("--val_freq",     default=5,          type=int)
    p.add_argument("--val_subset",   default=600,        type=int)
    p.add_argument("--num_workers",  default=2,          type=int)

    p.add_argument("--test_at_end",  action="store_true",
                   help="Đánh giá trên tập test sau khi training xong")

    p.add_argument("--output_dir",   default="runs/mmstn", type=str)
    p.add_argument("--metrics_csv",  default="metrics.csv", type=str)
    p.add_argument("--gpu_num",      default="0",           type=str)
    p.add_argument("--seed",         default=42,  type=int,
                   help="Random seed. Run 3-5 seeds for ESWA mean±std reporting, "
                        "same convention as train_flowmatching.py.")

    # DataLoader compat (same as the other two train scripts)
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
    print(f"  MMSTN BASELINE  (Social-GAN style, shared multi-modal encoder)")
    print(f"  Encoder: PaperEncoder (FNO3D + Mamba + Env_net)  ← same as ST-Trans/LSTM")
    print(f"  embedding_dim={args.embedding_dim}  enc_h_g={args.encoder_h_dim_g}"
          f"  dec_h_g={args.decoder_h_dim_g}  noise_dim={args.noise_dim}")
    print(f"  best_k={args.best_k}  d_steps={args.d_steps}  g_steps={args.g_steps}")
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
    model = MMSTN(
        obs_len         = args.obs_len,
        pred_len        = args.pred_len,
        unet_in_ch      = args.unet_in_ch,
        embedding_dim   = args.embedding_dim,
        encoder_h_dim_g = args.encoder_h_dim_g,
        decoder_h_dim_g = args.decoder_h_dim_g,
        encoder_h_dim_d = args.encoder_h_dim_d,
        mlp_dim         = args.mlp_dim,
        num_layers      = args.num_layers,
        noise_dim       = (args.noise_dim,),
        noise_type      = args.noise_type,
        dropout         = args.dropout,
        best_k          = args.best_k,
        l2_loss_weight  = args.l2_loss_weight,
    ).to(device)

    n_params_g = sum(p.numel() for p in model.generator.parameters() if p.requires_grad)
    n_params_d = sum(p.numel() for p in model.discriminator.parameters() if p.requires_grad)
    print(f"  params : G={n_params_g:,}  D={n_params_d:,}")

    # ── Optimizers (separate G/D, faithful to MMSTN train.py) ────────────
    optimizer_g = optim.Adam(model.generator.parameters(),
                             lr=args.g_learning_rate, weight_decay=args.weight_decay)
    optimizer_d = optim.Adam(model.discriminator.parameters(),
                             lr=args.d_learning_rate, weight_decay=args.weight_decay)

    best_ade     = float("inf")
    patience_cnt = 0
    train_start  = time.perf_counter()

    print("=" * 70)
    print(f"  TRAINING  ({len(train_loader)} steps/epoch)")
    print("=" * 70)

    for epoch in range(args.num_epochs):
        model.train()
        sum_g_loss, sum_d_loss = 0.0, 0.0
        n_g_steps, n_d_steps = 0, 0
        t0 = time.perf_counter()

        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        last_g_losses = {}
        last_d_losses = {}

        for i, batch in enumerate(train_loader):
            bl = move(list(batch), device)

            # ── faithful d_steps : g_steps alternation, MMSTN train.py ──
            if d_steps_left > 0:
                d_bd = model.discriminator_step_loss(bl)
                d_loss = d_bd["total"]
                optimizer_d.zero_grad()
                d_loss.backward()
                if args.clipping_threshold_d > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.discriminator.parameters(), args.clipping_threshold_d)
                optimizer_d.step()
                sum_d_loss += d_loss.item()
                n_d_steps  += 1
                last_d_losses = d_bd
                d_steps_left -= 1
            elif g_steps_left > 0:
                g_bd = model.generator_step_loss(bl)
                g_loss = g_bd["total"]
                optimizer_g.zero_grad()
                g_loss.backward()
                if args.clipping_threshold_g > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.generator.parameters(), args.clipping_threshold_g)
                optimizer_g.step()
                sum_g_loss += g_loss.item()
                n_g_steps  += 1
                last_g_losses = g_bd
                g_steps_left -= 1

            if d_steps_left == 0 and g_steps_left == 0:
                d_steps_left = args.d_steps
                g_steps_left = args.g_steps

            if i % 30 == 0:
                dpe_now = last_g_losses.get("ADE", float("nan"))
                print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
                      f"  D_loss={last_d_losses.get('D_data_loss', float('nan')):.4f}"
                      f"  G_l2={last_g_losses.get('G_l2_loss_rel', float('nan')):.4f}"
                      f"  G_adv={last_g_losses.get('G_discriminator_loss', float('nan')):.4f}"
                      f"  ADE~{dpe_now:.1f}km")

        avg_g = sum_g_loss / max(n_g_steps, 1)
        avg_d = sum_d_loss / max(n_d_steps, 1)

        # ── Val loss (G's total loss, used only for logging — early stop
        # and best-checkpoint selection use ADE like the other baselines) ─
        model.eval()
        val_g_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                bl_v = move(list(batch), device)
                bd_v = model.generator_step_loss(bl_v)
                val_g_loss += bd_v["total"].item()
                n_val += 1
        avg_val_g = val_g_loss / max(n_val, 1)

        ep_t = time.perf_counter() - t0
        print(f"  Epoch {epoch:>4}  G_loss={avg_g:.4f}  D_loss={avg_d:.4f}"
              f"  val_G_loss={avg_val_g:.4f}  t={ep_t:.0f}s")

        # ── ADE + ATE + CTE evaluation ────────────────────────────────────
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

            print(f"  [VAL ep{epoch}]"
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
                "model_type"     : "MMSTN",
                "G_loss"         : _fmt(avg_g),
                "D_loss"         : _fmt(avg_d),
                "val_G_loss"     : _fmt(avg_val_g),
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
                    "model_type" : "MMSTN",
                    "paper"      : "MMSTN (Social-GAN style, Faiaz lineage)",
                    "seed"       : args.seed,
                    "model_cfg"  : {
                        "obs_len":         args.obs_len,
                        "pred_len":        args.pred_len,
                        "unet_in_ch":      args.unet_in_ch,
                        "embedding_dim":   args.embedding_dim,
                        "encoder_h_dim_g": args.encoder_h_dim_g,
                        "decoder_h_dim_g": args.decoder_h_dim_g,
                        "encoder_h_dim_d": args.encoder_h_dim_d,
                        "mlp_dim":         args.mlp_dim,
                        "num_layers":      args.num_layers,
                        "noise_dim":       args.noise_dim,
                        "noise_type":      args.noise_type,
                        "dropout":         args.dropout,
                        "best_k":          args.best_k,
                        "l2_loss_weight":  args.l2_loss_weight,
                    },
                }, best_ckpt)
                print(f"  ✅ Best ADE {best_ade:.1f} km  (epoch {epoch})")
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
                "G_loss"     : avg_g,
                "D_loss"     : avg_d,
                "seed"       : args.seed,
            }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

    total_h = (time.perf_counter() - train_start) / 3600
    print("=" * 70)
    print(f"  Model   : MMSTN")
    print(f"  Best ADE: {best_ade:.1f} km")
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