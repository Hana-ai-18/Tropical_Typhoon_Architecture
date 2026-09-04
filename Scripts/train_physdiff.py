from __future__ import annotations

"""
Train Phys-Diff baseline (PIGA-augmented DDPM) with the SAME multi-modal
encoder (PaperEncoder: FNO3D + Mamba + Env_net) used by ST-Trans / paper
LSTM-GRU-RNN / MMSTN baselines, for a fair comparison.

Epoch loop / early-stopping / metrics-CSV / checkpoint format are IDENTICAL
in structure to train_st_trans.py / train_paper_baseline.py / train_mmstn.py
so all baselines are directly comparable. The DDPM mechanics themselves
(cosine/linear beta schedule, epsilon-prediction MSE loss with the original
repo's numerical-stability guards, PIGA-augmented transformer denoiser) are
kept faithful to Phys-Diff's own models/ddpm.py and networks/piga.py -- see
Model/phys_diff_model.py docstring for a full list of what was kept
verbatim vs. what was necessarily adapted (2-D-only prediction, PaperEncoder
context, DDIM-strided sampling for affordable per-epoch validation).
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
from Model.physdiff_model import PhysDiff
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
def evaluate(model: PhysDiff, loader, device) -> dict:
    """Identical metric definitions (ADE/FDE/ATE/CTE per horizon) to the
    other three baselines, so numbers are directly comparable. Uses
    DDIM-strided sampling (model.sample_steps) for speed."""
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
    print("  TEST SET EVALUATION  (Phys-Diff)")
    print("=" * 70)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
           "split": "test", "model_type": "PhysDiff"}
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
        description="Train Phys-Diff baseline (PIGA-augmented DDPM) with shared multi-modal encoder")

    p.add_argument("--dataset_root", default="TCND_vn",  type=str)
    p.add_argument("--obs_len",      default=8,          type=int)
    p.add_argument("--pred_len",     default=12,         type=int)

    # Phys-Diff architecture (defaults match configs/config.yaml where
    # meaningful; unet_in_ch is this project's Data3d channel count)
    p.add_argument("--unet_in_ch",   default=13,   type=int)
    p.add_argument("--d_model",      default=128,  type=int)
    p.add_argument("--d_embedding",  default=64,   type=int)
    p.add_argument("--enc_layers",   default=3,    type=int)
    p.add_argument("--enc_heads",    default=4,    type=int)
    p.add_argument("--enc_ff",       default=256,  type=int)
    p.add_argument("--enc_dropout",  default=0.1,  type=float)
    p.add_argument("--dec_layers",   default=3,    type=int)
    p.add_argument("--dec_heads",    default=4,    type=int)
    p.add_argument("--dec_ff",       default=256,  type=int)
    p.add_argument("--dec_dropout",  default=0.1,  type=float)
    p.add_argument("--d_sub",        default=16,   type=int)
    p.add_argument("--gate_mlp_dims", default="64,16,1", type=str,
                   help="comma-separated, matches configs/config.yaml piga.gate_mlp_dims")

    # DDPM schedule (matches configs/config.yaml model.ddpm defaults)
    p.add_argument("--num_timesteps", default=1000,   type=int)
    p.add_argument("--beta_schedule", default="cosine", type=str,
                   choices=["linear", "cosine"])
    p.add_argument("--beta_start",    default=0.0001, type=float)
    p.add_argument("--beta_end",      default=0.02,   type=float)
    p.add_argument("--sample_steps",  default=25,     type=int,
                   help="DDIM-strided reverse steps used for eval/inference "
                        "only (training always uses the full random-timestep "
                        "objective); lower = faster validation on Kaggle T4. "
                        "Table 8-style sweeps in the diffusion/CFM literature "
                        "typically show DPE/SSR plateauing well before 40-50 "
                        "steps for short (pred_len<=12) sequences; 25 keeps "
                        "each validation-time sample ~2x cheaper than the "
                        "previous default of 50 while remaining well past "
                        "the steep early-N accuracy gain (see docstring of "
                        "DDPMScheduler.sample_strided). Raise back to 50+ "
                        "for the FINAL best-checkpoint test-set evaluation "
                        "only (--test_at_end), where wall-clock no longer "
                        "compounds over many epochs.")

    # Loss weights
    p.add_argument("--coord_loss_weight",      default=1.0, type=float)
    p.add_argument("--diffusion_loss_weight",  default=1.0, type=float)

    # Training infra (epoch framework matching the other three baselines)
    p.add_argument("--num_epochs",   default=150,        type=int,
                   help="Original repo's own default (30) was tuned for a "
                        "much larger ERA5/FengWu dataset; this project's "
                        "dataset converges much faster (observed: val_loss "
                        "bottoms out around epoch 7-10, then overfits). 150 "
                        "gives headroom while keeping CosineAnnealingLR's "
                        "T_max=num_epochs meaningful (with T_max=1200, LR "
                        "barely decayed at all within the first 50 epochs "
                        "this project actually needs -- observed lr=9.96e-05 "
                        "at epoch 49 vs. lr=1.00e-04 at epoch 0, effectively "
                        "flat). Increase if your dataset/architecture combo "
                        "needs longer training; this is a hyperparameter "
                        "tuned to this dataset's convergence speed, not a "
                        "change to the DDPM/PIGA algorithm itself.")
    p.add_argument("--batch_size",   default=90,         type=int,
                   help="Original Phys-Diff repo uses batch_size=64 (config.yaml "
                        "training.batch_size), tuned for its own ERA5/FengWu "
                        "encoder. With the shared FNO3D+Mamba+Env_net encoder, "
                        "90 matches the convention used by train_st_trans.py/"
                        "train_paper_baseline.py for a consistent comparison; "
                        "lower it back to 64 if you hit GPU memory limits.")
    p.add_argument("--lr",           default=1e-4,       type=float)
    p.add_argument("--weight_decay", default=1e-4,       type=float)
    p.add_argument("--grad_clip",    default=0.1,        type=float,
                   help="Matches configs/config.yaml training.gradient_clip=0.1 "
                        "(the original repo clips very aggressively).")
    p.add_argument("--patience",     default=30,         type=int,
                   help="With val_freq=5, patience=30 tolerates 6 "
                        "consecutive non-improving validations (~30 epochs) "
                        "before stopping -- enough to ride out normal noise "
                        "but short enough to not waste most of the training "
                        "budget once the model has clearly started "
                        "overfitting (observed in this project's logs: ADE "
                        "degrades steadily and monotonically from epoch ~7 "
                        "onward with no recovery within 40+ subsequent "
                        "epochs). Note the original repo's own early-stopping "
                        "default is patience=2 (configs/config.yaml "
                        "training.early_stopping.patience), tuned for its "
                        "own num_epochs=30 -- proportionally even stricter "
                        "than this. 30 here is deliberately more lenient to "
                        "account for this project's much noisier per-batch "
                        "environment (small batch_size=90, complex "
                        "multi-modal encoder) and val_freq=5 stride.")
    p.add_argument("--min_epochs",   default=50,         type=int)
    p.add_argument("--lr_min",       default=1e-5,       type=float,
                   help="Matches configs/config.yaml training.min_lr=0.00001. "
                        "Used as eta_min for CosineAnnealingLR (the original "
                        "repo's own scheduler: 'cosine', not plateau-based).")
    p.add_argument("--val_freq",     default=5,          type=int)
    p.add_argument("--val_subset",   default=300,        type=int,
                   help="Smaller than the other baselines' default (600) "
                        "since DDIM sampling is heavier per-sample; raise "
                        "if your GPU/time budget allows.")
    p.add_argument("--val_loss_subset", default=0, type=int,
                   help="If >0, the cheap per-epoch val_loss pass (no "
                        "sampling, run every epoch) uses only this many "
                        "sequences instead of the full validation set. On "
                        "Kaggle T4 the full val set (~3436 sequences here) "
                        "still runs PaperEncoder's FNO3D 3D-FFT once per "
                        "batch every single epoch even without any DDIM "
                        "sampling, which is the single largest recurring "
                        "cost in this script at num_epochs=150+. This "
                        "value is 0 (= full val set, original behavior) "
                        "by default so nothing changes unless you opt in; "
                        "try 600-900 to cut this cost by ~4-5x on T4 while "
                        "keeping the val_loss curve representative enough "
                        "for early-stopping (early-stopping/best-checkpoint "
                        "selection itself is driven by ADE via evaluate(), "
                        "not by this val_loss number, so subsetting it does "
                        "not change which checkpoint gets kept as best).")
    p.add_argument("--num_workers",  default=2,          type=int)

    p.add_argument("--test_at_end",  action="store_true",
                   help="Đánh giá trên tập test sau khi training xong")

    p.add_argument("--output_dir",   default="runs/phys_diff", type=str)
    p.add_argument("--metrics_csv",  default="metrics.csv",    type=str)
    p.add_argument("--gpu_num",      default="0",              type=str)
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

    gate_mlp_dims = tuple(int(x) for x in args.gate_mlp_dims.split(","))

    print("=" * 70)
    print(f"  PHYS-DIFF BASELINE  (PIGA-augmented DDPM, shared multi-modal encoder)")
    print(f"  Encoder: PaperEncoder (FNO3D + Mamba + Env_net)  ← same as ST-Trans/LSTM/MMSTN")
    print(f"  d_model={args.d_model}  d_embedding={args.d_embedding}"
          f"  enc_layers={args.enc_layers}  dec_layers={args.dec_layers}")
    print(f"  num_timesteps={args.num_timesteps}  schedule={args.beta_schedule}"
          f"  sample_steps(eval)={args.sample_steps}")
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

    # Cheap per-epoch val_loss loader: full val set by default (original
    # behavior, args.val_loss_subset==0), or a smaller fixed subset if the
    # user opts in via --val_loss_subset for Kaggle T4 wall-clock budgets.
    # See --val_loss_subset help text for why this is the single largest
    # recurring per-epoch cost independent of DDIM sampling.
    if args.val_loss_subset > 0:
        val_loss_loader = make_subset_loader(
            val_dataset, args.val_loss_subset, args.batch_size, seq_collate,
            seed=args.seed)
        print(f"  val (loss only, per-epoch): {min(args.val_loss_subset, len(val_dataset))} seq"
              f"  (subset of {len(val_dataset)}; full val set still used"
              f"  nowhere else -- 'val   :' line above is the true total)")
    else:
        val_loss_loader = val_loader

    print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
    print(f"  val   : {len(val_dataset)} seq")

    # ── Model ─────────────────────────────────────────────────────────────
    model = PhysDiff(
        obs_len=args.obs_len, pred_len=args.pred_len, unet_in_ch=args.unet_in_ch,
        d_model=args.d_model, d_embedding=args.d_embedding,
        enc_layers=args.enc_layers, enc_heads=args.enc_heads,
        enc_ff=args.enc_ff, enc_dropout=args.enc_dropout,
        dec_layers=args.dec_layers, dec_heads=args.dec_heads,
        dec_ff=args.dec_ff, dec_dropout=args.dec_dropout,
        d_sub=args.d_sub, gate_mlp_dims=gate_mlp_dims,
        num_timesteps=args.num_timesteps, beta_schedule=args.beta_schedule,
        beta_start=args.beta_start, beta_end=args.beta_end,
        sample_steps=args.sample_steps,
        coord_loss_weight=args.coord_loss_weight,
        diffusion_loss_weight=args.diffusion_loss_weight,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params : {n_params:,}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    # Faithful to Phys-Diff's own scripts/train.py:
    #   scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #       optimizer, T_max=config['training']['num_epochs'],
    #       eta_min=config['training']['min_lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr_min)

    best_ade     = float("inf")
    patience_cnt = 0
    start_epoch  = 0

    last_ckpt_path = os.path.join(args.output_dir, "last_ckpt.pth")
    if os.path.exists(last_ckpt_path):
        print(f"  🔄 Found {last_ckpt_path} — resuming training")
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch  = ckpt["epoch"] + 1
        best_ade     = ckpt.get("best_ade", float("inf"))
        patience_cnt = ckpt.get("patience_cnt", 0)
        if ckpt.get("torch_rng_state") is not None:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu())
        if ckpt.get("numpy_rng_state") is not None:
            np.random.set_state(ckpt["numpy_rng_state"])
        if ckpt.get("python_rng_state") is not None:
            random.setstate(ckpt["python_rng_state"])
        print(f"  ▶ Resuming from epoch {start_epoch}  "
              f"(best_ade={best_ade:.1f} km, patience={patience_cnt})")

    train_start  = time.perf_counter()

    def _save_full_checkpoint(path: str, epoch: int):
        """Full state for exact resume: model, optimizer, scheduler,
        epoch, early-stop counters, best_ade, RNG states, args."""
        torch.save({
            "epoch"            : epoch,
            "model_state"      : model.state_dict(),
            "optimizer_state"  : optimizer.state_dict(),
            "scheduler_state"  : scheduler.state_dict(),
            "best_ade"         : best_ade,
            "patience_cnt"     : patience_cnt,
            "model_type"       : "PhysDiff",
            "seed"             : args.seed,
            "args"             : vars(args),
            "torch_rng_state"  : torch.get_rng_state(),
            "cuda_rng_state"   : torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy_rng_state"  : np.random.get_state(),
            "python_rng_state" : random.getstate(),
        }, path)

    print("=" * 70)
    print(f"  TRAINING  ({len(train_loader)} steps/epoch)")
    print("=" * 70)

    for epoch in range(start_epoch, args.num_epochs):
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
                      f"  loss={loss.item():.4f}"
                      f"  diff={bd.get('diffusion_loss', 0):.4f}"
                      f"  coord={bd.get('coord_loss', 0):.4f}"
                      f"  ae={bd.get('autoencode_loss', 0):.4f}"
                      f"  lr={lr:.2e}")

        avg_train = sum_loss / len(train_loader)

        # ── Val loss (same combined loss, no sampling — cheap) ────────────
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loss_loader:
                bl_v = move(list(batch), device)
                bd_v = model.get_loss_breakdown(bl_v)
                val_loss += bd_v["total"].item()
                n_val += 1
        avg_val = val_loss / max(n_val, 1)

        scheduler.step()  # CosineAnnealingLR: steps unconditionally each epoch

        ep_t = time.perf_counter() - t0
        print(f"  Epoch {epoch:>4}  train_loss={avg_train:.4f}"
              f"  val_loss={avg_val:.4f}  t={ep_t:.0f}s")

        # ── ADE + ATE + CTE evaluation (DDIM-strided sampling) ─────────────
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
                "model_type"     : "PhysDiff",
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
                    "model_type" : "PhysDiff",
                    "paper"      : "Phys-Diff (PIGA-augmented DDPM)",
                    "seed"       : args.seed,
                    "model_cfg"  : {
                        "obs_len": args.obs_len, "pred_len": args.pred_len,
                        "unet_in_ch": args.unet_in_ch,
                        "d_model": args.d_model, "d_embedding": args.d_embedding,
                        "enc_layers": args.enc_layers, "enc_heads": args.enc_heads,
                        "enc_ff": args.enc_ff, "enc_dropout": args.enc_dropout,
                        "dec_layers": args.dec_layers, "dec_heads": args.dec_heads,
                        "dec_ff": args.dec_ff, "dec_dropout": args.dec_dropout,
                        "d_sub": args.d_sub, "gate_mlp_dims": list(gate_mlp_dims),
                        "num_timesteps": args.num_timesteps,
                        "beta_schedule": args.beta_schedule,
                        "beta_start": args.beta_start, "beta_end": args.beta_end,
                        "sample_steps": args.sample_steps,
                        "coord_loss_weight": args.coord_loss_weight,
                        "diffusion_loss_weight": args.diffusion_loss_weight,
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

            # Full-state checkpoint at every validation epoch.
            val_ckpt_path = os.path.join(args.output_dir, f"val_ckpt_ep{epoch:04d}.pth")
            _save_full_checkpoint(val_ckpt_path, epoch)

        # Full-state checkpoint every epoch (overwritten) for exact resume.
        _save_full_checkpoint(last_ckpt_path, epoch)

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
    print(f"  Model   : Phys-Diff")
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