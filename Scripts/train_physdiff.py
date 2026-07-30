"""
scripts/train_physdiff.py  -- Train Phys-Diff Latent Diffusion Baseline
============================================================================
THUAT TOAN GOC: Liu et al. (2026) ICASSP, "Phys-Diff: A Physics-Inspired
Latent Diffusion Model for Tropical Cyclone Forecasting"
(arXiv:2603.00521, code: github.com/USTC-AI4EEE/Phys-Diff). Xem
Model/physdiff_model.py cho chi tiet kien truc.

DIEM KHAC BIET SO VOI train_st_trans.py:
  - Chi 1 optimizer (khong phai GAN).
  - Uncertainty-weighted multi-task loss (Eq.9) da tu dong hoc
    sigma_diff/sigma_recon qua nn.Parameter, khong can tune trong so
    thu cong giua L_diffusion va L_recon.
  - evaluate()/sample() dung DDPM reverse process tren khong gian
    LATENT (khac LT3P dung reverse process truc tiep tren toa do) --
    van can N buoc lap nhu LT3P nen van cham hon 1 forward pass thuong,
    val_freq/val_subset dieu chinh tuong tu train_lt3p.py.
"""
from __future__ import annotations

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
from Model.physdiff_model import PhysDiffModel
from Model.paper_baseline_model import (
    haversine_km, _norm_to_deg, _ate_cte_tensors,
    HORIZON_STEPS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers (giống hệt train_lt3p.py / train_st_trans.py)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation (ADE + ATE + CTE)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, num_ensemble: int = 1) -> dict:
    model.eval()

    all_ade, all_fde = [], []
    ade_buf = {h: [] for h in HORIZON_STEPS}
    ate_buf = {h: [] for h in HORIZON_STEPS}
    cte_buf = {h: [] for h in HORIZON_STEPS}
    all_ate_abs, all_cte_abs = [], []

    for batch in loader:
        bl = move(list(batch), device)
        pred, _, _ = model.sample(bl, num_ensemble=num_ensemble)
        gt = bl[1]
        T = min(pred.shape[0], gt.shape[0])

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

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}"
          f"  (best val ADE = {ckpt.get('best_ade', float('nan')):.1f} km)")

    model.n_sample_steps = args.test_n_sample_steps

    test_dataset, test_loader = data_loader(
        args, {"root": args.dataset_root, "type": "test"}, test=True)
    print(f"  test : {len(test_dataset)} sequences  ({len(test_loader)} batches)"
          f"  n_sample_steps={args.test_n_sample_steps}"
          f"  num_ensemble={args.eval_num_ensemble}")

    metrics = evaluate(model, test_loader, device, num_ensemble=args.eval_num_ensemble)

    print(f"\n  {'Metric':<20} {'Value (km)':>12}")
    print(f"  {'-'*34}")
    for key, val in metrics.items():
        print(f"  {key:<20} {_fmt(val):>12}")

    row = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "split": "test",
           "model_type": "PhysDiff"}
    row.update({k: _fmt(v) for k, v in metrics.items()})
    save_metrics_csv(row, csv_path)
    print(f"\n  Test metrics saved -> {csv_path}")
    print("=" * 70)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train Phys-Diff latent diffusion baseline (Liu et al. 2026 ICASSP)")

    p.add_argument("--dataset_root", default="TCND_vn",  type=str)
    p.add_argument("--obs_len",      default=8,          type=int)
    p.add_argument("--pred_len",     default=12,         type=int)

    # Model
    p.add_argument("--d_model",      default=64,         type=int)
    p.add_argument("--nhead",        default=4,          type=int)
    p.add_argument("--num_blocks",   default=3,          type=int,
                   help="Số Physics-Inspired Decoder block")
    p.add_argument("--dim_ff",       default=256,        type=int)
    p.add_argument("--dropout",      default=0.1,        type=float)
    p.add_argument("--unet_in_ch",   default=13,         type=int)

    # Diffusion schedule
    p.add_argument("--T_diffusion",  default=1000,       type=int,
                   help="[QUAN TRỌNG] Reverse sampling giờ LUÔN chạy đủ T_diffusion "
                        "bước liên tiếp (đúng Eq.2 paper Phys-Diff, không hỗ trợ "
                        "strided/DDIM-style skip-step). Tăng T_diffusion sẽ làm "
                        "val/test CHẬM HƠN TUYẾN TÍNH -- cân nhắc giảm --val_freq "
                        "và/hoặc --val_subset nếu T_diffusion lớn.")
    p.add_argument("--beta_start",   default=1e-4,       type=float)
    p.add_argument("--beta_end",     default=0.02,       type=float)
    p.add_argument("--n_sample_steps", default=50,       type=int,
                   help="[DEPRECATED, không còn ảnh hưởng] Trước đây dùng để strided-"
                        "sample nhanh hơn, nhưng công thức đó SAI so với paper (paper "
                        "không hỗ trợ skip-step) và đã bị loại bỏ khỏi "
                        "_ddpm_reverse_sample(). Giữ tham số này chỉ để tương thích "
                        "ngược với checkpoint cũ, không có tác dụng thực tế.")
    p.add_argument("--test_n_sample_steps", default=200, type=int,
                   help="[DEPRECATED, không còn ảnh hưởng] Cùng lý do trên.")
    p.add_argument("--eval_num_ensemble", default=20,    type=int)

    # Training
    p.add_argument("--num_epochs",   default=1200,       type=int)
    p.add_argument("--batch_size",   default=90,         type=int)
    p.add_argument("--lr",           default=1e-4,       type=float,
                   help="Đúng lr code gốc (1e-4, không phải 1e-3 như các baseline khác)")
    p.add_argument("--weight_decay", default=1e-4,       type=float)
    p.add_argument("--grad_clip",    default=0.5,        type=float)
    p.add_argument("--patience",     default=100,        type=int)
    p.add_argument("--min_epochs",   default=50,         type=int)
    p.add_argument("--lr_patience",  default=20,         type=int)
    p.add_argument("--lr_factor",    default=0.5,        type=float)
    p.add_argument("--lr_min",       default=1e-6,       type=float)
    p.add_argument("--val_freq",     default=25,         type=int,
                   help="[TĂNG từ 10] Reverse sampling giờ chạy đủ T_diffusion=1000 "
                        "bước liên tiếp thay vì strided 10-50 bước như trước -- MỖI "
                        "LẦN validate chậm hơn đáng kể, nên validate thưa hơn để không "
                        "làm chậm tổng thời gian train.")
    p.add_argument("--val_subset",   default=100,        type=int,
                   help="[GIẢM từ 300] Cùng lý do trên -- giảm số sample validate mỗi "
                        "lần để bù lại chi phí mỗi sample giờ đắt hơn nhiều.")
    p.add_argument("--num_workers",  default=2,          type=int)

    # Test
    p.add_argument("--test_at_end",  action="store_true")

    # I/O
    p.add_argument("--output_dir",   default="runs/physdiff", type=str)
    p.add_argument("--metrics_csv",  default="metrics.csv", type=str)
    p.add_argument("--gpu_num",      default="0",           type=str)
    p.add_argument("--seed",         default=42,  type=int,
                   help="Random seed. Run 3-5 seeds for ESWA mean±std reporting.")

    # DataLoader compat
    p.add_argument("--delim",        default=" ")
    p.add_argument("--skip",         default=1,   type=int)
    p.add_argument("--min_ped",      default=1,   type=int)
    p.add_argument("--threshold",    default=0.002, type=float)
    p.add_argument("--filter_region",  action="store_true", default=False)
    p.add_argument("--min_pct_in_scs", default=15.0, type=float)
    p.add_argument("--other_modal",  default="gph")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

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
    print(f"  PHYS-DIFF BASELINE  |  Latent Diffusion + PIGA Module")
    print(f"  Liu et al. (2026) ICASSP — arXiv:2603.00521")
    print(f"  Encoder: PaperEncoder (FNO3D + Mamba + Env_net) ← cùng các baseline khác")
    print(f"  Backbone: Latent Diffusion (T={args.T_diffusion}) + PIGA (traj/wind/pres)")
    print(f"  d_model={args.d_model}  num_blocks={args.num_blocks}  lr={args.lr}")
    print(f"  Loss: uncertainty-weighted (Eq.9, Kendall et al. 2018, learnable σ)")
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
    print(f"  val   : {len(val_dataset)} seq  (subset {args.val_subset} for eval)")

    # ── Model ─────────────────────────────────────────────────────────────
    model = PhysDiffModel(
        obs_len        = args.obs_len,
        pred_len       = args.pred_len,
        unet_in_ch     = args.unet_in_ch,
        d_model        = args.d_model,
        nhead          = args.nhead,
        num_blocks     = args.num_blocks,
        dim_ff         = args.dim_ff,
        dropout        = args.dropout,
        T_diffusion    = args.T_diffusion,
        beta_start     = args.beta_start,
        beta_end       = args.beta_end,
        n_sample_steps = args.n_sample_steps,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params : {n_params:,}")

    # ── Optimizer + Scheduler ────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.lr_min)

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
                      f"  loss={loss.item():.4f}"
                      f"  (Diff={bd['l_diffusion']:.4f}"
                      f" Coord={bd['l_traj']:.4f}"
                      f" MSW={bd['l_wind']:.4f}"
                      f" MLSP={bd['l_pres']:.4f})"
                      f"  σ_diff={bd['sigma_diff']:.3f}"
                      f"  σ_recon={bd['sigma_recon']:.3f}"
                      f"  lr={lr:.2e}")

        avg_train = sum_loss / len(train_loader)

        # ── Val loss (1 forward pass, không cần reverse) ────────────────────
        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for batch in val_loader:
                bl_v = move(list(batch), device)
                bd_v = model.get_loss_breakdown(bl_v)
                val_loss += bd_v["total"].item()
                n_val    += 1
        avg_val_loss = val_loss / max(n_val, 1)

        scheduler.step(avg_val_loss)

        ep_t = time.perf_counter() - t0
        print(f"  Epoch {epoch:>4}  train_loss={avg_train:.4f}"
              f"  val_loss={avg_val_loss:.4f}  t={ep_t:.0f}s")

        # ── ADE + ATE + CTE evaluation (DDPM reverse process THẬT) ───────────
        if epoch % args.val_freq == 0:
            r = evaluate(model, val_sub_loader, device, num_ensemble=1)

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

            print(f"  [VAL ep{epoch}]  (reverse process, {args.n_sample_steps} steps)"
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
                "val_loss"       : _fmt(avg_val_loss),
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
                "ATE_abs_72h_km" : _fmt(ate72),
                "CTE_abs_72h_km" : _fmt(cte72),
            }, metrics_csv)

            if ade < best_ade:
                best_ade     = ade
                patience_cnt = 0
                _model_cfg = {
                    "obs_len":        args.obs_len,
                    "pred_len":       args.pred_len,
                    "unet_in_ch":     args.unet_in_ch,
                    "d_model":        args.d_model,
                    "nhead":          args.nhead,
                    "num_blocks":     args.num_blocks,
                    "dim_ff":         args.dim_ff,
                    "dropout":        args.dropout,
                    "T_diffusion":    args.T_diffusion,
                    "beta_start":     args.beta_start,
                    "beta_end":       args.beta_end,
                    "n_sample_steps": args.n_sample_steps,
                }
                torch.save({
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "best_ade"   : best_ade,
                    "model_type" : "PhysDiff",
                    "paper"      : "Liu et al. 2026 (ICASSP) arXiv:2603.00521",
                    "seed"       : args.seed,
                    "model_cfg"  : _model_cfg,
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
                "train_loss" : avg_train,
                "val_loss"   : avg_val_loss,
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