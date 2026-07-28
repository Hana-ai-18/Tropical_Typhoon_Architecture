# """
# scripts/train_st_trans.py  ── ST-Trans Baseline Training
# =========================================================
# THUẬT TOÁN: Faiaz et al. (2026) Expert Systems With Applications 317, 131972

# THAY ĐỔI SO VỚI PHIÊN BẢN TRƯỚC:
#   ✅ Dùng cùng DataLoader / batch_list với paper_baseline (full modal: ảnh + env)
#   ✅ STTrans bây giờ sử dụng PaperEncoder (FNO3D + Mamba + Env_net) làm backbone
#   ✅ Thêm ATE / CTE trong mỗi lần eval
#   ✅ Đánh giá trên tập TEST ở cuối training (--test_at_end)
#   ✅ CSV log bao gồm đầy đủ ADE / ATE / CTE tại 12h / 24h / 48h / 72h

# HYPERPARAMETERS theo paper (§3.6):
#   - Epochs    : 1200 → early stopping
#   - Batch     : 90
#   - Optimizer : AdamW + weight decay
#   - LR Sched  : ReduceLROnPlateau (monitor val DPE)
#   - Grad clip : 0.5
#   - d_model   : 64, nhead : 4, dim_ff : 512
# """
# from __future__ import annotations

# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse
# import time
# import random
# import csv
# from datetime import datetime

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset

# from Model.data.loader_training import data_loader
# from Model.st_trans_model import (
#     STTrans, STTransAR,
# )
# from Model.paper_baseline_model import (
#     haversine_km, _norm_to_deg, _ate_cte_tensors,
#     HORIZON_STEPS,
# )


# # ══════════════════════════════════════════════════════════════════════════════
# #  Helpers
# # ══════════════════════════════════════════════════════════════════════════════

# def move(batch, device):
#     out = list(batch)
#     for i, x in enumerate(out):
#         if torch.is_tensor(x):
#             out[i] = x.to(device)
#         elif isinstance(x, dict):
#             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
#                       for k, v in x.items()}
#     return out


# def make_subset_loader(dataset, subset_size, batch_size, collate_fn, seed=42):
#     n   = len(dataset)
#     rng = random.Random(seed)
#     idx = rng.sample(range(n), min(subset_size, n))
#     return DataLoader(Subset(dataset, idx),
#                       batch_size=batch_size, shuffle=False,
#                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# def save_metrics_csv(row: dict, csv_path: str):
#     write_hdr = not os.path.exists(csv_path)
#     with open(csv_path, "a", newline="") as fh:
#         w = csv.DictWriter(fh, fieldnames=list(row.keys()))
#         if write_hdr:
#             w.writeheader()
#         w.writerow(row)


# def _fmt(v) -> str:
#     return f"{v:.2f}" if isinstance(v, float) and not np.isnan(v) else "nan"


# # ══════════════════════════════════════════════════════════════════════════════
# #  Evaluation  (ADE + ATE + CTE)
# # ══════════════════════════════════════════════════════════════════════════════

# @torch.no_grad()
# def evaluate(model, loader, device) -> dict:
#     """
#     Đánh giá model trên toàn bộ loader.
#     batch_list dùng cùng format với paper_baseline (full multi-modal).
#     """
#     model.eval()

#     all_ade, all_fde = [], []
#     ade_buf = {h: [] for h in HORIZON_STEPS}
#     ate_buf = {h: [] for h in HORIZON_STEPS}
#     cte_buf = {h: [] for h in HORIZON_STEPS}
#     all_ate_abs, all_cte_abs = [], []

#     for batch in loader:
#         bl         = move(list(batch), device)
#         pred, _, _ = model.sample(bl)           # batch_list đầy đủ
#         gt         = bl[1]
#         T          = min(pred.shape[0], gt.shape[0])

#         pred_d = _norm_to_deg(pred[:T])
#         gt_d   = _norm_to_deg(gt[:T])
#         dist   = haversine_km(pred_d, gt_d)     # [T, B]

#         ate, cte = _ate_cte_tensors(pred[:T], gt[:T])   # each [T, B]

#         all_ade.extend(dist.mean(0).tolist())
#         all_fde.extend(dist[-1].tolist())
#         all_ate_abs.extend(ate.abs().mean(0).tolist())
#         all_cte_abs.extend(cte.abs().mean(0).tolist())

#         for h, s in HORIZON_STEPS.items():
#             if s < T:
#                 ade_buf[h].extend(dist[s].tolist())
#                 ate_buf[h].extend(ate[s].abs().tolist())
#                 cte_buf[h].extend(cte[s].abs().tolist())

#     def _mean(lst):
#         return float(np.mean(lst)) if lst else float("nan")

#     result = dict(
#         ADE     = _mean(all_ade),
#         FDE     = _mean(all_fde),
#         ATE_abs = _mean(all_ate_abs),
#         CTE_abs = _mean(all_cte_abs),
#     )
#     for h in HORIZON_STEPS:
#         result[f"{h}h"]         = _mean(ade_buf[h])
#         result[f"ATE_abs_{h}h"] = _mean(ate_buf[h])
#         result[f"CTE_abs_{h}h"] = _mean(cte_buf[h])

#     return result


# # ══════════════════════════════════════════════════════════════════════════════
# #  Test set evaluation
# # ══════════════════════════════════════════════════════════════════════════════

# def run_test_evaluation(model, ckpt_path: str, args, device,
#                         collate_fn, csv_path: str):
#     """Load best checkpoint rồi đánh giá toàn bộ test set."""
#     print("\n" + "=" * 70)
#     print("  TEST SET EVALUATION  (ST-Trans)")
#     print("=" * 70)

#     ckpt = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(ckpt["model_state"])
#     print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}"
#           f"  (best val ADE = {ckpt.get('best_ade', float('nan')):.1f} km)")

#     test_dataset, test_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "test"}, test=True)
#     print(f"  test : {len(test_dataset)} sequences  ({len(test_loader)} batches)")

#     metrics = evaluate(model, test_loader, device)

#     print(f"\n  {'Metric':<20} {'Value (km)':>12}")
#     print(f"  {'-'*34}")
#     for key, val in metrics.items():
#         print(f"  {key:<20} {_fmt(val):>12}")

#     row = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
#            "split": "test",
#            "model_type": f"ST-Trans-{args.model_type}"}
#     row.update({k: _fmt(v) for k, v in metrics.items()})
#     save_metrics_csv(row, csv_path)
#     print(f"\n  Test metrics saved → {csv_path}")
#     print("=" * 70)
#     return metrics


# # ══════════════════════════════════════════════════════════════════════════════
# #  Args
# # ══════════════════════════════════════════════════════════════════════════════

# def get_args():
#     p = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#         description="Train ST-Trans baseline (Faiaz et al. 2026)")

#     p.add_argument("--dataset_root", default="TCND_vn",  type=str)
#     p.add_argument("--obs_len",      default=8,          type=int)
#     p.add_argument("--pred_len",     default=12,         type=int)

#     # Model
#     p.add_argument("--model_type",   default="non_ar",   type=str,
#                    choices=["non_ar", "ar"])
#     p.add_argument("--d_model",      default=64,         type=int)
#     p.add_argument("--nhead",        default=4,          type=int)
#     p.add_argument("--num_enc_layers", default=1,        type=int)
#     p.add_argument("--num_dec_layers", default=3,        type=int)
#     p.add_argument("--dim_ff",       default=512,        type=int)
#     p.add_argument("--dropout",      default=0.1,        type=float)
#     p.add_argument("--unet_in_ch",   default=13,         type=int)

#     # Physics loss
#     p.add_argument("--lambda_speed", default=0.1,        type=float)
#     p.add_argument("--lambda_accel", default=0.01,       type=float)
#     p.add_argument("--w_mse",        default=0.05,       type=float)
#     p.add_argument("--v_max_kmh",    default=80.0,       type=float)

#     # Training
#     p.add_argument("--num_epochs",   default=1200,       type=int)
#     p.add_argument("--batch_size",   default=90,         type=int)
#     p.add_argument("--lr",           default=1e-3,       type=float)
#     p.add_argument("--weight_decay", default=1e-4,       type=float)
#     p.add_argument("--grad_clip",    default=0.5,        type=float)
#     p.add_argument("--patience",     default=100,        type=int)
#     p.add_argument("--min_epochs",   default=50,         type=int)
#     p.add_argument("--lr_patience",  default=20,         type=int)
#     p.add_argument("--lr_factor",    default=0.5,        type=float)
#     p.add_argument("--lr_min",       default=1e-6,       type=float)
#     p.add_argument("--val_freq",     default=5,          type=int)
#     p.add_argument("--val_subset",   default=600,        type=int)
#     p.add_argument("--num_workers",  default=2,          type=int)

#     # Test
#     p.add_argument("--test_at_end",  action="store_true",
#                    help="Đánh giá trên tập test sau khi training xong")

#     # I/O
#     p.add_argument("--output_dir",   default="runs/st_trans", type=str)
#     p.add_argument("--metrics_csv",  default="metrics.csv",   type=str)
#     p.add_argument("--gpu_num",      default="0",             type=str)
#     p.add_argument("--seed",         default=42,  type=int,
#                    help="Random seed. Run 3-5 seeds for ESWA mean±std reporting, "
#                         "same convention as train_flowmatching.py.")

#     # DataLoader compat (cùng với train_paper_baseline)
#     p.add_argument("--delim",        default=" ")
#     p.add_argument("--skip",         default=1,   type=int)
#     p.add_argument("--min_ped",      default=1,   type=int)
#     p.add_argument("--threshold",    default=0.002, type=float)
#     p.add_argument("--other_modal",  default="gph")

#     return p.parse_args()


# # ══════════════════════════════════════════════════════════════════════════════
# #  MAIN
# # ══════════════════════════════════════════════════════════════════════════════

# def main(args):
#     # Seed/ablation output_dir suffixing (no-op for default seed=42) —
#     # same convention as train_flowmatching.py, so multi-seed runs don't
#     # overwrite each other.
#     if args.seed != 42:
#         args.output_dir = f"{args.output_dir}_seed{args.seed}"

#     if torch.cuda.is_available():
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # [FIX] Print device/GPU diagnostics immediately — without this, a
#     # silent CPU fallback (or a slow/shared GPU) is only noticed hours
#     # into training when epoch timing looks off, instead of in the first
#     # few seconds. This does not change any training behavior.
#     print(f"  Device: {device}")
#     if torch.cuda.is_available():
#         print(f"  GPU: {torch.cuda.get_device_name(0)}  "
#               f"(CUDA {torch.version.cuda}, {torch.cuda.device_count()} visible)")
#     else:
#         print(f"  ⚠ No GPU detected — training on CPU will be MUCH slower "
#               f"(often 10-50x). If you expected a GPU here, check Kaggle's "
#               f"Accelerator setting for this session.")
#     os.makedirs(args.output_dir, exist_ok=True)

#     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
#     best_ckpt   = os.path.join(args.output_dir, "best_model.pth")

#     print("=" * 70)
#     print(f"  ST-TRANS BASELINE  |  type={args.model_type.upper()}")
#     print(f"  Faiaz et al. (2026) Expert Systems With Applications 317, 131972")
#     print(f"  Encoder: PaperEncoder (FNO3D + Mamba + Env_net)  ← cùng với LSTM baseline")
#     print(f"  d_model={args.d_model}  nhead={args.nhead}  dim_ff={args.dim_ff}")
#     print(f"  λ_speed={args.lambda_speed}  λ_accel={args.lambda_accel}")
#     print(f"  Metrics: ADE / ATE / CTE @ 12h / 24h / 48h / 72h")
#     print("=" * 70)

#     # ── Data ── (cùng pipeline với train_paper_baseline) ──────────────────
#     train_dataset, train_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "train"}, test=False)
#     val_dataset, val_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "val"}, test=True)

#     from Model.data.trajectoriesWithMe_unet_training import seq_collate
#     val_sub_loader = make_subset_loader(
#         val_dataset, args.val_subset, args.batch_size, seq_collate)

#     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
#     print(f"  val   : {len(val_dataset)} seq")

#     # ── Model ─────────────────────────────────────────────────────────────
#     if args.model_type == "non_ar":
#         model = STTrans(
#             obs_len        = args.obs_len,
#             pred_len       = args.pred_len,
#             unet_in_ch     = args.unet_in_ch,
#             d_model        = args.d_model,
#             nhead          = args.nhead,
#             num_enc_layers = args.num_enc_layers,
#             num_dec_layers = args.num_dec_layers,
#             dim_ff         = args.dim_ff,
#             dropout        = args.dropout,
#             lambda_speed   = args.lambda_speed,
#             lambda_accel   = args.lambda_accel,
#             w_mse          = args.w_mse,
#             v_max_kmh      = args.v_max_kmh,
#         ).to(device)
#     else:
#         model = STTransAR(
#             obs_len        = args.obs_len,
#             pred_len       = args.pred_len,
#             unet_in_ch     = args.unet_in_ch,
#             d_model        = args.d_model,
#             nhead          = args.nhead,
#             num_enc_layers = args.num_enc_layers,
#             dim_ff         = args.dim_ff,
#             dropout        = args.dropout,
#             lambda_speed   = args.lambda_speed,
#             lambda_accel   = args.lambda_accel,
#             w_mse          = args.w_mse,
#             v_max_kmh      = args.v_max_kmh,
#         ).to(device)

#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"  params : {n_params:,}")

#     # ── Optimizer + Scheduler ─────────────────────────────────────────────
#     optimizer = optim.AdamW(model.parameters(),
#                             lr=args.lr, weight_decay=args.weight_decay)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=args.lr_factor,
#         patience=args.lr_patience, min_lr=args.lr_min)

#     best_ade     = float("inf")
#     patience_cnt = 0
#     train_start  = time.perf_counter()

#     print("=" * 70)
#     print(f"  TRAINING  ({len(train_loader)} steps/epoch)")
#     print("=" * 70)

#     for epoch in range(args.num_epochs):
#         model.train()
#         sum_loss = 0.0
#         t0 = time.perf_counter()

#         for i, batch in enumerate(train_loader):
#             bl   = move(list(batch), device)   # full batch_list (ảnh + env + ...)
#             bd   = model.get_loss_breakdown(bl)
#             loss = bd["total"]

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#             optimizer.step()

#             sum_loss += loss.item()

#             if i % 30 == 0:
#                 lr = optimizer.param_groups[0]["lr"]
#                 print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
#                       f"  loss={loss.item():.4f}"
#                       f"  dpe={bd.get('dpe', 0):.2f}km"
#                       f"  speed={bd.get('speed', 0):.4f}"
#                       f"  lr={lr:.2e}")

#         avg_train = sum_loss / len(train_loader)

#         # ── Val loss (DPE) ────────────────────────────────────────────────
#         model.eval()
#         val_dpe = 0.0
#         n_val   = 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 bl_v    = move(list(batch), device)
#                 bd_v    = model.get_loss_breakdown(bl_v)
#                 val_dpe += bd_v["dpe"]
#                 n_val   += 1
#         avg_val_dpe = val_dpe / max(n_val, 1)

#         scheduler.step(avg_val_dpe)

#         ep_t = time.perf_counter() - t0
#         print(f"  Epoch {epoch:>4}  train_loss={avg_train:.4f}"
#               f"  val_dpe={avg_val_dpe:.2f}km  t={ep_t:.0f}s")

#         # ── ADE + ATE + CTE evaluation ────────────────────────────────────
#         if epoch % args.val_freq == 0:
#             r = evaluate(model, val_sub_loader, device)

#             ade12 = r.get("12h",         float("nan"))
#             ade24 = r.get("24h",         float("nan"))
#             ade48 = r.get("48h",         float("nan"))
#             ade72 = r.get("72h",         float("nan"))
#             ade   = r.get("ADE",         float("nan"))
#             ate12 = r.get("ATE_abs_12h", float("nan"))
#             cte12 = r.get("CTE_abs_12h", float("nan"))
#             ate72 = r.get("ATE_abs_72h", float("nan"))
#             cte72 = r.get("CTE_abs_72h", float("nan"))

#             t12 = "🎯" if ade12 < 50  else "❌"
#             t24 = "🎯" if ade24 < 100 else "❌"
#             t48 = "🎯" if ade48 < 200 else "❌"
#             t72 = "🎯" if ade72 < 300 else "❌"

#             print(f"  [VAL ep{epoch}]"
#                   f"  ADE={ade:.1f}"
#                   f"  12h={ade12:.0f}{t12}"
#                   f"  24h={ade24:.0f}{t24}"
#                   f"  48h={ade48:.0f}{t48}"
#                   f"  72h={ade72:.0f}{t72} km")
#             print(f"           "
#                   f"  ATE@12h={ate12:.1f}  CTE@12h={cte12:.1f}"
#                   f"  ATE@72h={ate72:.1f}  CTE@72h={cte72:.1f} km")

#             save_metrics_csv({
#                 "timestamp"      : datetime.now().strftime("%Y%m%d_%H%M%S"),
#                 "split"          : "val",
#                 "epoch"          : epoch,
#                 "model_type"     : f"ST-Trans-{args.model_type}",
#                 "train_loss"     : _fmt(avg_train),
#                 "val_dpe_km"     : _fmt(avg_val_dpe),
#                 "ADE_km"         : _fmt(ade),
#                 "FDE_km"         : _fmt(r.get("FDE", float("nan"))),
#                 "12h_km"         : _fmt(ade12),
#                 "24h_km"         : _fmt(ade24),
#                 "48h_km"         : _fmt(ade48),
#                 "72h_km"         : _fmt(ade72),
#                 "ATE_abs_km"     : _fmt(r.get("ATE_abs", float("nan"))),
#                 "CTE_abs_km"     : _fmt(r.get("CTE_abs", float("nan"))),
#                 "ATE_abs_12h_km" : _fmt(ate12),
#                 "CTE_abs_12h_km" : _fmt(cte12),
#                 "ATE_abs_24h_km" : _fmt(r.get("ATE_abs_24h", float("nan"))),
#                 "CTE_abs_24h_km" : _fmt(r.get("CTE_abs_24h", float("nan"))),
#                 "ATE_abs_48h_km" : _fmt(r.get("ATE_abs_48h", float("nan"))),
#                 "CTE_abs_48h_km" : _fmt(r.get("CTE_abs_48h", float("nan"))),
#                 "ATE_abs_72h_km" : _fmt(ate72),
#                 "CTE_abs_72h_km" : _fmt(cte72),
#             }, metrics_csv)

#             if ade < best_ade:
#                 best_ade     = ade
#                 patience_cnt = 0
#                 # [FIX] STTrans (non_ar) and STTransAR have DIFFERENT
#                 # constructor signatures — STTransAR has no num_dec_layers
#                 # (it uses a GRU-cell decoder, not a Transformer decoder).
#                 # Saving the wrong key set would raise TypeError if this
#                 # model_cfg is later used to reconstruct the model.
#                 if args.model_type == "non_ar":
#                     _model_cfg = {
#                         "obs_len":        args.obs_len,
#                         "pred_len":       args.pred_len,
#                         "unet_in_ch":     args.unet_in_ch,
#                         "d_model":        args.d_model,
#                         "nhead":          args.nhead,
#                         "num_enc_layers": args.num_enc_layers,
#                         "num_dec_layers": args.num_dec_layers,
#                         "dim_ff":         args.dim_ff,
#                         "dropout":        args.dropout,
#                     }
#                 else:
#                     _model_cfg = {
#                         "obs_len":        args.obs_len,
#                         "pred_len":       args.pred_len,
#                         "unet_in_ch":     args.unet_in_ch,
#                         "d_model":        args.d_model,
#                         "nhead":          args.nhead,
#                         "num_enc_layers": args.num_enc_layers,
#                         "dim_ff":         args.dim_ff,
#                         "dropout":        args.dropout,
#                     }
#                 torch.save({
#                     "epoch"      : epoch,
#                     "model_state": model.state_dict(),
#                     "best_ade"   : best_ade,
#                     "model_type" : args.model_type,
#                     "paper"      : "Faiaz et al. 2026",
#                     "seed"       : args.seed,
#                     "model_cfg"  : _model_cfg,
#                 }, best_ckpt)
#                 print(f"  ✅ Best ADE {best_ade:.1f} km  (epoch {epoch})")
#             else:
#                 patience_cnt += args.val_freq
#                 print(f"  No improvement {patience_cnt}/{args.patience}"
#                       f"  (best={best_ade:.1f} km)")

#             if epoch >= args.min_epochs and patience_cnt >= args.patience:
#                 print(f"  ⛔ Early stop @ epoch {epoch}")
#                 break

#         if epoch % 100 == 0:
#             torch.save({
#                 "epoch"      : epoch,
#                 "model_state": model.state_dict(),
#                 "train_loss" : avg_train,
#                 "val_dpe"    : avg_val_dpe,
#                 "seed"       : args.seed,
#             }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

#     total_h = (time.perf_counter() - train_start) / 3600
#     print("=" * 70)
#     print(f"  Model   : ST-Trans-{args.model_type.upper()}")
#     print(f"  Best ADE: {best_ade:.1f} km")
#     print(f"  Total   : {total_h:.2f}h")
#     print(f"  Metrics : {metrics_csv}")
#     print("=" * 70)

#     # ── Test evaluation ────────────────────────────────────────────────────
#     if args.test_at_end and os.path.exists(best_ckpt):
#         from Model.data.trajectoriesWithMe_unet_training import seq_collate
#         run_test_evaluation(model, best_ckpt, args, device,
#                             seq_collate, metrics_csv)


# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.seed)
#     main(args)

"""
scripts/train_st_trans.py  ── ST-Trans Baseline Training
=========================================================
THUẬT TOÁN: Faiaz et al. (2026) Expert Systems With Applications 317, 131972

THAY ĐỔI SO VỚI PHIÊN BẢN TRƯỚC:
  ✅ Dùng cùng DataLoader / batch_list với paper_baseline (full modal: ảnh + env)
  ✅ STTrans bây giờ sử dụng PaperEncoder (FNO3D + Mamba + Env_net) làm backbone
  ✅ Thêm ATE / CTE trong mỗi lần eval
  ✅ Đánh giá trên tập TEST ở cuối training (--test_at_end)
  ✅ CSV log bao gồm đầy đủ ADE / ATE / CTE tại 12h / 24h / 48h / 72h

HYPERPARAMETERS theo paper (§3.6):
  - Epochs    : 1200 → early stopping
  - Batch     : 90
  - Optimizer : AdamW + weight decay
  - LR Sched  : ReduceLROnPlateau (monitor val DPE)
  - Grad clip : 0.5
  - d_model   : 64, nhead : 4, dim_ff : 512
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
from Model.st_trans_model import (
    STTrans, STTransAR,
)
from Model.paper_baseline_model import (
    haversine_km, _norm_to_deg, _ate_cte_tensors,
    HORIZON_STEPS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
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
#  Evaluation  (ADE + ATE + CTE)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """
    Đánh giá model trên toàn bộ loader.
    batch_list dùng cùng format với paper_baseline (full multi-modal).
    """
    model.eval()

    all_ade, all_fde = [], []
    ade_buf = {h: [] for h in HORIZON_STEPS}
    ate_buf = {h: [] for h in HORIZON_STEPS}
    cte_buf = {h: [] for h in HORIZON_STEPS}
    all_ate_abs, all_cte_abs = [], []

    for batch in loader:
        bl         = move(list(batch), device)
        pred, _, _ = model.sample(bl)           # batch_list đầy đủ
        gt         = bl[1]
        T          = min(pred.shape[0], gt.shape[0])

        pred_d = _norm_to_deg(pred[:T])
        gt_d   = _norm_to_deg(gt[:T])
        dist   = haversine_km(pred_d, gt_d)     # [T, B]

        ate, cte = _ate_cte_tensors(pred[:T], gt[:T])   # each [T, B]

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


# ══════════════════════════════════════════════════════════════════════════════
#  Test set evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_test_evaluation(model, ckpt_path: str, args, device,
                        collate_fn, csv_path: str):
    """Load best checkpoint rồi đánh giá toàn bộ test set."""
    print("\n" + "=" * 70)
    print("  TEST SET EVALUATION  (ST-Trans)")
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
           "split": "test",
           "model_type": f"ST-Trans-{args.model_type}"}
    row.update({k: _fmt(v) for k, v in metrics.items()})
    save_metrics_csv(row, csv_path)
    print(f"\n  Test metrics saved → {csv_path}")
    print("=" * 70)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train ST-Trans baseline (Faiaz et al. 2026)")

    p.add_argument("--dataset_root", default="TCND_vn",  type=str)
    p.add_argument("--obs_len",      default=8,          type=int)
    p.add_argument("--pred_len",     default=12,         type=int)

    # Model
    p.add_argument("--model_type",   default="non_ar",   type=str,
                   choices=["non_ar", "ar"])
    p.add_argument("--d_model",      default=64,         type=int)
    p.add_argument("--nhead",        default=4,          type=int)
    p.add_argument("--num_enc_layers", default=1,        type=int)
    p.add_argument("--num_dec_layers", default=3,        type=int)
    p.add_argument("--dim_ff",       default=512,        type=int)
    p.add_argument("--dropout",      default=0.1,        type=float)
    p.add_argument("--unet_in_ch",   default=13,         type=int)

    # Physics loss
    p.add_argument("--lambda_speed", default=0.1,        type=float)
    p.add_argument("--lambda_accel", default=0.01,       type=float)
    p.add_argument("--w_mse",        default=0.05,       type=float)
    p.add_argument("--v_max_kmh",    default=80.0,       type=float)

    # Training
    p.add_argument("--num_epochs",   default=1200,       type=int)
    p.add_argument("--batch_size",   default=90,         type=int)
    p.add_argument("--lr",           default=1e-3,       type=float)
    p.add_argument("--weight_decay", default=1e-4,       type=float)
    p.add_argument("--grad_clip",    default=0.5,        type=float)
    p.add_argument("--patience",     default=100,        type=int)
    p.add_argument("--min_epochs",   default=50,         type=int)
    p.add_argument("--lr_patience",  default=20,         type=int)
    p.add_argument("--lr_factor",    default=0.5,        type=float)
    p.add_argument("--lr_min",       default=1e-6,       type=float)
    p.add_argument("--val_freq",     default=5,          type=int)
    p.add_argument("--val_subset",   default=600,        type=int)
    p.add_argument("--num_workers",  default=2,          type=int)

    # Test
    p.add_argument("--test_at_end",  action="store_true",
                   help="Đánh giá trên tập test sau khi training xong")

    # I/O
    p.add_argument("--output_dir",   default="runs/st_trans", type=str)
    p.add_argument("--metrics_csv",  default="metrics.csv",   type=str)
    p.add_argument("--gpu_num",      default="0",             type=str)
    p.add_argument("--seed",         default=42,  type=int,
                   help="Random seed. Run 3-5 seeds for ESWA mean±std reporting, "
                        "same convention as train_flowmatching.py.")

    # DataLoader compat (cùng với train_paper_baseline)
    p.add_argument("--delim",        default=" ")
    p.add_argument("--skip",         default=1,   type=int)
    p.add_argument("--min_ped",      default=1,   type=int)
    p.add_argument("--threshold",    default=0.002, type=float)
    # [FIX-DATA-30] Region filter — see train_flowmatching.py for full
    # rationale. Off by default.
    p.add_argument("--filter_region",  action="store_true", default=False,
                   help="Keep only storms whose track substantially enters "
                        "the South China Sea / Vietnam region.")
    p.add_argument("--min_pct_in_scs", default=15.0, type=float,
                   help="Minimum %% of track points inside the SCS/Vietnam "
                        "box required to keep a storm when --filter_region.")
    p.add_argument("--other_modal",  default="gph")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    # Seed/ablation output_dir suffixing (no-op for default seed=42) —
    # same convention as train_flowmatching.py, so multi-seed runs don't
    # overwrite each other.
    if args.seed != 42:
        args.output_dir = f"{args.output_dir}_seed{args.seed}"

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # [FIX] Print device/GPU diagnostics immediately — without this, a
    # silent CPU fallback (or a slow/shared GPU) is only noticed hours
    # into training when epoch timing looks off, instead of in the first
    # few seconds. This does not change any training behavior.
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"(CUDA {torch.version.cuda}, {torch.cuda.device_count()} visible)")
    else:
        print(f"  ⚠ No GPU detected — training on CPU will be MUCH slower "
              f"(often 10-50x). If you expected a GPU here, check Kaggle's "
              f"Accelerator setting for this session.")
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
    best_ckpt   = os.path.join(args.output_dir, "best_model.pth")

    print("=" * 70)
    print(f"  ST-TRANS BASELINE  |  type={args.model_type.upper()}")
    print(f"  Faiaz et al. (2026) Expert Systems With Applications 317, 131972")
    print(f"  Encoder: PaperEncoder (FNO3D + Mamba + Env_net)  ← cùng với LSTM baseline")
    print(f"  d_model={args.d_model}  nhead={args.nhead}  dim_ff={args.dim_ff}")
    print(f"  λ_speed={args.lambda_speed}  λ_accel={args.lambda_accel}")
    print(f"  Metrics: ADE / ATE / CTE @ 12h / 24h / 48h / 72h")
    print("=" * 70)

    # ── Data ── (cùng pipeline với train_paper_baseline) ──────────────────
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
    if args.model_type == "non_ar":
        model = STTrans(
            obs_len        = args.obs_len,
            pred_len       = args.pred_len,
            unet_in_ch     = args.unet_in_ch,
            d_model        = args.d_model,
            nhead          = args.nhead,
            num_enc_layers = args.num_enc_layers,
            num_dec_layers = args.num_dec_layers,
            dim_ff         = args.dim_ff,
            dropout        = args.dropout,
            lambda_speed   = args.lambda_speed,
            lambda_accel   = args.lambda_accel,
            w_mse          = args.w_mse,
            v_max_kmh      = args.v_max_kmh,
        ).to(device)
    else:
        model = STTransAR(
            obs_len        = args.obs_len,
            pred_len       = args.pred_len,
            unet_in_ch     = args.unet_in_ch,
            d_model        = args.d_model,
            nhead          = args.nhead,
            num_enc_layers = args.num_enc_layers,
            dim_ff         = args.dim_ff,
            dropout        = args.dropout,
            lambda_speed   = args.lambda_speed,
            lambda_accel   = args.lambda_accel,
            w_mse          = args.w_mse,
            v_max_kmh      = args.v_max_kmh,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params : {n_params:,}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
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
            bl   = move(list(batch), device)   # full batch_list (ảnh + env + ...)
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
                      f"  dpe={bd.get('dpe', 0):.2f}km"
                      f"  speed={bd.get('speed', 0):.4f}"
                      f"  lr={lr:.2e}")

        avg_train = sum_loss / len(train_loader)

        # ── Val loss (DPE) ────────────────────────────────────────────────
        model.eval()
        val_dpe = 0.0
        n_val   = 0
        with torch.no_grad():
            for batch in val_loader:
                bl_v    = move(list(batch), device)
                bd_v    = model.get_loss_breakdown(bl_v)
                val_dpe += bd_v["dpe"]
                n_val   += 1
        avg_val_dpe = val_dpe / max(n_val, 1)

        scheduler.step(avg_val_dpe)

        ep_t = time.perf_counter() - t0
        print(f"  Epoch {epoch:>4}  train_loss={avg_train:.4f}"
              f"  val_dpe={avg_val_dpe:.2f}km  t={ep_t:.0f}s")

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
                "model_type"     : f"ST-Trans-{args.model_type}",
                "train_loss"     : _fmt(avg_train),
                "val_dpe_km"     : _fmt(avg_val_dpe),
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
                # [FIX] STTrans (non_ar) and STTransAR have DIFFERENT
                # constructor signatures — STTransAR has no num_dec_layers
                # (it uses a GRU-cell decoder, not a Transformer decoder).
                # Saving the wrong key set would raise TypeError if this
                # model_cfg is later used to reconstruct the model.
                if args.model_type == "non_ar":
                    _model_cfg = {
                        "obs_len":        args.obs_len,
                        "pred_len":       args.pred_len,
                        "unet_in_ch":     args.unet_in_ch,
                        "d_model":        args.d_model,
                        "nhead":          args.nhead,
                        "num_enc_layers": args.num_enc_layers,
                        "num_dec_layers": args.num_dec_layers,
                        "dim_ff":         args.dim_ff,
                        "dropout":        args.dropout,
                    }
                else:
                    _model_cfg = {
                        "obs_len":        args.obs_len,
                        "pred_len":       args.pred_len,
                        "unet_in_ch":     args.unet_in_ch,
                        "d_model":        args.d_model,
                        "nhead":          args.nhead,
                        "num_enc_layers": args.num_enc_layers,
                        "dim_ff":         args.dim_ff,
                        "dropout":        args.dropout,
                    }
                torch.save({
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "best_ade"   : best_ade,
                    "model_type" : args.model_type,
                    "paper"      : "Faiaz et al. 2026",
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
                "val_dpe"    : avg_val_dpe,
                "seed"       : args.seed,
            }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

    total_h = (time.perf_counter() - train_start) / 3600
    print("=" * 70)
    print(f"  Model   : ST-Trans-{args.model_type.upper()}")
    print(f"  Best ADE: {best_ade:.1f} km")
    print(f"  Total   : {total_h:.2f}h")
    print(f"  Metrics : {metrics_csv}")
    print("=" * 70)

    # ── Test evaluation ────────────────────────────────────────────────────
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