"""
scripts/train_mgtcf.py  -- Train MGTCF Multi-Generator GAN Baseline (v2)
============================================================================
NGUON: viet lai TU CHINH train_github.py goc cua tac gia (Huang et al.,
MGTCF/TropiCycloneNet), do nguoi dung cung cap truc tiep. 3-step training
(discriminator_step / generator_step / net_chooser_step) bam sat DUNG
thu tu va cong thuc loss trong file goc, KHONG con la don gian hoa nhu
ban v1.

3-STEP TRAINING (mỗi batch, đúng thứ tự train_github.py gốc):
  1. discriminator_step:
       - Generator sample 1 mẫu (all_g_out=False, num_samples=1) qua
         GC-Net categorical sampling.
       - scores_real = D(traj_real), scores_fake = D(traj_fake.detach())
       - loss_d = gan_d_loss(scores_real, scores_fake)  [BCE that/gia,
         label smoothing ngau nhien U(0.7,1.2)/U(0,0.3) dung cong thuc
         goc trong losses.py]
       - optimizer_d.zero_grad(); loss_d.backward(); optimizer_d.step()

  2. generator_step:
       - Generator sample best_k mẫu (all_g_out=False) qua GC-Net.
       - Variety/Best-of-K L2 loss: MIN trong best_k mẫu (dung
         g_l2_loss_rel goc, KHONG dung MSE thuong).
       - scores_fake = D(traj_fake[best mau cuoi]) -> discriminator_loss
         = gan_g_loss(scores_fake)  [G co gang danh lua D]
       - loss_g = l2_loss + discriminator_loss
         (BO image_loss vi khong trien khai lai Unet3D du bao anh --
         xem docstring mgtcf_model.py)
       - optimizer_g.zero_grad(); loss_g.backward(); optimizer_g.step()
         (CHU Y: optimizer_g CHUA bao gom net_chooser trong buoc nay ve
         mat GRADIENT THUC SU, vi net_chooser chi dung de SAMPLE
         (torch.no_grad trong _get_samples), khong nhan gradient tu
         g_l2_loss -- nhung VAN nam trong optimizer_g.param_groups vi
         no la 1 phan cua generator.parameters())

  3. net_chooser_step:
       - Generator chạy TẤT CẢ K generator (all_g_out=True, generator
         WEIGHTS duoi torch.no_grad() -- CHI GC-Net nhan gradient).
       - Winner-Takes-All: tim generator co L2 distance nho nhat (toan
         horizon, weighting_target='l2' theo dung code goc mac dinh).
       - loss_gc = F.cross_entropy(net_chooser_weights, min_idx)
       - optimizer_g.zero_grad(); loss_gc.backward(); optimizer_g.step()
         (dung CHUNG optimizer_g voi generator_step, vi net_chooser la
         1 submodule cua MGTCFGenerator -- dung ĐÚNG code goc: ca 3 ham
         deu dung optimizer_g, CHỈ discriminator_step dung optimizer_d
         rieng)

HYPERPARAMETERS (giu tuong tu train_st_trans.py cho cac tham so chung,
them cac tham so rieng cua GAN theo dung train_github.py goc):
  - best_k=6 (Variety Loss, dung so voi n_generators=6 mac dinh)
  - clipping_threshold_g / clipping_threshold_d (grad clip rieng cho
    G va D, dung ten bien giong code goc)
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from Model.data.loader_training import data_loader
from Model.mgtcf_model import MGTCFModel
from Model.paper_baseline_model import (
    haversine_km, _norm_to_deg, _ate_cte_tensors,
    HORIZON_STEPS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  GAN loss functions (đúng losses.py gốc: bce_loss, gan_g_loss, gan_d_loss)
# ══════════════════════════════════════════════════════════════════════════════

def bce_loss(inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Numerically stable BCE, đúng công thức trong losses.py gốc."""
    neg_abs = -inp.abs()
    loss = inp.clamp(min=0) - inp * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake: torch.Tensor) -> torch.Tensor:
    """Đúng gan_g_loss gốc — label smoothing ngẫu nhiên U(0.7, 1.2)."""
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real: torch.Tensor, scores_fake: torch.Tensor) -> torch.Tensor:
    """Đúng gan_d_loss gốc — label smoothing ngẫu nhiên cho cả real/fake."""
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


# ══════════════════════════════════════════════════════════════════════════════
#  3-step training (đúng train_github.py gốc)
# ══════════════════════════════════════════════════════════════════════════════

def discriminator_step(model: MGTCFModel, bl, optimizer_d, grad_clip_d: float) -> dict:
    obs_traj = bl[0]
    traj_gt  = bl[1]
    T = min(model.pred_len, traj_gt.shape[0])
    gt = traj_gt[:T]

    pred, _, _ = model.generator(bl, num_samples=1, all_g_out=False)
    pred_fake = pred[:T, 0, :, :]   # [T, B, 2] -- pred shape gốc [pred_len, num_samples, B, 2]

    traj_real = torch.cat([obs_traj, gt], dim=0)
    traj_fake = torch.cat([obs_traj, pred_fake.detach()], dim=0)

    # img_embed đơn giản: dùng lại last_img lặp lại cho toàn chuỗi (xấp xỉ,
    # vì không có ảnh "dự báo" tương lai thật — xem docstring model)
    B = obs_traj.shape[1]
    device = obs_traj.device
    img_embed_dummy_real = torch.zeros(
        obs_traj.shape[0] + T, B, model.generator.embedding_dim, device=device)
    img_embed_dummy_fake = img_embed_dummy_real.clone()

    scores_real = model.discriminator(traj_real, img_embed_dummy_real)
    scores_fake = model.discriminator(traj_fake, img_embed_dummy_fake)

    loss_d = gan_d_loss(scores_real, scores_fake)

    optimizer_d.zero_grad()
    loss_d.backward()
    if grad_clip_d > 0:
        nn.utils.clip_grad_norm_(model.discriminator.parameters(), grad_clip_d)
    optimizer_d.step()

    return {"d_loss": loss_d.item()}


def generator_step(model: MGTCFModel, bl, optimizer_g, grad_clip_g: float,
                    best_k: int, l2_loss_weight: float) -> dict:
    obs_traj = bl[0]
    traj_gt  = bl[1]
    T = min(model.pred_len, traj_gt.shape[0])
    gt = traj_gt[:T]
    B = obs_traj.shape[1]
    device = obs_traj.device

    pred, _, _ = model.generator(bl, num_samples=best_k, all_g_out=False)
    pred = pred[:T, :, :, :]   # [T, best_k, B, 2] -- pred shape gốc [pred_len, best_k, B, 2]

    # ── Variety/Best-of-K L2 loss (MIN trong best_k, đúng code gốc) ─────────
    l2_loss_val = torch.zeros(1, device=device)
    if l2_loss_weight > 0:
        per_sample_l2 = ((pred - gt.unsqueeze(1)) ** 2).sum(dim=(0, 3))   # [best_k, B]
        min_l2_per_batch = per_sample_l2.min(dim=0).values                # [B]
        l2_loss_val = l2_loss_weight * min_l2_per_batch.mean()

    # ── Adversarial loss (dùng mẫu cuối cùng trong best_k, đúng code gốc
    #    lấy pred_traj_fake[:,-1]) ────────────────────────────────────────
    pred_fake_last = pred[:, -1, :, :]   # [T, B, 2]
    traj_fake = torch.cat([obs_traj, pred_fake_last], dim=0)
    img_embed_dummy = torch.zeros(
        obs_traj.shape[0] + T, B, model.generator.embedding_dim, device=device)

    scores_fake = model.discriminator(traj_fake, img_embed_dummy)
    adv_loss = gan_g_loss(scores_fake)

    loss_g = l2_loss_val + adv_loss

    optimizer_g.zero_grad()
    loss_g.backward()
    if grad_clip_g > 0:
        nn.utils.clip_grad_norm_(model.generator.parameters(), grad_clip_g)
    optimizer_g.step()

    with torch.no_grad():
        best_idx = per_sample_l2.argmin(dim=0) if l2_loss_weight > 0 else torch.zeros(B, dtype=torch.long)
        best_pred = torch.stack([pred[:, best_idx[b], b, :] for b in range(B)], dim=1)
        pred_deg = _norm_to_deg(best_pred)
        gt_deg = _norm_to_deg(gt)
        dpe = haversine_km(pred_deg, gt_deg).mean().item()

    return {"g_l2_loss": l2_loss_val.item(), "g_adv_loss": adv_loss.item(),
            "g_total_loss": loss_g.item(), "g_dpe_bestofk": dpe}


def net_chooser_step(model: MGTCFModel, bl, optimizer_g, grad_clip_g: float) -> dict:
    obs_traj = bl[0]
    traj_gt  = bl[1]
    T = min(model.pred_len, traj_gt.shape[0])
    gt = traj_gt[:T]
    B = obs_traj.shape[1]

    # all_g_out=True: chạy TẤT CẢ K generator dưới torch.no_grad() bên
    # trong model.generator.forward() — chỉ GC-Net (net_chooser_out)
    # nhận gradient thực sự.
    pred_all, net_chooser_out, _ = model.generator(bl, num_samples=1, all_g_out=True)
    pred_all = pred_all[:T, :, :, :]   # [T, K, B, 2] -- pred_all shape gốc [pred_len, K, B, 2]

    with torch.no_grad():
        l2_dist = ((pred_all - gt.unsqueeze(1)) ** 2).sum(dim=(0, 3))   # [K, B]
        min_idx = l2_dist.argmin(dim=0)                                   # [B]

    loss_gc = F.cross_entropy(net_chooser_out, min_idx)

    optimizer_g.zero_grad()
    loss_gc.backward()
    if grad_clip_g > 0:
        nn.utils.clip_grad_norm_(model.generator.parameters(), grad_clip_g)
    optimizer_g.step()

    with torch.no_grad():
        gc_accuracy = (net_chooser_out.argmax(dim=-1) == min_idx).float().mean().item()

    return {"gc_ce_loss": loss_gc.item(), "gc_accuracy": gc_accuracy}


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers (giống hệt train_st_trans.py)
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
#  Evaluation (ADE + ATE + CTE) -- dùng model.sample() với Roulette thật
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, num_ensemble: int = 20) -> dict:
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
    print("  TEST SET EVALUATION  (MGTCF)")
    print("=" * 70)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}"
          f"  (best val ADE = {ckpt.get('best_ade', float('nan')):.1f} km)")

    test_dataset, test_loader = data_loader(
        args, {"root": args.dataset_root, "type": "test"}, test=True)
    print(f"  test : {len(test_dataset)} sequences  ({len(test_loader)} batches)")

    metrics = evaluate(model, test_loader, device, num_ensemble=args.eval_num_ensemble)

    print(f"\n  {'Metric':<20} {'Value (km)':>12}")
    print(f"  {'-'*34}")
    for key, val in metrics.items():
        print(f"  {key:<20} {_fmt(val):>12}")

    row = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "split": "test",
           "model_type": "MGTCF"}
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
        description="Train MGTCF multi-generator GAN baseline "
                    "(faithful port of Huang et al. 2023/2025)")

    p.add_argument("--dataset_root", default="TCND_vn",  type=str)
    p.add_argument("--obs_len",      default=8,          type=int)
    p.add_argument("--pred_len",     default=12,         type=int)

    # Model (đúng tên/giá trị mặc định code gốc nơi có thể)
    p.add_argument("--n_generators", default=6,          type=int,
                   help="Số generator K (code gốc: num_gs=6)")
    p.add_argument("--embedding_dim", default=64,        type=int)
    p.add_argument("--encoder_h_dim", default=64,        type=int)
    p.add_argument("--decoder_h_dim", default=128,       type=int)
    p.add_argument("--noise_dim",    default=8,          type=int)
    p.add_argument("--disc_h_dim",   default=64,         type=int)
    p.add_argument("--disc_mlp_dim", default=256,        type=int)
    p.add_argument("--dropout",      default=0.0,        type=float)
    p.add_argument("--unet_in_ch",   default=13,         type=int)
    p.add_argument("--best_k",       default=6,          type=int,
                   help="Số mẫu dùng cho Variety Loss (code gốc: args.best_k)")
    p.add_argument("--l2_loss_weight", default=1.0,      type=float)

    # Training -- Generator (2 optimizer, đúng code gốc)
    p.add_argument("--num_epochs",   default=1200,       type=int)
    p.add_argument("--batch_size",   default=90,         type=int)
    p.add_argument("--lr_g",         default=1e-3,       type=float)
    p.add_argument("--lr_d",         default=1e-3,       type=float)
    p.add_argument("--clipping_threshold_g", default=1.5, type=float,
                   help="Gradient clip cho Generator (đúng tên biến code gốc)")
    p.add_argument("--clipping_threshold_d", default=0.0, type=float,
                   help="Gradient clip cho Discriminator (0 = tắt, đúng code gốc)")

    p.add_argument("--patience",     default=100,        type=int)
    p.add_argument("--min_epochs",   default=50,         type=int)
    p.add_argument("--val_freq",     default=5,          type=int)
    p.add_argument("--val_subset",   default=600,        type=int)
    p.add_argument("--num_workers",  default=2,          type=int)
    p.add_argument("--eval_num_ensemble", default=20,    type=int)

    # Test
    p.add_argument("--test_at_end",  action="store_true")

    # I/O
    p.add_argument("--output_dir",   default="runs/mgtcf", type=str)
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
    print(f"  MGTCF BASELINE (v2, faithful port)  |  Multi-Generator GAN")
    print(f"  Huang et al. (2023) AAAI / (2025) Nat. Commun. (TropiCycloneNet)")
    print(f"  Encoder: PaperEncoder (FNO3D + Mamba + Env_net)  ← cùng các baseline khác")
    print(f"  K={args.n_generators} generators  best_k={args.best_k}  d_noise={args.noise_dim}")
    print(f"  lr_G={args.lr_g}  lr_D={args.lr_d}")
    print(f"  Training: 3-step (discriminator_step → generator_step → net_chooser_step)")
    print(f"  Metrics: ADE / ATE / CTE @ 12h / 24h / 48h / 72h (best-of-{args.eval_num_ensemble} Roulette)")
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
    model = MGTCFModel(
        obs_len       = args.obs_len,
        pred_len      = args.pred_len,
        unet_in_ch    = args.unet_in_ch,
        n_generators  = args.n_generators,
        embedding_dim = args.embedding_dim,
        encoder_h_dim = args.encoder_h_dim,
        decoder_h_dim = args.decoder_h_dim,
        noise_dim     = args.noise_dim,
        disc_h_dim    = args.disc_h_dim,
        disc_mlp_dim  = args.disc_mlp_dim,
        dropout       = args.dropout,
        best_k        = args.best_k,
    ).to(device)

    n_params_g = sum(p.numel() for p in model.generator.parameters() if p.requires_grad)
    n_params_d = sum(p.numel() for p in model.discriminator.parameters() if p.requires_grad)
    print(f"  params : G(encoder+gens+gc_net)={n_params_g:,}  D(discriminator)={n_params_d:,}")

    # ── HAI optimizer, đúng code gốc: optimizer_g (Adam), optimizer_d (Adam) ─
    optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr_g)
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr_d)

    best_ade     = float("inf")
    patience_cnt = 0
    train_start  = time.perf_counter()

    print("=" * 70)
    print(f"  TRAINING  ({len(train_loader)} steps/epoch)")
    print("=" * 70)

    for epoch in range(args.num_epochs):
        model.train()
        sum_d, sum_g, sum_gc = 0.0, 0.0, 0.0
        t0 = time.perf_counter()

        for i, batch in enumerate(train_loader):
            bl = move(list(batch), device)

            # ── 3-step training, đúng thứ tự train_github.py gốc ────────────
            d_log  = discriminator_step(model, bl, optimizer_d, args.clipping_threshold_d)
            g_log  = generator_step(model, bl, optimizer_g, args.clipping_threshold_g,
                                    args.best_k, args.l2_loss_weight)
            gc_log = net_chooser_step(model, bl, optimizer_g, args.clipping_threshold_g)

            sum_d  += d_log["d_loss"]
            sum_g  += g_log["g_total_loss"]
            sum_gc += gc_log["gc_ce_loss"]

            if i % 30 == 0:
                print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
                      f"  d_loss={d_log['d_loss']:.4f}"
                      f"  g_loss={g_log['g_total_loss']:.4f}"
                      f"  (l2={g_log['g_l2_loss']:.3f} adv={g_log['g_adv_loss']:.3f}"
                      f" dpe≈{g_log['g_dpe_bestofk']:.1f}km)"
                      f"  gc_ce={gc_log['gc_ce_loss']:.4f}"
                      f"  gc_acc={gc_log['gc_accuracy']:.2f}")

        avg_d  = sum_d  / len(train_loader)
        avg_g  = sum_g  / len(train_loader)
        avg_gc = sum_gc / len(train_loader)

        ep_t = time.perf_counter() - t0
        print(f"  Epoch {epoch:>4}  d_loss={avg_d:.4f}  g_loss={avg_g:.4f}"
              f"  gc_ce={avg_gc:.4f}  t={ep_t:.0f}s")

        # ── ADE + ATE + CTE evaluation (best-of-K Roulette) ─────────────────
        if epoch % args.val_freq == 0:
            r = evaluate(model, val_sub_loader, device, num_ensemble=args.eval_num_ensemble)

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
                "model_type"     : "MGTCF",
                "d_loss"         : _fmt(avg_d),
                "g_loss"         : _fmt(avg_g),
                "gc_ce_loss"     : _fmt(avg_gc),
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
                    "obs_len":       args.obs_len,
                    "pred_len":      args.pred_len,
                    "unet_in_ch":    args.unet_in_ch,
                    "n_generators":  args.n_generators,
                    "embedding_dim": args.embedding_dim,
                    "encoder_h_dim": args.encoder_h_dim,
                    "decoder_h_dim": args.decoder_h_dim,
                    "noise_dim":     args.noise_dim,
                    "disc_h_dim":    args.disc_h_dim,
                    "disc_mlp_dim":  args.disc_mlp_dim,
                    "dropout":       args.dropout,
                    "best_k":        args.best_k,
                }
                torch.save({
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "best_ade"   : best_ade,
                    "model_type" : "MGTCF",
                    "paper"      : "Huang et al. 2023/2025 (faithful port)",
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
                "d_loss"     : avg_d,
                "g_loss"     : avg_g,
                "gc_ce_loss" : avg_gc,
                "seed"       : args.seed,
            }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

    total_h = (time.perf_counter() - train_start) / 3600
    print("=" * 70)
    print(f"  Model   : MGTCF (v2, faithful port)")
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