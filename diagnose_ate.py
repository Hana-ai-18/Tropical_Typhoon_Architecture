"""
diagnose_ate_postcal.py
=========================
Khác diagnose_ate_deep.py: đo speed ratio SAU KHI đã chạy speed_calibrate_pred
+ physics re-rank (tức output cuối cùng của model.sample() — đúng cái
evaluate_multi_model.py dùng để tính ADE/ATE/CTE thật). Mục đích: biết bias
CÒN SÓT LẠI sau khi correction[t] hiện có đã áp, để không chồng thêm lớp
correction mới lên bias đã được sửa một phần rồi (bài học từ lần sửa
speed_calibrate_pred() trước đó làm ATE tăng thay vì giảm).

USAGE:
!python /kaggle/working/Tropical_Typhoon_Architecture/diagnose_ate_postcal.py \
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
from Model.flow_matching_model import TCFlowMatching, _norm_to_deg, _step_speeds_kmh


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
def diagnose(model, loader, device, n_batches: int, n_ensemble: int):
    model.eval()

    storm_obs_speed  = []
    storm_ratio_final = []   # ratio SAU sample() đầy đủ (calib + re-rank)
    per_step_ratio    = []   # [T, B] mỗi batch, để xem theo horizon

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        bl = move(list(batch), device)
        obs_traj = bl[0]

        # dùng ĐÚNG model.sample() — output cuối cùng, giống evaluate_multi_model.py
        pred_mean, _, _ = model.sample(bl, num_ensemble=n_ensemble,
                                        use_speed_calibration=True)
        # pred_mean: [T, B, 2]

        obs_deg = _norm_to_deg(obs_traj[:, :, :2])
        obs_spd_mu = _step_speeds_kmh(obs_deg).mean(0)   # [B]

        pred_deg = _norm_to_deg(pred_mean)   # [T, B, 2]
        last_deg = obs_deg[-1]
        pts = torch.cat([last_deg.unsqueeze(0), pred_deg], 0)   # [T+1, B, 2]
        pred_step_spd = _step_speeds_kmh(pts)                    # [T, B]

        ratio_tb = pred_step_spd / obs_spd_mu.clamp(min=1.0).unsqueeze(0)   # [T,B]
        per_step_ratio.append(ratio_tb.cpu())

        ratio_mean_over_t = ratio_tb.mean(0)   # [B]
        storm_ratio_final.append(ratio_mean_over_t.cpu())
        storm_obs_speed.append(obs_spd_mu.cpu())

        print(f"  batch {i+1}/{n_batches} done (B={obs_traj.shape[1]})")

    storm_obs_speed   = torch.cat(storm_obs_speed).numpy()
    storm_ratio_final = torch.cat(storm_ratio_final).numpy()
    all_ratio_tb = torch.cat(per_step_ratio, dim=1)   # [T, N]

    print()
    print("=" * 70)
    print("SAU sample() đầy đủ — Speed ratio theo HORIZON")
    print("=" * 70)
    T = all_ratio_tb.shape[0]
    for t in range(T):
        r = all_ratio_tb[t]
        print(f"  step {t:2d} ({(t+1)*6:3d}h): mean={r.mean():.4f}  median={r.median():.4f}  "
              f"%>1.15={100*(r>1.15).float().mean():.1f}%  %<0.85={100*(r<0.85).float().mean():.1f}%")

    print()
    print("=" * 70)
    print("SAU sample() đầy đủ — Speed ratio theo STORM SPEED CATEGORY")
    print("=" * 70)
    slow = storm_obs_speed < 8.0
    med  = (storm_obs_speed >= 8.0) & (storm_obs_speed < 15.0)
    fast = storm_obs_speed >= 15.0

    def _stat(mask, name):
        if mask.sum() == 0:
            print(f"  {name}: n=0")
            return None
        vals = storm_ratio_final[mask]
        print(f"  {name:12s}: n={mask.sum():3d}  mean_ratio={vals.mean():.4f}  "
              f"median={np.median(vals):.4f}  std={vals.std():.4f}")
        return vals.mean()

    m_slow = _stat(slow, "SLOW(<8kmh)")
    m_med  = _stat(med,  "MED(8-15)")
    m_fast = _stat(fast, "FAST(>=15)")

    corr = np.corrcoef(storm_obs_speed, storm_ratio_final)[0, 1]
    print(f"\n  corr(obs_speed, ratio SAU calib) = {corr:+.4f}")

    print()
    print("=" * 70)
    print("SO SÁNH TRƯỚC vs SAU calib hiện có (từ log Q3 trước đó)")
    print("=" * 70)
    print("  TRƯỚC (candidate thô):  SLOW=2.0568  MED=1.4163  FAST=0.9949  corr=-0.4824")
    before = {"SLOW": 2.0568, "MED": 1.4163, "FAST": 0.9949}
    after  = {"SLOW": m_slow, "MED": m_med, "FAST": m_fast}
    for k in ["SLOW", "MED", "FAST"]:
        if after[k] is not None:
            resid = after[k]
            print(f"  SAU  (đã có calib hiện tại): {k}={resid:.4f}   "
                  f"(bias còn sót lại: cần nhân thêm {1/resid:.3f} để về 1.0)")

    print()
    print("Diễn giải:")
    print("  - Nếu bias SAU vẫn còn LỆCH RÕ theo category (giống pattern TRƯỚC,")
    print("    chỉ là đỡ hơn) => correction[t] hiện có CHƯA đủ để sửa hết, còn dư")
    print("    địa để thêm 1 lớp scale THEO OBS_SPEED — nhưng phải tính scale factor")
    print("    từ CHÍNH số 'SAU' này (1/resid ở trên), KHÔNG dùng lại số 'TRƯỚC'")
    print("    (đó chính là lỗi double-correct đã gây ATE tăng lần trước).")
    print("  - Nếu bias SAU đã gần 1.0 ở mọi category => correction[t] hiện có ĐÃ")
    print("    bù gần đủ, không còn nhiều dư địa để cải thiện bằng scale tuyến tính")
    print("    đơn giản nữa — cần xem hướng khác (network architecture/conditioning).")


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

    diagnose(model, loader, device, n_batches=args.n_batches, n_ensemble=args.n_ensemble)


if __name__ == "__main__":
    main()