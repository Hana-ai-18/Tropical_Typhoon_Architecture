"""
evaluate.py — TC-FlowMatching evaluation script
════════════════════════════════════════════════════════════════════════════════

THIẾT KẾ: Script này KHÔNG override bất kỳ inference parameter nào
(sigma, clip, K...) vì từ v2.1-learn các tham số đó đã là LEARNABLE
(speed_correction_logits, log_sigma_reg/heading/calib...) và được lưu
trong checkpoint. Override thủ công sẽ BỎ QUA những gì model đã học.

Chỉ cần: --checkpoint path/to/best_model.pth --dataset_root ...
"""
from __future__ import annotations

import sys, os, argparse, time, json
import sys, os, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import defaultdict
from typing import Dict

import numpy as np
import torch

from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, EMAModel,
    _norm_to_deg, _haversine_deg, _forward_azimuth,
    _unwrap,
)

# ─────────────────────────────────────────────────────────────────────────────
ST_TRANS = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
            "12h": 65.0, "24h": 130.0, "48h": 205.0, "72h": 321.0}

HORIZON_STEPS = {"12h": 1, "24h": 3, "48h": 7, "72h": 11}

# ─────────────────────────────────────────────────────────────────────────────

def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


def _ate_cte(pred_deg, gt_deg):
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    bear_ref = _forward_azimuth(gt_deg[:T-1], gt_deg[1:T])
    bear_err = _forward_azimuth(gt_deg[1:T], pred_deg[1:T])
    dist_err = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang = bear_err - bear_ref
    return dist_err * torch.cos(ang), dist_err * torch.sin(ang)


@torch.no_grad()
def run_evaluation(model, loader, device, tag="TEST",
                    n_ensemble=20, ema=None, use_tta=False) -> Dict:
    """
    Chạy evaluation trên loader.
    - Không augmentation
    - Dùng model.sample() với các learnable params đã có từ checkpoint
    - KHÔNG override sigma/K/clip
    """
    bk = None
    if ema is not None:
        try: bk = ema.apply_to(model)
        except Exception as e: print(f"  ⚠ EMA apply: {e}")

    model.eval()
    all_ade, all_ate, all_cte = [], [], []
    step_dist = defaultdict(list)
    n_batches = len(loader)

    t0 = time.time()
    for i, batch in enumerate(loader):
        bl = move(list(batch), device)
        gt = bl[1]; B = bl[0].shape[1]

        if use_tta:
            # TTA: scale obs speed x5 levels, weighted average
            obs = bl[0]; anchor = obs[-1:, :, :2].detach()
            scales = [0.875, 0.9375, 1.0, 1.0625, 1.125]
            preds, weights = [], []
            for sc in scales:
                obs_s = obs.clone()
                obs_s[..., :2] = anchor + (obs[..., :2] - anchor) * sc
                bl_s = list(bl); bl_s[0] = obs_s
                try:
                    p, _, _ = model.sample(bl_s, num_ensemble=n_ensemble)
                    preds.append(p)
                    weights.append(2.0 if abs(sc - 1.0) < 1e-6 else 1.0)
                except Exception: continue
            if not preds: continue
            tw = sum(weights)
            pred = sum(w / tw * p for w, p in zip(weights, preds))
        else:
            try:
                pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
            except Exception as e:
                print(f"  [batch {i}] sample error: {e}"); continue

        T = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
        dist = _haversine_deg(pd, gd)
        ate, cte = _ate_cte(pd, gd)

        all_ade.extend(dist.mean(0).tolist())
        if ate.shape[0] > 0:
            all_ate.extend(ate.abs().mean(0).tolist())
            all_cte.extend(cte.abs().mean(0).tolist())
        for h, s in HORIZON_STEPS.items():
            if s < T: step_dist[h].extend(dist[s].tolist())

    if bk is not None:
        try: ema.restore(model, bk)
        except Exception: pass

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")
    elapsed = time.time() - t0

    result = {
        "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
        "n":   len(all_ade), "time_s": elapsed,
    }
    for h in HORIZON_STEPS: result[h] = _m(step_dist[h])

    # Print
    def _ok(k):
        v = result.get(k, float("nan"))
        return "✓" if np.isfinite(v) and v < ST_TRANS.get(k, 1e9) else "✗"

    tta_tag = " [TTA]" if use_tta else ""
    print(f"\n  {'='*72}")
    print(f"  [{tag}]{tta_tag}  n={result['n']}  ({elapsed:.1f}s)")
    print(f"  ADE={result['ADE']:7.1f}km {_ok('ADE')}  "
          f"ATE={result['ATE']:7.1f}km {_ok('ATE')}  "
          f"CTE={result['CTE']:7.1f}km {_ok('CTE')}")
    print(f"  12h={result['12h']:6.1f}  24h={result['24h']:6.1f}  "
          f"48h={result['48h']:6.1f}  72h={result['72h']:6.1f} km")
    beat = [f"{k}={result.get(k,999):.0f}<{ST_TRANS.get(k,999):.0f}"
            for k in ["ADE","ATE","CTE","12h","24h","48h","72h"]
            if np.isfinite(result.get(k, float("nan")))
            and result.get(k, 999) < ST_TRANS.get(k, 1e9)]
    print(f"  BEAT: {' | '.join(beat) if beat else 'none yet'}")

    # So sánh với ST-Trans
    print(f"\n  COMPARISON vs ST-Trans:")
    print(f"  {'Metric':<8} {'ST-Trans':>10} {'Model':>10} {'Δ':>8} Status")
    print(f"  {'─'*50}")
    for k in ["ADE","ATE","CTE"]:
        v = result.get(k, float("nan"))
        ref = ST_TRANS[k]
        delta = v - ref
        status = "✓ BEAT" if v < ref else "✗"
        print(f"  {k:<8} {ref:>10.1f} {v:>10.1f} {delta:>+8.1f}km  {status}")
    print(f"  {'='*72}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="TC-FlowMatching evaluation")
    parser.add_argument("--checkpoint",    required=True, help="Path to checkpoint .pth")
    parser.add_argument("--dataset_root",  required=True, help="Dataset root directory")
    parser.add_argument("--split",         default="test", choices=["test","val","train"])
    parser.add_argument("--n_ensemble",    type=int,   default=20,
                        help="K ensemble samples (default: 20, same as training)")
    parser.add_argument("--use_ema",       action="store_true", default=True,
                        help="Use EMA weights if available in checkpoint (default: True)")
    parser.add_argument("--no_ema",        action="store_true",
                        help="Force disable EMA even if available")
    parser.add_argument("--tta",           action="store_true",
                        help="Test-time augmentation (speed scale TTA)")
    parser.add_argument("--save_json",     type=str, default="",
                        help="Save results to JSON file")
    parser.add_argument("--gpu",           type=int, default=0)
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n  Checkpoint: {args.checkpoint}")
    print(f"  Split:      {args.split}  |  Device: {device}")
    print(f"  n_ensemble: {args.n_ensemble} (model defaults used, no override)")
    print("="*72)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ck = torch.load(args.checkpoint, map_location="cpu")
    is_swa = ck.get("is_swa", False)

    # Reconstruct model từ checkpoint config
    model_cfg = ck.get("model_cfg", {})
    model = TCFlowMatching(**model_cfg).to(device)

    # Load weights
    state = ck.get("model", ck)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"  ⚠ Missing keys: {missing[:5]}...")
    if unexpected: print(f"  ⚠ Unexpected keys: {unexpected[:5]}...")

    ep = ck.get("epoch", "?")
    print(f"  Loaded ep{ep} (is_swa={is_swa})")

    # In các giá trị đã học để kiểm tra
    raw = _unwrap(model)
    import torch.nn.functional as F
    if hasattr(raw, "speed_correction_logits"):
        corr = (torch.sigmoid(raw.speed_correction_logits) * 2.0).tolist()
        print(f"  [LEARN] speed_correction (12h-72h): {[f'{v:.3f}' for v in corr[:4]]}...")
    if hasattr(raw, "log_sigma_reg"):
        print(f"  [LEARN] log_sigma: reg={raw.log_sigma_reg.item():.3f}  "
              f"heading={raw.log_sigma_heading.item():.3f}  "
              f"calib={raw.log_sigma_calib.item():.3f}")
        prec_r = torch.exp(-2.0*raw.log_sigma_reg.clamp(min=-3.0))
        prec_h = torch.exp(-2.0*raw.log_sigma_heading.clamp(min=-3.0))
        prec_c = torch.exp(-2.0*raw.log_sigma_calib.clamp(min=-3.0))
        print(f"  [LEARN] eff_lambda: reg={0.5*prec_r.item():.3f}  "
              f"heading={0.5*prec_h.item():.3f}  "
              f"calib={0.5*prec_c.item():.3f}")

    # ── EMA ──────────────────────────────────────────────────────────────────
    ema = None
    use_ema = args.use_ema and not args.no_ema and not is_swa
    if use_ema and ck.get("ema"):
        try:
            ema = EMAModel(model)
            for k, v in ck["ema"].items():
                if k in ema.shadow:
                    ema.shadow[k].copy_(v.to(device))
            print(f"  EMA loaded ({len(ema.shadow)} params)")
        except Exception as e:
            print(f"  ⚠ EMA load failed: {e}"); ema = None

    # ── Data ──────────────────────────────────────────────────────────────────
    try:
        # _, loader = data_loader(
        #     argparse.Namespace(dataset_root=args.dataset_root),
        #     {"root": args.dataset_root, "type": args.split},
        #     test=(args.split != "train"))
        _, loader = data_loader(
            argparse.Namespace(
                dataset_root=args.dataset_root,
                obs_len=8,
                pred_len=12,
                num_workers=2,
                other_modal="gph",
                delim=" ",
                skip=1,
                min_ped=1,
                threshold=0.002,
                batch_size=64,
            ),
            {"root": args.dataset_root, "type": args.split},
            test=(args.split != "train"))
        print(f"  {args.split}: {sum(1 for _ in loader)*loader.batch_size} sequences, "
              f"{len(loader)} batches")
    except Exception as e:
        print(f"  Data loader error: {e}"); return

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = {}

    # Standard
    r = run_evaluation(model, loader, device,
                       tag=f"{args.split.upper()} (ep{ep})",
                       n_ensemble=args.n_ensemble, ema=ema, use_tta=False)
    results["standard"] = {k: v for k, v in r.items()
                            if isinstance(v, (int, float))}

    # TTA nếu yêu cầu
    if args.tta:
        r_tta = run_evaluation(model, loader, device,
                                tag=f"{args.split.upper()} TTA",
                                n_ensemble=args.n_ensemble, ema=ema, use_tta=True)
        results["tta"] = {k: v for k, v in r_tta.items()
                           if isinstance(v, (int, float))}
        print(f"  TTA Δ: ADE {r['ADE']:.1f}→{r_tta['ADE']:.1f}km  "
              f"ATE {r['ATE']:.1f}→{r_tta['ATE']:.1f}km  "
              f"CTE {r['CTE']:.1f}→{r_tta['CTE']:.1f}km")

    # Save JSON
    if args.save_json:
        results["checkpoint"] = args.checkpoint
        results["split"] = args.split
        results["epoch"] = ep
        results["n_ensemble"] = args.n_ensemble
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {args.save_json}")


if __name__ == "__main__":
    main()