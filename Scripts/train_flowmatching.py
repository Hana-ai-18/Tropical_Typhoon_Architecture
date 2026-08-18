from __future__ import annotations
import json, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_TRAIN_SCRIPT_PATCH_VERSION = "v3-gradnan-guard-2026-08-06"
print(f">>> train_flowmatching.py PATCH VERSION: {_TRAIN_SCRIPT_PATCH_VERSION} <<<")

import argparse, math, time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler

from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _haversine_deg,
    EMAModel, augment_batch, hard_score_from_obs,
    compute_ensemble_uncertainty,
    compute_heading_deviation, compute_cte_contribution,
    compute_obs_attribution,
)

HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}
ST_TRANS_VAL  = {"ADE": 172.68, "ATE": 142.21, "CTE": 42.04,
                 "12h": 65.42, "24h": 104.67, "48h": 205.10, "72h": 321.39}
ST_TRANS_TEST = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
                 "12h": 77.5, "24h": 130.5, "48h": 269.9, "72h": 423.3}

def _unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m

def move(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
    return out

def _ate_cte(pred_deg, gt_deg):
    """Decompose error into along-track and cross-track components."""
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    lo1 = torch.deg2rad(gt_deg[:T-1,:,0]); la1 = torch.deg2rad(gt_deg[:T-1,:,1])
    lo2 = torch.deg2rad(gt_deg[1:T, :,0]); la2 = torch.deg2rad(gt_deg[1:T, :,1])
    lo3 = torch.deg2rad(pred_deg[1:T,:,0]); la3 = torch.deg2rad(pred_deg[1:T,:,1])
    ya  = torch.sin(lo2-lo1)*torch.cos(la2)
    xa  = torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
    be  = torch.atan2(ya, xa)
    ye  = torch.sin(lo3-lo2)*torch.cos(la3)
    xe  = torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
    bee = torch.atan2(ye, xe)
    tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang = bee - be
    return tot*torch.cos(ang), tot*torch.sin(ang)

def build_optimizer(model, lr_velocity, lr_encoder, weight_decay,
                     lr_logits_scale: float = 0.2,
                     lr_extra_scale: float = 0.2):
    raw = _unwrap(model)

    encoder_ids  = {id(p) for p in raw.encoder.parameters()}
    velocity_ids = {id(p) for p in raw.velocity.parameters()}
    covered_ids  = encoder_ids | velocity_ids

    # [EMA-NORMALIZED-HORIZON UPDATE] reg_step_logits and heading_step_logits
    # were removed from the model (replaced by non-learnable EMA buffers
    # reg_dist_ema / heading_err_ema, which are NOT nn.Parameters and
    # therefore never appear in named_parameters() or any optimizer group
    # below -- they update via their own no_grad EMA rule inside the loss
    # functions, exactly like BatchNorm running stats). Only
    # hard_score_weight_logits remains a real softmax-based learnable
    # logit vector in this category.
    softmax_logit_names = {"hard_score_weight_logits"}
    softmax_logit_params, rest_extra_params = [], []
    for name, p in raw.named_parameters():
        if id(p) in covered_ids:
            continue
        short = name.rsplit(".", 1)[-1]
        (softmax_logit_params if short in softmax_logit_names
         else rest_extra_params).append(p)

    groups = [
        {"params": list(raw.encoder.parameters()),  "lr": lr_encoder,  "name": "encoder"},
        {"params": list(raw.velocity.parameters()), "lr": lr_velocity, "name": "velocity"},
    ]
    if len(rest_extra_params) > 0:
        # [ROOT-CAUSE FIX] rest_extra_params contains speed_correction_logits,
        # score_weight_logits, score_v_sigma_scale_logit,
        # score_kernel_scale_logits, log_sigma_reg/heading/calib/score —
        # all LOW-DIMENSIONAL (1-12 elements each), highly sensitive
        # scalars/vectors that directly parameterize exp()/softplus()/
        # sigmoid() calls feeding physics-score selection and speed
        # calibration. Previously these shared lr_velocity (2e-4) — the
        # SAME learning rate used for the velocity network's millions of
        # parameters. For a network with millions of dimensions, a given
        # gradient magnitude is naturally "diluted"; for a 1-12 element
        # parameter, the same gradient magnitude produces a much larger
        # relative step. Combined with AdamW's per-parameter adaptive
        # moment estimates, this repeatedly produced anomalously large
        # updates to exactly this parameter group (confirmed empirically:
        # first_bad_params in training logs consistently listed ONLY
        # members of this group, epoch after epoch, never encoder/velocity
        # weights). Scaling down to lr_velocity * lr_extra_scale (same
        # 0.2x factor already proven safe for the structurally similar
        # softmax_logits group below) directly addresses the mechanism
        # that was producing NaN gradients at the source, rather than
        # only cleaning up after the fact.
        groups.append({"params": rest_extra_params,
                        "lr": lr_velocity * lr_extra_scale,
                        "name": "learnable_extra"})
        print(f"  [build_optimizer] learnable_extra group: {len(rest_extra_params)} tensors "
              f"({sum(p.numel() for p in rest_extra_params)} params) — "
              f"speed_correction/log_sigma*/score_* @ lr×{lr_extra_scale} "
              f"(reduced from 1.0x — low-dim scalars, prone to NaN at full velocity-net lr)")
    if len(softmax_logit_params) > 0:
        groups.append({"params": softmax_logit_params,
                        "lr": lr_velocity * lr_logits_scale,
                        "name": "softmax_logits"})
        print(f"  [build_optimizer] softmax_logits group: {len(softmax_logit_params)} tensors "
              f"({sum(p.numel() for p in softmax_logit_params)} params) — "
              f"hard_score_weight_logits @ lr×{lr_logits_scale} "
              f"(slower convergence, less seed-init sensitivity)")

    return optim.AdamW(groups, weight_decay=weight_decay)

def get_lrs(opt):
    lr_enc = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "encoder")
    lr_vel = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "velocity")
    return lr_enc, lr_vel

class TwoGroupScheduler:
    def __init__(self, opt, warmup_epochs, total_epochs,
                 lr_vel, lr_vel_min, freeze_end_ep,
                 lr_enc_peak, encoder_warmup_epochs=5):
        self.opt         = opt
        self.warmup      = warmup_epochs
        self.total       = total_epochs
        self.lr_vel      = lr_vel
        self.lr_vel_min  = lr_vel_min
        self.freeze_end  = freeze_end_ep
        self.lr_enc_peak = lr_enc_peak
        self.enc_warmup  = encoder_warmup_epochs
        self.epoch       = 0
        
        self._lr_ratio = {}
        for pg in self.opt.param_groups:
            name = pg.get("name")
            if name not in (None, "encoder"):
                self._lr_ratio[name] = pg["lr"] / lr_vel if lr_vel > 0 else 1.0

    def _cosine(self, ep_from, ep_to, lr_s, lr_e, ep):
        t = max(0., min(1., (ep - ep_from) / max(ep_to - ep_from, 1)))
        return lr_e + 0.5 * (lr_s - lr_e) * (1 + math.cos(math.pi * t))

    def step(self):
        ep = self.epoch
        if ep < self.warmup:
            lr_vel = self.lr_vel * (0.1 + 0.9 * ep / max(self.warmup - 1, 1))
        else:
            lr_vel = self._cosine(self.warmup, self.total, self.lr_vel, self.lr_vel_min, ep)
        if ep < self.freeze_end:
            lr_enc = 0.0
        elif ep < self.freeze_end + self.enc_warmup:
            lr_enc = self.lr_enc_peak * (ep - self.freeze_end) / self.enc_warmup
        else:
            lr_enc = self._cosine(self.freeze_end + self.enc_warmup, self.total,
                                  self.lr_enc_peak, self.lr_vel_min, ep)
        for pg in self.opt.param_groups:
            name = pg.get("name")
            if name == "encoder":
                pg["lr"] = lr_enc
            elif name in self._lr_ratio:
                # Preserve this group's original ratio to lr_vel (e.g.
                # softmax_logits at lr_logits_scale=0.2) through the same
                # cosine schedule velocity follows, instead of resetting to
                # lr_vel outright.
                pg["lr"] = lr_vel * self._lr_ratio[name]
            else:
                pg["lr"] = lr_vel
        self.epoch += 1
        return lr_vel, lr_enc

class SWAHandler:

    def __init__(self, swa_lr: float = 2e-6):
        self.swa_lr    = swa_lr
        self.active    = False
        self.start_ep  = None
        self.n_updates = 0
        self.avg_state = {}

    def should_activate(self, ade_history: List[float],
                         window: int = 3, threshold: float = 1.5) -> bool:
        if len(ade_history) < window: return False
        return (ade_history[-window] - ade_history[-1]) < threshold

    def activate(self, model, opt, ep: int):
        self.active = True; self.start_ep = ep
        for pg in opt.param_groups: pg["lr"] = self.swa_lr
        m = _unwrap(model)
        # [BUG FOUND, SAME CLASS AS EMAModel's earlier fix]
        # reg_dist_ema/heading_err_ema (and their *_warmed companion flags)
        # are running STATISTICS of per-horizon prediction error, updated
        # by their own no_grad EMA rule inside _reg_loss/_heading_loss_ms
        # (same role as BatchNorm's running_mean) -- NOT learned weights
        # that should be smoothed across checkpoints the way SWA smooths
        # the rest of the model. EMAModel already excludes these by name
        # (_HORIZON_EMA_BUFFER_NAMES) for exactly this reason, but
        # SWAHandler used a plain dtype filter with no equivalent
        # exclusion, so SWA-averaging these buffers would corrupt the
        # per-horizon error-difficulty signal (averaging together error
        # magnitudes from potentially quite different points across the
        # SWA-active window) instead of leaving it as the live, single
        # most-recent-batch statistic it is designed to be. Mirrors
        # EMAModel's exclusion set exactly so both mechanisms agree on
        # what counts as "real" model state vs. a running diagnostic.
        _excluded = {"reg_dist_ema", "heading_err_ema",
                     "reg_dist_ema_warmed", "heading_err_ema_warmed"}
        self.avg_state = {k: v.detach().clone().float()
                          for k, v in m.state_dict().items()
                          if v.dtype.is_floating_point
                          and k.rsplit(".", 1)[-1] not in _excluded}
        self.n_updates = 1
        print(f"  *** SWA ACTIVATED @ ep{ep} (lr → {self.swa_lr:.1e}) ***")

    def update(self, model):
        if not self.active: return
        m = _unwrap(model); sd = m.state_dict(); n = self.n_updates
        for k in self.avg_state:
            if k in sd:
                self.avg_state[k] = (n * self.avg_state[k] + sd[k].detach().float()) / (n + 1)
        self.n_updates += 1

    def apply_to_model(self, model):
        if not self.active or not self.avg_state: return
        m = _unwrap(model); sd = m.state_dict()
        for k in self.avg_state:
            if k in sd: sd[k].copy_(self.avg_state[k].to(sd[k].device))

    def restore_from_backup(self, model, backup):
        m = _unwrap(model); sd = m.state_dict()
        for k, v in backup.items():
            if k in sd: sd[k].copy_(v)

    def save_avg_state(self, path: str, epoch: int, best_score: float,
                       extra: Optional[dict] = None, model_cfg=None):
        payload = {"epoch": epoch, "model": self.avg_state,
                   "best_score": best_score, "is_swa": True,
                   "swa_updates": self.n_updates,
                   "model_cfg": model_cfg}
        if extra: payload.update(extra)
        torch.save(payload, path)


@torch.no_grad()
def evaluate_hard_val(model, val_loader, device, hard_threshold: float = 0.35,
                       n_ensemble: int = 20, ema=None, epoch_for_loss: int = 9999):
 
    bk = None
    if ema is not None:
        try: bk = ema.apply_to(model)
        except Exception: pass

    model.eval()
    all_ade, all_ate, all_cte = [], [], []
    n_hard = 0

    for batch in val_loader:
        bl = move(list(batch), device)
        B  = bl[0].shape[1]
        h_score   = hard_score_from_obs(bl[0][:, :, :2])
        hard_mask = h_score > hard_threshold
        if hard_mask.sum() == 0: continue
        hard_idx  = hard_mask.nonzero(as_tuple=True)[0]

        bl_h = list(bl)
        for i, item in enumerate(bl_h):
            if torch.is_tensor(item):
                if item.dim() >= 2 and item.shape[1] == B:
                    bl_h[i] = item[:, hard_idx, ...]
                elif item.dim() >= 1 and item.shape[0] == B:
                    bl_h[i] = item[hard_idx, ...]

        try:
            # [CURV-SCORE-ENABLE] use_curvature_score=True was coded, tested,
            # and marked "opt-in, A/B-testable on any existing checkpoint,
            # no retraining needed" — but was never actually turned on at
            # any eval/test call site in this file (checked: every prior
            # model.sample() call used the default False). It is the ONLY
            # scoring component that checks whether a candidate's turning
            # RATE over the full horizon matches the storm's observed
            # turning rate; head_score only checks step-0 direction and
            # smooth_score actively penalizes turning everywhere, so for
            # recurving storms the best-of-K selection was systematically
            # biased toward straight-line candidates at 48h-72h. Enabling
            # this requires no retraining to test its effect on an existing
            # checkpoint.
            pred, _, _ = model.sample(bl_h, num_ensemble=n_ensemble, use_curvature_score=True)
        except Exception as e:
            print(f"  hard val error: {e}"); continue

        gt = bl_h[1]
        T  = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
        dist = _haversine_deg(pd, gd)
        ate, cte = _ate_cte(pd, gd)
        all_ade.extend(dist.mean(0).tolist())
        if ate.shape[0] > 0:
            all_ate.extend(ate.abs().mean(0).tolist())
            all_cte.extend(cte.abs().mean(0).tolist())
        n_hard += len(hard_idx)

    if bk is not None:
        try: ema.restore(model, bk)
        except Exception: pass

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")
    return {
        "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
        "n_hard": n_hard,
        "combined_score": 0.6*_m(all_ade) + 0.2*_m(all_ate) + 0.2*_m(all_cte),
    }

@torch.no_grad()
def evaluate(model, loader, device, tag: str = "",
             n_ensemble: int = 20, ema=None,
             ref_targets=None, use_tta: bool = False,
             n_tta: int = 5, epoch_for_loss: int = 9999,
             run_xai: bool = False, xai_batch=None) -> Dict:
  
    bk = None
    if ema is not None:
        try: bk = ema.apply_to(model)
        except Exception as e: print(f"  ⚠ EMA: {e}")

    model.eval()
    all_ade, all_ate, all_cte = [], [], []
    step_dist = defaultdict(list)
    sum_loss = sum_cfm = sum_head = 0.0
    sum_n = 0

    for batch in loader:
        bl = move(list(batch), device)
        gt = bl[1]; B = bl[0].shape[1]

       
        try:
            bd = model.get_loss_breakdown(bl, epoch=epoch_for_loss)
            if torch.isfinite(bd["total"]):
                sum_loss += bd["total"].item() * B
                sum_cfm  += bd["l_cfm"] * B
                sum_head += bd["l_heading"] * B
                sum_n    += B
        except Exception: pass

        if use_tta:
            obs = bl[0]; anchor = obs[-1:, :, :2].detach()
            scales = [0.875, 0.9375, 1.0, 1.0625, 1.125][:n_tta]
            preds_t, weights_t = [], []
            for sc in scales:
                obs_s = obs.clone(); obs_s[..., :2] = anchor + (obs[..., :2] - anchor) * sc
                bl_s = list(bl); bl_s[0] = obs_s
                try:
                    p, _, _ = model.sample(bl_s, num_ensemble=n_ensemble, use_curvature_score=True)
                    preds_t.append(p)
                    weights_t.append(2.0 if abs(sc - 1.0) < 1e-6 else 1.0)
                except Exception: continue
            if not preds_t: continue
            tw   = sum(weights_t)
            pred = sum(w / tw * p for w, p in zip(weights_t, preds_t))
        else:
            try:
                pred, _, _ = model.sample(bl, num_ensemble=n_ensemble, use_curvature_score=True)
            except Exception as e:
                print(f"  sample error: {e}"); continue

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
    val_loss = sum_loss / max(sum_n, 1)

    result = {
        "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
        "n": len(all_ade),
        "val_loss": val_loss,
        "val_cfm_loss":  sum_cfm / max(sum_n, 1),
        "val_head_loss": sum_head / max(sum_n, 1),
        "val_mom_loss":  0.0,  
    }
    for h in HORIZON_STEPS: result[f"{h}h"] = _m(step_dist[h])
    ade, ate_, cte_ = result["ADE"], result["ATE"], result["CTE"]
    result["combined_score"] = (
        0.6 * ade + 0.2 * ate_ + 0.2 * cte_
        if all(np.isfinite(x) for x in [ade, ate_, cte_]) else ade)

    ref = ref_targets or ST_TRANS_VAL
    def _v(k): return result.get(k, float("nan"))
    def _ok(k): return "✓" if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9) else "✗"

    tta_str = " [TTA]" if use_tta else ""
    print(f"\n  {'='*72}")
    print(f"  [{tag}]{tta_str}  n={result['n']}")
    print(f"  Val Loss : {val_loss:.6f}  cfm={result['val_cfm_loss']:.6f}  "
          f"head4s={result['val_head_loss']:.6f}  [mom=DISABLED]")
    print(f"  ADE={_v('ADE'):7.1f}km {_ok('ADE')}  "
          f"ATE={_v('ATE'):7.1f}km {_ok('ATE')}  "
          f"CTE={_v('CTE'):7.1f}km {_ok('CTE')}")
    print(f"  Combined = {_v('combined_score'):.1f}")
    print(f"  12h={_v('12h'):6.1f}  24h={_v('24h'):6.1f}  "
          f"48h={_v('48h'):6.1f}  72h={_v('72h'):6.1f} km")
    beat = [f"{k}={_v(k):.0f}<{ref.get(k,999):.0f}"
            for k in ["ADE","ATE","CTE","12h","24h","48h","72h"]
            if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9)]
    print(f"  BEAT: {' | '.join(beat) if beat else 'none yet'}")

 
    if run_xai and xai_batch is not None:
        try:
            _, _, _, xai = _unwrap(model).sample(xai_batch, return_xai=True, use_curvature_score=True)

            print(f"  {'─'*60}")
            print(f"  XAI Summary (fixed val batch)")
            print(f"  {'─'*60}")

            # XAI-4: Uncertainty
            print(f"  [XAI-4] Uncertainty:"
                  f" 12h={xai['mean_12h_std']:.1f}km"
                  f"  72h={xai['mean_72h_std']:.1f}km"
                  f"  ratio={float(xai['uncertainty_ratio'].mean()):.2f}×"
                  f"  high_uncert={xai['high_uncertainty'].sum().item()}")

            # XAI-2: Hard score components
            hc = xai.get("hard_components", {})
            if hc:
                print(f"  [XAI-2] HardScore:"
                      f" curv={float(hc['curvature'].mean()):.3f}"
                      f"  spd_var={float(hc['speed_var'].mean()):.3f}"
                      f"  dir_chg={float(hc['dir_change'].mean()):.3f}"
                      f"  obs_spd_n={float(hc.get('obs_speed_norm', torch.zeros(1)).mean()):.3f}")

            # XAI-3: Physics score components
            pc = xai.get("physics_components", {})
            if pc:
                print(f"  [XAI-3] Physics:"
                      f" speed={float(pc['speed'].mean()):.3f}"
                      f"  smooth={float(pc['smooth'].mean()):.3f}"
                      f"  heading={float(pc['heading'].mean()):.3f}")

            # XAI-5: Speed comparison (shows calibration effect)
            sc = xai.get("speed_comparison", {})
            if sc:
                ratio = sc.get("speed_ratio", 1.0)
                flag  = ("OVER" if ratio > 1.15 else
                         "UNDER" if ratio < 0.85 else "OK")
                n_over  = sc.get("over_predict",  torch.zeros(1)).sum().item()
                n_under = sc.get("under_predict", torch.zeros(1)).sum().item()
                print(f"  [XAI-5] Speed (post-calibration):"
                      f" obs={sc['obs_speed_mean']:.1f}km/h"
                      f"  pred={sc['pred_speed_mean']:.1f}km/h"
                      f"  ratio={ratio:.2f} {flag}"
                      f"  (over:{int(n_over)} under:{int(n_under)})")

            # XAI-6: Heading deviation — key metric for heading loss efficacy
            hd = xai.get("heading_deviation_deg")
            if hd is not None and hd.shape[0] >= 1:
                hd_mean = hd.mean(1)
                print(f"  [XAI-6] Heading deviation:"
                      f" 12h={hd_mean[0].item():.1f}°"
                      f"  24h={hd_mean[min(2,len(hd_mean)-1)].item():.1f}°"
                      f"  72h={hd_mean[min(10,len(hd_mean)-1)].item():.1f}°")

            # XAI-7: ATE/CTE decomposition
            ac = xai.get("ate_cte_decomp", {})
            if ac:
                print(f"  [XAI-7] Error:"
                      f" ATE={ac['ate_abs_mean']:.1f}km"
                      f"  CTE={ac['cte_abs_mean']:.1f}km"
                      f"  ratio={ac['ate_abs_mean']/(ac['cte_abs_mean']+1e-3):.2f}")

            # XAI-8: Per-horizon speed 
            sph = xai.get("speed_per_horizon", {})
            if sph and "pred_kmh" in sph:
                ps = sph["pred_kmh"]; gs = sph["gt_kmh"]; rs = sph["ratio"]
                hz = [(0,"12h"),(2,"24h"),(6,"48h"),(10,"72h")]
                parts = []
                for idx, lbl in hz:
                    if idx < len(rs):
                        flag = "FAIL" if rs[idx] > 1.3 or rs[idx] < 0.7 else "OK"
                        parts.append(f"{lbl}:r={rs[idx]:.2f}{flag}")
                print(f"  [XAI-8] Speed/horizon: {' | '.join(parts)}")

            # XAI-9: Storm category 
            sc9 = xai.get("storm_categories", {})
            if sc9:
                print(f"  [XAI-9] Storms:"
                      f" slow={sc9.get('n_slow',0)}"
                      f"  med={sc9.get('n_medium',0)}"
                      f"  fast={sc9.get('n_fast',0)}"
                      f"  spd={sc9.get('speed_mean',0):.1f}±{sc9.get('speed_std',0):.1f}km/h")

         
            lp = xai.get("learned_params", {})
            if lp:
                sc_str  = ",".join(f"{v:.2f}" for v in lp.get("speed_correction", [])[:4])
                # [BUG FOUND] These previously read "reg_step_weights",
                # "b_horizon", and "heading_step_weights" -- keys that no
                # longer exist in learned_params after EMA-NORMALIZED-
                # HORIZON replaced the old softmax-weight mechanism (see
                # flow_matching_model.py's get_xai/learned_params dict,
                # which now exposes "reg_dist_ema_km_per_horizon" and
                # "heading_err_ema_per_horizon" instead). Confirmed
                # directly from the seed2 training log: reg_w and
                # b_horizon printed as empty ([] and [?]) for all 220
                # epochs, because .get(old_key, []) always fell through to
                # the empty-list default. Updated to read the current key
                # names; values now reflect the true per-horizon EMA error
                # (km for reg, dimensionless relative angular error for
                # heading) rather than a defunct softmax weight.
                rw_full = lp.get("reg_dist_ema_km_per_horizon", [])
                rw_str  = ",".join(f"{v:.1f}" for v in rw_full[:4])
                rw_far  = ",".join(f"{v:.1f}" for v in rw_full[7:12]) if len(rw_full) >= 12 else "?"
                hdw_full = lp.get("heading_err_ema_per_horizon", [])
                hdw_str = ",".join(f"{v:.3f}" for v in hdw_full[:4])
                hw_str  = ",".join(f"{v:.3f}" for v in lp.get("hard_score_weights", []))
                sig_inf = lp.get("sigma_inf", float("nan"))
                ls_r    = lp.get("log_sigma_reg",     float("nan"))
                ls_h    = lp.get("log_sigma_heading",  float("nan"))
                ls_c    = lp.get("log_sigma_calib",    float("nan"))
                el_r    = lp.get("eff_lambda_reg",     float("nan"))
                el_h    = lp.get("eff_lambda_heading", float("nan"))
                el_c    = lp.get("eff_lambda_calib",   float("nan"))
                print(f"  [LEARN] speed_corr(12h-24h)=[{sc_str}]"
                      f"  reg_ema_km(6h-24h)=[{rw_str}]")
                print(f"  [LEARN] reg_ema_km(48h-72h,idx7-11)=[{rw_far}]"
                      f"  heading_ema(6h-24h)=[{hdw_str}]")
                print(f"  [LEARN] hard_w(curv,spdvar,dirchg,obsspd)=[{hw_str}]"
                      f"  sigma_inf={sig_inf:.4f}")
                print(f"  [LEARN] log_sigma: reg={ls_r:.3f}  heading={ls_h:.3f}  calib={ls_c:.3f}"
                      f"  |  eff_lambda: reg={el_r:.3f}  heading={el_h:.3f}  calib={el_c:.3f}")

            print(f"  {'─'*60}")
            result["xai"] = xai
        except Exception as e:
            print(f"  XAI error: {e}")
            import traceback; traceback.print_exc()

    print(f"  {'='*72}\n")
    return result


def _save(path, epoch, model, opt, sched, best_score,
          ema=None, scaler=None, extra=None, model_cfg=None):
    m   = _unwrap(model)
    esd = None
    if ema is not None:
        try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
        except Exception: pass
    payload = {
        "epoch": epoch, "model": m.state_dict(),
        "optimizer": opt.state_dict(), "scheduler": sched.epoch,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_score": best_score, "best_ade": best_score, "ema": esd,
        # [FIX] self-describing checkpoint — see call site comment in main()
        "model_cfg": model_cfg,
    }
    if extra: payload.update(extra)
    torch.save(payload, path)

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   
    p.add_argument("--dataset_root",           default="TCND_vn")
    p.add_argument("--obs_len",                default=8,      type=int)
    p.add_argument("--pred_len",               default=12,     type=int)
    p.add_argument("--num_workers",            default=2,      type=int)
    p.add_argument("--other_modal",            default="gph")
    p.add_argument("--delim",                  default=" ")
    p.add_argument("--skip",                   default=1,      type=int)
    p.add_argument("--min_ped",                default=1,      type=int)
    p.add_argument("--threshold",              default=0.002,  type=float)
    p.add_argument("--filter_region",          action="store_true", default=False,
                   help="Keep only storms whose track substantially enters "
                        "the South China Sea / Vietnam region (see "
                        "--min_pct_in_scs). Off by default.")
    p.add_argument("--min_pct_in_scs",         default=15.0,   type=float,
                   help="Minimum %% of a storm's track points that must fall "
                        "inside the South China Sea / Vietnam box (lon "
                        "99-121E, lat 0-23N) for it to be kept when "
                        "--filter_region is set.")
    p.add_argument("--d_cond",                 default=256,    type=int)
    p.add_argument("--d_model",                default=256,    type=int)
    p.add_argument("--nhead",                  default=8,      type=int)
    p.add_argument("--num_dec_layers",         default=4,      type=int)
    p.add_argument("--dim_ff",                 default=512,    type=int)
    p.add_argument("--dropout",                default=0.1,    type=float)
    p.add_argument("--unet_in_ch",             default=13,     type=int)
    p.add_argument("--sigma_min",              default=0.06,   type=float)
    p.add_argument("--sigma_max",              default=0.15,   type=float)
    p.add_argument("--sigma_decay_end",        default=100,    type=int)
    p.add_argument("--lambda_reg",             default=0.2,    type=float)
    p.add_argument("--lambda_heading",         default=0.07,   type=float)
    p.add_argument("--use_curvature_score_train", action="store_true", default=True,
                   help="[FIX-CURVATURE-WEIGHT-NEVER-TRAINED] Include curvature "
                        "(whole-path turning-rate match) as a 5th component in "
                        "L_score during training, so score_weight_logits[4] "
                        "actually receives gradient instead of staying frozen "
                        "at its ~1%% init value. Independent of and complementary "
                        "to sample()'s own use_curvature_score at inference time "
                        "-- enabling only at eval-time on a checkpoint trained "
                        "without this flag has no effect (confirmed empirically).")
    p.add_argument("--lambda_momentum",        default=0.0,    type=float,
                   help="[v2.6] DISABLED — hurt test ATE by +7.9km")
    p.add_argument("--lambda_hard_reg",        default=0.02,   type=float)
    p.add_argument("--log_sigma_reg_min_clamp", default=-3.0,   type=float)
    p.add_argument("--disable_horizon_nll",    action="store_true", default=False)
    p.add_argument("--use_ot",                 default=True,   action="store_true")
    p.add_argument("--no_ot",                  dest="use_ot",  action="store_false")
    p.add_argument("--ot_epsilon",             default=0.05,   type=float)
    p.add_argument("--n_ensemble",             default=20,     type=int)
    p.add_argument("--sigma_inference",        default=0.04,   type=float)
    p.add_argument("--n_inference_steps",      default=10,     type=int)
  
    p.add_argument("--num_epochs",             default=250,    type=int)
    p.add_argument("--batch_size",             default=64,     type=int)
    p.add_argument("--lr",                     default=2e-4,   type=float)
    p.add_argument("--lr_logits_scale",        default=0.2,    type=float)
    p.add_argument("--lr_extra_scale",         default=0.2,    type=float,
                   help="[ROOT-CAUSE FIX] LR multiplier for the "
                        "learnable_extra group (speed_correction_logits, "
                        "score_weight_logits, score_v_sigma_scale_logit, "
                        "score_kernel_scale_logits, log_sigma_*). These "
                        "are low-dimensional (1-12 elements), sensitive "
                        "scalars feeding exp()/softplus()/sigmoid() calls; "
                        "previously shared the full velocity-network LR "
                        "(2e-4), which repeatedly produced NaN gradients "
                        "in exactly this group (confirmed via "
                        "first_bad_params logging). Reduced to match the "
                        "already-proven-safe softmax_logits scale.")
    p.add_argument("--lr_min",                 default=1e-6,   type=float)
    p.add_argument("--warmup_epochs",          default=5,      type=int)
    p.add_argument("--weight_decay",           default=1e-4,   type=float)
    p.add_argument("--grad_clip",              default=1.0,    type=float)
    p.add_argument("--use_amp",                action="store_true", default=False)
    p.add_argument("--use_ema",                default=True,   action="store_true")
    p.add_argument("--no_ema",                 dest="use_ema", action="store_false")
    # Encoder freeze
    p.add_argument("--freeze_encoder_epochs",  default=10,     type=int)
    p.add_argument("--encoder_warmup_epochs",  default=5,      type=int)
    p.add_argument("--lr_enc_peak",            default=5e-5,   type=float)
    # Eval
    p.add_argument("--val_freq",               default=5,      type=int)
    p.add_argument("--patience",               default=40,     type=int)
    p.add_argument("--min_ep",                 default=20,     type=int)
    p.add_argument("--hard_val_threshold",     default=0.35,   type=float)
    p.add_argument("--hard_val_freq",          default=10,     type=int)
    # SWA
    p.add_argument("--swa_lr",                 default=2e-6,   type=float)
    p.add_argument("--swa_window",             default=3,      type=int)
    p.add_argument("--swa_threshold",          default=1.5,    type=float)
    p.add_argument("--swa_min_ep",             default=50,     type=int)
    # Test
    p.add_argument("--tta_test",               default=True,   action="store_true")
    p.add_argument("--n_tta",                  default=5,      type=int)
    p.add_argument("--multiscale_test",        default=True,   action="store_true")
    # IO
    p.add_argument("--output_dir",             default="runs/fm_v26")
    p.add_argument("--gpu_num",                default="0")
    p.add_argument("--resume",                 default=None)
    p.add_argument("--test_at_end",            action="store_true", default=True)
    p.add_argument("--no_test",                dest="test_at_end", action="store_false")
   
    p.add_argument("--seed",                   type=int, default=42)
    p.add_argument("--disable_l_heading",      action="store_true", default=False,
                   help="Ablation: disable L_heading_ms")
    p.add_argument("--disable_l_calib",        action="store_true", default=False,
                   help="Ablation: disable L_calib (speed correction training)")
    p.add_argument("--disable_l_reg",          action="store_true", default=False,
                   help="Ablation: disable L_reg (CFM-only variant)")
    p.add_argument("--disable_aug_c",          action="store_true", default=False,
                   help="Ablation: disable AUG-C recurvature")
    p.add_argument("--disable_learned_weights",action="store_true", default=False,
                   help="Ablation: fixed lambda (no Kendall weighting)")
    p.add_argument("--disable_hard_reg",       action="store_true", default=False,
                   help="Ablation: disable hard_score_weight_logits uniform "
                        "regularizer (same as --lambda_hard_reg 0.0)")
    p.add_argument("--ablation_name",          type=str, default="")
    return p.parse_args()

def main(args):
   
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
   
    if args.ablation_name:
        args.output_dir = f"{args.output_dir}_{args.ablation_name}"
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
        print(f" No GPU detected — training on CPU will be MUCH slower "
              f"(often 10-50x). If you expected a GPU here, check Kaggle's "
              f"Accelerator setting for this session.")
    os.makedirs(args.output_dir, exist_ok=True)

    # [SAFETY GUARD] exist_ok=True above means re-running with the same
    # --output_dir (e.g. default seed=42 with no --ablation_name, since
    # the seed suffix is only appended when seed != 42) silently starts
    # writing into a directory that may already contain a previous run's
    # checkpoints -- every _save() call uses the same fixed filenames
    # (best_model.pth, last_model.pth, etc.), so a second run overwrites
    # the first with no warning. This does not block the run (a fresh
    # directory is the common case and should proceed silently), it only
    # flags the specific situation where stale checkpoints already exist
    # and --resume was NOT requested, since that combination is the one
    # most likely to be an accidental overwrite rather than an intended
    # fresh start.
    _existing_best = os.path.join(args.output_dir, "best_model.pth")
    if os.path.exists(_existing_best) and not args.resume:
        print(f"  ⚠ WARNING: {_existing_best} already exists and --resume "
              f"was not passed -- this run will OVERWRITE it as training "
              f"progresses. If this is unintentional (e.g. seed=42 reused "
              f"the default --output_dir from a previous run), stop now "
              f"and pass a distinct --output_dir or --resume the prior run.")

  
    _wall_start = time.time()
    best_ckpt      = os.path.join(args.output_dir, "best_model.pth")
    hard_best_ckpt = os.path.join(args.output_dir, "hard_best_model.pth")
    swa_ckpt       = os.path.join(args.output_dir, "swa_model.pth")
    last_ckpt      = os.path.join(args.output_dir, "last_model.pth")

    print("=" * 72)
    print("  TC-FlowMatching v2.1-XAI")
    print(f"  [AUG-C]  Recurvature ±20° (PROVEN -7.1km CTE in v2.5)")
    print(f"  [L_HDG]  L_heading: EMA-normalized-horizon weighting, weight={args.lambda_heading}")
    # [BUG FOUND, same class as the training-loop summary print fixed
    # earlier] This line described speed calibration as a fixed
    # clip(0.85, 1.15) correction, but speed_calibrate_pred's own
    # docstring documents why that was replaced: XAI-8 showed the needed
    # correction varies substantially by horizon (roughly 0.77 at 12h,
    # 0.62 at 24h, 1.06 at 48h, 1.04 at 72h), which a single shared clip
    # range cannot satisfy simultaneously -- it is now a per-horizon
    # LEARNED scale, not a fixed clip. Confirmed against the training
    # logs too: speed_corr values there (e.g. 0.994/0.997/0.998/0.998)
    # are learned parameters, not clipped constants.
    print(f"  [CALIB]  Speed calibration at inference (per-horizon LEARNED, not fixed clip)")
    print(f"  [NO-D]   AUG-D removed (proven +4.5km ATE worse)")
    print(f"  KEEP:    sigma_inf={args.sigma_inference} FIXED, L_reg EMA-normalized-horizon, OT")
    print("=" * 72)

    
    print("\n  Loading data...")
    trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
    vd, val_loader = data_loader(args, {"root": args.dataset_root, "type": "val"}, test=True)
    print(f"  train: {len(trd)} ({len(trl)} batches/ep)")
    print(f"  val:   {len(vd)} ({len(val_loader)} batches)")

    
    try:
        xai_batch = move(list(next(iter(val_loader))), device)
    except Exception:
        xai_batch = None

    model = TCFlowMatching(
        pred_len=args.pred_len,        obs_len=args.obs_len,
        unet_in_ch=args.unet_in_ch,   d_cond=args.d_cond,
        d_model=args.d_model,          nhead=args.nhead,
        num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
        dropout=args.dropout,
        sigma_min=args.sigma_min,      sigma_max=args.sigma_max,
        sigma_decay_end=args.sigma_decay_end,
        lambda_reg=args.lambda_reg,    lambda_heading=args.lambda_heading,
        lambda_momentum=0.0,
        lambda_hard_reg=(0.0 if args.disable_hard_reg else args.lambda_hard_reg),
        log_sigma_reg_min_clamp=args.log_sigma_reg_min_clamp,
        enable_horizon_nll=not args.disable_horizon_nll,
        use_ot=args.use_ot,            ot_epsilon=args.ot_epsilon,
        use_ema=args.use_ema,          n_ensemble=args.n_ensemble,
        n_inference_steps=args.n_inference_steps,
        sigma_inference=args.sigma_inference,
        use_curvature_score_train=args.use_curvature_score_train,
    ).to(device)

    model_cfg = dict(
        pred_len=args.pred_len,        obs_len=args.obs_len,
        unet_in_ch=args.unet_in_ch,   d_cond=args.d_cond,
        d_model=args.d_model,          nhead=args.nhead,
        num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
        dropout=args.dropout,
        sigma_min=args.sigma_min,      sigma_max=args.sigma_max,
        sigma_decay_end=args.sigma_decay_end,
        lambda_reg=args.lambda_reg,    lambda_heading=args.lambda_heading,
        lambda_momentum=0.0,
        lambda_hard_reg=(0.0 if args.disable_hard_reg else args.lambda_hard_reg),
        log_sigma_reg_min_clamp=args.log_sigma_reg_min_clamp,
        enable_horizon_nll=not args.disable_horizon_nll,
        use_ot=args.use_ot,            ot_epsilon=args.ot_epsilon,
        use_ema=args.use_ema,          n_ensemble=args.n_ensemble,
        n_inference_steps=args.n_inference_steps,
        sigma_inference=args.sigma_inference,
        use_curvature_score_train=args.use_curvature_score_train,
    )

    model.init_ema()
    ema = getattr(_unwrap(model), "_ema", None)
    raw = _unwrap(model)
    n_enc = sum(p.numel() for p in raw.encoder.parameters())
    n_vel = sum(p.numel() for p in raw.velocity.parameters())
    encoder_ids  = {id(p) for p in raw.encoder.parameters()}
    velocity_ids = {id(p) for p in raw.velocity.parameters()}
    n_extra = sum(p.numel() for p in raw.parameters()
                  if id(p) not in encoder_ids and id(p) not in velocity_ids)
    n_total = n_enc + n_vel + n_extra
    mem_mb  = sum(p.numel()*p.element_size() for p in model.parameters()) / 1e6
    print(f"\n  Encoder: {n_enc:,}  VelocityTrans: {n_vel:,}  "
          f"LearnableExtra: {n_extra:,}  Total: {n_total:,}  Mem: {mem_mb:.1f}MB")
   
    footprint_info = {
        "n_encoder": n_enc, "n_velocity": n_vel, "n_extra": n_extra,
        "n_total": n_total, "mem_mb": mem_mb,
        "seed": args.seed,
        "ablation_name": args.ablation_name or "full",
        "disable_l_heading": getattr(args, "disable_l_heading", False),
        "disable_l_calib":   getattr(args, "disable_l_calib", False),
        "disable_l_reg":     getattr(args, "disable_l_reg", False),
        "disable_aug_c":     getattr(args, "disable_aug_c", False),
        "disable_hard_reg":  getattr(args, "disable_hard_reg", False),
        "lambda_hard_reg":   getattr(args, "lambda_hard_reg", 0.02),
        
        "disable_horizon_nll": getattr(args, "disable_horizon_nll", False),
        "use_ot":              getattr(args, "use_ot", True),
    }

    if args.disable_horizon_nll or not args.use_ot:
        print(f"  [ABLATION] Model-level: "
              f"{'no_horizon_nll(raw dist, no log_b_horizon) ' if args.disable_horizon_nll else ''}"
              f"{'no_OT(random x0/x1 pairing) ' if not args.use_ot else ''}")

    if any([args.disable_l_heading, args.disable_l_calib, args.disable_l_reg,
            args.disable_aug_c, args.disable_learned_weights, args.disable_hard_reg]):
        print(f"  [ABLATION] Disabled: "
              f"{'L_heading ' if args.disable_l_heading else ''}"
              f"{'L_calib ' if args.disable_l_calib else ''}"
              f"{'L_reg ' if args.disable_l_reg else ''}"
              f"{'AUG-C ' if args.disable_aug_c else ''}"
              f"{'Kendall_weights ' if args.disable_learned_weights else ''}"
              f"{'HardScoreReg ' if args.disable_hard_reg else ''}")
        _orig_glb = raw.get_loss_breakdown.__func__
        _abl_flags = {
            "disable_l_heading": args.disable_l_heading,
            "disable_l_calib":   args.disable_l_calib,
            "disable_l_reg":     args.disable_l_reg,
            "disable_learned_weights": args.disable_learned_weights,
            "disable_hard_reg":  args.disable_hard_reg,
        }
        def _patched_glb(self, batch_list, epoch=0, **kwargs):
            bd = _orig_glb(self, batch_list, epoch=epoch, **kwargs)
            import math as _m
          
            l_cfm      = bd["_t_l_cfm"]
            l_reg      = bd["_t_l_reg"]
            l_heading  = bd["_t_l_heading"]
            l_calib    = bd["_t_l_calib"]
           
            l_hard_reg = bd["_t_l_hard_reg"]
            if _abl_flags["disable_l_reg"]:     l_reg      = l_reg * 0.0
            if _abl_flags["disable_l_heading"]: l_heading  = l_heading * 0.0
            if _abl_flags["disable_l_calib"]:   l_calib    = l_calib * 0.0
            if _abl_flags["disable_hard_reg"]:  l_hard_reg = l_hard_reg * 0.0
            HALF = 0.5 * _m.log(2.0 * _m.pi)
            if _abl_flags["disable_learned_weights"]:
                total = (l_cfm
                         + bd["lam_reg"]   * 0.20 * l_reg
                         + bd["lam_dir"]   * 0.07 * l_heading
                         + bd["lam_calib"] * 0.10 * l_calib)
            else:
                prec_r = torch.exp(-2.0 * self.log_sigma_reg.clamp(min=-3.0))
                prec_h = torch.exp(-2.0 * self.log_sigma_heading.clamp(min=-3.0))
                prec_c = torch.exp(-2.0 * self.log_sigma_calib.clamp(min=-3.0))
                total = (l_cfm
                         + bd["lam_reg"]   * (0.5*prec_r*l_reg     + self.log_sigma_reg.clamp(min=-3.0)     + HALF)
                         + bd["lam_dir"]   * (0.5*prec_h*l_heading + self.log_sigma_heading.clamp(min=-3.0) + HALF)
                         + bd["lam_calib"] * (0.5*prec_c*l_calib   + self.log_sigma_calib.clamp(min=-3.0)   + HALF))
            
            total = total + self.lambda_hard_reg * l_hard_reg
            if not torch.isfinite(total):
                total = total.new_zeros(())
            bd.update({"total": total,
                       "l_reg":      float(l_reg.detach()),
                       "l_heading":  float(l_heading.detach()),
                       "l_calib":    float(l_calib.detach()),
                       "l_hard_reg": float(l_hard_reg.detach())})
            return bd
        import types
        raw.get_loss_breakdown = types.MethodType(_patched_glb, raw)


    opt    = build_optimizer(model, lr_velocity=args.lr, lr_encoder=0.0,
                             weight_decay=args.weight_decay,
                             lr_logits_scale=args.lr_logits_scale,
                             lr_extra_scale=args.lr_extra_scale)
    scaler = GradScaler("cuda", enabled=args.use_amp)
    sched  = TwoGroupScheduler(
        opt=opt, warmup_epochs=args.warmup_epochs, total_epochs=args.num_epochs,
        lr_vel=args.lr, lr_vel_min=args.lr_min,
        freeze_end_ep=args.freeze_encoder_epochs, lr_enc_peak=args.lr_enc_peak,
        encoder_warmup_epochs=args.encoder_warmup_epochs)
    print(f"\n  LR vel: {args.lr:.0e} → {args.lr_min:.0e}  "
          f"LR enc: 0 ({args.freeze_encoder_epochs}ep) → {args.lr_enc_peak:.0e}")

    swa = SWAHandler(swa_lr=args.swa_lr)

    start_ep = 0; best_score = float("inf"); best_hard = float("inf")
    patience_cnt = 0; val_ade_history = []

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        _unwrap(model).load_state_dict(ck["model"], strict=False)
        try: opt.load_state_dict(ck["optimizer"])
        except Exception as e: print(f"  ⚠ Opt: {e}")
        sched.epoch  = ck.get("scheduler", 0)
        start_ep     = ck.get("epoch", 0) + 1
        best_score   = ck.get("best_score", ck.get("best_ade", float("inf")))
        patience_cnt = ck.get("patience_cnt", 0)
        if scaler and ck.get("scaler"):
            try: scaler.load_state_dict(ck["scaler"])
            except Exception: pass
        if ema and ck.get("ema"):
            for k, v in ck["ema"].items():
                if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
        print(f"  ↩ Resume ep{start_ep}  best={best_score:.1f}  patience={patience_cnt}")

    try:
        # [EMA-NORMALIZED-HORIZON NOTE] reg_dist_ema/heading_err_ema are
        # updated via BRANCHLESS in-place buffer mutation (tensor-weighted
        # blend, not a Python if/else on a tensor's value) inside a
        # `if self.training:` branch (self.training is a plain Python bool
        # attribute, which TorchDynamo guards on natively -- same pattern
        # nn.BatchNorm uses for running_mean/running_var, and recompiles
        # once per train/eval mode switch, not per-batch). The warm-start
        # logic (first-batch-after-freeze overwrite vs subsequent EMA
        # blend) deliberately avoids any `bool(tensor)`/`.item()` read
        # inside the forward pass -- an earlier version of this mechanism
        # DID use such a read and was verified (via
        # torch.compile(fullgraph=True), which raises a hard error on any
        # graph break instead of silently falling back) to force a
        # CPU-GPU sync and TorchDynamo graph break. The current tensor-
        # weighted-blend version was verified with the same
        # fullgraph=True test to compile with ZERO graph breaks, so it is
        # safe under mode="reduce-overhead" CUDA-graph capture.
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: ok")
    except Exception: pass

    nstep = len(trl)
    print(f"\n  TRAINING ({nstep} steps/ep × {args.num_epochs} ep)")
    # [BUG FOUND] These three summary lines previously hardcoded a
    # description of augmentation/loss/inference that had drifted out of
    # sync with the actual code: augment_batch's real distribution is
    # A=25% (shift), B=20% (speed scale), C=20% (recurvature), D-E=25%
    # (no-op), F=10% (Gaussian noise) -- six branches, not the
    # "shift+speed+recurv+no-aug(35%)" four-branch description previously
    # printed here (which also omitted branch F entirely and mislabeled
    # the no-op share as 35% instead of the correct 25%). Likewise,
    # "L_heading_ms(4steps,decay=0.5)" describes the OLD fixed-decay
    # heading loss that has since been replaced by EMA-NORMALIZED-HORIZON
    # weighting across the full 12-step horizon (see _heading_loss_ms's
    # docstring in flow_matching_model.py). A hardcoded print string has
    # no way to track code changes made in a different file, so it will
    # silently go stale again the next time augment_batch or the loss
    # design changes -- flagging this explicitly rather than just fixing
    # the current text, since the SAME mismatch could recur.
    print(f"  Aug: shift±5km(25%) + speed×[0.85,1.15](20%) + recurv±20°(20%) "
          f"+ no-aug(25%) + noise±3km(10%)"
          f"{'  [recurv DISABLED via --disable_aug_c]' if args.disable_aug_c else ''}")
    print(f"  Loss: L_CFM + L_reg(EMA-normalized-horizon) "
          f"+ L_heading(EMA-normalized-horizon, full 12-step)")
    print(f"  Inf:  1-shot sigma=0.04 + speed_calibrate(learned) + top3 physics"
          f"{' + curvature_score' if args.use_curvature_score_train else ''}")
    print()

    for ep in range(start_ep, start_ep + args.num_epochs):
        rel_ep = ep - start_ep
        freeze = rel_ep < args.freeze_encoder_epochs
        for p in _unwrap(model).encoder.parameters():
            p.requires_grad_(not freeze)

        if rel_ep == 0 and freeze:
            print(f"  *** Ep{ep}: encoder frozen ***")
        if rel_ep == args.freeze_encoder_epochs:
            print(f"\n  *** Ep{ep}: encoder unfrozen ***")

        model.train()
        sum_loss = sum_cfm = sum_reg = sum_head = sum_ade1 = 0.0
        n_sanitized_batches = 0
        t0_ep = time.perf_counter()

        for i, batch in enumerate(trl):
            bl = move(list(batch), device)
            bl_aug = augment_batch(bl, disable_c=args.disable_aug_c)   
            opt.zero_grad()
            with autocast(device_type="cuda", enabled=args.use_amp):
                bd = model.get_loss_breakdown(bl_aug, epoch=ep)

            # ── NaN/no-grad SANITIZE (not skip) ─────────────────────────
            # [USER REQUIREMENT] Every batch must run backward()/step() —
            # no batch is ever skipped via `continue`. If the model's own
            # internal guard already fired (get_loss_breakdown returns a
            # graph-detached x0.new_zeros(()) — see [REVERTED] comment in
            # flow_matching_model.py — which has requires_grad=False),
            # `total` has no gradient path at all, so there is nothing
            # meaningful to backward() on for this specific loss value.
            # Instead of skipping the batch, replace it with a
            # graph-CONNECTED zero: 0.0 * l_cfm, built from the actual
            # l_cfm tensor computed this iteration (which DOES have a
            # valid grad_fn from the velocity network's forward pass).
            # backward() on this produces an all-zero gradient for every
            # parameter that contributed to l_cfm, and a well-defined
            # (zero) gradient for the loss overall — mathematically a
            # complete no-op update (equivalent in effect to a skip), but
            # structurally still a full, real backward()+step() call, so
            # no batch is ever dropped from the loop.
            _total_has_grad = bd["total"].requires_grad and torch.isfinite(bd["total"])
            if not _total_has_grad:
                n_sanitized_batches += 1
                print(f"  ⚠ [{ep}][{i}] total loss had no valid gradient "
                      f"(finite={torch.isfinite(bd['total']).item()} "
                      f"requires_grad={bd['total'].requires_grad} "
                      f"cfm={bd['l_cfm']:.4f} reg={bd['l_reg']:.4f} "
                      f"h4s={bd['l_heading']:.4f} calib={bd['l_calib']:.4f} "
                      f"score={bd['l_score']:.4f} hreg={bd.get('l_hard_reg',0.0):.4f}) "
                      f"→ using zero-effect gradient (backward/step still run, batch NOT skipped)")
                loss_for_backward = 0.0 * bd["_t_l_cfm"]
            else:
                loss_for_backward = bd["total"]

            scaler.scale(loss_for_backward).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # ── Gradient NaN/Inf SANITIZE (not skip) ────────────────────
            # clip_grad_norm_ only rescales gradient MAGNITUDE — it does
            # NOT remove NaN/Inf values. If any parameter's gradient is
            # already NaN/Inf despite the fixes in flow_matching_model.py
            # (e.g. from a submodule this patch cannot see, such as
            # FNO3DEncoder/DataEncoder1D_Mamba/Env_net), letting
            # optimizer.step() run would permanently corrupt that weight
            # with NaN, and every subsequent forward pass through it would
            # be NaN forever. [USER REQUIREMENT] Instead of skipping this
            # batch's step() entirely, zero out ONLY the NaN/Inf gradients
            # (leaving any other, valid gradients in the SAME batch
            # intact) and still call scaler.step(opt) — so the batch is
            # never dropped from the loop, and any parameter that got a
            # valid gradient this batch still receives its real update.
            _bad_grad = not torch.isfinite(grad_norm)
            if _bad_grad:
                n_sanitized_batches += 1
                _bad_names = []
                for _name, _p in _unwrap(model).named_parameters():
                    if _p.grad is not None and not torch.isfinite(_p.grad).all():
                        _p.grad = torch.nan_to_num(_p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                        if len(_bad_names) < 5:
                            _bad_names.append(_name)
                        # NOTE: no early break — every NaN/Inf-carrying
                        # parameter's gradient must be sanitized, not just
                        # the first 5. The len(<5) check above only limits
                        # how many NAMES get logged, not how many
                        # parameters get their gradient cleaned.
                print(f"  ⚠ [{ep}][{i}] non-finite grad_norm={grad_norm.item()} "
                      f"sanitized_params(first 5)={_bad_names[:5]} "
                      f"→ NaN/Inf gradients zeroed, step() still runs (batch NOT skipped)")
                # Re-clip after sanitizing so the reported/used grad_norm
                # for logging purposes is finite and consistent.
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if freeze:
                for p in _unwrap(model).encoder.parameters():
                    if p.grad is not None: p.grad.zero_()

            scaler.step(opt); scaler.update()
            model.ema_update()
            swa.update(model)

            sum_loss += bd["total"].item(); sum_cfm  += bd["l_cfm"]
            sum_reg  += bd["l_reg"];        sum_head += bd["l_heading"]
            sum_ade1 += bd["ade_1step"]

            if i % 30 == 0:
                _, lr_vel = get_lrs(opt)
                enc_s = "frozen" if freeze else "active"
                swa_s = " [SWA]" if swa.active else ""
                print(f"  [{ep:>3}][{i:>3}/{nstep}]"
                      f"  tot={bd['total'].item():.4f}"
                      f"  cfm={bd['l_cfm']:.4f}"
                      f"  reg={bd['l_reg']:.4f}"
                      f"  h4s={bd['l_heading']:.4f}"
                      f"  hreg={bd.get('l_hard_reg', 0.0):.4f}"
                      f"  lam_d={bd['lam_dir']:.2f}"
                      f"  ade1={bd['ade_1step']:.0f}km"
                      f"  enc={enc_s}{swa_s}"
                      f"  lr={lr_vel:.2e}")

        # [USER REQUIREMENT] Every batch now runs backward()/step() — none
        # are skipped — so the epoch average uses the full nstep count,
        # not a reduced "effective" count. n_sanitized_batches is kept
        # purely as a diagnostic counter (how many batches needed their
        # loss/gradient sanitized this epoch), not a divisor.
        train_loss = sum_loss / nstep
        _, lr_vel_used = get_lrs(opt)
        # [BUG FOUND, SWA-LR-OVERRIDE] sched.step() previously ran
        # unconditionally every epoch, including AFTER swa.activate() had
        # already set every param_group's lr to args.swa_lr (2e-6). The
        # cosine scheduler has no knowledge that SWA just overrode its LR,
        # so calling .step() on it recomputes LR according to the
        # scheduler's OWN unchanged schedule -- confirmed directly in the
        # training logs (both seed0 and seed1): SWA activates at ep90 with
        # "lr -> 2.0e-06", and by ep92 (one scheduler step later) lr is
        # back to ~1.4e-04, roughly 70x higher. This defeats the entire
        # purpose of SWA, which requires a low, STABLE lr to average
        # weights around a single local minimum -- instead the optimizer
        # kept taking large steps for the rest of training, and SWA's
        # score never surpassed the pre-SWA best in either seed (up to
        # 3301 updates, still behind best) despite running for the last
        # ~40% of the configured epoch budget. Root-caused, not guessed:
        # traced from the exact LR values printed in both training logs
        # to this line before making the fix.
        if not swa.active:
            sched.step()

        sanitize_s = (f"  sanitized={n_sanitized_batches}/{nstep}"
                      if n_sanitized_batches > 0 else "")
        print(f"\n  ── Ep{ep:>3}"
              f"  train={train_loss:.6f}"
              f"  cfm={sum_cfm/nstep:.4f}"
              f"  reg={sum_reg/nstep:.4f}"
              f"  h4s={sum_head/nstep:.4f}"
              f"  ade1={sum_ade1/nstep:.0f}km"
              f"  lr={lr_vel_used:.2e}"
              f"  t={time.perf_counter()-t0_ep:.0f}s"
              f"{sanitize_s}")

        _save(last_ckpt, ep, model, opt, sched, best_score, ema, scaler,
              model_cfg=model_cfg)
        if ep % 5 == 0:
            ep_ckpt = os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth")
            _save(ep_ckpt, ep, model, opt, sched, best_score, ema, scaler,
                  model_cfg=model_cfg)
            print(f"  💾 {ep_ckpt}")
        if rel_ep % args.val_freq == 0:
            run_xai_this = (rel_ep % 10 == 0)
            r = evaluate(model, val_loader, device, tag=f"VAL ep{ep}",
                         n_ensemble=args.n_ensemble, ema=ema,
                         ref_targets=ST_TRANS_VAL, epoch_for_loss=ep,
                         run_xai=run_xai_this, xai_batch=xai_batch)

            val_ade = r["ADE"]; score = r["combined_score"]
            val_ade_history.append(val_ade)

            if len(val_ade_history) >= 4:
                trend = float(np.mean(val_ade_history[-2:])) - float(np.mean(val_ade_history[-4:-2]))
                trend_s = f"↑{trend:+.1f}km⚠" if trend > 5 else f"↓{trend:+.1f}km✓" if trend < -5 else f"→{trend:+.1f}flat"
            else:
                trend_s = "—"
            print(f"  train={train_loss:.6f}  val_ADE={val_ade:.1f}  combined={score:.1f}  trend={trend_s}")

            if (not swa.active and ep >= args.swa_min_ep
                    and swa.should_activate(val_ade_history, args.swa_window, args.swa_threshold)):
                swa.activate(model, opt, ep)

            if score < best_score:
                best_score = score; patience_cnt = 0
                _save(best_ckpt, ep, model, opt, sched, best_score, ema, scaler,
                      extra={"val_ade": r["ADE"], "val_ate": r["ATE"],
                             "val_cte": r["CTE"], "patience_cnt": 0},
                      model_cfg=model_cfg)
                print(f"  ✅ Best! score={best_score:.2f}"
                      f"  ADE={r['ADE']:.1f} ATE={r['ATE']:.1f} CTE={r['CTE']:.1f}")
            else:
                # [BUG FOUND, PATIENCE-VS-SWA] patience_cnt previously
                # incremented unconditionally on every no-improvement
                # epoch, including while swa.active is True. Now that the
                # sched.step()/swa LR-override bug above is fixed, SWA
                # genuinely needs a stretch of epochs with weights
                # averaging around a stable minimum before its score has
                # a chance to surpass best_score -- during that stretch,
                # ordinary per-epoch validation noise can easily produce
                # several consecutive "no improve" epochs even though SWA
                # is working exactly as intended. Letting patience run
                # during this window risks early-stopping SWA before it
                # has had a fair chance, which is what the training logs
                # showed happening in practice (early stop at ep110, only
                # ~20 epochs after SWA activated at ep90 -- barely enough
                # time for a handful of SWA updates, let alone
                # convergence). Patience now only accumulates for
                # non-SWA epochs; once SWA is active, its own separate
                # [SWA] score-vs-best comparison (printed at swa
                # checkpoints elsewhere in this loop) is the relevant
                # signal, not this counter.
                if rel_ep >= args.min_ep and not swa.active:
                    patience_cnt += args.val_freq
                print(f"  No improve {patience_cnt}/{args.patience} (best={best_score:.1f})"
                      f"{'  [SWA active -- patience frozen]' if swa.active else ''}")
                if rel_ep >= args.min_ep and not swa.active and patience_cnt >= args.patience:
                    print(f"  ⛔ Early stop @ ep{ep}")
                    break

     
        if rel_ep % args.hard_val_freq == 0 and rel_ep >= args.min_ep:
            r_h = evaluate_hard_val(model, val_loader, device,
                                     hard_threshold=args.hard_val_threshold,
                                     n_ensemble=args.n_ensemble, ema=ema,
                                     epoch_for_loss=ep)
            print(f"  [HVAL] n={r_h['n_hard']}"
                  f"  ADE={r_h['ADE']:.1f} ATE={r_h['ATE']:.1f} CTE={r_h['CTE']:.1f}"
                  f"  combined={r_h['combined_score']:.1f}")
            if r_h["combined_score"] < best_hard and r_h["n_hard"] >= 10:
                best_hard = r_h["combined_score"]
                _save(hard_best_ckpt, ep, model, opt, sched, best_hard, ema, scaler,
                      extra={"hard_val_ade": r_h["ADE"], "selection_criterion": "hard_val"},
                      model_cfg=model_cfg)
                print(f"  💎 Hard-best! score={best_hard:.2f} ADE={r_h['ADE']:.1f}")

      
        if swa.active and rel_ep % args.val_freq == 0 and swa.n_updates >= 10:
            backup = {k: v.detach().clone()
                      for k, v in _unwrap(model).state_dict().items()
                      if v.dtype.is_floating_point}
            swa.apply_to_model(model)
            r_swa = evaluate(model, val_loader, device, tag=f"SWA ep{ep}",
                              n_ensemble=args.n_ensemble, ema=None,
                              ref_targets=ST_TRANS_VAL, epoch_for_loss=ep)
            swa.restore_from_backup(model, backup)

            swa_score = r_swa["combined_score"]
            print(f"  [SWA] score={swa_score:.2f} ({swa.n_updates} updates) vs best={best_score:.2f}")
            if swa_score < best_score:
                best_score = swa_score; patience_cnt = 0
                swa.save_avg_state(swa_ckpt, ep, best_score,
                                   extra={"val_ade": r_swa["ADE"],
                                          "val_ate": r_swa["ATE"],
                                          "val_cte": r_swa["CTE"]},
                                   model_cfg=model_cfg)
                import shutil; shutil.copy(swa_ckpt, best_ckpt)
                print(f"  ✅ SWA best! score={best_score:.2f} ADE={r_swa['ADE']:.1f}")

    _wall_total = time.time() - _wall_start
    print(f"\n  Training wall-clock: {_wall_total/3600:.2f}h ({_wall_total:.0f}s)")
    try:
        footprint_info.update({
            "training_wall_clock_s": _wall_total,
            "training_wall_clock_h": round(_wall_total/3600, 3),
            "num_epochs": args.num_epochs,
            "best_score": best_score,
        })
        import json as _json
        fp_path = os.path.join(args.output_dir, "footprint.json")
        with open(fp_path, "w") as _fp:
            _json.dump(footprint_info, _fp, indent=2)
        print(f"  Footprint saved → {fp_path}")
    except Exception as _fe:
        print(f"  Footprint save failed: {_fe}")

  
    print(f"\n  Done! best_score={best_score:.2f}")
    if not args.test_at_end: return

    print("\n  Loading best checkpoint for TEST...")
    if not os.path.exists(best_ckpt):
        print("  No checkpoint found."); return

    ck = torch.load(best_ckpt, map_location=device)
    is_swa = ck.get("is_swa", False)
    _unwrap(model).load_state_dict(ck["model"], strict=False)
    if not is_swa and ema and ck.get("ema"):
        for k, v in ck["ema"].items():
            if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
    print(f"  Loaded ep{ck.get('epoch','?')} (is_swa={is_swa})")

    try:
        _, test_loader = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
        print(f"  Test: {len(test_loader)} batches")
    except Exception:
        print("  No test set → using val"); test_loader = val_loader

    # Standard test
    r_test = evaluate(model, test_loader, device, tag="TEST (best ckpt)",
                      n_ensemble=args.n_ensemble, ema=None if is_swa else ema,
                      ref_targets=ST_TRANS_TEST, run_xai=True, xai_batch=xai_batch)

    # TTA test
    if args.tta_test:
        r_tta = evaluate(model, test_loader, device, tag="TEST+TTA",
                         n_ensemble=args.n_ensemble, ema=None if is_swa else ema,
                         ref_targets=ST_TRANS_TEST, use_tta=True, n_tta=args.n_tta)
        print(f"\n  TTA: ADE {r_test['ADE']:.1f}→{r_tta['ADE']:.1f}  "
              f"ATE {r_test['ATE']:.1f}→{r_tta['ATE']:.1f}  "
              f"CTE {r_test['CTE']:.1f}→{r_tta['CTE']:.1f}")

    # Multi-scale test
    if args.multiscale_test:
        raw_m = _unwrap(model)
        if hasattr(raw_m, "sample_multiscale"):
            print("\n  Multi-scale sigma test...")
            ms_ades, ms_ates, ms_ctes = [], [], []
            ms_steps = defaultdict(list)
            raw_m.eval()
            bk_ms = (ema.apply_to(model) if (ema and not is_swa) else None)
            with torch.no_grad():
                for batch in test_loader:
                    bl = move(list(batch), device)
                    try:
                        pred, _, _ = raw_m.sample_multiscale(bl)
                    except Exception: continue
                    gt = bl[1]; T = min(pred.shape[0], gt.shape[0])
                    pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
                    dist = _haversine_deg(pd, gd)
                    ate, cte = _ate_cte(pd, gd)
                    ms_ades.extend(dist.mean(0).tolist())
                    if ate.shape[0] > 0:
                        ms_ates.extend(ate.abs().mean(0).tolist())
                        ms_ctes.extend(cte.abs().mean(0).tolist())
                    for h, s in HORIZON_STEPS.items():
                        if s < T: ms_steps[h].extend(dist[s].tolist())
            if bk_ms: ema.restore(model, bk_ms)
            def _mm(lst): return float(np.mean(lst)) if lst else float("nan")
            print(f"  Multi-scale: ADE={_mm(ms_ades):.1f} ATE={_mm(ms_ates):.1f} CTE={_mm(ms_ctes):.1f}")

    # Final comparison
    val_ade_b = ck.get("val_ade", float("nan"))
    val_ate_b = ck.get("val_ate", float("nan"))
    val_cte_b = ck.get("val_cte", float("nan"))
    v21 = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4}  # ST-Trans targets
    print("\n" + "=" * 72)
    print("  v2.1-XAI FINAL RESULTS vs ST-Trans target")
    print("=" * 72)
    print(f"  {'Metric':<10} {'Val':>10} {'Test':>10} {'Gap':>10} {'v2.1 Test':>12} Status")
    print("  " + "─"*60)
    for m_n, val_v, test_v, v21_v in [
        ("ADE", val_ade_b, r_test["ADE"], v21["ADE"]),
        ("ATE", val_ate_b, r_test["ATE"], v21["ATE"]),
        ("CTE", val_cte_b, r_test["CTE"], v21["CTE"]),
    ]:
        gap  = test_v - val_v if (np.isfinite(test_v) and np.isfinite(val_v)) else float("nan")
        impr = v21_v - test_v
        flag = (f"↓{impr:+.1f}" if np.isfinite(impr) and impr > 0 else
                f"↑{-impr:+.1f}" if np.isfinite(impr) else "?")
        print(f"  {m_n:<10} {val_v:>10.2f} {test_v:>10.2f} {gap:>+10.2f} {v21_v:>12.1f}  {flag}")
    print("=" * 72)

    _auto_eval(args, best_ckpt, device)


def _auto_eval(args, best_ckpt: str, device):
    import subprocess, sys, json as _json

    print("\n" + "="*72)
    print("  AUTO EVALUATE + STATISTICAL TESTS")
    print("="*72)

    eval_dir  = os.path.join(args.output_dir, "eval")
    stats_dir = os.path.join(args.output_dir, "stats")
    os.makedirs(eval_dir,  exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)


    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    candidates_eval = [
        os.path.join(project_root, "evaluate_full.py"),
        os.path.join(os.getcwd(),  "evaluate_full.py"),
        os.path.join(script_dir,   "evaluate_full.py"),
    ]
    candidates_stat = [
        os.path.join(project_root, "statistical_tests.py"),
        os.path.join(os.getcwd(),  "statistical_tests.py"),
        os.path.join(script_dir,   "statistical_tests.py"),
    ]
    eval_script = next((p for p in candidates_eval if os.path.exists(p)), None)
    stat_script = next((p for p in candidates_stat if os.path.exists(p)), None)

  
    if eval_script is None:
        print("  ⚠ evaluate_full.py không tìm thấy → bỏ qua auto-eval")
        print(f"    Đặt file tại: {candidates_eval[0]}")
    else:
        print(f"  ▶ evaluate_full.py ({eval_script})")
        ep = "best"
        try:
            import torch as _torch
            ck_info = _torch.load(best_ckpt, map_location="cpu")
            ep = ck_info.get("epoch", "best")
        except Exception:
            pass

        cmd_eval = [
            sys.executable, eval_script,
            "--checkpoint",   best_ckpt,
            "--dataset_root", args.dataset_root,
            "--split",        "test",
            "--output_dir",   eval_dir,
            "--n_ensemble",   str(args.n_ensemble),
            "--no_crps",     
            "--gpu",          str(args.gpu_num),
        ]
        try:
            result = subprocess.run(cmd_eval, capture_output=False, timeout=1800)
            if result.returncode == 0:
                print(f"  ✅ evaluate_full done → {eval_dir}/")
            else:
                print(f"  ❌ evaluate_full failed (code {result.returncode})")
        except subprocess.TimeoutExpired:
            print("  ⚠ evaluate_full timeout (30min) → bỏ qua")
        except Exception as e:
            print(f"  ⚠ evaluate_full error: {e}")

  
    eval_json = None
    if os.path.exists(eval_dir):
        # [BUG FOUND, PATH MISMATCH] Previously filtered for files starting
        # with "eval_test" -- but evaluate_full.py actually writes files
        # named "eval_fm_test_ep{N}.json" (confirmed directly from training
        # logs: both seed0 and seed1 print "Saved →
        # .../eval/eval_fm_test_ep70.json" immediately before this block
        # runs, yet the very next line unconditionally reports "Không tìm
        # thấy eval JSON" in both logs). "eval_fm_test_ep70.json" does NOT
        # start with "eval_test" (it starts with "eval_fm_"), so the old
        # filter matched zero files on every single run, silently skipping
        # the statistical significance test step every time regardless of
        # whether evaluate_full.py succeeded. Broadened to match any
        # filename containing "test" and ending in ".json" under eval_dir,
        # which covers the confirmed real pattern (eval_fm_test_ep*.json)
        # without over-fitting to one exact prefix in case the model-type
        # portion of the filename varies (e.g. a different --model_type
        # producing eval_lstm_test_ep*.json, eval_sttrans_test_ep*.json,
        # etc., for other scripts/checkpoints that might write into the
        # same eval_dir).
        jsons = sorted([
            os.path.join(eval_dir, f) for f in os.listdir(eval_dir)
            if "test" in f and f.endswith(".json")
        ])
        if jsons:
            eval_json = jsons[-1]  # file mới nhất

    if stat_script is None:
        print("  ⚠ statistical_tests.py không tìm thấy → bỏ qua")
    elif eval_json is None:
        print("  ⚠ Không tìm thấy eval JSON → bỏ qua statistical tests")
        print(f"    Thử chạy lại evaluate_full.py trước")
    else:
        print(f"  ▶ statistical_tests.py ({stat_script})")
        print(f"    FM result: {eval_json}")
        cmd_stat = [
            sys.executable, stat_script,
            "--fm_results",       eval_json,
            "--use_st_trans_ref",
            "--fm_n_storms",      "420",
            "--baseline_name",    "ST-Trans",
            "--output_dir",       stats_dir,
            "--n_bootstrap",      "10000",
        ]
        try:
            result = subprocess.run(cmd_stat, capture_output=False, timeout=600)
            if result.returncode == 0:
                print(f"  ✅ statistical_tests done → {stats_dir}/")
            else:
                print(f"  ❌ statistical_tests failed (code {result.returncode})")
        except subprocess.TimeoutExpired:
            print("  ⚠ statistical_tests timeout (10min) → bỏ qua")
        except Exception as e:
            print(f"  ⚠ statistical_tests error: {e}")

    summary_path = os.path.join(args.output_dir, "auto_eval_summary.json")
    try:
        summary = {
            "checkpoint":    best_ckpt,
            "eval_dir":      eval_dir,
            "stats_dir":     stats_dir,
            "eval_json":     eval_json,
            "seed":          getattr(args, "seed", 42),
            "ablation_name": getattr(args, "ablation_name", ""),
        }
      
        if eval_json and os.path.exists(eval_json):
            with open(eval_json) as _f:
                ev = _json.load(_f)
            summary["test_ADE"] = ev.get("ADE")
            summary["test_ATE"] = ev.get("ATE")
            summary["test_CTE"] = ev.get("CTE")
            summary["test_RMSE"] = ev.get("RMSE")
            summary["crps_mean"] = ev.get("crps", {}).get("mean")
        with open(summary_path, "w") as _f:
            _json.dump(summary, _f, indent=2)
        print(f"\n  Summary → {summary_path}")
        if summary.get("test_ADE"):
            print(f"  ADE={summary['test_ADE']:.2f}  "
                  f"ATE={summary['test_ATE']:.2f}  "
                  f"CTE={summary['test_CTE']:.2f}")
    except Exception as e:
        print(f"  Summary save failed: {e}")

    print("="*72)


if __name__ == "__main__":
    args = get_args()
    # [BUG FOUND] This previously hardcoded seed=42 BEFORE main() runs its
    # own args.seed-based seeding (random.seed(args.seed) etc., near the
    # top of main()) -- meaning any code that touches global RNG state
    # between this point and main()'s own seeding (e.g. a library import
    # triggered lazily, or torch.compile's internal setup) would see
    # seed=42's state regardless of --seed. main()'s own seeding call
    # immediately overwrites this for everything AFTER it runs, so
    # results reported from training itself were not affected -- but this
    # is dead, misleading code (a global seed=42 that has no effect once
    # main() starts) and a latent bug if anything before main() ever
    # becomes RNG-sensitive. Seed with the ACTUAL requested value instead
    # of a hardcoded constant, consistent with the rest of this file's
    # principle of not using hand-picked constants where a real value is
    # available.
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.dataset_root == "TCND_vn":
        _auto = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
        if os.path.isdir(_auto):
            args.dataset_root = _auto
    main(args)