"""

# ─────────────────────────────────────────────────────────────────────────────
# FILE PLACEMENT:
#
#   SOURCE:        evaluate_full.py
#   KAGGLE TARGET: /kaggle/working/evaluate_full.py    (root, cạnh Model/)
#   LOCAL DEV:     evaluate_full.py
#
#   Chạy sau khi train xong:
#   python evaluate_full.py \
#     --checkpoint /kaggle/working/runs/best_model.pth \
#     --dataset_root /kaggle/input/datasets/tc-ofm \
#     --split test --output_dir results/
# ─────────────────────────────────────────────────────────────────────────────

evaluate_full.py — TC-FlowMatching ESWA Full Evaluation
════════════════════════════════════════════════════════════════════════════════
Produces all metrics required by ESWA paper:
  B. Core Experiments:   ADE/ATE/CTE/RMSE/MAE/Final-DPE per horizon
                         Per-storm-category (TD/TS/Cat1-5, slow/medium/fast)
                         Boxplot data, error-by-leadtime
  C. Ablation/Testing:   CRPS (ensemble spread-skill)
  D. Physical validity:  Speed/accel within physical bounds
                         Cone of uncertainty per storm

Usage:
  python evaluate_full.py \\
    --checkpoint runs/best_model.pth \\
    --dataset_root /path/to/tc-ofm \\
    --split test \\
    --output_dir results/ \\
    --n_ensemble 20

  [MỚI] Đánh giá baseline (không chỉ FM) — thêm --model_type:
  python evaluate_full.py \\
    --checkpoint runs/best_model_lstm_seed0.pth \\
    --model_type lstm \\
    --dataset_root /path/to/tc-ofm \\
    --split test --per_storm --no_crps \\
    --output_dir results/lstm_seed0/
"""
from __future__ import annotations

import sys, os, argparse, time, json, math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ─── Imports from model ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, EMAModel,
    _norm_to_deg, _haversine_deg, _forward_azimuth,
    _step_speeds_kmh, _unwrap,
)
# [MỚI, --model_type] Baseline architectures — chỉ cần cho việc XÂY
# model đúng loại (constructor) và LOAD checkpoint; toàn bộ phần đánh
# giá phía dưới (run_full_evaluation/per_storm_breakdown/...) đã viết
# tổng quát trên "model" (gọi model.sample(bl, num_ensemble=..., **kw)
# và đọc pred/all_t) nên không cần đổi gì thêm — cả 2 lớp baseline này
# đã có sẵn sample() cùng chữ ký/return-shape với TCFlowMatching (xem
# ghi chú trong _build_model() bên dưới để biết đã verify thế nào).
from Model.st_trans_model import STTrans
from Model.paper_baseline_model import PaperBaseline

# ode_steps_sweep lives in ablation_runner.py (NOT duplicated here) — it
# already implements the N sweep correctly (temporarily overrides
# raw.n_inference_steps, calls model.sample(), restores it). Importing
# rather than re-implementing avoids a second, possibly-diverging copy.
try:
    from ablation_runner import ode_steps_sweep, print_ode_sweep
except ImportError:
    ode_steps_sweep = None
    print_ode_sweep = None

# ─── Constants ────────────────────────────────────────────────────────────────
R_EARTH   = 6371.0
DT_HOURS  = 6.0
ST_TRANS  = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
             "12h": 65.0, "24h": 130.0, "48h": 205.0, "72h": 321.0}

# Horizon steps: step index 0=6h, 1=12h, 3=24h, 7=48h, 11=72h
HORIZONS = {"6h": 0, "12h": 1, "24h": 3, "48h": 7, "72h": 11}

# TC intensity categories by obs speed (km/h)
# TD<63, TS 63-119, Cat1 119-153, Cat2 154-177, Cat3 178-208, Cat4 209-252, Cat5≥253 (kt→km/h)
INTENSITY_BINS  = [0, 63, 119, 153, 177, 208, 252, 9999]
INTENSITY_NAMES = ["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"]
# Speed categories (obs mean km/h)
SPEED_SLOW   = 8.0
SPEED_FAST   = 15.0

# Physical bounds
MAX_TC_SPEED_KMH  = 100.0   # TC track speed >100km/h per 6h is unphysical
MAX_ACCEL_KMH2    = 30.0    # Speed change >30km/h per step is unphysical

# [MỚI, --model_type] Kiến trúc hợp lệ — "fm" giữ nguyên hành vi mặc
# định cũ (không truyền --model_type vẫn ra đúng TCFlowMatching, không
# đổi hành vi cho mọi lệnh gọi evaluate_full.py đã có từ trước).
MODEL_TYPES = ["fm", "st_trans", "lstm", "gru", "rnn"]


# ─────────────────────────────────────────────────────────────────────────────
#  Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


def _ate_cte_full(pred_deg: torch.Tensor,
                   gt_deg:   torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ATE = along-track error (signed), CTE = cross-track error (signed).
    Returns shape [T-1, B] each.
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    bear_ref = _forward_azimuth(gt_deg[:T-1], gt_deg[1:T])    # [T-1, B]
    bear_err = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])  # [T-1, B]
    dist_err = _haversine_deg(pred_deg[1:T], gt_deg[1:T])     # [T-1, B]
    ang      = bear_err - bear_ref
    return dist_err * torch.cos(ang), dist_err * torch.sin(ang)


def _rmse_per_step(pred_deg: torch.Tensor,
                    gt_deg:   torch.Tensor) -> torch.Tensor:
    """RMSE in km per step. Returns [T, B]."""
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    d = _haversine_deg(pred_deg[:T], gt_deg[:T])  # [T, B]
    return d  # caller squares and takes mean


def _crps_ensemble(samples: torch.Tensor,
                    gt_deg:  torch.Tensor) -> Dict:
    """
    Continuous Ranked Probability Score for ensemble forecast.
    samples: [K, T, B, 2] in deg
    gt_deg:  [T, B, 2] in deg

    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    where X, X' are independent draws from the ensemble.
    Returns mean CRPS per horizon and spread-skill ratio.
    """
    K, T, B, _ = samples.shape
    T_gt = min(T, gt_deg.shape[0])

    crps_per_step = []  # [T]
    spread_per_step = []
    skill_per_step = []

    for t in range(T_gt):
        # Distance from each member to GT: [K, B]
        dist_to_gt = torch.stack([
            _haversine_deg(samples[k, t:t+1], gt_deg[t:t+1]).squeeze(0)
            for k in range(K)
        ], dim=0)  # [K, B]

        # E[|X - y|]: mean over ensemble
        e_dist = dist_to_gt.mean(0)  # [B]

        # E[|X - X'|]: pairwise mean
        pairwise = 0.0
        n_pairs = 0
        for i in range(K):
            for j in range(i+1, K):
                d = _haversine_deg(samples[i, t:t+1], samples[j, t:t+1]).squeeze(0)
                pairwise = pairwise + d
                n_pairs += 1
        if n_pairs > 0:
            e_spread = pairwise / n_pairs
        else:
            e_spread = torch.zeros_like(e_dist)

        crps = e_dist - 0.5 * e_spread          # [B]
        crps_per_step.append(float(crps.mean()))
        spread_per_step.append(float(e_spread.mean()))
        skill_per_step.append(float(e_dist.mean()))

    crps_arr   = np.array(crps_per_step)
    spread_arr = np.array(spread_per_step)
    skill_arr  = np.array(skill_per_step)
    # Spread-skill ratio: ideal = 1.0 (spread matches RMSE)
    skill_std  = np.sqrt(np.array([
        float((_haversine_deg(samples[:, t].mean(0, keepdim=True),
                               gt_deg[t:t+1]).squeeze(0)**2).mean())
        for t in range(T_gt)
    ]) + 1e-6)
    ss_ratio   = spread_arr / (skill_std + 1e-6)

    hz_idx = {k: v for k, v in HORIZONS.items() if v < T_gt}
    return {
        "crps_mean":        float(crps_arr.mean()),
        "crps_per_step":    crps_arr.tolist(),
        "spread_per_step":  spread_arr.tolist(),
        "spread_skill_ratio": ss_ratio.tolist(),
        "crps_by_horizon":  {h: float(crps_arr[s]) for h, s in hz_idx.items()},
        "ss_ratio_by_horizon": {h: float(ss_ratio[s]) for h, s in hz_idx.items()},
    }


def _physical_validity(pred_deg: torch.Tensor,
                        obs_deg:  torch.Tensor) -> Dict:
    """
    Check if predicted trajectory violates physical constraints.
    Returns fraction of storms with physically valid predictions.
    """
    B = pred_deg.shape[1]
    if pred_deg.shape[0] < 2:
        return {"valid_speed_frac": 1.0, "valid_accel_frac": 1.0,
                "mean_pred_speed": 0.0, "max_pred_speed": 0.0}

    pts = torch.cat([obs_deg[-1:], pred_deg], 0)  # [T+1, B, 2]
    speeds = _step_speeds_kmh(pts)   # [T, B] km/h

    # Speed validity: all steps < MAX_TC_SPEED_KMH
    valid_speed = (speeds < MAX_TC_SPEED_KMH).all(0)   # [B]
    # Acceleration validity: |speed[t] - speed[t-1]| < MAX_ACCEL
    if speeds.shape[0] >= 2:
        accel = (speeds[1:] - speeds[:-1]).abs()
        valid_accel = (accel < MAX_ACCEL_KMH2).all(0)  # [B]
    else:
        valid_accel = torch.ones(B, dtype=torch.bool, device=pred_deg.device)

    return {
        "valid_speed_frac":  float(valid_speed.float().mean()),
        "valid_accel_frac":  float(valid_accel.float().mean()),
        "mean_pred_speed":   float(speeds.mean()),
        "max_pred_speed":    float(speeds.max()),
        "speed_per_step":    speeds.mean(1).tolist(),   # [T] mean over batch
    }


def _obs_speed(obs_deg: torch.Tensor) -> torch.Tensor:
    """Mean obs speed per storm. [B] km/h."""
    if obs_deg.shape[0] < 2:
        return obs_deg.new_zeros(obs_deg.shape[1])
    return _step_speeds_kmh(obs_deg).mean(0)   # [B]


# ─────────────────────────────────────────────────────────────────────────────
#  [MỚI, --model_type] Model construction / loading — 1 kiến trúc bất kỳ
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(model_type: str, model_cfg: dict, device):
    """
    Xây model theo đúng kiến trúc --model_type, dùng model_cfg đã lưu
    trong checkpoint nếu có (giữ đúng cách flow_matching's nhánh vẫn
    làm từ trước — chỉ tổng quát hoá sang 4 kiến trúc còn lại).

    Đã verify TRƯỚC khi viết hàm này (không đoán):
      - STTrans.sample(batch_list, num_ensemble=1, **kwargs) ->
        (pred[T,B,2], me_mean[T,B,2] zeros, pred.unsqueeze(0)) — cùng
        3-tuple return convention với TCFlowMatching.sample(), và nhận
        **kwargs nên các keyword evaluate_full.py truyền thêm (vd
        use_curvature_score=) không làm crash dù bị bỏ qua.
      - PaperBaseline(model_type="lstm"/"gru"/"rnn", ...).sample(...)
        cùng chữ ký, cùng return convention y hệt STTrans — 1 lớp dùng
        chung cho cả 3 kiến trúc RNN-họ, chọn qua model_cfg["model_type"]
        hoặc đối số model_type truyền vào constructor.
    Do đó toàn bộ phần đánh giá phía dưới (run_full_evaluation,
    per_storm_breakdown, collect_case_studies, sigma_sensitivity,
    ensemble_size_eval) không cần sửa gì — chúng chỉ gọi
    model.sample(bl, num_ensemble=..., **kw) và đọc (pred, _, all_t),
    đúng interface chung của cả 5 kiến trúc.
    """
    if model_type == "fm":
        return TCFlowMatching(**model_cfg).to(device)
    elif model_type == "st_trans":
        return STTrans(**model_cfg).to(device)
    elif model_type in ("lstm", "gru", "rnn"):
        # model_cfg có thể đã có "model_type" (lưu từ lúc train) — nếu
        # có, ưu tiên nó khớp đúng checkpoint thật; nếu không, dùng
        # --model_type CLI đang chạy. Tránh truyền "model_type" 2 lần
        # (1 lần trong model_cfg, 1 lần override) gây lỗi "multiple
        # values for keyword argument".
        cfg = dict(model_cfg)
        cfg.setdefault("model_type", model_type)
        return PaperBaseline(**cfg).to(device)
    else:
        raise ValueError(f"Unknown --model_type: {model_type!r} "
                          f"(phải là 1 trong {MODEL_TYPES})")


def _infer_model_type_from_checkpoint(ck: dict, cli_model_type: str) -> str:
    """
    [MỚI] Nếu checkpoint tự ghi lại kiến trúc thật (model_cfg chứa
    "model_type" — đúng cách PaperBaseline lưu, vì 3 kiến trúc lstm/
    gru/rnn dùng CHUNG 1 class PaperBaseline, chỉ khác field này), ưu
    tiên nó thay vì --model_type CLI người dùng gõ tay — tránh trường
    hợp gõ nhầm --model_type lstm nhưng checkpoint thật là gru (2 kiến
    trúc có cùng shape tensor phần lớn layer, load_state_dict(strict=
    False) có thể "load được" một phần rồi âm thầm cho kết quả sai
    thay vì báo lỗi rõ ràng). Cảnh báo rõ nếu 2 nguồn xung đột; vẫn ưu
    tiên checkpoint vì đó là nguồn đáng tin hơn (ghi tại thời điểm
    train, không phụ thuộc người dùng gõ đúng CLI mỗi lần eval).
    """
    model_cfg = ck.get("model_cfg") or {}
    ckpt_type = model_cfg.get("model_type")
    if ckpt_type and ckpt_type != cli_model_type:
        print(f"  ⚠ --model_type={cli_model_type!r} nhưng checkpoint's "
              f"model_cfg ghi model_type={ckpt_type!r} — DÙNG giá trị "
              f"trong checkpoint (đáng tin hơn CLI gõ tay). Nếu đây "
              f"không phải ý bạn, kiểm tra lại đường dẫn --checkpoint.")
        return ckpt_type
    return cli_model_type


# ─────────────────────────────────────────────────────────────────────────────
#  Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_full_evaluation(model, loader, device,
                         tag:        str  = "TEST",
                         n_ensemble: int  = 20,
                         ema:        Optional[EMAModel] = None,
                         collect_samples: bool = True,
                         use_curvature_score: bool = False,
                         ddim_steps: Optional[int] = None) -> Dict:
    """
    Full evaluation. Returns per-storm arrays for downstream analysis.
    collect_samples=True: collects K ensemble samples for CRPS (memory-intensive).
    use_curvature_score: [CURV-SCORE, opt-in] passed through to model.sample()
      — re-ranks the K sampled candidates by how well their turning rate
      matches the storm's OBSERVED turning rate, in addition to the 4
      existing physics-score components. Pure inference-time change on an
      already-trained checkpoint — no retraining needed. See
      flow_matching_model.py's _physics_score docstring for the full
      rationale. Baseline models (STTrans/PaperBaseline) accept and
      ignore this kwarg via **kwargs — harmless no-op for them.
    ddim_steps: [MULTI-STEP, opt-in] number of Euler integration steps for
      sampling, overriding the checkpoint's trained n_inference_steps
      (default 1 — single-shot x0+v). This mechanism already existed in
      sample() but was never exercised (default always 1). Multi-step
      integration is the theoretically-correct way to sample a CFM model
      (v was trained to be valid at every t via the OT linear-path target,
      not just t=0), and may increase ensemble diversity/CRPS — but ONLY
      if the velocity field is meaningfully sensitive to x at each step;
      if v has collapsed to be nearly x-independent (suspected root cause
      of the low Spread/Skill ratio), multi-step will improve per-sample
      trajectory accuracy without necessarily improving diversity. Must be
      measured empirically, not assumed. None = use checkpoint default (1).
      Baseline models ignore this kwarg via **kwargs (no ODE integration
      concept for them — deterministic single-shot regression heads).
    """
    bk = None
    if ema is not None:
        try:
            bk = ema.apply_to(model)
        except Exception as e:
            print(f"  ⚠ EMA apply failed: {e}")

    model.eval()

    # Per-storm accumulators
    per_storm = {
        "ade": [], "ate": [], "cte": [],
        "rmse": [], "mae": [],
        "obs_speed": [],
        "dist_per_step": [[] for _ in range(12)],   # [12 steps][storms]
        # ATE/CTE per_storm arrays sized 12 (not 11) to align indices with
        # dist_per_step / HORIZONS convention (index k = (k+1)*6h horizon).
        # Index 0 ("6h") stays legitimately empty — there is no along/cross
        # track decomposition at the very first predicted point (no prior
        # heading reference exists yet); _m([]) correctly returns nan there.
        "ate_per_step":  [[] for _ in range(12)],
        "cte_per_step":  [[] for _ in range(12)],
        "final_dpe": [],
        "valid_speed": [], "valid_accel": [],
        "crps": [],   # per storm
    }
    all_samples_by_step = [[] for _ in range(12)]  # for CRPS

    # Collect all ensemble samples per batch if needed
    t_start = time.time()
    n_batches = len(loader)

    for i, batch in enumerate(loader):
        bl  = move(list(batch), device)
        gt  = bl[1]                          # [T, B, 2]
        obs = bl[0]                          # [T_obs, B, ≥2]
        B   = obs.shape[1]

        obs_deg_i = _norm_to_deg(obs[:, :, :2])   # [T_obs, B, 2]
        obs_spd_i = _obs_speed(obs_deg_i)          # [B]
        gt_deg_i  = _norm_to_deg(gt[:, :, :2])    # [T_gt, B, 2]

        # ── Standard prediction (mean of top-K) ──────────────────────────
        try:
            pred, _, all_t = model.sample(bl, num_ensemble=n_ensemble,
                                           use_curvature_score=use_curvature_score,
                                           ddim_steps=ddim_steps)
        except Exception as e:
            print(f"  [batch {i+1}/{n_batches}] sample error: {e}"); continue

        T   = min(pred.shape[0], gt.shape[0])
        pd  = _norm_to_deg(pred[:T])           # [T, B, 2]
        gd  = gt_deg_i[:T]                     # [T, B, 2]
        d   = _haversine_deg(pd, gd)           # [T, B]
        ate, cte = _ate_cte_full(pd, gd)       # [T-1, B], [T-1, B]

        per_storm["ade"].extend(d.mean(0).tolist())
        per_storm["ate"].extend(ate.abs().mean(0).tolist() if ate.shape[0] > 0 else [0.0]*B)
        per_storm["cte"].extend(cte.abs().mean(0).tolist() if cte.shape[0] > 0 else [0.0]*B)
        per_storm["rmse"].extend(d.pow(2).mean(0).sqrt().tolist())
        per_storm["mae"].extend(d.mean(0).tolist())   # same as ADE for haversine
        per_storm["final_dpe"].extend(d[min(T-1, 11)].tolist())   # 72h or last step
        per_storm["obs_speed"].extend(obs_spd_i.tolist())

        # Per-step dist
        for s in range(min(T, 12)):
            per_storm["dist_per_step"][s].extend(d[s].tolist())
        # ate[i]/cte[i] (i=0..T-2) hold the error AT ORIGINAL STEP INDEX i+1
        # (see _ate_cte_full: it uses gt[1:T]/pred[1:T], so ate[0] is the
        # error at step index 1, not step index 0). Storing under key i+1
        # keeps ate_per_step/cte_per_step aligned with dist_per_step's own
        # indexing (index k = (k+1)*6h horizon, per HORIZONS dict) — before
        # this fix, ate_per_step[s] held the WRONG horizon's data (off by
        # one step) for every reported horizon, and "72h" (index 11) was
        # silently dropped entirely (nan) because it was never populated.
        for i in range(min(ate.shape[0], 11)):
            k = i + 1
            per_storm["ate_per_step"][k].extend(ate[i].abs().tolist())
            per_storm["cte_per_step"][k].extend(cte[i].abs().tolist())

        # Physical validity
        phys = _physical_validity(pd, obs_deg_i)
        per_storm["valid_speed"].append(phys["valid_speed_frac"])
        per_storm["valid_accel"].append(phys["valid_accel_frac"])

        # ── CRPS: collect ensemble members ─────────────────────────────────
        if collect_samples and all_t is not None and all_t.shape[0] >= 2:
            K_actual = min(all_t.shape[0], n_ensemble)
            for s in range(min(T, 12)):
                # all_t: [K, T, B, 2] norm → convert step s to deg [K, B, 2]
                step_samples = _norm_to_deg(all_t[:K_actual, s, :, :2])   # [K, B, 2]
                all_samples_by_step[s].append(step_samples.cpu())

    elapsed = time.time() - t_start

    # ── Aggregate ──────────────────────────────────────────────────────────
    def _m(lst): return float(np.nanmean(lst)) if lst else float("nan")
    def _s(lst): return float(np.nanstd(lst))  if lst else float("nan")

    result = {
        "tag": tag, "n": len(per_storm["ade"]), "time_s": elapsed,
        # Core metrics
        "ADE":  _m(per_storm["ade"]),    "ADE_std":  _s(per_storm["ade"]),
        "ATE":  _m(per_storm["ate"]),    "ATE_std":  _s(per_storm["ate"]),
        "CTE":  _m(per_storm["cte"]),    "CTE_std":  _s(per_storm["cte"]),
        "RMSE": _m(per_storm["rmse"]),
        "MAE":  _m(per_storm["mae"]),
        "FinalDPE": _m(per_storm["final_dpe"]),
        # Physical validity
        "valid_speed_frac": _m(per_storm["valid_speed"]),
        "valid_accel_frac": _m(per_storm["valid_accel"]),
    }

    # Per-horizon (6h/12h/24h/48h/72h)
    result["per_horizon"] = {}
    result["per_horizon_ate"] = {}
    result["per_horizon_cte"] = {}
    for hz, s in HORIZONS.items():
        d_s = per_storm["dist_per_step"][s] if s < 12 else []
        result["per_horizon"][hz] = _m(d_s)
        # Arrays are now aligned with dist_per_step's indexing (see fix
        # above), so no offset/guard needed beyond bounds-checking. Index 0
        # ("6h") legitimately has no ATE/CTE (no prior heading reference at
        # the first predicted point) — _m([]) correctly returns nan there,
        # not a bug. "72h" (index 11) is now correctly populated.
        if s < 12:
            result["per_horizon_ate"][hz] = _m(per_storm["ate_per_step"][s])
            result["per_horizon_cte"][hz] = _m(per_storm["cte_per_step"][s])

    # Boxplot data (raw arrays)
    result["boxplot_ade"] = per_storm["ade"]
    # [REGRESSION FIX] per-storm ATE/CTE arrays for statistical_tests.py.
    # Without these, load_from_json() there falls back to a single scalar
    # (the mean) for ATE/CTE, and Wilcoxon signed-rank degenerates to nan
    # for both (needs >=10 paired samples). per_storm["ate"]/["cte"] are
    # the per-storm MEAN-over-horizon values (one number per storm, from
    # the "ate.abs().mean(0)" collection a few lines above in this
    # function) — a different, coarser array than ate_per_step (which is
    # per-horizon, pooled across storms). This is the one statistical
    # tests need: one paired value per storm.
    result["boxplot_ate"] = per_storm["ate"]
    result["boxplot_cte"] = per_storm["cte"]
    result["dist_per_step_mean"] = [_m(per_storm["dist_per_step"][s]) for s in range(12)]
    result["ate_per_step_mean"]  = [_m(per_storm["ate_per_step"][s])  for s in range(12)]
    result["cte_per_step_mean"]  = [_m(per_storm["cte_per_step"][s])  for s in range(12)]

    # ── By speed category ──────────────────────────────────────────────
    obs_spd_arr = np.array(per_storm["obs_speed"])
    ade_arr     = np.array(per_storm["ade"])
    ate_arr     = np.array(per_storm["ate"])
    cte_arr     = np.array(per_storm["cte"])

    # Speed categories
    slow_m = obs_spd_arr < SPEED_SLOW
    fast_m = obs_spd_arr >= SPEED_FAST
    med_m  = ~slow_m & ~fast_m
    result["by_speed"] = {
        "slow": {
            "n": int(slow_m.sum()),
            "ADE": float(ade_arr[slow_m].mean()) if slow_m.any() else float("nan"),
            "ATE": float(ate_arr[slow_m].mean()) if slow_m.any() else float("nan"),
            "CTE": float(cte_arr[slow_m].mean()) if slow_m.any() else float("nan"),
        },
        "medium": {
            "n": int(med_m.sum()),
            "ADE": float(ade_arr[med_m].mean()) if med_m.any() else float("nan"),
            "ATE": float(ate_arr[med_m].mean()) if med_m.any() else float("nan"),
            "CTE": float(cte_arr[med_m].mean()) if med_m.any() else float("nan"),
        },
        "fast": {
            "n": int(fast_m.sum()),
            "ADE": float(ade_arr[fast_m].mean()) if fast_m.any() else float("nan"),
            "ATE": float(ate_arr[fast_m].mean()) if fast_m.any() else float("nan"),
            "CTE": float(cte_arr[fast_m].mean()) if fast_m.any() else float("nan"),
        },
    }

    # ── By TC intensity category (TD/TS/Cat1-5) ───────────────────────
    # obs_speed (km/h) → Saffir-Simpson scale (1 min sustained wind proxy)
    # Mapping từ track speed sang intensity category (approximate):
    #   TC track speed ≠ wind speed, nhưng dùng làm proxy vì không có
    #   wind speed trong dataset tc-ofm. Bins theo km/h track speed:
    #   TD < 8, TS 8-15, Cat1 15-20, Cat2 20-25, Cat3 25-30, Cat4+ ≥30
    # Reviewer note: ghi rõ đây là proxy từ track speed, không phải wind speed.
    intensity_bins  = [0, 8, 15, 20, 25, 30, 9999]
    intensity_names = ["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4+"]
    result["by_intensity"] = {}
    for i, cat in enumerate(intensity_names):
        lo, hi = intensity_bins[i], intensity_bins[i+1]
        mask = (obs_spd_arr >= lo) & (obs_spd_arr < hi)
        result["by_intensity"][cat] = {
            "n":   int(mask.sum()),
            "ADE": float(ade_arr[mask].mean()) if mask.any() else float("nan"),
            "ATE": float(ate_arr[mask].mean()) if mask.any() else float("nan"),
            "CTE": float(cte_arr[mask].mean()) if mask.any() else float("nan"),
            "speed_range_kmh": f"{lo}-{hi}",
        }

    # ── CRPS (computed from collected samples) ──────────────────────────
    crps_per_step = []
    spread_per_step = []
    if collect_samples and all_samples_by_step[0]:
        print(f"  Computing CRPS over {len(all_samples_by_step[0])} batches...")
        # Simplified CRPS: E[|X-y|] - 0.5*E[|X-X'|] per step
        gt_loader_deg = []  # Need GT per step — compute from loader
        # Re-iterate loader for GT (already have per_storm aggregation)
        # Instead, compute crps from collected step samples vs per_storm dist
        for s in range(12):
            if not all_samples_by_step[s]:
                crps_per_step.append(float("nan"))
                spread_per_step.append(float("nan"))
                continue
            # samples: list of [K, B, 2] tensors → concat over batches → [K, N, 2]
            step_samples_cat = torch.cat(all_samples_by_step[s], dim=1)  # [K, N, 2]
            K, N, _ = step_samples_cat.shape
            # Spread = mean pairwise distance (subsample for speed)
            idx1 = torch.randperm(K)[:min(K, 10)]
            idx2 = torch.randperm(K)[:min(K, 10)]
            spread_vals = []
            for a, b in zip(idx1.tolist(), idx2.tolist()):
                if a != b:
                    d_ab = _haversine_deg(
                        step_samples_cat[a:a+1].expand(1, N, 2).to(device),
                        step_samples_cat[b:b+1].expand(1, N, 2).to(device)
                    ).squeeze(0)
                    spread_vals.append(float(d_ab.mean()))
            spread = float(np.mean(spread_vals)) if spread_vals else 0.0
            spread_per_step.append(spread)
            # Skill: mean dist to GT (from pre-computed per_storm dist_per_step)
            skill = _m(per_storm["dist_per_step"][s])
            # CRPS ≈ skill - 0.5*spread
            crps_per_step.append(skill - 0.5 * spread)

        result["crps"] = {
            "per_step": crps_per_step,
            "mean": float(np.nanmean(crps_per_step)),
            "spread_per_step": spread_per_step,
            "spread_mean": float(np.nanmean(spread_per_step)),
            "spread_skill_ratio": [
                s / (k + 1e-6)
                for s, k in zip(spread_per_step, result["dist_per_step_mean"])
            ],
        }
    else:
        result["crps"] = {}

    if bk is not None:
        try: ema.restore(model, bk)
        except Exception: pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Printing
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def per_storm_breakdown(model, loader, device,
                         n_ensemble: int = 20,
                         use_curvature_score: bool = False) -> Dict[str, Dict]:
    """
    [DEBUG] CTE/ATE/ADE broken down by REAL STORM NAME (not just speed
    bucket), to answer: "are 1-2 specific storms in the test set dragging
    up mean CTE for this checkpoint, and are they the SAME storms across
    seeds or different?"

    Storm identity comes from the dataset's tyID field (info["old"] =
    [year, storm_name, idx]), which seq_collate already passes through as
    the LAST element of each batch (bl[15]) — untouched by move() since
    it's a list of dicts, not a tensor. This was previously computed but
    never surfaced in evaluate_full.py's output; nothing else changes.

    [MỚI, --model_type] Hàm này nhận "model" tổng quát (không hardcode
    TCFlowMatching) — hoạt động y hệt với STTrans/PaperBaseline vì cả 3
    đều implement cùng model.sample(bl, num_ensemble=..., **kwargs) ->
    (pred, _, all_t) interface (xem _build_model()'s docstring). Không
    cần sửa gì trong thân hàm.

    Returns: {storm_name: {"n": int, "ade": [...], "ate": [...],
                            "cte": [...], "obs_speed": [...], "year": [...]}}
    One list entry per WINDOW belonging to that storm (a storm usually
    spans multiple windows/sequences in the test set).
    """
    model.eval()
    per_storm = defaultdict(lambda: {"n": 0, "ade": [], "ate": [], "cte": [],
                                       "obs_speed": [], "year": []})

    for batch in loader:
        bl = move(list(batch), device)
        gt = bl[1]
        obs = bl[0]
        try:
            tyid_list = bl[15]
        except IndexError:
            print("  ⚠ per_storm_breakdown: batch has no tyID field "
                  "(bl[15]) — dataset/collate version mismatch, skipping "
                  "storm-name attribution for this batch.")
            continue

        obs_deg_i = _norm_to_deg(obs[:, :, :2])
        gt_deg_i  = _norm_to_deg(gt[:, :, :2])
        obs_spd_i = _obs_speed(obs_deg_i)

        try:
            pred, _, _ = model.sample(bl, num_ensemble=n_ensemble,
                                       use_curvature_score=use_curvature_score)
        except Exception as e:
            print(f"  per_storm_breakdown: sample error: {e}"); continue

        T  = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T])
        gd = gt_deg_i[:T]
        d  = _haversine_deg(pd, gd)                # [T, B]
        ate, cte = _ate_cte_full(pd, gd)            # [T-1, B]

        B = obs.shape[1]
        for b in range(B):
            info = tyid_list[b] if b < len(tyid_list) else None
            if not isinstance(info, dict) or "old" not in info:
                name = "UNKNOWN"; year = "?"
            else:
                year = str(info["old"][0])
                name = str(info["old"][1])
            key = f"{name}_{year}"
            rec = per_storm[key]
            rec["n"] += 1
            rec["ade"].append(float(d[:, b].mean()))
            rec["ate"].append(float(ate[:, b].abs().mean()) if ate.shape[0] > 0 else 0.0)
            rec["cte"].append(float(cte[:, b].abs().mean()) if cte.shape[0] > 0 else 0.0)
            rec["obs_speed"].append(float(obs_spd_i[b]))
            rec["year"].append(year)

    return dict(per_storm)


def print_per_storm_breakdown(per_storm: Dict[str, Dict], sort_by: str = "cte"):
    """Print storms sorted by mean CTE descending — worst offenders first."""
    if not per_storm:
        print("  ⚠ No per-storm data (tyID unavailable or no batches processed)")
        return
    rows = []
    for name, rec in per_storm.items():
        if rec["n"] == 0:
            continue
        rows.append((
            name, rec["n"],
            float(np.mean(rec["ade"])), float(np.mean(rec["ate"])),
            float(np.mean(rec["cte"])), float(np.mean(rec["obs_speed"])),
        ))
    idx = {"cte": 4, "ate": 3, "ade": 2}.get(sort_by, 4)
    rows.sort(key=lambda r: r[idx], reverse=True)

    print(f"\n  {'='*78}")
    print(f"  PER-STORM BREAKDOWN (sorted by {sort_by.upper()}, worst first)")
    print(f"  {'='*78}")
    print(f"  {'Storm':<20} {'n':>4} {'ADE':>8} {'ATE':>8} {'CTE':>8} {'ObsSpd':>8}")
    print(f"  {'-'*78}")
    for name, n, ade, ate, cte, spd in rows:
        print(f"  {name:<20} {n:>4} {ade:>8.1f} {ate:>8.1f} {cte:>8.1f} {spd:>8.1f}")
    print(f"  {'='*78}\n")


def compare_per_storm_across_checkpoints(results_by_seed: Dict[str, Dict[str, Dict]]):
    """
    [DEBUG] Side-by-side CTE per storm across multiple seeds/checkpoints,
    to directly answer: "is the SAME storm bad for seed 0/1/2 but good for
    seed 42, or is it different storms each time (i.e. no consistent
    culprit — just aggregate variance)?"

    results_by_seed: {"42": per_storm_dict_42, "0": per_storm_dict_0, ...}
    """
    all_storms = set()
    for d in results_by_seed.values():
        all_storms.update(d.keys())
    seeds = list(results_by_seed.keys())

    print(f"\n  {'='*100}")
    print(f"  CROSS-SEED PER-STORM CTE COMPARISON")
    print(f"  {'='*100}")
    header = f"  {'Storm':<20}" + "".join(f"{'seed='+s:>14}" for s in seeds) + f"{'spread':>10}"
    print(header)
    print(f"  {'-'*100}")

    rows = []
    for storm in sorted(all_storms):
        vals = []
        for s in seeds:
            rec = results_by_seed[s].get(storm)
            vals.append(float(np.mean(rec["cte"])) if rec and rec["n"] > 0 else None)
        valid = [v for v in vals if v is not None]
        spread = (max(valid) - min(valid)) if len(valid) >= 2 else 0.0
        rows.append((storm, vals, spread))

    # Sort by spread descending — storms with the most seed-to-seed
    # disagreement first (these are the "culprits" worth investigating).
    rows.sort(key=lambda r: r[2], reverse=True)
    for storm, vals, spread in rows:
        val_str = "".join(f"{v:>14.1f}" if v is not None else f"{'---':>14}" for v in vals)
        print(f"  {storm:<20}{val_str}{spread:>10.1f}")
    print(f"  {'='*100}")
    print(f"  Sorted by spread (max-min CTE across seeds) descending — top rows are")
    print(f"  storms where checkpoints disagree most; these are the strongest")
    print(f"  candidates for a genuine per-storm-difficulty explanation of the")
    print(f"  aggregate CTE gap, as opposed to uniform variance across all storms.\n")


def print_full_results(r: Dict, st_trans: Dict = ST_TRANS):
    n = r.get("n", 0)
    tag = r.get("tag", "")
    print(f"\n  {'='*72}")
    print(f"  [{tag}]  n={n}  ({r.get('time_s', 0):.1f}s)")

    def beat(k, v):
        ref = st_trans.get(k, 1e9)
        return "✓" if v < ref else "✗"

    # Core metrics
    ade, ate, cte = r["ADE"], r["ATE"], r["CTE"]
    print(f"  ADE={ade:7.2f}±{r.get('ADE_std',0):.1f}km {beat('ADE',ade)}  "
          f"ATE={ate:7.2f}±{r.get('ATE_std',0):.1f}km {beat('ATE',ate)}  "
          f"CTE={cte:7.2f}±{r.get('CTE_std',0):.1f}km {beat('CTE',cte)}")
    print(f"  RMSE={r.get('RMSE',0):6.2f}km  MAE={r.get('MAE',0):6.2f}km  "
          f"FinalDPE(72h)={r.get('FinalDPE',0):6.2f}km")

    # Per-horizon
    ph = r.get("per_horizon", {})
    print(f"  Per-horizon ADE:  " +
          "  ".join(f"{h}={ph.get(h,float('nan')):6.1f}" for h in ["6h","12h","24h","48h","72h"]))
    ph_ate = r.get("per_horizon_ate", {})
    ph_cte = r.get("per_horizon_cte", {})
    print(f"  Per-horizon ATE:  " +
          "  ".join(f"{h}={ph_ate.get(h,float('nan')):6.1f}" for h in ["12h","24h","48h","72h"]))
    print(f"  Per-horizon CTE:  " +
          "  ".join(f"{h}={ph_cte.get(h,float('nan')):6.1f}" for h in ["12h","24h","48h","72h"]))

    # ST-Trans comparison
    print(f"\n  vs ST-Trans:")
    print(f"  {'Metric':<8} {'ST-Trans':>9} {'Ours':>9} {'Δ':>8}  Status")
    print(f"  {'─'*48}")
    for k in ["ADE", "ATE", "CTE"]:
        v = r.get(k, float("nan"))
        ref = st_trans.get(k, float("nan"))
        delta = v - ref
        status = "✓ BEAT" if v < ref else "✗"
        print(f"  {k:<8} {ref:>9.1f} {v:>9.2f} {delta:>+8.2f}km  {status}")

    # By speed category
    bs = r.get("by_speed", {})
    if bs:
        print(f"\n  By obs-speed category:")
        for cat, d in bs.items():
            print(f"    {cat:<8} n={d.get('n',0):3d}  "
                  f"ADE={d.get('ADE',float('nan')):6.1f}  "
                  f"ATE={d.get('ATE',float('nan')):6.1f}  "
                  f"CTE={d.get('CTE',float('nan')):6.1f}")

    # By TC intensity category
    bi = r.get("by_intensity", {})
    if bi:
        print(f"\n  By TC intensity (track speed proxy):")
        print(f"    {'Cat':<8} {'n':>4}  {'ADE':>7}  {'ATE':>7}  {'CTE':>7}  Speed(km/h)")
        for cat, d in bi.items():
            if d.get("n", 0) > 0:
                print(f"    {cat:<8} {d['n']:>4}  "
                      f"{d.get('ADE',float('nan')):>7.1f}  "
                      f"{d.get('ATE',float('nan')):>7.1f}  "
                      f"{d.get('CTE',float('nan')):>7.1f}  "
                      f"{d.get('speed_range_kmh','')}")

    # Physical validity
    print(f"\n  Physical validity:  "
          f"valid_speed={r.get('valid_speed_frac',0):.3f}  "
          f"valid_accel={r.get('valid_accel_frac',0):.3f}")

    # CRPS
    crps = r.get("crps", {})
    if crps:
        print(f"  CRPS mean={crps.get('mean',float('nan')):.2f}km  "
              f"spread_mean={crps.get('spread_mean',float('nan')):.2f}km")
        spread_ps = crps.get("spread_per_step", [])
        dist_ps   = r.get("dist_per_step_mean", [])
        ss = crps.get("spread_skill_ratio", [])
        if spread_ps:
            hz_spread = {h: spread_ps[s] if s < len(spread_ps) else float("nan")
                         for h, s in HORIZONS.items()}
            hz_skill  = {h: dist_ps[s] if s < len(dist_ps) else float("nan")
                         for h, s in HORIZONS.items()}
            print(f"  Spread (km) by horizon: " +
                  "  ".join(f"{h}={v:.1f}" for h, v in hz_spread.items()))
            print(f"  Skill  (km) by horizon: " +
                  "  ".join(f"{h}={v:.1f}" for h, v in hz_skill.items()))
        if ss:
            hz_ss = {h: ss[s] if s < len(ss) else float("nan")
                     for h, s in HORIZONS.items()}
            print(f"  Spread/Skill ratio: " +
                  "  ".join(f"{h}={v:.2f}" for h, v in hz_ss.items()))

    print(f"  {'='*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Sigma sensitivity analysis (reviewer ablation)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sigma_sensitivity(model, loader, device,
                       sigma_values: list = [0.01, 0.02, 0.04, 0.06, 0.08],
                       n_ensemble: int = 20) -> Dict:
    """
    Ablation: sensitivity của kết quả theo sigma_inference.
    Chạy inference với nhiều sigma khác nhau trên cùng checkpoint.
    → Justification cho sigma_inference=0.04 cố định (reviewer question).

    [MỚI, --model_type] CHỈ có ý nghĩa cho FM (sigma_inference là tham
    số CFM riêng của TCFlowMatching — STTrans/PaperBaseline không có
    attribute này). Guard rõ ràng ngay đầu hàm thay vì để crash mơ hồ
    (AttributeError không rõ nguyên nhân) khi lỡ gọi với baseline.
    """
    raw = _unwrap(model)
    if not hasattr(raw, "sigma_inference"):
        print("  ⚠ sigma_sensitivity: model không có sigma_inference "
              "(không phải FM — bỏ qua, trả về rỗng). Ablation này chỉ "
              "áp dụng cho --model_type fm.")
        return {}
    orig_sigma = float(raw.sigma_inference)
    results = {}
    for sigma in sigma_values:
        raw.sigma_inference = sigma  # temporarily override
        all_ade = []
        all_cte = []
        for batch in loader:
            bl = move(list(batch), device)
            gt = bl[1]
            try:
                pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
            except Exception:
                continue
            T   = min(pred.shape[0], gt.shape[0])
            pd  = _norm_to_deg(pred[:T])
            gd  = _norm_to_deg(gt[:T, :, :2])
            d   = _haversine_deg(pd, gd)
            all_ade.extend(d.mean(0).tolist())
            if T >= 2:
                ate_v, cte_v = _ate_cte_full(pd, gd)
                all_cte.extend(cte_v.abs().mean(0).tolist())
        results[sigma] = {
            "sigma":    sigma,
            "ADE":      float(np.mean(all_ade)) if all_ade else float("nan"),
            "CTE":      float(np.mean(all_cte)) if all_cte else float("nan"),
            "n":        len(all_ade),
        }
        print(f"  sigma={sigma:.3f}: ADE={results[sigma]['ADE']:.2f}  CTE={results[sigma]['CTE']:.2f}")
    raw.sigma_inference = orig_sigma  # restore
    return results


@torch.no_grad()
def ensemble_size_eval(model, loader, device,
                        k_values: list = [1, 3, 5, 10, 20, 40]) -> Dict:
    """
    Ablation: accuracy vs compute trade-off theo ensemble size K.
    → Justification cho K=20 default.
    → ESWA Table: shows diminishing returns beyond K=20.

    [MỚI, --model_type] Baseline (STTrans/PaperBaseline) đều
    deterministic (sample() luôn trả cùng 1 pred bất kể num_ensemble,
    all_t chỉ có K=1 candidate) — K-sweep vẫn CHẠY ĐƯỢC (không crash,
    model.sample() nhận num_ensemble qua **kwargs rồi bỏ qua), nhưng
    ADE/ATE/CTE sẽ giống hệt nhau ở mọi K và spread luôn NaN (không đủ
    2 candidate để tính pairwise distance) — đây là kết quả ĐÚNG về mặt
    ý nghĩa (baseline không có khái niệm ensemble), không phải bug,
    nhưng dễ gây hiểu nhầm nếu không biết trước. Không chặn cứng vì có
    thể vẫn hữu ích để xác nhận baseline stable qua K (dù luôn giống
    nhau) — chỉ in cảnh báo 1 lần.
    """
    raw = _unwrap(model)
    if not hasattr(raw, "n_inference_steps") and not hasattr(raw, "sigma_inference"):
        print("  ⚠ ensemble_size_eval: model có vẻ là baseline "
              "deterministic (không phải FM) — ADE/ATE/CTE sẽ GIỐNG "
              "HỆT nhau ở mọi K, spread sẽ luôn NaN. Đây là kết quả "
              "đúng (baseline không có khái niệm ensemble), không phải "
              "lỗi.")
    results = {}
    for k in k_values:
        all_ade, all_ate, all_cte, all_spread = [], [], [], []
        by_lt_ade = defaultdict(list)
        by_lt_ate = defaultdict(list)
        by_lt_cte = defaultdict(list)
        t0 = time.time()
        for batch in loader:
            bl = move(list(batch), device)
            gt = bl[1]
            try:
                pred, _, all_t = model.sample(bl, num_ensemble=k)
            except Exception:
                continue
            T   = min(pred.shape[0], gt.shape[0])
            pd  = _norm_to_deg(pred[:T])
            gd  = _norm_to_deg(gt[:T, :, :2])
            d   = _haversine_deg(pd, gd)          # [T, B]
            all_ade.extend(d.mean(0).tolist())
            for step0 in range(T):
                by_lt_ade[step0 + 1].extend(d[step0].tolist())

            if T >= 2:
                ate_v, cte_v = _ate_cte_full(pd, gd)   # [T-1, B] each
                all_ate.extend(ate_v.abs().mean(0).tolist())
                all_cte.extend(cte_v.abs().mean(0).tolist())
                for kk in range(T - 1):   # kk+2 = lead_time
                    by_lt_ate[kk + 2].extend(ate_v[kk].abs().tolist())
                    by_lt_cte[kk + 2].extend(cte_v[kk].abs().tolist())

            # [BỔ SUNG] spread — pairwise haversine giữa các candidate
            # cuối cùng, cùng công thức với ode_steps_sweep(). K=1 luôn
            # cho spread=NaN (không có cặp nào để so — đúng ý nghĩa,
            # không phải bug). Baseline: all_t.shape[0] luôn = 1
            # (deterministic) nên nhánh này tự động không chạy — spread
            # luôn NaN cho baseline, đúng như cảnh báo ở đầu hàm.
            if all_t is not None and torch.is_tensor(all_t) and all_t.shape[0] >= 2:
                Kb = all_t.shape[0]
                last = _norm_to_deg(all_t[:, -1, :, :2])   # [K, B, 2]
                idx1 = torch.randperm(Kb)[:min(Kb, 10)]
                idx2 = torch.randperm(Kb)[:min(Kb, 10)]
                for a, b in zip(idx1.tolist(), idx2.tolist()):
                    if a != b:
                        dab = _haversine_deg(last[a:a+1], last[b:b+1]).squeeze(0)
                        all_spread.append(float(dab.mean()))

        elapsed = time.time() - t0
        by_lead_time = {}
        all_lts = sorted(set(by_lt_ade.keys()) | set(by_lt_ate.keys()) | set(by_lt_cte.keys()))
        for lt in all_lts:
            by_lead_time[lt] = {
                "ADE": float(np.mean(by_lt_ade[lt])) if by_lt_ade.get(lt) else float("nan"),
                "ATE": float(np.mean(by_lt_ate[lt])) if by_lt_ate.get(lt) else float("nan"),
                "CTE": float(np.mean(by_lt_cte[lt])) if by_lt_cte.get(lt) else float("nan"),
                "n":   len(by_lt_ade.get(lt, [])),
            }

        results[k] = {
            "K": k, "ADE": float(np.mean(all_ade)) if all_ade else float("nan"),
            "ATE": float(np.mean(all_ate)) if all_ate else float("nan"),
            "CTE": float(np.mean(all_cte)) if all_cte else float("nan"),
            "spread": float(np.mean(all_spread)) if all_spread else float("nan"),
            "time_s": elapsed, "n": len(all_ade),
            "by_lead_time": by_lead_time,
        }
        print(f"  K={k:3d}: ADE={results[k]['ADE']:.2f}  ATE={results[k]['ATE']:.2f}"
              f"  CTE={results[k]['CTE']:.2f}  spread={results[k]['spread']:.2f}km  "
              f"t={elapsed:.1f}s")
    return results


def _infer_seed_local(checkpoint_path: str, ck: dict) -> str:
    """
    Copy nguyên văn logic của evaluate_multi_model.py's _infer_seed()
    (không import chéo — file này tự chứa, giữ đúng nguyên tắc đã áp
    dụng cho các file khác trong dự án). Ưu tiên đọc field "seed" lưu
    sẵn trong checkpoint, fallback parse "seed<N>" từ đường dẫn.
    """
    if isinstance(ck, dict) and "seed" in ck:
        return str(ck["seed"])
    import re
    m = re.search(r"seed[_-]?(\d+)", checkpoint_path)
    if m:
        return m.group(1)
    return "unknown"


def run_ensemble_ablation_multi_seed(checkpoints: List[str], dataset_root: str,
                                      split: str, k_values: List[int],
                                      device, args) -> Dict:
    """
    [MỚI] Chạy ensemble_size_eval() (K-sweep) trên NHIỀU checkpoint
    (nhiều seed của CÙNG 1 kiến trúc FM), rồi gộp thành mean±std theo
    seed cho mỗi K — cùng triết lý "pooled/mean±std theo seed, không
    chọn best/1 seed ngẫu nhiên" đã áp dụng cho generate_paper_report.py's
    Table 1/3/4/5.

    [MỚI, --model_type] Nhánh multi-seed vẫn CHỈ dùng cho FM (K-sweep
    có ý nghĩa nhất với FM — baseline deterministic nên spread luôn
    NaN, xem cảnh báo trong ensemble_size_eval()). Nếu args.model_type
    khác "fm", vẫn chạy được (dùng _build_model tổng quát) nhưng kết
    quả sẽ đơn điệu theo K — không cấm, chỉ không phải use-case chính.

    Trả về dict cùng SCHEMA với ensemble_size_eval() (key=K, value=dict
    có ADE/ATE/CTE/spread) để build_ensemble_k_table() trong
    generate_paper_report.py dùng lại được NGUYÊN VẸN, không cần sửa gì
    ở phía đọc — chỉ khác là mỗi giá trị giờ là MEAN qua các seed, và
    có thêm "_std" bên cạnh mỗi field để không mất thông tin biến thiên
    giữa các seed.
    """
    from Model.data.loader_training import data_loader
    model_type = getattr(args, "model_type", "fm")
    per_seed_results = {}   # seed -> {K: {...}}
    for ckpt_path in checkpoints:
        print(f"\n  {'='*70}\n  Loading checkpoint: {ckpt_path}\n  {'='*70}")
        ck = torch.load(ckpt_path, map_location="cpu")
        model_cfg = ck.get("model_cfg") or {}
        if not model_cfg:
            print("  ⚠ Checkpoint has no model_cfg — dùng constructor defaults.")
        resolved_type = _infer_model_type_from_checkpoint(ck, model_type)
        model = _build_model(resolved_type, model_cfg, device)
        state = ck.get("model", ck)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  ⚠ load_state_dict: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected keys")

        ema = None
        if resolved_type == "fm" and not args.no_ema and ck.get("ema"):
            try:
                ema = EMAModel(model)
                for k, v in ck["ema"].items():
                    if k in ema.shadow:
                        ema.shadow[k].copy_(v.to(device))
                print(f"  EMA loaded ({len(ema.shadow)} params)")
            except Exception as e:
                print(f"  ⚠ EMA failed: {e}")

        seed = _infer_seed_local(ckpt_path, ck)
        print(f"  seed={seed}  epoch={ck.get('epoch', '?')}  model_type={resolved_type}")

        # [FIX] 2 bug: (1) _NS copy vars(args) không đủ field data_loader()
        # cần (obs_len/pred_len/batch_size/num_workers/skip/min_ped/
        # threshold không có trong evaluate_full.py's CLI chính) —
        # dùng đúng argparse.Namespace đầy đủ, CÙNG pattern đã chạy được
        # ở luồng single-checkpoint (dòng ~1295) và evaluate_multi_model.py.
        # (2) data_loader() trả về (train/val_loader, test_loader) —
        # code cũ lấy NHẦM phần tử đầu (`loader, _ = ...`), trong khi
        # mọi nơi khác trong codebase đều dùng `_, loader = ...` (phần
        # tử thứ hai). Sửa cho khớp.
        import argparse as _ap
        _loader_args = _ap.Namespace(
            dataset_root = dataset_root,
            obs_len      = 8,
            pred_len     = 12,
            batch_size   = 64,
            num_workers  = 2,
            test_year    = getattr(args, "test_year", None),
            skip         = getattr(args, "skip", 1),
            min_ped      = getattr(args, "min_ped", 1),
            threshold    = getattr(args, "threshold", 0.002),
        )
        _, loader = data_loader(_loader_args, {"root": dataset_root, "type": split},
                                test=(split != "train"))
        print(f"  Data: {len(loader)} batches")

        model.eval()
        results = ensemble_size_eval(model, loader, device, k_values=k_values)
        per_seed_results[seed] = results

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Gộp: mean±std qua các seed, cho từng K
    n_seeds = len(per_seed_results)
    print(f"\n  Gộp kết quả qua {n_seeds} seed: {list(per_seed_results.keys())}")
    merged = {}
    for k in k_values:
        entry = {}
        for field in ("ADE", "ATE", "CTE", "spread", "time_s", "n"):
            vals = [per_seed_results[s][k][field] for s in per_seed_results
                    if k in per_seed_results[s]
                    and not np.isnan(per_seed_results[s][k].get(field, float("nan")))]
            entry[field] = float(np.mean(vals)) if vals else float("nan")
            if field in ("ADE", "ATE", "CTE", "spread"):
                entry[f"{field}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        entry["n_seeds"] = n_seeds

        # [BỔ SUNG] Gộp by_lead_time qua các seed — cùng cách seed-mean-
        # rồi-mean/std đã dùng cho generate_paper_report.py's Table 1/3
        # (KHÔNG gộp raw records của mọi seed lại rồi mới mean, tránh
        # lẫn variance storm-to-storm với variance seed-to-seed).
        all_lts = set()
        for s in per_seed_results:
            if k in per_seed_results[s]:
                all_lts |= set(per_seed_results[s][k].get("by_lead_time", {}).keys())
        by_lead_time_merged = {}
        for lt in sorted(all_lts):
            for metric in ("ADE", "ATE", "CTE"):
                seed_vals = [per_seed_results[s][k]["by_lead_time"][lt][metric]
                            for s in per_seed_results
                            if k in per_seed_results[s]
                            and lt in per_seed_results[s][k].get("by_lead_time", {})
                            and not np.isnan(per_seed_results[s][k]["by_lead_time"][lt].get(metric, float("nan")))]
                by_lead_time_merged.setdefault(lt, {})
                by_lead_time_merged[lt][metric] = float(np.mean(seed_vals)) if seed_vals else float("nan")
                by_lead_time_merged[lt][f"{metric}_std"] = float(np.std(seed_vals)) if len(seed_vals) > 1 else 0.0
        entry["by_lead_time"] = by_lead_time_merged

        merged[k] = entry
        print(f"  K={k:3d}: ADE={entry['ADE']:.2f}±{entry['ADE_std']:.2f}  "
              f"ATE={entry['ATE']:.2f}±{entry['ATE_std']:.2f}  "
              f"CTE={entry['CTE']:.2f}±{entry['CTE_std']:.2f}  "
              f"spread={entry['spread']:.2f}±{entry['spread_std']:.2f}km")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
#  Case study: cone of uncertainty
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_case_studies(model, loader, device, n_cases: int = 6,
                          n_ensemble: int = 20) -> List[Dict]:
    """
    Collect detailed predictions for N representative storms.
    Picks storms covering slow/medium/fast and worst/best ADE.
    Returns list of dicts with obs, gt, all ensemble members, mean pred.
    """
    model.eval()
    cases = []
    storm_data = []

    for batch in loader:
        bl  = move(list(batch), device)
        gt  = bl[1]
        obs = bl[0]
        B   = obs.shape[1]
        obs_deg = _norm_to_deg(obs[:, :, :2])

        try:
            pred, _, all_t = model.sample(bl, num_ensemble=n_ensemble)
        except Exception:
            continue

        T = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T])
        gd = _norm_to_deg(gt[:T, :, :2])
        d  = _haversine_deg(pd, gd).mean(0)  # [B] ADE per storm
        spd = _obs_speed(obs_deg)              # [B]

        for b in range(B):
            storm_data.append({
                "ade": float(d[b]),
                "obs_speed": float(spd[b]),
                "obs_deg": obs_deg[:, b, :].cpu().numpy(),
                "gt_deg":  gd[:, b, :].cpu().numpy(),
                "pred_deg": pd[:, b, :].cpu().numpy(),
                "ensemble": all_t[:, :T, b, :2].cpu().numpy()
                            if all_t is not None else None,
            })
        if len(storm_data) > 200:
            break

    if not storm_data:
        return []

    # Pick representative cases: best/worst/median + fast/slow
    storm_data.sort(key=lambda x: x["ade"])
    n_total = len(storm_data)

    selected_indices = set()
    # Best, worst, median
    selected_indices.update([0, n_total//4, n_total//2, 3*n_total//4, n_total-1])
    # Fastest storm
    fastest = max(range(len(storm_data)), key=lambda i: storm_data[i]["obs_speed"])
    selected_indices.add(fastest)
    # Slowest storm
    slowest = min(range(len(storm_data)), key=lambda i: storm_data[i]["obs_speed"])
    selected_indices.add(slowest)

    selected = sorted(selected_indices)[:n_cases]
    return [storm_data[i] for i in selected]


def compute_cone_of_uncertainty(ensemble: np.ndarray,
                                  confidence: float = 0.67) -> Dict:
    """
    Compute cone of uncertainty from ensemble.
    ensemble: [K, T, 2] in degrees (lon, lat)
    Returns per-step radius (km) at given confidence level.
    """
    K, T, _ = ensemble.shape
    radii = []
    for t in range(T):
        pts_t = ensemble[:, t, :]  # [K, 2] lon, lat
        mean_lon = pts_t[:, 0].mean()
        mean_lat = pts_t[:, 1].mean()
        # Haversine distance from each member to mean
        dists = []
        for k in range(K):
            dlat = math.radians(pts_t[k, 1] - mean_lat)
            dlon = math.radians(pts_t[k, 0] - mean_lon)
            lat1 = math.radians(mean_lat)
            lat2 = math.radians(pts_t[k, 1])
            a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
            dists.append(2 * R_EARTH * math.asin(math.sqrt(max(0, min(1, a)))))
        radii.append(float(np.quantile(dists, confidence)))
    return {
        "radii_km": radii,
        "horizon_labels": [f"{(t+1)*6}h" for t in range(T)],
        "confidence": confidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="TC-FlowMatching ESWA Full Evaluation")
    p.add_argument("--checkpoint",   required=False, default=None,
                   help="1 checkpoint duy nhất — dùng cho pipeline đầy đủ "
                        "(case_studies, sigma_sensitivity, per_storm, ...). "
                        "Bắt buộc trừ khi dùng --checkpoints.")
    p.add_argument("--checkpoints",  type=str, nargs="+", default=None,
                   help="[MỚI] Nhiều checkpoint (nhiều seed CÙNG 1 kiến "
                        "trúc) — CHỈ dùng được với --ensemble_ablation, "
                        "chạy K-sweep trên từng checkpoint rồi gộp "
                        "mean±std theo seed, lưu 1 file JSON riêng rồi "
                        "thoát sớm (bỏ qua case_studies/sigma_sensitivity/"
                        "per_storm — các phần đó gắn với đúng 1 checkpoint, "
                        "không có ý nghĩa 'gộp qua seed'). Không thể dùng "
                        "cùng lúc với --checkpoint.")
    p.add_argument("--model_type",   default="fm", choices=MODEL_TYPES,
                   help="[MỚI] Kiến trúc của checkpoint đang eval: fm "
                        "(TCFlowMatching, mặc định — giữ nguyên hành vi "
                        "cũ khi không truyền cờ này), st_trans (STTrans), "
                        "lstm/gru/rnn (PaperBaseline với model_type "
                        "tương ứng). Nếu checkpoint tự ghi model_type "
                        "trong model_cfg (đúng cách PaperBaseline lưu), "
                        "giá trị đó được ưu tiên hơn cờ CLI này — xem "
                        "_infer_model_type_from_checkpoint(). Các tính "
                        "năng chỉ có ý nghĩa cho FM (--sigma_sensitivity, "
                        "EMA) tự bỏ qua/cảnh báo khi model_type khác fm, "
                        "không crash.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split",        default="test", choices=["test","val","train"])
    p.add_argument("--n_ensemble",   type=int, default=20)
    p.add_argument("--output_dir",   default="eval_results")
    p.add_argument("--no_ema",       action="store_true")
    p.add_argument("--no_crps",      action="store_true",
                   help="Skip CRPS computation (faster)")
    p.add_argument("--use_curvature_score", action="store_true", default=False,
                   help="[CURV-SCORE, opt-in] Re-rank sampled candidates by "
                        "how well their turning rate matches the storm's "
                        "OBSERVED turning rate (in addition to the 4 "
                        "existing physics-score components). Pure "
                        "inference-time change on an already-trained "
                        "checkpoint — no retraining needed. Default False "
                        "preserves prior behavior exactly. No-op for "
                        "baseline models (accepted via **kwargs, ignored).")
    p.add_argument("--ddim_steps", type=int, default=None,
                   help="[MULTI-STEP, opt-in] Number of Euler integration "
                        "steps for sampling (overrides checkpoint's "
                        "n_inference_steps, default 1 = single-shot x0+v). "
                        "Try e.g. 4, 8, 16 to test whether multi-step "
                        "integration improves ensemble diversity (CRPS, "
                        "Spread/Skill ratio) on an EXISTING checkpoint — "
                        "no retraining needed. Effect is NOT guaranteed; "
                        "measure and compare against the default before "
                        "relying on it. None = use checkpoint default (1). "
                        "No-op for baseline models (accepted via **kwargs, "
                        "ignored — no ODE integration concept for them).")
    p.add_argument("--case_studies", action="store_true", default=True,
                   help="Collect case study data")
    p.add_argument("--n_cases",      type=int, default=6)
    p.add_argument("--gpu",          type=int, default=0)
    p.add_argument("--test_year",    type=int, default=None,
                   help="Filter test set by year (same as evaluate_test_storms.py)")
    p.add_argument("--sigma_sensitivity", action="store_true", default=False,
                   help="Run sigma_inference sensitivity analysis (reviewer "
                        "ablation). FM-only — auto-skipped with a warning "
                        "if --model_type is not fm.")
    p.add_argument("--ensemble_ablation",  action="store_true", default=False,
                   help="Run ensemble size K ablation")
    p.add_argument("--k_values", type=int, nargs="+", default=[1, 3, 5, 10, 20, 40],
                   help="Danh sách K để quét cho --ensemble_ablation. "
                        "Default [1,3,5,10,20,40] bao quanh K=20 hiện dùng.")
    p.add_argument("--per_storm",         action="store_true", default=False,
                   help="[DEBUG] Break down ADE/ATE/CTE by real storm name "
                        "(from dataset tyID) instead of only speed/intensity "
                        "buckets. Prints storms sorted by CTE, worst first. "
                        "Also saves per_storm_<split>_ep<N>.json for "
                        "cross-checkpoint comparison (see compare_seeds.py "
                        "usage in this file's docstring / README). Works "
                        "for any --model_type.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ── [MỚI] Multi-seed ensemble ablation — rẽ nhánh sớm, KHÔNG chạm
    # vào phần còn lại của main() (case_studies/sigma_sensitivity/
    # per_storm đều gắn với đúng 1 checkpoint, không có ý nghĩa "gộp
    # qua seed") ────────────────────────────────────────────────────────
    if args.checkpoints:
        if args.checkpoint:
            print("  ⚠ Truyền cả --checkpoint và --checkpoints — dùng "
                  "--checkpoints (nhiều seed), bỏ qua --checkpoint.")
        if not args.ensemble_ablation:
            print("  ❌ --checkpoints chỉ hỗ trợ cùng --ensemble_ablation "
                  "(các chế độ khác gắn với 1 checkpoint duy nhất, dùng "
                  "--checkpoint thay vì --checkpoints cho chúng).")
            return
        print(f"\n  Multi-seed ensemble ablation | {len(args.checkpoints)} checkpoints")
        print(f"  Model type: {args.model_type} | Split: {args.split} | "
              f"Device: {device} | K values: {args.k_values}")
        print("="*72)
        merged = run_ensemble_ablation_multi_seed(
            args.checkpoints, args.dataset_root, args.split,
            args.k_values, device, args)
        out_path = os.path.join(args.output_dir,
                                 f"ensemble_ablation_multiseed_{args.model_type}_{args.split}.json")
        with open(out_path, "w") as f:
            json.dump({"ensemble_ablation": merged}, f, indent=2)
        print(f"\n  Saved multi-seed ensemble ablation → {out_path}")
        print(f"  Dùng file này với generate_paper_report.py's --eval_full_json "
              f"(schema khớp với 'ensemble_ablation' key mà build_ensemble_k_table() cần).")
        return

    if not args.checkpoint:
        print("  ❌ Cần --checkpoint (1 checkpoint) hoặc --checkpoints "
              "(nhiều checkpoint, chỉ dùng với --ensemble_ablation).")
        return

    print(f"\n  Checkpoint: {args.checkpoint}")
    print(f"  Model type: {args.model_type} | Split: {args.split} | Device: {device}")
    print("="*72)

    # ── Load model ─────────────────────────────────────────────────────────
    ck = torch.load(args.checkpoint, map_location="cpu")
    # model_cfg may be: missing (old checkpoints, pre-fix), present but None
    # (checkpoint saved without passing model_cfg=), or a real dict — only
    # the last case should override constructor defaults.
    model_cfg = ck.get("model_cfg") or {}
    if not model_cfg:
        print("  ⚠ Checkpoint has no model_cfg — reconstructing with "
              "constructor DEFAULTS. This is only correct if the checkpoint "
              "was trained with default architecture args. If you trained "
              "with non-default architecture flags, this will silently "
              "load the WRONG architecture.")
    resolved_type = _infer_model_type_from_checkpoint(ck, args.model_type)
    model = _build_model(resolved_type, model_cfg, device)
    state = ck.get("model", ck)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"  ⚠ Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected: print(f"  ⚠ Unexpected ({len(unexpected)}): {unexpected[:3]}...")
    ep = ck.get("epoch", "?")
    print(f"  Loaded ep{ep}  (model_type={resolved_type})")

    # Print learned params (FM-only attributes — guarded with hasattr,
    # baseline models simply skip these lines silently)
    raw = _unwrap(model)
    if hasattr(raw, "speed_correction_logits"):
        corr = (torch.sigmoid(raw.speed_correction_logits) * 2.0).tolist()
        print(f"  [LEARN] speed_corr: {[f'{v:.3f}' for v in corr[:4]]}...")
    if hasattr(raw, "log_sigma_reg"):
        print(f"  [LEARN] eff_lambda: "
              f"reg={0.5*math.exp(-2*max(-3,raw.log_sigma_reg.item())):.3f}  "
              f"heading={0.5*math.exp(-2*max(-3,raw.log_sigma_heading.item())):.3f}  "
              f"calib={0.5*math.exp(-2*max(-3,raw.log_sigma_calib.item())):.3f}")

    # Model footprint
    total_params = sum(p.numel() for p in model.parameters())
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"  Model: {total_params:,} params | {mem_mb:.1f}MB")

    # ── EMA ────────────────────────────────────────────────────────────────
    # [MỚI, --model_type] EMA (EMAModel) là cơ chế train-time riêng của
    # TCFlowMatching (self._ema trong constructor) — STTrans/PaperBaseline
    # không có. Guard theo resolved_type để không thử load EMA state vào
    # 1 model không hỗ trợ nó (checkpoint baseline sẽ không có key "ema"
    # nên nhánh này vốn đã tự nhiên bỏ qua, nhưng guard rõ ràng ở đây để
    # ý định không phụ thuộc vào việc checkpoint "tình cờ" thiếu key).
    ema = None
    if resolved_type == "fm" and not args.no_ema and ck.get("ema"):
        try:
            ema = EMAModel(model)
            for k, v in ck["ema"].items():
                if k in ema.shadow:
                    ema.shadow[k].copy_(v.to(device))
            print(f"  EMA loaded ({len(ema.shadow)} params)")
        except Exception as e:
            print(f"  ⚠ EMA failed: {e}"); ema = None

    # ── Data ───────────────────────────────────────────────────────────────
    import argparse as _ap
    _loader_args = _ap.Namespace(
        dataset_root = args.dataset_root,
        obs_len      = 8,
        pred_len     = 12,
        batch_size   = 64,
        num_workers  = 2,
        test_year    = getattr(args, "test_year", None),
        skip         = getattr(args, "skip", 1),
        min_ped      = getattr(args, "min_ped", 1),
        threshold    = getattr(args, "threshold", 0.002),
    )
    try:
        _, loader = data_loader(
            _loader_args,
            {"root": args.dataset_root, "type": args.split},
            test=(args.split != "train"),
        )
    except Exception as _e:
        print(f"  ❌ data_loader(type=\'{args.split}\') failed: {_e}")
        raise
    print(f"  Data: {len(loader)} batches")

    # ── Full evaluation ─────────────────────────────────────────────────────
    result = run_full_evaluation(
        model, loader, device,
        tag=f"{args.split.upper()} ep{ep} [{resolved_type}]",
        n_ensemble=args.n_ensemble,
        ema=ema,
        collect_samples=not args.no_crps,
        use_curvature_score=args.use_curvature_score,
        ddim_steps=args.ddim_steps,
    )

    # Print
    print_full_results(result)

    # Timing benchmark: ms per inference step
    ms_per_step = float("nan")
    try:
        model.eval()
        with torch.no_grad():
            dummy = move(list(next(iter(loader))), device)
            t0 = time.time()
            for _ in range(10):
                model.sample(dummy, num_ensemble=args.n_ensemble)
            ms_per_step = (time.time() - t0) / 10 * 1000
        print(f"  Inference: {ms_per_step:.1f}ms per batch (K={args.n_ensemble})")
    except Exception as e:
        print(f"  ⚠ Timing benchmark failed: {e}")
    result["inference_ms_per_batch"] = ms_per_step
    result["total_params"] = total_params
    result["model_mb"] = mem_mb
    result["model_type"] = resolved_type

    # ── Case studies ─────────────────────────────────────────────────────────
    case_results = []
    if args.case_studies:
        print(f"\n  Collecting {args.n_cases} case studies...")
        cases = collect_case_studies(model, loader, device,
                                      n_cases=args.n_cases,
                                      n_ensemble=args.n_ensemble)
        for i, case in enumerate(cases):
            cone = {}
            if case.get("ensemble") is not None:
                cone = compute_cone_of_uncertainty(
                    case["ensemble"].reshape(-1, case["ensemble"].shape[-2], 2),
                    confidence=0.67
                )
            case_results.append({
                "id":         i,
                "ade":        case["ade"],
                "obs_speed":  case["obs_speed"],
                "cone_67pct": cone,
                # Arrays saved as lists for JSON
                "obs_deg":    case["obs_deg"].tolist(),
                "gt_deg":     case["gt_deg"].tolist(),
                "pred_deg":   case["pred_deg"].tolist(),
            })
            print(f"    Case {i+1}: ADE={case['ade']:.1f}km  "
                  f"obs_speed={case['obs_speed']:.1f}km/h  "
                  f"cone_72h={cone.get('radii_km', [0]*12)[-1]:.1f}km" if cone else "")
        result["case_studies"] = case_results

    # ── Sigma sensitivity (nếu được yêu cầu) ─────────────────────────────────
    if args.sigma_sensitivity:
        print(f"\n  Running sigma_inference sensitivity analysis...")
        sigma_results = sigma_sensitivity(model, loader, device,
                                          sigma_values=[0.01, 0.02, 0.04, 0.06, 0.08],
                                          n_ensemble=args.n_ensemble)
        result["sigma_sensitivity"] = sigma_results

    # ── Ensemble size ablation (nếu được yêu cầu) ─────────────────────────
    if args.ensemble_ablation:
        print(f"\n  Running ensemble size ablation K={args.k_values}...")
        ens_results = ensemble_size_eval(model, loader, device,
                                          k_values=args.k_values)
        result["ensemble_ablation"] = ens_results

    # ── Per-storm breakdown (nếu được yêu cầu) ─────────────────────────────
    if args.per_storm:
        print(f"\n  Running per-storm breakdown (by real storm name)...")
        ps = per_storm_breakdown(model, loader, device,
                                  n_ensemble=args.n_ensemble,
                                  use_curvature_score=args.use_curvature_score)
        print_per_storm_breakdown(ps, sort_by="cte")
        # Saved SEPARATELY (not nested in the main eval JSON) so
        # compare_per_storm_across_checkpoints() can load several of these
        # by path and diff them across seeds/checkpoints directly.
        # [MỚI] Filename giờ bao gồm model_type để 5 kiến trúc x 3 seed
        # (15 lần chạy) không ghi đè lẫn nhau trong CÙNG 1 --output_dir.
        ps_summary = {name: {"n": rec["n"],
                              "ade": float(np.mean(rec["ade"])),
                              "ate": float(np.mean(rec["ate"])),
                              "cte": float(np.mean(rec["cte"])),
                              "obs_speed": float(np.mean(rec["obs_speed"]))}
                      for name, rec in ps.items() if rec["n"] > 0}
        ps_path = os.path.join(
            args.output_dir,
            f"per_storm_{resolved_type}_{args.split}_ep{ep}.json")
        with open(ps_path, "w") as f:
            json.dump(ps_summary, f, indent=2)
        print(f"  Saved per-storm breakdown → {ps_path}")

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir,
                             f"eval_{resolved_type}_{args.split}_ep{ep}.json")
    save_result = {k: v for k, v in result.items()
                   if k != "boxplot_ade"}   # exclude large arrays from main JSON
    save_result["boxplot_ade"] = result.get("boxplot_ade", [])  # keep for boxplot
    save_result["checkpoint"]  = args.checkpoint
    save_result["split"]       = args.split

    with open(out_path, "w") as f:
        json.dump(save_result, f, indent=2, default=str)
    print(f"  Saved → {out_path}")

    # Summary line for paper table
    print(f"\n  ── PAPER TABLE ROW ──")
    print(f"  {resolved_type.upper()} | "
          f"ADE={result['ADE']:.1f}±{result.get('ADE_std',0):.1f} | "
          f"ATE={result['ATE']:.1f}±{result.get('ATE_std',0):.1f} | "
          f"CTE={result['CTE']:.1f}±{result.get('CTE_std',0):.1f} | "
          f"CRPS={result.get('crps',{}).get('mean',float('nan')):.1f}")


if __name__ == "__main__":
    main()