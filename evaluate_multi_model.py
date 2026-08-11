"""
evaluate_multi_model.py
=========================
Runs test-set evaluation for RNN / GRU / LSTM / ST-Trans / FM checkpoints,
using ONE SHARED ATE/CTE formula for all five models.

WHY A SEPARATE SCRIPT, NOT REUSING EACH MODEL'S OWN METRIC CODE
-----------------------------------------------------------------
paper_baseline_model.py (used by RNN/GRU/LSTM/ST-Trans) has its own
_ate_cte_tensors() with a DIFFERENT convention than flow_matching_model.py's
_ate_cte_full() (used by FM):
  - flat-earth approx (111*cos(lat)) vs haversine — small effect (~0.2%)
  - reference heading computed OUTGOING from each point (gt[t]->gt[t+1])
    vs INCOMING to each point (gt[t-1]->gt[t]) — this is NOT a small
    effect: verified numerically on a synthetic turning trajectory, the
    two conventions can even disagree on the SIGN of CTE at a given step.
Comparing FM's ATE/CTE (computed one way) against RNN/GRU/LSTM/ST-Trans's
ATE/CTE (computed the other way) would not be a fair like-for-like
comparison — any difference could be an artifact of the formula, not the
model. This script computes ADE/ATE/CTE for ALL FIVE models using the
SAME function (_ate_cte_full, _haversine_deg, _forward_azimuth, imported
from flow_matching_model.py — the version with the off-by-one fix already
verified in evaluate_full.py), so Table-10-style comparisons are sound.

USAGE
-----
python evaluate_multi_model.py \
    --dataset_root <root> \
    --fm_checkpoint runs/fm_seed42/best_model.pth \
    --st_trans_checkpoint runs/st_trans/best_model.pth \
    --lstm_checkpoint runs/lstm/best_model.pth \
    --gru_checkpoint runs/gru/best_model.pth \
    --rnn_checkpoint runs/rnn/best_model.pth \
    --output_dir eval_multi/

Any checkpoint arg can be omitted to skip that model. Produces one
combined JSON of per-window records (one row per (model, storm, window))
that generate_comparison_table.py consumes directly for the Table-10-style
statistical significance tables.
"""
from __future__ import annotations
import sys, os, argparse, json
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _haversine_deg, _forward_azimuth, _unwrap,
)
from Model.paper_baseline_model import PaperBaseline, MODEL_TYPES
from Model.st_trans_model import STTrans


def _infer_seed(checkpoint_path: str, ck: dict) -> str:
    """
    Best-effort seed extraction: prefer an explicit 'seed' key saved in
    the checkpoint (train_flowmatching.py / train_st_trans.py /
    train_paper_baseline.py all save this per the project's multi-seed
    convention). Falls back to parsing "_seed<N>" from the path/dirname
    (the convention those same scripts use for output_dir), and finally
    "unknown" if neither is found — records with seed="unknown" will
    NOT be treated as genuinely pooled multi-seed data by
    generate_paper_table.py (it warns explicitly in that case).
    """
    if isinstance(ck, dict) and "seed" in ck:
        return str(ck["seed"])
    import re
    m = re.search(r"seed[_-]?(\d+)", checkpoint_path)
    if m:
        return m.group(1)
    return "unknown"


def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


def ate_cte_full(pred_deg: torch.Tensor, gt_deg: torch.Tensor):
    """
    SAME formula as evaluate_full.py's _ate_cte_full (off-by-one already
    fixed there — see that file's own comments for the fix history).
    Returns [T-1, B] each; index k holds the error at ORIGINAL step k+1.
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    bear_ref = _forward_azimuth(gt_deg[:T - 1], gt_deg[1:T])
    bear_err = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
    dist_err = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang = bear_err - bear_ref
    return dist_err * torch.cos(ang), dist_err * torch.sin(ang)


def load_fm(checkpoint: str, device):
    ck = torch.load(checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg") or {}
    if not model_cfg:
        print(f"  ⚠ FM checkpoint has no model_cfg — using constructor "
              f"defaults (correct only if trained with default architecture).")
    model = TCFlowMatching(**model_cfg).to(device)
    state = ck.get("model", ck.get("model_state"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ FM load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
    model.eval()
    seed = _infer_seed(checkpoint, ck)
    return model, seed


def load_paper_baseline(checkpoint: str, model_type: str, device,
                         hidden_dim: int = 256, n_layers: int = 3,
                         obs_len: int = 8, pred_len: int = 12,
                         unet_in_ch: int = 13, dropout: float = 0.20):
    """
    [UPDATED] train_paper_baseline.py now saves a full model_cfg dict in
    newer checkpoints (added alongside --seed support). If present, it is
    used directly and the CLI-default args above are ignored — this is
    the reliable path. If absent (checkpoint trained before that fix),
    falls back to the CLI-default-matching values passed in here, with a
    warning, same caveat as FM's missing-model_cfg case.
    """
    assert model_type in MODEL_TYPES, f"model_type must be one of {MODEL_TYPES}"
    ck = torch.load(checkpoint, map_location="cpu")
    saved_type = ck.get("model_type", model_type)
    if saved_type != model_type:
        print(f"  ⚠ Checkpoint's saved model_type='{saved_type}' differs "
              f"from requested '{model_type}' — using requested value. "
              f"Verify this checkpoint is really the {model_type.upper()} one.")

    model_cfg = ck.get("model_cfg")
    if model_cfg:
        model = PaperBaseline(**model_cfg).to(device)
    else:
        print(f"  ⚠ {model_type.upper()} checkpoint has no model_cfg — "
              f"using CLI-default-matching args (hidden_dim={hidden_dim}, "
              f"n_layers={n_layers}, dropout={dropout}). Only correct if "
              f"trained with train_paper_baseline.py's own defaults.")
        model = PaperBaseline(model_type=model_type, pred_len=pred_len,
                               obs_len=obs_len, hidden_dim=hidden_dim,
                               n_layers=n_layers, unet_in_ch=unet_in_ch,
                               dropout=dropout).to(device)
    state = ck.get("model_state", ck.get("model"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ {model_type.upper()} load_state_dict: {len(missing)} "
              f"missing, {len(unexpected)} unexpected keys")
    model.eval()
    seed = _infer_seed(checkpoint, ck)
    return model, seed


def load_st_trans(checkpoint: str, device,
                   obs_len: int = 8, pred_len: int = 12, unet_in_ch: int = 13,
                   d_model: int = 64, nhead: int = 4, num_enc_layers: int = 1,
                   num_dec_layers: int = 3, dim_ff: int = 512, dropout: float = 0.1):
    """
    [UPDATED] Same model_cfg pattern as load_paper_baseline. Note STTrans
    (non_ar) and STTransAR have DIFFERENT constructor signatures (AR has
    no num_dec_layers) — the saved model_cfg already accounts for this
    (see train_st_trans.py's checkpoint-save branch), so **kwargs here
    naturally works for either. This loader only instantiates STTrans
    (non-AR) — STTransAR support would need its own loader if you also
    want to evaluate that variant.
    """
    ck = torch.load(checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg")
    if model_cfg:
        model = STTrans(**model_cfg).to(device)
    else:
        print(f"  ⚠ ST-Trans checkpoint has no model_cfg — using "
              f"CLI-default-matching args (d_model={d_model}, nhead={nhead}, "
              f"num_dec_layers={num_dec_layers}). Only correct if trained "
              f"with train_st_trans.py's own defaults.")
        model = STTrans(obs_len=obs_len, pred_len=pred_len, unet_in_ch=unet_in_ch,
                         d_model=d_model, nhead=nhead, num_enc_layers=num_enc_layers,
                         num_dec_layers=num_dec_layers, dim_ff=dim_ff,
                         dropout=dropout).to(device)
    state = ck.get("model_state", ck.get("model"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ ST-Trans load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
    model.eval()
    seed = _infer_seed(checkpoint, ck)
    return model, seed


@torch.no_grad()
def evaluate_one_model(model, loader, device, model_name: str,
                        seed: str = "unknown",
                        n_ensemble: int = 20) -> List[Dict]:
    """
    Returns a list of PER-LEAD-TIME records:
      {"model": name, "seed": seed, "storm": storm_key, "window": idx,
       "lead_time": t, "ade": .., "ate": .., "cte": .., "obs_speed": ..}
    One record per (storm, window, lead_time) triple — matches the
    paper's Table 10 pairing granularity (140 windows x 16 lead-times =
    2240 matched pairs, i.e. paired PER FORECAST STEP, not averaged over
    the whole trajectory first).

    lead_time convention (1-indexed, 1..T; T=pred_len, e.g. 1=6h...12=72h
    when T=12): this is the SAME convention as generate_paper_report.py's
    HORIZON_LEAD_TIMES = {"6h":1,...,"72h":12}. ADE (d) has a value for
    EVERY lead_time 1..T. ATE/CTE do not: there is no heading reference
    at the very first predicted step, so ate/cte are only defined for
    lead_time 2..T (None at lead_time=1/6h). [FIX] An earlier version of
    this loop bounded lead_time by ate/cte's shorter range (T-1 instead
    of T), which silently dropped the LAST lead_time (T, i.e. 72h when
    T=12) from ADE too, and additionally mislabeled lead_time=1 as if it
    were the first step (6h) when it was actually the second step (12h,
    0-indexed step 1) — both bugs are fixed by this version: ADE now
    covers the full 1..T range, and lead_time=1 genuinely is the first
    predicted step.
    """
    records = []
    is_fm = isinstance(model, TCFlowMatching) or hasattr(model, "sigma_inference")

    for bi, batch in enumerate(loader):
        bl = move(list(batch), device)
        gt = bl[1]
        obs = bl[0]
        try:
            tyid_list = bl[15]
        except IndexError:
            tyid_list = None

        try:
            if is_fm:
                pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
            else:
                pred, _, _ = model.sample(bl, num_ensemble=1)
        except Exception as e:
            print(f"  [{model_name}] batch {bi}: sample error: {e}")
            continue

        T = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T])
        gd = _norm_to_deg(gt[:T, :, :2])
        d  = _haversine_deg(pd, gd)                  # [T, B] -- steps 0..T-1 (0=6h ... T-1=72h when T=12)
        ate, cte = ate_cte_full(pd, gd)               # [T-1, B] -- ate[k] = error at step k+1 (0-indexed)
        T_valid = ate.shape[0]                        # = T-1

        obs_deg = _norm_to_deg(obs[:, :, :2])
        if obs_deg.shape[0] >= 2:
            step_km = _haversine_deg(obs_deg[:-1], obs_deg[1:])
            obs_speed = step_km.mean(0) / 6.0
        else:
            obs_speed = torch.zeros(obs.shape[1], device=device)

        B = obs.shape[1]
        for b in range(B):
            if tyid_list is not None and b < len(tyid_list) and \
               isinstance(tyid_list[b], dict) and "old" in tyid_list[b]:
                info = tyid_list[b]
                storm_key = f"{info['old'][1]}_{info['old'][0]}"
            else:
                storm_key = f"UNKNOWN_batch{bi}"
            # [FIX] Bug thật đã tìm ra: trước đây vòng lặp chạy
            # `for i in range(T_valid)` (T_valid = T-1), với
            # lead_time = i+1 (i=0..T_valid-1 => lead_time=1..T_valid=1..T-1).
            # Với T=12, lead_time chỉ chạy 1..11 -- KHÔNG BAO GIỜ đạt 12.
            # generate_paper_report.py's HORIZON_LEAD_TIMES tra "72h"->12,
            # nên luôn ra n=0 ở 72h (khớp đúng hiện tượng đã quan sát).
            # Đồng thời "lead_time=1" trước đây thực chất ứng 0-indexed
            # step 1 (=12h theo evaluate_full.py's HORIZONS convention),
            # KHÔNG PHẢI 6h -- tên horizon "6h" ở nơi đọc dữ liệu cũng bị
            # lệch 1 bước so với dữ liệu thật.
            #
            # Sửa: lead_time giờ là 1-indexed THẬT trên toàn bộ T bước
            # (lead_time = step_0indexed + 1, chạy 1..T, tức 1=6h...T=72h
            # khi T=12) -- khớp đúng HORIZON_LEAD_TIMES = {"6h":1,...,
            # "72h":12} sau khi sửa ở generate_paper_report.py.
            # ADE (d) có đủ giá trị cho MỌI lead_time 1..T.
            # ATE/CTE (ate/cte) chỉ có giá trị cho lead_time 2..T (không
            # định nghĩa được ở lead_time=1/6h, vì cần bước trước đó để
            # biết hướng đi) -- ghi None thay vì bỏ hẳn record.
            for step0 in range(T):           # step0 = 0-indexed step, 0..T-1
                lead_time = step0 + 1        # 1-indexed, 1..T (1=6h...T=72h)
                has_atecte = step0 >= 1      # ate/cte defined for step0=1..T-1
                ate_i = step0 - 1            # ate/cte array index when has_atecte
                records.append({
                    "model":     model_name,
                    "seed":      seed,
                    "storm":     storm_key,
                    "window":    b,
                    "lead_time": lead_time,
                    "ade":       float(d[step0, b]),
                    "ate":       float(ate[ate_i, b].abs()) if has_atecte else None,
                    "cte":       float(cte[ate_i, b].abs()) if has_atecte else None,
                    "obs_speed": float(obs_speed[b]),
                })
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--output_dir", default="eval_multi")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_ensemble", type=int, default=20)
    p.add_argument("--test_year", type=int, default=None)

    p.add_argument("--fm_checkpoints",       nargs="+", default=None,
                   help="One or more FM checkpoint paths, one per seed")
    p.add_argument("--st_trans_checkpoints", nargs="+", default=None,
                   help="One or more ST-Trans checkpoint paths, one per seed")
    p.add_argument("--lstm_checkpoints",     nargs="+", default=None,
                   help="One or more LSTM checkpoint paths, one per seed")
    p.add_argument("--gru_checkpoints",      nargs="+", default=None,
                   help="One or more GRU checkpoint paths, one per seed")
    p.add_argument("--rnn_checkpoints",      nargs="+", default=None,
                   help="One or more RNN checkpoint paths, one per seed")
    # Backward-compat singular aliases (old single-checkpoint usage still works)
    p.add_argument("--fm_checkpoint",       default=None, help="[legacy] single checkpoint, use --fm_checkpoints instead")
    p.add_argument("--st_trans_checkpoint", default=None, help="[legacy] single checkpoint")
    p.add_argument("--lstm_checkpoint",     default=None, help="[legacy] single checkpoint")
    p.add_argument("--gru_checkpoint",      default=None, help="[legacy] single checkpoint")
    p.add_argument("--rnn_checkpoint",      default=None, help="[legacy] single checkpoint")

    p.add_argument("--paper_hidden_dim", type=int, default=256)
    p.add_argument("--paper_n_layers",   type=int, default=3)
    p.add_argument("--paper_dropout",    type=float, default=0.20)
    p.add_argument("--st_d_model",        type=int, default=64)
    p.add_argument("--st_nhead",          type=int, default=4)
    p.add_argument("--st_num_enc_layers", type=int, default=1)
    p.add_argument("--st_num_dec_layers", type=int, default=3)
    p.add_argument("--st_dim_ff",         type=int, default=512)
    p.add_argument("--st_dropout",        type=float, default=0.1)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"  Loading test data...")
    import argparse as _ap
    _loader_args = _ap.Namespace(
        dataset_root = args.dataset_root,
        obs_len      = 8,
        pred_len     = 12,
        batch_size   = 64,
        num_workers  = 2,
        test_year    = args.test_year,
        skip         = 1,
        min_ped      = 1,
        threshold    = 0.002,
    )
    _, loader = data_loader(_loader_args,
                             {"root": args.dataset_root, "type": args.split},
                             test=(args.split != "train"))
    print(f"  Data: {len(loader)} batches")

    def _collect(multi, single):
        """Merge --xxx_checkpoints (list) and legacy --xxx_checkpoint (str) into one list."""
        paths = list(multi) if multi else []
        if single and single not in paths:
            paths.append(single)
        return paths

    jobs = []  # (display_name, kind, checkpoint_path)
    for display_name, kind, multi, single in [
        ("FM",       "fm",       args.fm_checkpoints,       args.fm_checkpoint),
        ("ST-Trans", "st_trans", args.st_trans_checkpoints, args.st_trans_checkpoint),
        ("LSTM",     "lstm",     args.lstm_checkpoints,     args.lstm_checkpoint),
        ("GRU",      "gru",      args.gru_checkpoints,      args.gru_checkpoint),
        ("RNN",      "rnn",      args.rnn_checkpoints,      args.rnn_checkpoint),
    ]:
        for ckpt_path in _collect(multi, single):
            jobs.append((display_name, kind, ckpt_path))

    if not jobs:
        print("  No checkpoints given — nothing to do.")
        return

    all_records = []
    for display_name, kind, ckpt_path in jobs:
        print(f"\n  {'='*70}\n  Loading {display_name}: {ckpt_path}\n  {'='*70}")
        if kind == "fm":
            model, seed = load_fm(ckpt_path, device)
        elif kind == "st_trans":
            model, seed = load_st_trans(ckpt_path, device,
                                   d_model=args.st_d_model, nhead=args.st_nhead,
                                   num_enc_layers=args.st_num_enc_layers,
                                   num_dec_layers=args.st_num_dec_layers,
                                   dim_ff=args.st_dim_ff, dropout=args.st_dropout)
        else:
            model, seed = load_paper_baseline(ckpt_path, kind, device,
                                         hidden_dim=args.paper_hidden_dim,
                                         n_layers=args.paper_n_layers,
                                         dropout=args.paper_dropout)

        n_params = sum(pm.numel() for pm in model.parameters())
        print(f"  {display_name} (seed={seed}): {n_params:,} params")

        recs = evaluate_one_model(model, loader, device, display_name,
                                   seed=seed, n_ensemble=args.n_ensemble)
        all_records.extend(recs)

        # [FIX] ate/cte là None ở lead_time=1 (6h) theo convention đã sửa
        # (xem evaluate_one_model's docstring) — np.mean crash nếu None
        # lẫn trong list. Lọc trước khi tính, giống mọi chỗ khác trong
        # generate_paper_report.py đã áp dụng cùng bộ lọc này.
        ade = np.mean([r["ade"] for r in recs if r["ade"] is not None])
        ate_vals = [r["ate"] for r in recs if r["ate"] is not None]
        cte_vals = [r["cte"] for r in recs if r["cte"] is not None]
        ate = np.mean(ate_vals) if ate_vals else float("nan")
        cte = np.mean(cte_vals) if cte_vals else float("nan")
        print(f"  {display_name} seed={seed}: n={len(recs)}  ADE={ade:.2f}  "
              f"ATE={ate:.2f}  CTE={cte:.2f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = os.path.join(args.output_dir, f"multi_model_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n  Saved {len(all_records)} records → {out_path}")
    print(f"  Run generate_comparison_table.py --records {out_path} "
          f"to produce the Table-10-style significance table.")


if __name__ == "__main__":
    main()