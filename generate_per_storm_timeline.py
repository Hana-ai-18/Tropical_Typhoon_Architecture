"""
generate_per_storm_timeline.py
================================
For EVERY storm in the test set, and for EACH of the 5 architectures
(FM, ST-Trans, LSTM, GRU, RNN), prints a per-timestep table:

    Storm: RITA_1975
      Model: FM
        1975-08-22 12:00   ADE=12.3km  ATE=10.1km  CTE=7.0km
        1975-08-22 18:00   ADE=25.6km  ATE=22.0km  CTE=13.2km
        ...
      Model: ST-Trans
        ...

Timestamps are read from the dataset's own tydate field (real
YYYYMMDDHH per timestep), NOT synthetic "+6h, +12h..." labels — this
was the whole point of using the real per-storm data instead of an
aggregate table.

Also writes a paper-ready CSV (one row per storm x model x lead_time)
and a compact markdown summary (one row per storm x model, with ADE
broken out every 6h in separate columns) for direct inclusion in a
paper appendix.

WHY A NEW SCRIPT RATHER THAN EXTENDING per_storm_breakdown()
(evaluate_full.py) OR evaluate_multi_model.py:
  - per_storm_breakdown() only returns ONE aggregate number per storm
    (averaged over all lead-times) — this script needs one row PER
    TIMESTEP, a finer granularity it was never designed for.
  - evaluate_multi_model.py already computes per-lead-time records
    (storm, window, lead_time) for the 5-architecture statistical
    comparison, but does not carry the real tydate timestamp or print
    a human-readable per-storm table — this script reuses its loader
    functions (load_fm/load_st_trans/load_paper_baseline, ate_cte_full)
    to avoid re-deriving that logic, and ADDS the real-timestamp
    lookup + the paper-table formatting on top.

USAGE
-----
python generate_per_storm_timeline.py \
    --dataset_root <root> --split test \
    --fm_checkpoint <ckpt> \
    --st_trans_checkpoint <ckpt> \
    --lstm_checkpoint <ckpt> \
    --gru_checkpoint <ckpt> \
    --rnn_checkpoint <ckpt> \
    --output_dir per_storm_timeline/

Any checkpoint arg can be omitted to skip that architecture.
"""
from __future__ import annotations
import sys, os, argparse, json, csv
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _haversine_deg, _forward_azimuth,
)
from Model.paper_baseline_model import PaperBaseline, MODEL_TYPES
from Model.st_trans_model import STTrans


def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


def ate_cte_full(pred_deg: torch.Tensor, gt_deg: torch.Tensor):
    """
    Same convention established throughout this codebase's evaluation
    scripts (evaluate_full.py / evaluate_multi_model.py): returns
    [T-1, B] each; index i holds the error at ORIGINAL step index i+1
    (no ATE/CTE is defined at the very first predicted step — no prior
    heading reference exists there).
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


def format_timestamp(yyyymmddhh: str) -> str:
    """'1975082212' -> '1975-08-22 12:00'. Falls back to raw string if
    the value doesn't parse (e.g. placeholder/malformed dates in the
    source file) — never crashes the whole run over one bad timestamp."""
    s = str(yyyymmddhh).strip()
    try:
        dt = datetime.strptime(s[:10], "%Y%m%d%H")
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return s


def load_fm(checkpoint: str, device):
    ck = torch.load(checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg") or {}
    if not model_cfg:
        print(f"  ⚠ FM checkpoint has no model_cfg — using constructor "
              f"defaults (correct only if trained with default architecture).")
    model = TCFlowMatching(**model_cfg).to(device)
    state = ck.get("model", ck.get("model_state"))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ck.get("seed")


def load_st_trans(checkpoint: str, device):
    ck = torch.load(checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg") or {}
    if model_cfg:
        model = STTrans(**model_cfg).to(device)
    else:
        print(f"  ⚠ ST-Trans checkpoint has no model_cfg — using constructor defaults.")
        model = STTrans().to(device)
    state = ck.get("model_state", ck.get("model"))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ck.get("seed")


def load_paper_baseline(checkpoint: str, model_type: str, device):
    assert model_type in MODEL_TYPES, f"model_type must be one of {MODEL_TYPES}"
    ck = torch.load(checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg") or {}
    if model_cfg:
        model = PaperBaseline(**model_cfg).to(device)
    else:
        print(f"  ⚠ {model_type.upper()} checkpoint has no model_cfg — using constructor defaults.")
        model = PaperBaseline(model_type=model_type).to(device)
    state = ck.get("model_state", ck.get("model"))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ck.get("seed")


@torch.no_grad()
def collect_timeline(model, loader, device, model_name: str,
                      obs_len: int, n_ensemble: int = 20,
                      seed: Optional[int] = None) -> List[Dict]:
    """
    Returns one record per (storm, window, lead_time), each carrying the
    REAL timestamp (from the dataset's tydate field) alongside
    ADE/ATE/CTE at that exact timestep — the raw material for the
    per-storm timeline table.

    [MULTI-SEED] Now also tags each record with the seed it came from,
    so records from multiple checkpoints of the SAME architecture (e.g.
    FM seed 0/1/2) can be grouped by (storm, model, lead_time) to compute
    a mean±std across seeds at EACH TIMESTEP — not just an aggregate
    mean±std over the whole trajectory. See aggregate_across_seeds().
    """
    records = []
    is_fm = isinstance(model, TCFlowMatching) or hasattr(model, "sigma_inference")

    for bi, batch in enumerate(loader):
        bl = move(list(batch), device)
        gt = bl[1]
        try:
            tyid_list = bl[15]
        except IndexError:
            tyid_list = None
            print(f"  ⚠ batch {bi}: no tyID field (bl[15]) — cannot recover "
                  f"real timestamps or storm names for this batch, skipping.")
            continue
        if tyid_list is None:
            continue

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
        d  = _haversine_deg(pd, gd)          # [T, B]
        ate, cte = ate_cte_full(pd, gd)       # [T-1, B]
        T_valid = ate.shape[0]

        B = gt.shape[1]
        for b in range(B):
            info = tyid_list[b] if b < len(tyid_list) else None
            if not isinstance(info, dict) or "old" not in info:
                storm_key, tydates = f"UNKNOWN_batch{bi}_{b}", None
            else:
                year, name = info["old"][0], info["old"][1]
                storm_key = f"{name}_{year}"
                tydates = info.get("tydate")

            for t in range(T):
                if tydates is not None and (obs_len + t) < len(tydates):
                    real_ts = tydates[obs_len + t]
                else:
                    real_ts = None  # dataset variant without tydate — fall back below

                rec = {
                    "model":     model_name,
                    "seed":      seed,
                    "storm":     storm_key,
                    "window":    b,
                    "lead_time_idx": t,             # 0-indexed step within forecast
                    "lead_time_h":   (t + 1) * 6,    # hours from last observation
                    "timestamp": format_timestamp(real_ts) if real_ts is not None
                                 else f"+{(t+1)*6}h (no tydate in dataset)",
                    "ade": float(d[t, b]),
                }
                # ATE/CTE undefined at t=0 (no prior heading reference) —
                # see ate_cte_full's docstring. Leave as None, not 0.0, so
                # downstream tables/CSV don't silently imply a perfect
                # cross-track score at the first step.
                if t >= 1 and (t - 1) < T_valid:
                    rec["ate"] = float(ate[t - 1, b].abs())
                    rec["cte"] = float(cte[t - 1, b].abs())
                else:
                    rec["ate"] = None
                    rec["cte"] = None
                records.append(rec)

    return records


def aggregate_across_seeds(records: List[Dict]) -> List[Dict]:
    """
    [MULTI-SEED] Groups records by (storm, window, model, lead_time_idx)
    — i.e. across different seeds of the SAME architecture — and reduces
    each group to mean±std for ADE/ATE/CTE at that exact timestep.

    CORRECTNESS CAVEAT: this assumes "window=b" (the batch-index position
    a storm's sequence lands in) is IDENTICAL across seed runs for the
    same architecture — true as long as the test DataLoader is built with
    shuffle=False and the same batch_size/dataset ordering every run
    (the standard, expected setup for a held-out test split; training
    loaders that shuffle would NOT satisfy this and should not be fed
    into this function). If window indices drift between seeds (e.g. a
    dataset re-filtering that changes sequence count), records with no
    matching (storm, window) across ALL seeds are simply left as
    single-seed entries (n=1, std=0) rather than silently dropped or
    mismatched — check the returned "n_seeds" field per row before
    trusting a std of exactly 0.0 as "seeds agreed perfectly" vs. "only
    one seed had data here".

    Returns records in the SAME shape as collect_timeline's output, but
    with "ade"/"ate"/"cte" replaced by "{metric}_mean"/"{metric}_std",
    plus "n_seeds" (how many seeds contributed to this row) and "seeds"
    (the actual seed values, for traceability).
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        key = (r["storm"], r["window"], r["model"], r["lead_time_idx"])
        groups[key].append(r)

    out = []
    for (storm, window, model, lt_idx), group in groups.items():
        ade_vals = [g["ade"] for g in group]
        ate_vals = [g["ate"] for g in group if g["ate"] is not None]
        cte_vals = [g["cte"] for g in group if g["cte"] is not None]
        seeds = [g.get("seed") for g in group]

        row = {
            "storm": storm, "window": window, "model": model,
            "lead_time_idx": lt_idx, "lead_time_h": group[0]["lead_time_h"],
            # Timestamp SHOULD be identical across seeds for the same
            # (storm, window, lead_time) — they all read the same
            # underlying dataset file. Take the first non-placeholder one.
            "timestamp": group[0]["timestamp"],
            "n_seeds": len(group),
            "seeds": seeds,
            "ade_mean": float(np.mean(ade_vals)),
            "ade_std":  float(np.std(ade_vals, ddof=0)) if len(ade_vals) > 1 else 0.0,
        }
        if ate_vals:
            row["ate_mean"] = float(np.mean(ate_vals))
            row["ate_std"]  = float(np.std(ate_vals, ddof=0)) if len(ate_vals) > 1 else 0.0
        else:
            row["ate_mean"] = None
            row["ate_std"]  = None
        if cte_vals:
            row["cte_mean"] = float(np.mean(cte_vals))
            row["cte_std"]  = float(np.std(cte_vals, ddof=0)) if len(cte_vals) > 1 else 0.0
        else:
            row["cte_mean"] = None
            row["cte_std"]  = None
        out.append(row)

    return out



    """
    Human-readable console output: one block per storm, one sub-block
    per model, one line per timestep — exactly the layout requested.
    """
    storms = sorted(set(r["storm"] for r in records))
    models_seen = sorted(set(r["model"] for r in records))

    for storm in storms:
        print(f"\n{'='*78}")
        print(f"Storm: {storm}")
        print(f"{'='*78}")
        storm_recs = [r for r in records if r["storm"] == storm]
        windows = sorted(set(r["window"] for r in storm_recs))
        for w in windows:
            w_recs = [r for r in storm_recs if r["window"] == w]
            if len(windows) > 1:
                print(f"\n  -- window {w} --")
            for model in models_seen:
                m_recs = sorted([r for r in w_recs if r["model"] == model],
                                 key=lambda r: r["lead_time_idx"])
                if not m_recs:
                    continue
                print(f"\n  Model: {model}")
                for r in m_recs:
                    ate_s = f"{r['ate']:.1f}km" if r["ate"] is not None else "  n/a "
                    cte_s = f"{r['cte']:.1f}km" if r["cte"] is not None else "  n/a "
                    print(f"    {r['timestamp']:<20} (+{r['lead_time_h']:>3}h)  "
                          f"ADE={r['ade']:7.1f}km  ATE={ate_s:>9}  CTE={cte_s:>9}")


def print_storm_tables(records: List[Dict]):
    """
    [RESTORED] Human-readable console output for the SINGLE-seed case:
    one block per storm, one sub-block per model, one line per timestep
    — exactly the layout requested. This function was accidentally
    dropped when print_storm_tables_agg() (the multi-seed mean±std
    variant) was added in a later edit — main() still calls this name
    for the is_multi_seed=False branch, so its absence was a real bug
    (NameError at runtime), not just a Pylance false positive.
    """
    storms = sorted(set(r["storm"] for r in records))
    models_seen = sorted(set(r["model"] for r in records))

    for storm in storms:
        print(f"\n{'='*78}")
        print(f"Storm: {storm}")
        print(f"{'='*78}")
        storm_recs = [r for r in records if r["storm"] == storm]
        windows = sorted(set(r["window"] for r in storm_recs))
        for w in windows:
            w_recs = [r for r in storm_recs if r["window"] == w]
            if len(windows) > 1:
                print(f"\n  -- window {w} --")
            for model in models_seen:
                m_recs = sorted([r for r in w_recs if r["model"] == model],
                                 key=lambda r: r["lead_time_idx"])
                if not m_recs:
                    continue
                print(f"\n  Model: {model}")
                for r in m_recs:
                    ate_s = f"{r['ate']:.1f}km" if r["ate"] is not None else "  n/a "
                    cte_s = f"{r['cte']:.1f}km" if r["cte"] is not None else "  n/a "
                    print(f"    {r['timestamp']:<20} (+{r['lead_time_h']:>3}h)  "
                          f"ADE={r['ade']:7.1f}km  ATE={ate_s:>9}  CTE={cte_s:>9}")


def print_storm_tables_agg(agg_rows: List[Dict]):
    """
    [MULTI-SEED] Same layout as print_storm_tables(), but for
    aggregate_across_seeds()'s output — each line shows mean±std across
    seeds instead of a single value, plus n_seeds so a reader can spot
    rows backed by fewer seeds than expected (e.g. one checkpoint failed
    to sample for that batch).
    """
    storms = sorted(set(r["storm"] for r in agg_rows))
    models_seen = sorted(set(r["model"] for r in agg_rows))

    for storm in storms:
        print(f"\n{'='*88}")
        print(f"Storm: {storm}")
        print(f"{'='*88}")
        storm_recs = [r for r in agg_rows if r["storm"] == storm]
        windows = sorted(set(r["window"] for r in storm_recs))
        for w in windows:
            w_recs = [r for r in storm_recs if r["window"] == w]
            if len(windows) > 1:
                print(f"\n  -- window {w} --")
            for model in models_seen:
                m_recs = sorted([r for r in w_recs if r["model"] == model],
                                 key=lambda r: r["lead_time_idx"])
                if not m_recs:
                    continue
                n_seeds_here = set(r["n_seeds"] for r in m_recs)
                warn = "" if n_seeds_here == {m_recs[0]["n_seeds"]} and \
                             m_recs[0]["n_seeds"] > 1 else \
                       f"  ⚠ n_seeds varies or =1: {sorted(n_seeds_here)}"
                print(f"\n  Model: {model}{warn}")
                for r in m_recs:
                    ate_s = (f"{r['ate_mean']:6.1f}±{r['ate_std']:4.1f}km"
                             if r["ate_mean"] is not None else "      n/a       ")
                    cte_s = (f"{r['cte_mean']:6.1f}±{r['cte_std']:4.1f}km"
                             if r["cte_mean"] is not None else "      n/a       ")
                    print(f"    {r['timestamp']:<20} (+{r['lead_time_h']:>3}h, "
                          f"n={r['n_seeds']})  "
                          f"ADE={r['ade_mean']:6.1f}±{r['ade_std']:4.1f}km  "
                          f"ATE={ate_s}  CTE={cte_s}")


def write_csv(records: List[Dict], path: str):
    """
    One row per (storm, model, window, lead_time) — the finest-grain,
    paper-appendix-ready table. Sort so a reader scanning the file sees
    each storm's models grouped together, each model's timesteps in
    chronological order — matches print_storm_tables' layout.
    """
    fieldnames = ["storm", "window", "model", "lead_time_h", "timestamp",
                  "ade_km", "ate_km", "cte_km"]
    rows = sorted(records, key=lambda r: (r["storm"], r["window"], r["model"], r["lead_time_idx"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "storm": r["storm"], "window": r["window"], "model": r["model"],
                "lead_time_h": r["lead_time_h"], "timestamp": r["timestamp"],
                "ade_km": f"{r['ade']:.2f}",
                "ate_km": f"{r['ate']:.2f}" if r["ate"] is not None else "",
                "cte_km": f"{r['cte']:.2f}" if r["cte"] is not None else "",
            })
    print(f"  Saved per-timestep CSV → {path}  ({len(rows)} rows)")


STANDARD_HORIZONS_H = [12, 24, 48, 72]  # matches paper's own reporting convention
                                          # (Table 7/8 style) — not every 6h step


def write_markdown_summary(records: List[Dict], path: str,
                            horizons_h: List[int] = None):
    """
    [REDESIGNED, compact] ONE flat table per metric (ADE/ATE/CTE), NOT one
    table per storm — a reader scans rows top-to-bottom to compare storms
    and models directly, instead of scrolling through dozens of small
    per-storm tables. Columns are the STANDARD reporting horizons
    (12h/24h/48h/72h, matching the paper's own Table 7/8 convention),
    not every single 6h step — the full per-6h detail is still in the
    CSV (write_csv) for anyone who needs it, but a paper table showing
    12 columns per metric is not readable; 4 is.

    horizons_h defaults to STANDARD_HORIZONS_H (12/24/48/72h). Pass a
    different list only if your data genuinely only has other lead
    times available (e.g. a shorter forecast horizon).
    """
    if horizons_h is None:
        horizons_h = STANDARD_HORIZONS_H

    storms = sorted(set(r["storm"] for r in records))
    models_seen = sorted(set(r["model"] for r in records))

    lines = ["# Per-storm error summary — all architectures\n",
             f"Standard horizons: {', '.join(f'{h}h' for h in horizons_h)}. "
             f"Full per-6h detail available in the companion CSV.\n"]

    for metric, label in [("ade", "ADE"), ("ate", "ATE"), ("cte", "CTE")]:
        lines.append(f"\n## {label} (km)\n")
        header = "| Storm | Model | " + " | ".join(f"{h}h" for h in horizons_h) + " |"
        sep    = "|---|---|" + "---|" * len(horizons_h)
        lines.append(header)
        lines.append(sep)
        for storm in storms:
            storm_recs = [r for r in records if r["storm"] == storm]
            # A storm can have multiple windows (multiple forecast issue
            # times) in the test set — average ADE/ATE/CTE across windows
            # at each horizon so ONE row represents the whole storm,
            # keeping the table storm-per-row rather than
            # storm*window-per-row (which would defeat the "compact"
            # goal for storms with many windows).
            for model in models_seen:
                m_recs = [r for r in storm_recs if r["model"] == model]
                if not m_recs:
                    continue
                cells = []
                for h in horizons_h:
                    vals = [r[metric] for r in m_recs
                            if r["lead_time_h"] == h and r[metric] is not None]
                    cells.append(f"{np.mean(vals):.1f}" if vals else "—")
                lines.append(f"| {storm} | {model} | " + " | ".join(cells) + " |")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved compact markdown summary → {path}")


def write_markdown_summary_merged(records: List[Dict], path: str,
                                   horizons_h: List[int] = None):
    """
    [MERGED] Single table combining ADE/ATE/CTE, instead of 3 separate
    tables (write_markdown_summary()'s layout) — one row per
    (storm, model), one column PER HORIZON, each cell showing all three
    metrics as "ADE/ATE/CTE" (km). This is the table requested to sit
    alongside the full per-timestep output: same row granularity
    (storm x model) as write_markdown_summary(), but all three metrics
    readable in one pass instead of scrolling between three tables.

    ATE/CTE are undefined at the very first predicted step in general
    (see ate_cte_full's docstring) — this cannot occur for the standard
    horizons here (12h/24h/48h/72h are never the first step for
    obs_len=8/pred_len=12), so "—/—" for ATE/CTE at these columns would
    indicate missing DATA (e.g. a sample_error during inference), not
    the expected first-step gap — worth investigating if seen.
    """
    if horizons_h is None:
        horizons_h = STANDARD_HORIZONS_H

    storms = sorted(set(r["storm"] for r in records))
    models_seen = sorted(set(r["model"] for r in records))

    lines = ["# Per-storm error summary (ADE/ATE/CTE, km) — all architectures\n",
             f"Each cell: ADE/ATE/CTE (km) at that horizon. "
             f"Standard horizons: {', '.join(f'{h}h' for h in horizons_h)}. "
             f"Full per-6h detail available in the companion CSV.\n"]
    header = "| Storm | Model | " + " | ".join(f"{h}h (ADE/ATE/CTE)" for h in horizons_h) + " |"
    sep    = "|---|---|" + "---|" * len(horizons_h)
    lines.append(header)
    lines.append(sep)

    for storm in storms:
        storm_recs = [r for r in records if r["storm"] == storm]
        for model in models_seen:
            m_recs = [r for r in storm_recs if r["model"] == model]
            if not m_recs:
                continue
            cells = []
            for h in horizons_h:
                h_recs = [r for r in m_recs if r["lead_time_h"] == h]
                ade_vals = [r["ade"] for r in h_recs if r["ade"] is not None]
                ate_vals = [r["ate"] for r in h_recs if r["ate"] is not None]
                cte_vals = [r["cte"] for r in h_recs if r["cte"] is not None]
                ade_s = f"{np.mean(ade_vals):.1f}" if ade_vals else "—"
                ate_s = f"{np.mean(ate_vals):.1f}" if ate_vals else "—"
                cte_s = f"{np.mean(cte_vals):.1f}" if cte_vals else "—"
                cells.append(f"{ade_s}/{ate_s}/{cte_s}")
            lines.append(f"| {storm} | {model} | " + " | ".join(cells) + " |")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved merged (ADE/ATE/CTE) markdown summary → {path}")


def write_csv_agg(agg_rows: List[Dict], path: str):
    """[MULTI-SEED] Same as write_csv() but for mean±std rows, with
    separate mean/std columns per metric plus n_seeds for traceability."""
    fieldnames = ["storm", "window", "model", "lead_time_h", "timestamp", "n_seeds", "seeds",
                  "ade_mean_km", "ade_std_km", "ate_mean_km", "ate_std_km",
                  "cte_mean_km", "cte_std_km"]
    rows = sorted(agg_rows, key=lambda r: (r["storm"], r["window"], r["model"], r["lead_time_idx"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "storm": r["storm"], "window": r["window"], "model": r["model"],
                "lead_time_h": r["lead_time_h"], "timestamp": r["timestamp"],
                "n_seeds": r["n_seeds"], "seeds": ";".join(str(s) for s in r["seeds"]),
                "ade_mean_km": f"{r['ade_mean']:.2f}", "ade_std_km": f"{r['ade_std']:.2f}",
                "ate_mean_km": f"{r['ate_mean']:.2f}" if r["ate_mean"] is not None else "",
                "ate_std_km":  f"{r['ate_std']:.2f}"  if r["ate_std"]  is not None else "",
                "cte_mean_km": f"{r['cte_mean']:.2f}" if r["cte_mean"] is not None else "",
                "cte_std_km":  f"{r['cte_std']:.2f}"  if r["cte_std"]  is not None else "",
            })
    print(f"  Saved per-timestep mean±std CSV → {path}  ({len(rows)} rows)")


def write_csv_trajectory(records: List[Dict], path: str):
    """
    [NEW] One row per (storm, model) — ADE/ATE/CTE averaged over the
    WHOLE trajectory (all lead times, not broken out per horizon like
    write_csv()/write_markdown_summary()), plus FINAL ADE (the ADE at
    the LAST lead time only, e.g. 72h for pred_len=12 — the endpoint
    error, a standard metric distinct from the trajectory-average ADE:
    see the paper's own "Final-Step DPE" convention). This is the
    per-storm summary table layout requested (storm as a row group,
    model as sub-rows, one mean±std-style value per metric) — separate
    from, and complementary to, the per-lead-time CSV (write_csv()),
    which stays available for the full per-6h detail.

    A storm can have multiple windows (multiple forecast issue times)
    in the test set — this function averages across windows too, so
    each (storm, model) pair collapses to exactly one row. If you need
    per-window trajectory averages instead, filter records by "window"
    before calling this function.
    """
    storms = sorted(set(r["storm"] for r in records))
    models_seen = sorted(set(r["model"] for r in records))
    max_lead_h = max(r["lead_time_h"] for r in records)

    fieldnames = ["storm", "model", "n_lead_times", "n_windows",
                  "ade_km", "ate_km", "cte_km", "final_ade_km"]
    rows = []
    for storm in storms:
        storm_recs = [r for r in records if r["storm"] == storm]
        n_windows = len(set(r["window"] for r in storm_recs))
        for model in models_seen:
            m_recs = [r for r in storm_recs if r["model"] == model]
            if not m_recs:
                continue
            ade_vals = [r["ade"] for r in m_recs if r["ade"] is not None]
            ate_vals = [r["ate"] for r in m_recs if r["ate"] is not None]
            cte_vals = [r["cte"] for r in m_recs if r["cte"] is not None]
            final_vals = [r["ade"] for r in m_recs
                          if r["lead_time_h"] == max_lead_h and r["ade"] is not None]
            rows.append({
                "storm": storm, "model": model,
                "n_lead_times": len(set(r["lead_time_h"] for r in m_recs)),
                "n_windows": n_windows,
                "ade_km": f"{np.mean(ade_vals):.2f}" if ade_vals else "",
                "ate_km": f"{np.mean(ate_vals):.2f}" if ate_vals else "",
                "cte_km": f"{np.mean(cte_vals):.2f}" if cte_vals else "",
                "final_ade_km": f"{np.mean(final_vals):.2f}" if final_vals else "",
            })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved per-storm trajectory-summary CSV → {path}  ({len(rows)} rows)")


def write_csv_trajectory_agg(agg_rows: List[Dict], path: str):
    """
    [MULTI-SEED] Same as write_csv_trajectory() but takes
    aggregate_across_seeds()'s output (already mean±std PER lead_time,
    across seeds) and further collapses ACROSS lead_times into one row
    per (storm, model) — i.e. this mixes two different sources of
    variance (seed-to-seed AND lead-time-to-lead-time) into a single
    reported std. That is exactly what the requested table layout
    (image: one mean±std value per metric per storm x model, no
    separate per-horizon breakdown) calls for, but it means this std is
    NOT the same quantity as aggregate_across_seeds()'s own per-lead-
    time std — it is coarser, "how much does ADE vary across BOTH seeds
    and forecast horizon for this storm" rather than "how much does ADE
    at exactly 24h vary across seeds". If you need the seed-only std at
    a fixed horizon, use write_csv_agg() (per-lead-time) instead; this
    function is for the compact whole-trajectory summary specifically.

    FINAL ADE mean±std is computed from the LAST lead_time's per-seed
    values directly (via the underlying per-seed ade_mean/ade_std at
    that horizon) — same endpoint-error convention as
    write_csv_trajectory()'s FINAL ADE column.
    """
    storms = sorted(set(r["storm"] for r in agg_rows))
    models_seen = sorted(set(r["model"] for r in agg_rows))
    max_lead_h = max(r["lead_time_h"] for r in agg_rows)

    fieldnames = ["storm", "model", "n_lead_times", "n_windows", "n_seeds",
                  "ade_mean_km", "ade_std_km", "ate_mean_km", "ate_std_km",
                  "cte_mean_km", "cte_std_km", "final_ade_mean_km", "final_ade_std_km"]
    rows = []
    for storm in storms:
        storm_recs = [r for r in agg_rows if r["storm"] == storm]
        n_windows = len(set(r["window"] for r in storm_recs))
        for model in models_seen:
            m_recs = [r for r in storm_recs if r["model"] == model]
            if not m_recs:
                continue
            n_seeds_here = m_recs[0]["n_seeds"] if m_recs else 0

            # [NOTE, see docstring] Pooling per-lead-time means/stds into
            # one trajectory-level mean/std here: mean of means is exact,
            # but "mean of per-horizon stds" is a simplification (not the
            # true combined std of the underlying pooled distribution) —
            # accurate enough for a compact summary table, not a
            # substitute for the per-lead-time CSV if exact variance
            # decomposition matters for your analysis.
            ade_m = [r["ade_mean"] for r in m_recs if r["ade_mean"] is not None]
            ade_s = [r["ade_std"]  for r in m_recs if r["ade_std"]  is not None]
            ate_m = [r["ate_mean"] for r in m_recs if r["ate_mean"] is not None]
            ate_s = [r["ate_std"]  for r in m_recs if r["ate_std"]  is not None]
            cte_m = [r["cte_mean"] for r in m_recs if r["cte_mean"] is not None]
            cte_s = [r["cte_std"]  for r in m_recs if r["cte_std"]  is not None]
            final_m = [r["ade_mean"] for r in m_recs
                       if r["lead_time_h"] == max_lead_h and r["ade_mean"] is not None]
            final_s = [r["ade_std"]  for r in m_recs
                       if r["lead_time_h"] == max_lead_h and r["ade_std"]  is not None]

            rows.append({
                "storm": storm, "model": model,
                "n_lead_times": len(set(r["lead_time_h"] for r in m_recs)),
                "n_windows": n_windows, "n_seeds": n_seeds_here,
                "ade_mean_km": f"{np.mean(ade_m):.2f}" if ade_m else "",
                "ade_std_km":  f"{np.mean(ade_s):.2f}" if ade_s else "",
                "ate_mean_km": f"{np.mean(ate_m):.2f}" if ate_m else "",
                "ate_std_km":  f"{np.mean(ate_s):.2f}" if ate_s else "",
                "cte_mean_km": f"{np.mean(cte_m):.2f}" if cte_m else "",
                "cte_std_km":  f"{np.mean(cte_s):.2f}" if cte_s else "",
                "final_ade_mean_km": f"{np.mean(final_m):.2f}" if final_m else "",
                "final_ade_std_km":  f"{np.mean(final_s):.2f}" if final_s else "",
            })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved per-storm trajectory-summary mean±std CSV → {path}  ({len(rows)} rows)")



def write_markdown_summary_agg(agg_rows: List[Dict], path: str,
                                horizons_h: List[int] = None):
    """
    [MULTI-SEED, REDESIGNED, compact] Same restructuring as
    write_markdown_summary(): ONE flat table per metric (ADE/ATE/CTE),
    storm+model as rows, standard horizons (12/24/48/72h) as columns —
    not one small table per storm. Cells show "mean±std" across seeds.
    This is the table to paste into a paper appendix; the full per-6h,
    per-seed detail stays in the companion CSV/JSON.
    """
    if horizons_h is None:
        horizons_h = STANDARD_HORIZONS_H

    storms = sorted(set(r["storm"] for r in agg_rows))
    models_seen = sorted(set(r["model"] for r in agg_rows))

    lines = ["# Per-storm error summary (mean±std across seeds) — all architectures\n",
             f"Standard horizons: {', '.join(f'{h}h' for h in horizons_h)}. "
             f"Full per-6h, per-seed detail available in the companion CSV/JSON.\n"]

    for metric, mean_key, std_key, label in [
        ("ade", "ade_mean", "ade_std", "ADE"),
        ("ate", "ate_mean", "ate_std", "ATE"),
        ("cte", "cte_mean", "cte_std", "CTE"),
    ]:
        lines.append(f"\n## {label} (km)\n")
        header = "| Storm | Model | " + " | ".join(f"{h}h" for h in horizons_h) + " |"
        sep    = "|---|---|" + "---|" * len(horizons_h)
        lines.append(header)
        lines.append(sep)
        for storm in storms:
            storm_recs = [r for r in agg_rows if r["storm"] == storm]
            for model in models_seen:
                m_recs = [r for r in storm_recs if r["model"] == model]
                if not m_recs:
                    continue
                cells = []
                for h in horizons_h:
                    matches = [r for r in m_recs if r["lead_time_h"] == h
                               and r[mean_key] is not None]
                    if matches:
                        # Multiple windows for this storm at this horizon
                        # — average the per-window means (std is dropped
                        # at this aggregation level since it would mix
                        # cross-seed and cross-window variance; the
                        # per-window, per-seed std is in the CSV/JSON for
                        # anyone who needs that finer breakdown).
                        cells.append(f"{np.mean([r[mean_key] for r in matches]):.1f}"
                                     f"±{np.mean([r[std_key] for r in matches]):.1f}")
                    else:
                        cells.append("—")
                lines.append(f"| {storm} | {model} | " + " | ".join(cells) + " |")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved compact mean±std markdown summary → {path}")


def write_markdown_summary_agg_merged(agg_rows: List[Dict], path: str,
                                       horizons_h: List[int] = None):
    """
    [MULTI-SEED, MERGED] Single table combining ADE/ATE/CTE mean±std,
    instead of 3 separate tables — one row per (storm, model), one
    column per horizon, each cell showing all three metrics as
    "ADE±std/ATE±std/CTE±std" (km). Same idea as
    write_markdown_summary_merged() but for the multi-seed case.

    Cells get visually dense with 3 mean±std values each — if that
    reads as too cramped once you see it, write_markdown_summary_agg()
    (3 separate per-metric tables) remains available and is still
    written alongside this one; use whichever fits your paper's layout
    better. This function does not replace that one, only supplements it.
    """
    if horizons_h is None:
        horizons_h = STANDARD_HORIZONS_H

    storms = sorted(set(r["storm"] for r in agg_rows))
    models_seen = sorted(set(r["model"] for r in agg_rows))

    lines = ["# Per-storm error summary (ADE/ATE/CTE, mean±std across seeds, km) — all architectures\n",
             f"Each cell: ADE±std/ATE±std/CTE±std (km) at that horizon. "
             f"Standard horizons: {', '.join(f'{h}h' for h in horizons_h)}. "
             f"Full per-6h, per-seed detail available in the companion CSV/JSON.\n"]
    header = "| Storm | Model | " + " | ".join(f"{h}h (ADE/ATE/CTE)" for h in horizons_h) + " |"
    sep    = "|---|---|" + "---|" * len(horizons_h)
    lines.append(header)
    lines.append(sep)

    for storm in storms:
        storm_recs = [r for r in agg_rows if r["storm"] == storm]
        for model in models_seen:
            m_recs = [r for r in storm_recs if r["model"] == model]
            if not m_recs:
                continue
            cells = []
            for h in horizons_h:
                h_recs = [r for r in m_recs if r["lead_time_h"] == h]
                ade_m = [r["ade_mean"] for r in h_recs if r["ade_mean"] is not None]
                ade_s = [r["ade_std"]  for r in h_recs if r["ade_std"]  is not None]
                ate_m = [r["ate_mean"] for r in h_recs if r["ate_mean"] is not None]
                ate_s = [r["ate_std"]  for r in h_recs if r["ate_std"]  is not None]
                cte_m = [r["cte_mean"] for r in h_recs if r["cte_mean"] is not None]
                cte_s = [r["cte_std"]  for r in h_recs if r["cte_std"]  is not None]
                ade_c = f"{np.mean(ade_m):.1f}±{np.mean(ade_s):.1f}" if ade_m else "—"
                ate_c = f"{np.mean(ate_m):.1f}±{np.mean(ate_s):.1f}" if ate_m else "—"
                cte_c = f"{np.mean(cte_m):.1f}±{np.mean(cte_s):.1f}" if cte_m else "—"
                cells.append(f"{ade_c}/{ate_c}/{cte_c}")
            lines.append(f"| {storm} | {model} | " + " | ".join(cells) + " |")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved merged mean±std (ADE/ATE/CTE) markdown summary → {path}")


def _parse_seed_from_path(path: str, fallback_idx: int) -> int:
    """Best-effort seed extraction from a checkpoint path (e.g.
    '.../fm_seed0/best_model.pth' -> 0), used only when the checkpoint
    itself has no saved 'seed' field (older checkpoints predating
    --seed support in the training scripts)."""
    import re
    m = re.search(r"seed[_]?(\d+)", path, re.IGNORECASE)
    return int(m.group(1)) if m else fallback_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--output_dir", default="per_storm_timeline")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_ensemble", type=int, default=20)
    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--test_year", type=int, default=None)

    # [MULTI-SEED] Each accepts ONE OR MORE checkpoint paths (nargs="+"),
    # one per seed of that architecture — same convention as
    # evaluate_multi_model.py. Passing exactly one path per flag
    # reproduces the original single-checkpoint behavior exactly (no
    # mean±std, per_storm_timeline_test.csv/json/md as before); passing
    # 2+ switches this script to also emit the *_meanstd.* outputs.
    p.add_argument("--fm_checkpoints",       nargs="+", default=None)
    p.add_argument("--st_trans_checkpoints", nargs="+", default=None)
    p.add_argument("--lstm_checkpoints",     nargs="+", default=None)
    p.add_argument("--gru_checkpoints",      nargs="+", default=None)
    p.add_argument("--rnn_checkpoints",      nargs="+", default=None)

    p.add_argument("--storm_filter", default=None,
                   help="Only process storms whose name contains this "
                        "substring (case-insensitive) — useful for a "
                        "quick look at one storm without waiting for the "
                        "full test set. Default: all storms.")

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("  Loading test data...")
    _loader_args = argparse.Namespace(
        dataset_root=args.dataset_root, obs_len=args.obs_len, pred_len=args.pred_len,
        batch_size=64, num_workers=2, test_year=args.test_year,
        skip=1, min_ped=1, threshold=0.002,
    )
    _, loader = data_loader(_loader_args,
                             {"root": args.dataset_root, "type": args.split},
                             test=(args.split != "train"))
    print(f"  Data: {len(loader)} batches")
    print(f"  ⚠ Multi-seed aggregation assumes this loader's storm/window "
          f"ordering is IDENTICAL across every checkpoint below (shuffle="
          f"False, same batch_size, same dataset files) — see "
          f"aggregate_across_seeds()'s docstring for the exact assumption.")

    # [MULTI-SEED] Flatten (architecture, [ckpt_seed0, ckpt_seed1, ...])
    # into one job per (architecture, checkpoint).
    jobs = []
    for display_name, kind, ckpts in [
        ("FM",       "fm",       args.fm_checkpoints),
        ("ST-Trans", "st_trans", args.st_trans_checkpoints),
        ("LSTM",     "lstm",     args.lstm_checkpoints),
        ("GRU",      "gru",      args.gru_checkpoints),
        ("RNN",      "rnn",      args.rnn_checkpoints),
    ]:
        if not ckpts:
            continue
        for i, ckpt_path in enumerate(ckpts):
            jobs.append((display_name, kind, ckpt_path, i))

    if not jobs:
        print("  No checkpoints given — nothing to do.")
        return

    n_seeds_by_arch = {}
    for name, kind, ckpts in [("FM", "fm", args.fm_checkpoints),
                               ("ST-Trans", "st_trans", args.st_trans_checkpoints),
                               ("LSTM", "lstm", args.lstm_checkpoints),
                               ("GRU", "gru", args.gru_checkpoints),
                               ("RNN", "rnn", args.rnn_checkpoints)]:
        if ckpts:
            n_seeds_by_arch[name] = len(ckpts)
    is_multi_seed = any(n > 1 for n in n_seeds_by_arch.values())

    all_records = []
    for display_name, kind, ckpt_path, fallback_idx in jobs:
        print(f"\n  {'='*70}\n  Loading {display_name}: {ckpt_path}\n  {'='*70}")
        if kind == "fm":
            model, seed = load_fm(ckpt_path, device)
        elif kind == "st_trans":
            model, seed = load_st_trans(ckpt_path, device)
        else:
            model, seed = load_paper_baseline(ckpt_path, kind, device)

        if seed is None:
            seed = _parse_seed_from_path(ckpt_path, fallback_idx)
            print(f"  ⚠ Checkpoint has no saved 'seed' field — labeling "
                  f"records with seed={seed} guessed from path/order. "
                  f"This is a DISPLAY/GROUPING LABEL ONLY.")

        print(f"  {display_name} (seed={seed}): "
              f"{sum(pm.numel() for pm in model.parameters()):,} params")

        recs = collect_timeline(model, loader, device, display_name,
                                 obs_len=args.obs_len, n_ensemble=args.n_ensemble,
                                 seed=seed)
        all_records.extend(recs)
        print(f"  {display_name} (seed={seed}): {len(recs)} "
              f"(storm, window, lead_time) records")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.storm_filter:
        before = len(all_records)
        f = args.storm_filter.upper()
        all_records = [r for r in all_records if f in r["storm"].upper()]
        print(f"\n  --storm_filter '{args.storm_filter}': {before} -> "
              f"{len(all_records)} records")

    # Raw per-seed records always saved, regardless of single/multi-seed —
    # this is the traceable source of truth the mean±std numbers derive from.
    json_path = os.path.join(args.output_dir, f"per_storm_timeline_{args.split}.json")
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n  Saved raw per-seed records → {json_path}")

    horizons_h = sorted(set(r["lead_time_h"] for r in all_records))

    if not is_multi_seed:
        # Exactly one checkpoint per architecture — original behavior,
        # unchanged output filenames/format.
        print_storm_tables(all_records)
        write_csv(all_records, os.path.join(args.output_dir, f"per_storm_timeline_{args.split}.csv"))
        # [NEW] Per-storm trajectory-summary CSV — one row per (storm,
        # model), ADE/ATE/CTE averaged over the WHOLE trajectory plus
        # FINAL ADE (endpoint error), instead of broken out per horizon
        # like the CSV above. Separate file, does not replace it.
        write_csv_trajectory(all_records, os.path.join(args.output_dir, f"per_storm_trajectory_{args.split}.csv"))
        # [COMPACT] Markdown table uses STANDARD horizons (12/24/48/72h),
        # not every 6h step — the full per-6h detail is in the CSV above.
        # Only fall back to whatever horizons the data actually has if
        # none of the standard ones are present (e.g. a shorter forecast).
        md_horizons = [h for h in STANDARD_HORIZONS_H if h in horizons_h] or horizons_h
        write_markdown_summary(all_records, os.path.join(args.output_dir, f"per_storm_summary_{args.split}.md"),
                                md_horizons)
        # [MERGED] Same data, ADE/ATE/CTE combined into ONE table instead
        # of three — supplements (does not replace) the per-metric file above.
        write_markdown_summary_merged(all_records, os.path.join(args.output_dir, f"per_storm_summary_{args.split}_merged.md"),
                                       md_horizons)
    else:
        print(f"\n  Multi-seed input detected ({n_seeds_by_arch}) — "
              f"aggregating to mean±std per (storm, window, model, lead_time)...")
        agg_rows = aggregate_across_seeds(all_records)

        print_storm_tables_agg(agg_rows)

        agg_json_path = os.path.join(args.output_dir, f"per_storm_timeline_{args.split}_meanstd.json")
        with open(agg_json_path, "w") as f:
            json.dump(agg_rows, f, indent=2)
        print(f"\n  Saved mean±std records → {agg_json_path}")

        write_csv_agg(agg_rows, os.path.join(args.output_dir, f"per_storm_timeline_{args.split}_meanstd.csv"))
        # [NEW] Per-storm trajectory-summary CSV, multi-seed mean±std
        # version — one row per (storm, model), matching the requested
        # layout (image: storm as a row group, model as sub-rows,
        # mean±std per metric, no per-horizon breakdown).
        write_csv_trajectory_agg(agg_rows, os.path.join(args.output_dir, f"per_storm_trajectory_{args.split}_meanstd.csv"))
        md_horizons = [h for h in STANDARD_HORIZONS_H if h in horizons_h] or horizons_h
        write_markdown_summary_agg(agg_rows, os.path.join(args.output_dir, f"per_storm_summary_{args.split}_meanstd.md"),
                                    md_horizons)
        # [MERGED] Same data, ADE/ATE/CTE combined into ONE table.
        write_markdown_summary_agg_merged(agg_rows, os.path.join(args.output_dir, f"per_storm_summary_{args.split}_meanstd_merged.md"),
                                           md_horizons)


if __name__ == "__main__":
    main()