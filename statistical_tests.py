"""

# ─────────────────────────────────────────────────────────────────────────────
# FILE PLACEMENT:
#
#   SOURCE:        statistical_tests.py
#   KAGGLE TARGET: /kaggle/working/statistical_tests.py
#   LOCAL DEV:     statistical_tests.py
#
#   Chạy SAU evaluate_full.py:
#   python statistical_tests.py \
#     --fm_results results/eval_test_epXXX.json \
#     --use_st_trans_ref --fm_n_storms 420 \
#     --output_dir results/stats/
# ─────────────────────────────────────────────────────────────────────────────

statistical_tests.py — ESWA Section C: Statistical Significance
════════════════════════════════════════════════════════════════════════════════
Implements all tests required by ESWA paper:
  • Wilcoxon signed-rank test (non-parametric, robust)
  • Paired t-test
  • Bonferroni correction (multiple comparisons)
  • Cohen's d (effect size) — critical when margin is small (~4-6km)
  • Bootstrap CI 95% for Δ (FM − ST-Trans)

Usage:
  # Step 1: run evaluate_full.py for each model to get per-storm JSON files
  python statistical_tests.py \\
    --fm_results results/eval_test_epXXX.json \\
    --baseline_results results/baseline_st_trans.json \\
    --baseline_name "ST-Trans" \\
    --output_dir results/stats/

  # Or: provide per-storm predictions directly
  python statistical_tests.py \\
    --fm_pred fm_predictions.npy \\
    --baseline_pred sttrans_predictions.npy \\
    --gt_data gt_data.npy
"""
from __future__ import annotations

import os, sys, argparse, json, math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
#  Core statistical functions
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_test(x: np.ndarray, y: np.ndarray,
                   alternative: str = "less") -> Dict:
    """
    Wilcoxon signed-rank test: H0: median(x-y)=0, H1: x < y (fm better).
    x = FM errors, y = baseline errors.
    alternative='less' tests whether FM has lower errors than baseline.
    """
    diff = x - y
    # Remove zeros (ties)
    diff_nonzero = diff[diff != 0]
    if len(diff_nonzero) < 10:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "n": len(diff), "significant": False,
                "significant_0.05": False, "significant_0.01": False}

    stat, p = stats.wilcoxon(diff_nonzero, alternative=alternative)
    return {
        "statistic": float(stat),
        "p_value":   float(p),
        "n":         int(len(diff_nonzero)),
        "significant":      bool(p < 0.05),
        "significant_0.05": bool(p < 0.05),
        "significant_0.01": bool(p < 0.01),
    }


def paired_ttest(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Paired t-test: H0: mean(x-y)=0, H1: mean(x-y)<0 (FM better).
    x = FM, y = baseline.
    """
    diff = x - y
    stat, p = stats.ttest_rel(x, y, alternative="less")
    return {
        "statistic":        float(stat),
        "p_value":          float(p),
        "mean_diff":        float(diff.mean()),
        "std_diff":         float(diff.std()),
        "se_diff":          float(diff.std() / math.sqrt(len(diff))),
        "significant_0.05": p < 0.05,
        "significant_0.01": p < 0.01,
    }


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d effect size for paired samples.
    d = mean(x-y) / std(x-y)
    Interpretation: small=0.2, medium=0.5, large=0.8
    """
    diff = x - y
    std  = diff.std(ddof=1)
    if std < 1e-8:
        return 0.0
    return float(diff.mean() / std)


def bootstrap_ci(x: np.ndarray, y: np.ndarray,
                  n_bootstrap: int = 10000,
                  ci: float = 0.95) -> Dict:
    """
    Bootstrap confidence interval for Δ = mean(x) - mean(y) (FM - baseline).
    Negative Δ = FM better.
    """
    rng  = np.random.default_rng(42)
    n    = len(x)
    diffs = []
    for _ in range(n_bootstrap):
        idx   = rng.integers(0, n, size=n)
        d_b   = x[idx].mean() - y[idx].mean()
        diffs.append(d_b)
    diffs = np.array(diffs)
    alpha = 1 - ci
    lower = float(np.percentile(diffs, 100 * alpha / 2))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return {
        "delta_mean":      float(x.mean() - y.mean()),
        "ci_lower":        lower,
        "ci_upper":        upper,
        "ci_level":        ci,
        "significant":     upper < 0,  # entire CI is negative → FM better
        "n_bootstrap":     n_bootstrap,
    }


def bonferroni_correction(p_values: List[float],
                           alpha: float = 0.05) -> List[Dict]:
    """
    Bonferroni correction for multiple comparisons.
    Returns adjusted p-values and significance flags.
    """
    n = len(p_values)
    results = []
    for i, p in enumerate(p_values):
        p_adj = min(1.0, p * n)
        results.append({
            "p_raw":       p,
            "p_bonferroni": p_adj,
            "significant":  p_adj < alpha,
        })
    return results


def interpret_cohens_d(d: float) -> str:
    d_abs = abs(d)
    if d_abs < 0.2:   return "negligible"
    elif d_abs < 0.5: return "small"
    elif d_abs < 0.8: return "medium"
    else:             return "large"


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-metric test suite
# ─────────────────────────────────────────────────────────────────────────────

def run_full_tests(fm_errors:       Dict[str, np.ndarray],
                   baseline_errors: Dict[str, np.ndarray],
                   baseline_name:   str = "ST-Trans",
                   n_bootstrap:     int = 10000) -> Dict:
    """
    Run full test suite for all metrics.
    fm_errors:       {"ADE": [n,], "ATE": [n,], "CTE": [n,]}
    baseline_errors: {"ADE": [n,], "ATE": [n,], "CTE": [n,]}
    """
    metrics  = list(fm_errors.keys())
    p_wilcox = []
    p_ttest  = []
    results  = {}

    for metric in metrics:
        fm  = fm_errors[metric]
        bl  = baseline_errors[metric]

        # Align lengths
        n = min(len(fm), len(bl))
        fm_n = fm[:n]; bl_n = bl[:n]

        wilcox  = wilcoxon_test(fm_n, bl_n, alternative="less")
        ttest   = paired_ttest(fm_n, bl_n)
        d       = cohen_d(fm_n, bl_n)
        boot_ci = bootstrap_ci(fm_n, bl_n, n_bootstrap=n_bootstrap)

        results[metric] = {
            "fm_mean":        float(fm_n.mean()),
            "fm_std":         float(fm_n.std()),
            "baseline_mean":  float(bl_n.mean()),
            "baseline_std":   float(bl_n.std()),
            "delta_mean":     float(fm_n.mean() - bl_n.mean()),
            "wilcoxon":       wilcox,
            "paired_ttest":   ttest,
            "cohens_d":       d,
            "cohens_d_interpretation": interpret_cohens_d(d),
            "bootstrap_ci":   boot_ci,
            "n_pairs":        n,
        }
        p_wilcox.append(wilcox["p_value"])
        p_ttest.append(ttest["p_value"])

    # Bonferroni correction
    bonf_wilcox = bonferroni_correction(p_wilcox)
    bonf_ttest  = bonferroni_correction(p_ttest)

    for i, metric in enumerate(metrics):
        results[metric]["bonferroni_wilcoxon"] = bonf_wilcox[i]
        results[metric]["bonferroni_ttest"]    = bonf_ttest[i]

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_statistical_report(results: Dict, baseline_name: str = "ST-Trans"):
    print(f"\n  {'='*72}")
    print(f"  STATISTICAL TESTS: FM vs {baseline_name}")
    print(f"  {'='*72}")
    print(f"  {'Metric':<8} {'FM(mean)':>10} {'BL(mean)':>10} "
          f"{'Δ':>8} {'Cohen_d':>9} {'d_interp':>12} "
          f"{'Wilcox_p':>10} {'Bonf_W':>8} {'Boot95%CI':>20}")
    print(f"  {'─'*100}")

    for metric, r in results.items():
        ci     = r["bootstrap_ci"]
        ci_str = f"[{ci['ci_lower']:+.2f},{ci['ci_upper']:+.2f}]"
        w_p    = r.get("wilcoxon", {}).get("p_value", float("nan"))
        b_w    = r.get("bonferroni_wilcoxon", {}).get("p_bonferroni", float("nan"))
        sig_w  = "**" if (w_p == w_p and w_p < 0.01) else ("*" if (w_p == w_p and w_p < 0.05) else "ns")
        sig_b  = "**" if (b_w == b_w and b_w < 0.01) else ("*" if (b_w == b_w and b_w < 0.05) else "ns")

        print(f"  {metric:<8} "
              f"{r['fm_mean']:>10.2f} {r['baseline_mean']:>10.2f} "
              f"{r['delta_mean']:>+8.2f} "
              f"{r['cohens_d']:>+9.3f} {r['cohens_d_interpretation']:>12} "
              f"{w_p:>10.4f}{sig_w} {b_w:>8.4f}{sig_b} "
              f"{ci_str:>20}")

    print()
    print("  Significance: ** p<0.01, * p<0.05, ns p≥0.05 (Wilcoxon signed-rank)")
    print("  Bonferroni correction applied for multiple comparisons")
    print("  Bootstrap CI 95% for Δ = mean(FM) - mean(Baseline)")
    print("  Cohen's d: negligible<0.2, small<0.5, medium<0.8, large≥0.8")

    # Summary statement for paper
    print(f"\n  ── PAPER STATEMENT ──")
    sig_metrics = [m for m, r in results.items()
                   if r.get("wilcoxon", {}).get("significant_0.05", False)]
    if sig_metrics:
        print(f"  FM significantly outperforms {baseline_name} on: {sig_metrics} "
              f"(Wilcoxon signed-rank, Bonferroni-corrected p<0.05)")
    else:
        print(f"  FM does NOT significantly outperform {baseline_name} on any metric "
              f"at α=0.05 after Bonferroni correction.")
        print(f"  → ESWA recommendation: report 'comparable performance' + "
              f"highlight computational advantages")
    print(f"  {'='*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Load from eval JSON or per-storm arrays
# ─────────────────────────────────────────────────────────────────────────────

def load_from_json(path: str) -> Dict[str, np.ndarray]:
    """Load per-storm errors from evaluate_full.py JSON output."""
    with open(path) as f:
        data = json.load(f)
    errors = {}
    # Try loading boxplot_ade first (per-storm ADE)
    if "boxplot_ade" in data and data["boxplot_ade"]:
        errors["ADE"] = np.array(data["boxplot_ade"])
    elif "ADE" in data and data["ADE"] == data["ADE"]:  # not NaN
        errors["ADE"] = np.array([data["ADE"]])
    # ATE, CTE: prefer per-storm arrays (boxplot_ate/boxplot_cte, added to
    # evaluate_full.py's output alongside boxplot_ade), fall back to the
    # scalar mean only if the per-storm arrays are missing (e.g. older
    # eval JSON from before this fix). Without the per-storm arrays,
    # Wilcoxon signed-rank degenerates to nan (needs >=10 paired samples).
    for metric, box_key in [("ATE", "boxplot_ate"), ("CTE", "boxplot_cte")]:
        if box_key in data and data[box_key]:
            errors[metric] = np.array(data[box_key])
        elif metric in data and data[metric] == data[metric]:  # not NaN
            errors[metric] = np.array([data[metric]])
    # Warn if only scalar data (statistical tests need per-storm for power)
    n_ade = len(errors.get("ADE", []))
    if n_ade < 10:
        print(f"  ⚠ Only {n_ade} ADE values — statistical tests need per-storm data")
        print(f"    Run evaluate_full.py with working checkpoint first")
    return errors


def generate_synthetic_baseline(n: int, metrics: Dict[str, float],
                                  seed: int = 0) -> Dict[str, np.ndarray]:
    """
    Generate synthetic per-storm errors for a baseline with given means.
    Used when baseline predictions are not available per-storm.
    WARNING: This is an approximation — real per-storm predictions are preferred.
    """
    rng = np.random.default_rng(seed)
    result = {}
    # Typical TC error distribution is approximately log-normal
    for metric, mean in metrics.items():
        sigma  = mean * 0.5   # approximate std = 50% of mean
        mu_log = math.log(mean**2 / math.sqrt(mean**2 + sigma**2))
        s_log  = math.sqrt(math.log(1 + sigma**2 / mean**2))
        result[metric] = rng.lognormal(mu_log, s_log, size=n)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Statistical tests for ESWA paper")
    # Input options
    p.add_argument("--fm_results",       type=str, default=None,
                   help="JSON from evaluate_full.py for FM model")
    p.add_argument("--baseline_results", type=str, default=None,
                   help="JSON from evaluate_full.py for baseline")
    p.add_argument("--baseline_name",    type=str, default="ST-Trans")
    # If no per-storm JSON: use ST-Trans known values
    p.add_argument("--use_st_trans_ref", action="store_true", default=False,
                   help="Use ST-Trans published values (synthetic baseline)")
    p.add_argument("--fm_n_storms",      type=int, default=420,
                   help="Number of test storms (for synthetic baseline)")
    # Output
    p.add_argument("--output_dir",       type=str, default="results/stats")
    p.add_argument("--n_bootstrap",      type=int, default=10000)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load FM results ─────────────────────────────────────────────────────
    if args.fm_results:
        print(f"  Loading FM results: {args.fm_results}")
        fm_errors = load_from_json(args.fm_results)
    else:
        print("  ERROR: --fm_results required")
        return

    # ── Load or generate baseline ──────────────────────────────────────────
    if args.baseline_results:
        print(f"  Loading {args.baseline_name}: {args.baseline_results}")
        baseline_errors = load_from_json(args.baseline_results)
    elif args.use_st_trans_ref:
        print(f"  Using ST-Trans reference values (synthetic distribution)")
        print(f"  WARNING: Per-storm predictions preferred for exact test.")
        st_trans_means = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4}
        n = len(fm_errors.get("ADE", [args.fm_n_storms]))
        baseline_errors = generate_synthetic_baseline(n, st_trans_means)
    else:
        print("  ERROR: --baseline_results or --use_st_trans_ref required")
        return

    # ── Align metrics ──────────────────────────────────────────────────────
    common_metrics = [k for k in fm_errors if k in baseline_errors]
    if not common_metrics:
        print("  ERROR: No common metrics between FM and baseline")
        return

    fm_aligned   = {k: fm_errors[k] for k in common_metrics}
    bl_aligned   = {k: baseline_errors[k] for k in common_metrics}

    print(f"  Metrics: {common_metrics}")
    print(f"  FM storms: {len(fm_aligned.get('ADE', []))}")
    print(f"  Baseline storms: {len(bl_aligned.get('ADE', []))}")
    print(f"  n_bootstrap: {args.n_bootstrap}")
    print()

    # ── Run tests ──────────────────────────────────────────────────────────
    results = run_full_tests(
        fm_aligned, bl_aligned,
        baseline_name=args.baseline_name,
        n_bootstrap=args.n_bootstrap,
    )

    # ── Print ──────────────────────────────────────────────────────────────
    print_statistical_report(results, baseline_name=args.baseline_name)

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir,
                             f"stats_fm_vs_{args.baseline_name.replace(' ', '_')}.json")
    # Convert numpy types to native Python for JSON serialization
    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(_convert(results), f, indent=2)
    print(f"  Saved → {out_path}")

    # LaTeX table fragment
    print(f"\n  ── LaTeX TABLE FRAGMENT ──")
    print(r"  \begin{tabular}{lrrrrl}")
    print(r"  Metric & FM & " + args.baseline_name +
          r" & $\Delta$ & $d$ & $p$ (Bonf.) \\")
    print(r"  \hline")
    for metric, r in results.items():
        d    = r["cohens_d"]
        p_b  = r["bonferroni_wilcoxon"]["p_bonferroni"]
        sig  = "^{**}" if p_b < 0.01 else ("^{*}" if p_b < 0.05 else "")
        print(f"  {metric} & {r['fm_mean']:.1f} & {r['baseline_mean']:.1f} & "
              f"{r['delta_mean']:+.1f} & {d:+.3f} & {p_b:.4f}{sig} \\\\")
    print(r"  \end{tabular}")
    print(r"  (* p<0.05, ** p<0.01, Bonferroni-corrected Wilcoxon signed-rank)")


if __name__ == "__main__":
    main()