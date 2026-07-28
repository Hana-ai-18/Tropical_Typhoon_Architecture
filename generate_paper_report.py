"""
generate_paper_report.py
==============================
GỘP từ generate_paper_table.py (3 bảng thống kê: main, pooled
significance, per-horizon) và generate_comparison_plots.py (toàn bộ
plot so sánh) — 1 lệnh chạy ra CẢ bảng lẫn hình từ cùng 1 file
multi_model_<split>.json (từ evaluate_multi_model.py).

Không còn import chéo generate_comparison_table.py — 5 hàm thống kê
lõi (cohen_d, wilcoxon_test, paired_ttest, bonferroni_correction,
interpret_cohens_d) được copy nguyên văn vào thẳng file này (xem khối
"STATISTICAL HELPERS" bên dưới), để file này tự chứa hoàn toàn.

TABLES (từ generate_paper_table.py)
------------------------------------
  1. MAIN TABLE      — ADE/ATE/CTE mean±std ACROSS SEEDS, per architecture.
  2. SIGNIFICANCE     — pooled-3-seed paired tests, FM vs each baseline.
  3. PER-HORIZON      — ADE at 6h/12h/24h/48h/72h, FM vs strongest baseline.

PLOTS (từ generate_comparison_plots.py)
-----------------------------------------
  - error_vs_leadtime_{ade,ate,cte}.png + error_vs_leadtime_grid.png
  - error_boxplots.png, boxplot_by_horizon_ade.png
  - seed_variance_{ade,cte}.png
  - ode_n_sweep.png (nếu --ode_sweep)
  - ablation_bars_{ADE,CTE}.png (nếu --ablation_dir)
  - sigma_sensitivity.png, ensemble_size_ablation.png (nếu --eval_full_json)
  - per_storm_cte_worst.png (nếu --per_storm_json)

WHY POOLED, NOT "BEST SEED" OR "1 RANDOM SEED" (bảng thống kê)
-----------------------------------------------------------------
- "Best seed" là post-hoc selection bias — chọn sau khi đã thấy kết quả,
  làm phồng Cohen's d và giảm p-value giả tạo, không tái lập được.
- "1 seed ngẫu nhiên" lãng phí 2/3 compute đã tốn, kết luận phụ thuộc
  seed nào được chọn.
- Pooled 3-seed giữ mọi thông tin, phản ánh đúng biến thiên seed-to-seed
  thật, là lựa chọn "trung thực" nhất. Bảng MAIN (mean±std theo seed)
  đi kèm SONG SONG để người đọc cũng thấy được độ ổn định riêng.

USAGE
-----
python generate_paper_report.py \
    --records eval_multi/multi_model_test.json \
    --output_dir eval_multi/ \
    --ode_sweep ablations/ode_steps_sweep.json \
    --ablation_dir ablations/ \
    --eval_full_json results/eval_test_ep120.json \
    --per_storm_json results/per_storm_test_ep120.json

Chỉ --records là bắt buộc; các --ode_sweep/--ablation_dir/
--eval_full_json/--per_storm_json là optional, thiếu cái nào thì bỏ
qua đúng plot cần cái đó (in cảnh báo, không crash).
"""
from __future__ import annotations
import sys, os, argparse, json
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICAL HELPERS — copy nguyên văn từ generate_comparison_table.py,
#  không import chéo (file này tự chứa hoàn toàn).
# ─────────────────────────────────────────────────────────────────────────────

def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def wilcoxon_test(x: np.ndarray, y: np.ndarray, alternative: str = "less") -> Dict:
    diff = x - y
    diff_nonzero = diff[diff != 0]
    if len(diff_nonzero) < 10:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": len(diff)}
    stat, p = stats.wilcoxon(diff_nonzero, alternative=alternative)
    return {"statistic": float(stat), "p_value": float(p), "n": int(len(diff_nonzero))}


def paired_ttest(x: np.ndarray, y: np.ndarray) -> Dict:
    stat, p = stats.ttest_rel(x, y, alternative="less")
    return {"statistic": float(stat), "p_value": float(p)}


def bonferroni_correction(p_values: List[float]) -> List[float]:
    n = len(p_values)
    return [min(1.0, p * n) for p in p_values]


def interpret_cohens_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2: return "negligible"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    return "large"

HORIZON_LEAD_TIMES = {"6h": 1, "12h": 2, "24h": 4, "48h": 8, "72h": 12}
# lead_time convention: records store 1-indexed step (1=6h ... 12=72h),
# matching evaluate_full.py's HORIZONS dict (0-indexed step -> +1 here).
# If your evaluate_multi_model.py emits 0-indexed lead_time instead,
# pass --lead_time_zero_indexed to shift this mapping by -1.

ALL_MODELS = ["FM", "ST-Trans", "LSTM", "GRU", "RNN"]

# ─── Constants dùng riêng cho phần PLOT (giữ tách khỏi HORIZON_LEAD_TIMES
# ở trên, vốn chỉ có 5 mốc cho bảng per-horizon — phần plot cần đủ 12 mốc
# 6h→72h cho boxplot_by_horizon và error_vs_leadtime) ───────────────────
MODEL_COLORS = {
    "FM":       "#D62728",
    "ST-Trans": "#FF7F0E",
    "LSTM":     "#2CA02C",
    "GRU":      "#9467BD",
    "RNN":      "#8C564B",
}
MODEL_PLOT_ORDER = ["RNN", "LSTM", "GRU", "ST-Trans", "FM"]  # yếu->mạnh, khớp trục x Fig.4/5

HORIZON_LEAD_TIMES_FULL = {"6h": 1, "12h": 2, "18h": 3, "24h": 4, "30h": 5, "36h": 6,
                            "42h": 7, "48h": 8, "54h": 9, "60h": 10, "66h": 11, "72h": 12}


def load_records(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  Table 1: main mean±std-across-seeds table
# ─────────────────────────────────────────────────────────────────────────────

def build_main_table(records: List[Dict], models: List[str]) -> List[Dict]:
    """
    For each model: group by seed, compute per-seed mean ADE/ATE/CTE
    (mean over all (storm, window, lead_time) records for that seed),
    then report mean±std OF THOSE PER-SEED MEANS across seeds.

    This is deliberately NOT "mean±std of all raw per-record errors" —
    that would conflate within-seed forecast variance (storm-to-storm
    difficulty) with between-seed variance (training instability), and
    the latter is what a paper table's "±" is meant to communicate.

    [BỔ SUNG] Ngoài cột "overall" (gộp mọi lead_time), giờ có thêm cột
    "final_step" (chỉ 72h, tức lead_time == HORIZON_LEAD_TIMES["72h"]).
    Đây là con số quan trọng cho paper vì 72h là horizon dự báo XA NHẤT
    — sai số ở đây thường được dùng làm tiêu chí so sánh chính giữa các
    kiến trúc (khác với "overall mean" vốn bị pha loãng bởi các horizon
    gần, dễ dự báo hơn). Cùng cách tính seed-mean-rồi-mean/std như cột
    overall, chỉ lọc thêm điều kiện lead_time == 72h trước khi gộp theo
    seed.
    """
    final_lt = HORIZON_LEAD_TIMES.get("72h")
    rows = []
    for model in models:
        model_recs = [r for r in records if r["model"] == model]
        if not model_recs:
            print(f"  ⚠ No records for model={model}, skipping")
            continue
        by_seed = defaultdict(lambda: {"ade": [], "ate": [], "cte": []})
        by_seed_final = defaultdict(lambda: {"ade": [], "ate": [], "cte": []})
        for r in model_recs:
            s = r.get("seed", "unknown")
            for m in ("ade", "ate", "cte"):
                if m in r and r[m] is not None:
                    by_seed[s][m].append(r[m])
                    if r.get("lead_time") == final_lt:
                        by_seed_final[s][m].append(r[m])

        seed_means = {"ade": [], "ate": [], "cte": []}
        seed_means_final = {"ade": [], "ate": [], "cte": []}
        n_seeds = 0
        for seed, vals in sorted(by_seed.items()):
            n_seeds += 1
            for m in ("ade", "ate", "cte"):
                if vals[m]:
                    seed_means[m].append(float(np.mean(vals[m])))
        for seed, vals in sorted(by_seed_final.items()):
            for m in ("ade", "ate", "cte"):
                if vals[m]:
                    seed_means_final[m].append(float(np.mean(vals[m])))

        row = {"model": model, "n_seeds": n_seeds,
               "n_records": len(model_recs)}
        for m in ("ade", "ate", "cte"):
            vals = seed_means[m]
            row[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{m}_std"]  = float(np.std(vals))  if vals else float("nan")
            row[f"{m}_per_seed"] = vals

            vals_f = seed_means_final[m]
            row[f"{m}_final_mean"] = float(np.mean(vals_f)) if vals_f else float("nan")
            row[f"{m}_final_std"]  = float(np.std(vals_f))  if vals_f else float("nan")
        rows.append(row)
    return rows


def print_main_table(rows: List[Dict]):
    print(f"\n  {'='*140}")
    print(f"  TABLE 1 — MAIN RESULTS (mean ± std across seeds) — overall vs final step (72h)")
    print(f"  {'='*140}")
    print(f"  {'Model':<12} {'#seeds':>7} "
          f"{'ADE overall':>16} {'ADE@72h':>16} "
          f"{'ATE overall':>16} {'ATE@72h':>16} "
          f"{'CTE overall':>16} {'CTE@72h':>16}")
    print(f"  {'-'*140}")
    for r in rows:
        print(f"  {r['model']:<12} {r['n_seeds']:>7} "
              f"{r['ade_mean']:>9.2f}±{r['ade_std']:<5.2f} "
              f"{r['ade_final_mean']:>9.2f}±{r['ade_final_std']:<5.2f} "
              f"{r['ate_mean']:>9.2f}±{r['ate_std']:<5.2f} "
              f"{r['ate_final_mean']:>9.2f}±{r['ate_final_std']:<5.2f} "
              f"{r['cte_mean']:>9.2f}±{r['cte_std']:<5.2f} "
              f"{r['cte_final_mean']:>9.2f}±{r['cte_final_std']:<5.2f}")
    print(f"  {'='*140}")
    print(f"  'overall' = mean qua mọi lead_time (6h-72h) | '@72h' = chỉ final step "
          f"(horizon dự báo xa nhất, thường dùng làm tiêu chí so sánh chính)\n")


def print_main_table_latex(rows: List[Dict]):
    print(r"  \begin{table}")
    print(r"  \caption{Main results: ADE/ATE/CTE (km), mean $\pm$ std across seeds.}")
    print(r"  \begin{tabular}{lccc}")
    print(r"  \hline")
    print(r"  Model & ADE (km) & ATE (km) & CTE (km) \\")
    print(r"  \hline")
    for r in rows:
        print(f"  {r['model']} & "
              f"{r['ade_mean']:.2f} $\\pm$ {r['ade_std']:.2f} & "
              f"{r['ate_mean']:.2f} $\\pm$ {r['ate_std']:.2f} & "
              f"{r['cte_mean']:.2f} $\\pm$ {r['cte_std']:.2f} \\\\")
    print(r"  \hline")
    print(r"  \end{tabular}")
    print(r"  \end{table}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Table 2: pooled significance table (FM vs each baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_pooled_paired_arrays(records: List[Dict], model_a: str, model_b: str,
                                metric: str):
    """
    Pairs records at (seed, storm, window, lead_time) granularity so
    that the SAME seed's forecast on the SAME storm/window/lead_time is
    matched between model_a and model_b — i.e. each seed contributes its
    own set of paired observations, and all seeds' pairs are pooled
    together into one paired test. This is what "pooled 3-seed" means
    concretely: n_pairs = sum over seeds of matched (storm,window,
    lead_time) pairs for that seed, not n_pairs from a single seed and
    not an average collapsed across seeds first.

    Falls back to (seed, storm, window) if lead_time is absent, and
    further to (storm, window) if seed is absent (older single-seed
    records) — but pooling with genuinely multi-seed data requires the
    seed field, so absence of "seed" on records means you are NOT
    actually pooling seeds, just replicating generate_comparison_table's
    single-seed behavior. A warning is printed in that case.
    """
    has_seed = any("seed" in r for r in records)
    has_lead_time = any("lead_time" in r for r in records)
    if not has_seed:
        print("  ⚠ Records have no 'seed' field — pooled test is NOT "
              "actually pooling multiple seeds. Re-check evaluate_multi_model.py "
              "output.")
    if has_seed and has_lead_time:
        key_fn = lambda r: (r.get("seed"), r["storm"], r["window"], r.get("lead_time"))
    elif has_seed:
        key_fn = lambda r: (r.get("seed"), r["storm"], r["window"])
    elif has_lead_time:
        key_fn = lambda r: (r["storm"], r["window"], r.get("lead_time"))
    else:
        key_fn = lambda r: (r["storm"], r["window"])

    by_key_a = {key_fn(r): r[metric] for r in records
                if r["model"] == model_a and r.get(metric) is not None}
    by_key_b = {key_fn(r): r[metric] for r in records
                if r["model"] == model_b and r.get(metric) is not None}
    # [FIX] r.get(metric) is not None: ate/cte là None ở lead_time=1 (6h)
    # cho MỌI model (không định nghĩa được toán học ở bước dự báo đầu
    # tiên — cần bước trước đó để biết hướng đi). Không lọc None ở đây
    # sẽ khiến common-key set chứa cặp (None, None) hoặc lỗi khi ép
    # np.array (None lẫn trong mảng float => dtype=object, các phép
    # tính thống kê phía sau âm thầm ra NaN/lỗi khó dò thay vì bị loại
    # đúng chỗ). ADE không bị ảnh hưởng (luôn có giá trị, không phải None).
    common = sorted(set(by_key_a) & set(by_key_b), key=lambda k: str(k))
    if len(common) < 10:
        return None, None, len(common)
    x = np.array([by_key_a[k] for k in common])
    y = np.array([by_key_b[k] for k in common])
    return x, y, len(common)


def build_significance_table(records: List[Dict], baseline_model: str,
                              compare_against: List[str], metric: str) -> List[Dict]:
    """
    FM "achieves" a comparison per the project's agreed threshold:
    ALL FOUR of:
      mean_diff < 0  (FM lower error)
      |Cohen's d| >= 0.2  (at least small effect)
      Wilcoxon p (Bonferroni) < 0.05
      t-test p (Bonferroni) < 0.05
    """
    rows = []
    p_wilcox, p_ttest = [], []
    row_data = []
    for other in compare_against:
        x, y, n = build_pooled_paired_arrays(records, baseline_model, other, metric)
        if x is None:
            print(f"  ⚠ {baseline_model} vs {other} ({metric}): only {n} matched "
                  f"pairs (need >=10) — skipping.")
            continue
        mean_diff = float((x - y).mean())
        d = cohen_d(x, y)
        wt = wilcoxon_test(x, y, alternative="less")
        tt = paired_ttest(x, y)
        row_data.append((other, n, mean_diff, d, wt["p_value"], tt["p_value"]))
        p_wilcox.append(wt["p_value"])
        p_ttest.append(tt["p_value"])

    if not row_data:
        return []

    bonf_w = bonferroni_correction(p_wilcox)
    bonf_t = bonferroni_correction(p_ttest)
    for i, (other, n, mean_diff, d, pw, pt) in enumerate(row_data):
        achieved = (mean_diff < 0 and abs(d) >= 0.2
                    and bonf_w[i] < 0.05 and bonf_t[i] < 0.05)
        rows.append({
            "comparison":      f"{baseline_model} vs {other}",
            "n_pairs":         n,
            "mean_diff_km":    mean_diff,
            "cohens_d":        d,
            "cohens_d_interp": interpret_cohens_d(d),
            "wilcoxon_p":      pw,
            "wilcoxon_p_bonf": bonf_w[i],
            "ttest_p":         pt,
            "ttest_p_bonf":    bonf_t[i],
            "fm_achieves":     achieved,
        })
    return rows


def print_significance_table(rows: List[Dict], metric: str, baseline_model: str):
    print(f"\n  {'='*128}")
    print(f"  TABLE 2 — POOLED SIGNIFICANCE (3-seed pooled paired test) for "
          f"{metric.upper()} ({baseline_model} vs baselines)")
    print(f"  {'='*128}")
    print(f"  {'Comparison':<24} {'n (pairs)':>10} {'Mean diff':>12} "
          f"{'Cohen d':>9} {'Wilcoxon p':>12} {'Wilcoxon p(Bonf.)':>18} "
          f"{'t-test p':>12} {'t-test p(Bonf.)':>16}  {'FM achieves?':>13}")
    print(f"  {'-'*128}")
    for r in rows:
        mark = "✓ YES" if r["fm_achieves"] else "✗ no"
        print(f"  {r['comparison']:<24} {r['n_pairs']:>10} "
              f"{r['mean_diff_km']:>12.4f} {r['cohens_d']:>9.4f} "
              f"{r['wilcoxon_p']:>12.2E} {r['wilcoxon_p_bonf']:>18.2E} "
              f"{r['ttest_p']:>12.2E} {r['ttest_p_bonf']:>16.2E}  {mark:>13}")
    print(f"  {'='*128}")
    print(f"  'FM achieves' requires ALL FOUR: mean_diff<0, |d|>=0.2, "
          f"Wilcoxon p(Bonf.)<0.05, t-test p(Bonf.)<0.05.\n")


def print_significance_table_latex(rows: List[Dict], metric: str, baseline_model: str):
    print(r"  \begin{table}")
    print(f"  \\caption{{Pooled 3-seed significance tests for {metric.upper()} "
          f"({baseline_model} vs RNN/GRU/LSTM/ST-Trans).}}")
    print(r"  \begin{tabular}{lrrrrrrr}")
    print(r"  \hline")
    print(r"  Comparison & n (pairs) & Mean diff (km) & Cohen's d & "
          r"Wilcoxon $p$ & Wilcoxon $p$ (Bonf.) & $t$-test $p$ & $t$-test $p$ (Bonf.) \\")
    print(r"  \hline")
    for r in rows:
        print(f"  {r['comparison']} & {r['n_pairs']} & "
              f"{r['mean_diff_km']:.4f} & {r['cohens_d']:.4f} & "
              f"{r['wilcoxon_p']:.2E} & {r['wilcoxon_p_bonf']:.2E} & "
              f"{r['ttest_p']:.2E} & {r['ttest_p_bonf']:.2E} \\\\")
    print(r"  \hline")
    print(r"  \end{tabular}")
    print(r"  \end{table}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Table 3: per-horizon table (FM vs strongest baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_per_horizon_table(records: List[Dict], models: List[str],
                             zero_indexed: bool = False) -> List[Dict]:
    """
    TABLE 3 (mới) — mean±std ADE/ATE/CTE across seeds, cho TỪNG model,
    TÁCH RIÊNG theo từng horizon (6h/12h/24h/48h/72h). Giống hệt cách
    Table 1 tính mean±std (nhóm theo seed trước, rồi mean/std của các
    seed-mean — không phải mean±std của raw per-record errors, để không
    lẫn variance storm-to-storm với variance seed-to-seed) — chỉ khác ở
    chỗ Table 1 gộp toàn bộ horizon lại, còn bảng này tách riêng từng
    horizon để thấy độ ổn định của mỗi model đổi thế nào theo lead time.

    Thay thế bản cũ (chỉ so FM vs 1 baseline mạnh nhất) — giờ show ĐỦ
    mọi model đã evaluate, không giới hạn số lượng so sánh.

    ATE/CTE là None ở lead_time=1 (6h, xem evaluate_multi_model.py's
    docstring) — bị lọc ra trước khi tính mean/std, is not None cho mỗi
    (model, seed, horizon, metric) riêng biệt.
    """
    offset = -1 if zero_indexed else 0
    rows = []
    for h, lt in HORIZON_LEAD_TIMES.items():
        lt_key = lt + offset
        row = {"horizon": h}
        for model in models:
            by_seed = defaultdict(lambda: {"ade": [], "ate": [], "cte": []})
            for r in records:
                if r["model"] != model or r.get("lead_time") != lt_key:
                    continue
                s = r.get("seed", "unknown")
                for m in ("ade", "ate", "cte"):
                    if r.get(m) is not None:
                        by_seed[s][m].append(r[m])

            seed_means = {"ade": [], "ate": [], "cte": []}
            n_seeds = 0
            for seed, vals in sorted(by_seed.items()):
                n_seeds += 1
                for m in ("ade", "ate", "cte"):
                    if vals[m]:
                        seed_means[m].append(float(np.mean(vals[m])))

            for m in ("ade", "ate", "cte"):
                vals = seed_means[m]
                row[f"{model}_{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
                row[f"{model}_{m}_std"]  = float(np.std(vals))  if vals else float("nan")
            row[f"{model}_n_seeds"] = n_seeds
        rows.append(row)
    return rows


def print_per_horizon_table(rows: List[Dict], models: List[str], metric: str = "ade"):
    """In bảng cho 1 metric tại 1 thời điểm (gọi 3 lần cho ade/ate/cte nếu cần)."""
    metric_upper = metric.upper()
    col_w = 20
    print(f"\n  {'='*(12 + col_w * len(models))}")
    print(f"  TABLE 3 — PER-HORIZON {metric_upper} (mean±std across seeds, mọi model)")
    print(f"  {'='*(12 + col_w * len(models))}")
    header = f"  {'Horizon':<10}" + "".join(f"{m:>{col_w}}" for m in models)
    print(header)
    print(f"  {'-'*(12 + col_w * len(models))}")
    any_missing = False
    for r in rows:
        line = f"  {r['horizon']:<10}"
        for model in models:
            mean = r.get(f"{model}_{metric}_mean", float("nan"))
            std  = r.get(f"{model}_{metric}_std", float("nan"))
            n    = r.get(f"{model}_n_seeds", 0)
            if n == 0:
                any_missing = True
            cell = f"{mean:.1f}±{std:.1f}" if n > 0 else "n/a"
            line += f"{cell:>{col_w}}"
        print(line)
    print(f"  {'='*(12 + col_w * len(models))}")
    if any_missing:
        print(f"  ⚠ Một số model/horizon không có dữ liệu (n_seeds=0) — "
              f"kiểm tra lại checkpoint đã evaluate đủ seed cho model đó chưa, "
              f"hoặc metric={metric} không định nghĩa ở horizon 6h (đúng với "
              f"ate/cte, xem evaluate_multi_model.py's docstring).\n")
    else:
        print()




# ─────────────────────────────────────────────────────────────────────────────
#  PLOTS (từ generate_comparison_plots.py) — gộp thẳng vào đây, không import chéo
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def _present_models(records, order=MODEL_PLOT_ORDER):
    present = set(r["model"] for r in records)
    return [m for m in order if m in present] + \
           [m for m in sorted(present) if m not in order]


# ─────────────────────────────────────────────────────────────────────────────
#  Fig.3/8-style: error vs lead time, one line per model, 3 metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_vs_leadtime(records: List[Dict], output_dir: str,
                            metrics=("ade", "ate", "cte")):
    """
    One figure per metric: mean error at each 6h lead-time step,
    one colored line per model. Matches the reference Fig.3/8 layout
    (mirrored here as separate single-panel figures rather than one
    combined 2x3 grid, so each can also be dropped individually into
    the paper).
    """
    models = _present_models(records)
    lead_times = sorted(set(r["lead_time"] for r in records))
    saved = []

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 5))
        for model in models:
            means = []
            for lt in lead_times:
                # [FIX] ate/cte là None ở lead_time=1 (6h) — lọc trước
                # khi np.mean, tránh TypeError/NaN âm thầm khi None lẫn
                # trong list truyền vào np.mean.
                vals = [r[metric] for r in records
                        if r["model"] == model and r["lead_time"] == lt
                        and r.get(metric) is not None]
                means.append(np.mean(vals) if vals else np.nan)
            hours = [lt * 6 for lt in lead_times]
            ax.plot(hours, means, "o-", color=MODEL_COLORS.get(model, "#333"),
                    label=model, linewidth=1.8, markersize=4)

        ax.set_xlabel("Forecast Lead Time (hours)", fontsize=10)
        ax.set_ylabel(f"{metric.upper()} Error (km)", fontsize=10)
        ax.set_title(f"{metric.upper()} vs Forecast Lead Time", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        out = os.path.join(output_dir, f"error_vs_leadtime_{metric}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        saved.append(out)
        print(f"  Saved → {out}")
    return saved


def plot_error_vs_leadtime_grid(records: List[Dict], output_dir: str):
    """Combined 1x3 grid (ADE/ATE/CTE side by side) — single figure for the paper."""
    models = _present_models(records)
    lead_times = sorted(set(r["lead_time"] for r in records))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric in zip(axes, ["ade", "ate", "cte"]):
        for model in models:
            means = []
            for lt in lead_times:
                vals = [r[metric] for r in records
                        if r["model"] == model and r["lead_time"] == lt
                        and r.get(metric) is not None]
                means.append(np.mean(vals) if vals else np.nan)
            hours = [lt * 6 for lt in lead_times]
            ax.plot(hours, means, "o-", color=MODEL_COLORS.get(model, "#333"),
                    label=model, linewidth=1.8, markersize=4)
        ax.set_xlabel("Forecast Lead Time (h)", fontsize=10)
        ax.set_ylabel(f"{metric.upper()} (km)", fontsize=10)
        ax.set_title(metric.upper(), fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
    axes[0].legend(fontsize=9, framealpha=0.9)
    plt.suptitle("Track Forecast Errors Across Lead Time", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "error_vs_leadtime_grid.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Fig.4/5-style: boxplot of error distribution per model
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_boxplots(records: List[Dict], output_dir: str,
                         metrics=("ade", "ate", "cte")):
    """
    One figure, 1x3 subplots (ADE/CTE/ATE), boxplot of ALL per-record
    errors (pooled over storm/window/lead_time/seed) per model — matches
    Fig.4/5's "Distribution of forecast errors" layout.
    """
    models = _present_models(records)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    metric_titles = {"ade": "Direct Position Error", "ate": "Along-Track Error",
                     "cte": "Cross-Track Error"}

    for ax, metric in zip(axes, metrics):
        data = [[r[metric] for r in records
                 if r["model"] == m and r.get(metric) is not None] for m in models]
        colors = [MODEL_COLORS.get(m, "#888") for m in models]
        bp = ax.boxplot(data, tick_labels=models, patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=2, alpha=0.4))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_title(metric_titles.get(metric, metric.upper()), fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{metric.upper()} (km)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Distribution of Forecast Errors", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "error_boxplots.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Per-horizon boxplots (24h/48h/72h) — extra detail beyond the reference figs
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplot_by_horizon(records: List[Dict], output_dir: str,
                             horizons=("24h", "48h", "72h"), metric="ade"):
    """
    [EXTRA, not in reference figs] Boxplot of ADE per model, SPLIT by
    horizon (24h/48h/72h) rather than pooled across all lead times —
    shows whether a model's relative advantage/variance changes with
    forecast range, which the pooled Fig.4/5-style boxplot cannot show.
    """
    models = _present_models(records)
    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 5), sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    for ax, hz in zip(axes, horizons):
        lt = HORIZON_LEAD_TIMES_FULL.get(hz)
        data = [[r[metric] for r in records
                 if r["model"] == m and r["lead_time"] == lt
                 and r.get(metric) is not None]
                for m in models]
        colors = [MODEL_COLORS.get(m, "#888") for m in models]
        bp = ax.boxplot(data, tick_labels=models, patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=2, alpha=0.4))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_title(f"{metric.upper()} @ {hz}", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    axes[0].set_ylabel(f"{metric.upper()} (km)", fontsize=10)

    plt.suptitle(f"{metric.upper()} Distribution by Horizon", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, f"boxplot_by_horizon_{metric}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  [MỚI] Violin plot — chi tiết hơn boxplot, thấy được hình dạng phân phối
#  (đa đỉnh, lệch...) mà boxplot không thể hiện được.
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_violin(records: List[Dict], output_dir: str,
                       metrics=("ade", "ate", "cte")):
    """
    [MỚI, FM-specific] Violin plot phân phối lỗi theo model — bổ sung
    cho plot_error_boxplots(). Boxplot chỉ cho biết median/IQR/outlier;
    violin cho thấy CẢ HÌNH DẠNG phân phối (đa đỉnh/lệch/độ rộng) — quan
    trọng khi argue rằng FM's error distribution không chỉ có mean/median
    thấp hơn mà còn ÍT ĐUÔI DÀI hơn (ít trường hợp dự báo cực tệ), điều
    boxplot dễ bỏ sót nếu chỉ nhìn median.
    """
    models = _present_models(records)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    metric_titles = {"ade": "Direct Position Error", "ate": "Along-Track Error",
                     "cte": "Cross-Track Error"}

    for ax, metric in zip(axes, metrics):
        data = [[r[metric] for r in records
                 if r["model"] == m and r.get(metric) is not None] for m in models]
        # violinplot lỗi nếu 1 nhóm rỗng hoặc toàn giá trị giống hệt nhau
        # (variance=0) — lọc trước để không crash cả figure vì 1 model lỗi.
        valid_idx = [i for i, d in enumerate(data) if len(d) >= 2 and np.std(d) > 0]
        if not valid_idx:
            print(f"  ⚠ plot_error_violin: không đủ dữ liệu hợp lệ cho metric={metric}, skip")
            continue
        valid_data = [data[i] for i in valid_idx]
        valid_models = [models[i] for i in valid_idx]
        colors = [MODEL_COLORS.get(m, "#888") for m in valid_models]

        parts = ax.violinplot(valid_data, showmeans=True, showmedians=True)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.55)
            pc.set_edgecolor("black")
            pc.set_linewidth(0.6)
        for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_edgecolor("black")
                parts[key].set_linewidth(0.8)

        ax.set_xticks(range(1, len(valid_models) + 1))
        ax.set_xticklabels(valid_models)
        ax.set_title(metric_titles.get(metric, metric.upper()), fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{metric.upper()} (km)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Error Distribution Shape (Violin Plot)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "error_violin.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  [MỚI] Scatter: obs_speed (tốc độ bão quan sát) vs lỗi dự báo — kiểm
#  tra model có yếu đi rõ rệt với bão di chuyển nhanh không (motivation
#  gốc cho speed_correction/speed-calib trong kiến trúc FM).
# ─────────────────────────────────────────────────────────────────────────────

def plot_speed_vs_error(records: List[Dict], output_dir: str, metric="ade"):
    """
    [MỚI] Scatter obs_speed (km/h, tốc độ di chuyển bão quan sát được
    trước thời điểm dự báo) vs lỗi dự báo, 1 panel mỗi model, kèm
    đường hồi quy tuyến tính đơn giản để thấy xu hướng. Cần field
    "obs_speed" trong records (evaluate_multi_model.py đã ghi sẵn).
    Đây là cơ sở thực nghiệm trực tiếp cho lý do kiến trúc FM có riêng
    cơ chế speed_correction_logits — nếu lỗi tăng rõ theo obs_speed ở
    baseline nhưng phẳng hơn ở FM, đó là bằng chứng trực quan cho việc
    speed-calibration có tác dụng.
    """
    models = _present_models(records)
    has_speed = any(r.get("obs_speed") is not None for r in records)
    if not has_speed:
        print(f"  ⚠ plot_speed_vs_error: records không có field 'obs_speed', skip")
        return None

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        xs = [r["obs_speed"] for r in records
              if r["model"] == model and r.get("obs_speed") is not None
              and r.get(metric) is not None]
        ys = [r[metric] for r in records
              if r["model"] == model and r.get("obs_speed") is not None
              and r.get(metric) is not None]
        color = MODEL_COLORS.get(model, "#888")
        if xs:
            ax.scatter(xs, ys, s=6, alpha=0.25, color=color)
            if len(xs) >= 2 and np.std(xs) > 0:
                z = np.polyfit(xs, ys, 1)
                xline = np.linspace(min(xs), max(xs), 50)
                ax.plot(xline, np.poly1d(z)(xline), "-", color="black", linewidth=1.5)
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_xlabel("Observed storm speed (km/h)", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
    axes[0].set_ylabel(f"{metric.upper()} (km)", fontsize=10)

    plt.suptitle(f"{metric.upper()} vs Observed Storm Speed", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, f"speed_vs_{metric}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  [MỚI] Trực quan hoá thống kê: histogram phân phối (FM - baseline) diff,
#  minh hoạ trực quan cho Table 2's Wilcoxon/t-test/Cohen's d.
# ─────────────────────────────────────────────────────────────────────────────

def plot_significance_diff_hist(records: List[Dict], baseline_model: str,
                                  compare_against: List[str], metric: str,
                                  output_dir: str):
    """
    [MỚI] Với mỗi so sánh (baseline_model vs other), vẽ histogram phân
    phối (baseline_model - other) trên đúng các cặp ĐÃ GHÉP giống hệt
    cách build_significance_table() ghép (cùng seed/storm/window/
    lead_time) — không phải diff của 2 phân phối marginal độc lập.
    Đường thẳng đứng đỏ đánh dấu diff=0 (không khác biệt); nếu phần lớn
    khối histogram nằm bên trái 0, đó là minh chứng trực quan cho
    "baseline_model thắng" khớp với mean_diff<0 trong Table 2.

    Dùng lại build_pooled_paired_arrays() (không viết lại logic ghép
    cặp) để đảm bảo con số trên hình KHỚP CHÍNH XÁC với Table 2, không
    lệch do 2 cách ghép cặp khác nhau.
    """
    others = [m for m in compare_against if m != baseline_model]
    if not others:
        return None
    n_panels = len(others)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    saved_any = False
    for ax, other in zip(axes, others):
        x, y, n = build_pooled_paired_arrays(records, baseline_model, other, metric)
        if x is None:
            ax.text(0.5, 0.5, f"Không đủ cặp ghép\n(n={n} < 10)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{baseline_model} vs {other}", fontsize=11, fontweight="bold")
            continue
        diff = x - y
        ax.hist(diff, bins=40, color=MODEL_COLORS.get(baseline_model, "#D62728"),
                alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="diff = 0")
        ax.axvline(diff.mean(), color="black", linestyle="-", linewidth=1.5,
                   label=f"mean diff = {diff.mean():.2f}")
        ax.set_title(f"{baseline_model} vs {other} (n={n})", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{metric.upper()} diff ({baseline_model} − {other}) [km]", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        saved_any = True

    if not saved_any:
        plt.close()
        print(f"  ⚠ plot_significance_diff_hist: không có so sánh nào đủ dữ liệu cho metric={metric}")
        return None

    axes[0].set_ylabel("Số lượng cặp (count)", fontsize=10)
    plt.suptitle(f"Phân phối sai khác theo cặp — {metric.upper()} "
                f"({baseline_model} vs baseline khác)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, f"significance_diff_hist_{metric}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  [MỚI] Loss-comparison-style (giống Fig.6 paper tham chiếu): so sánh
#  ADE/ATE/CTE giữa các model theo lead time, dạng lưới 2x3 hoặc 1x3
#  tách theo split — ở đây ta chỉ có 1 split (test), nên rút gọn còn
#  1x3, khác plot_error_vs_leadtime_grid() ở chỗ trục y KHÔNG share và
#  có thêm vùng tô bóng ±std giữa các seed (nếu có multi-seed).
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_vs_leadtime_with_band(records: List[Dict], output_dir: str):
    """
    [MỚI] Giống plot_error_vs_leadtime_grid() nhưng có thêm dải bóng mờ
    ±1 std (tính qua CÁC SEED, không phải qua storm) quanh mỗi đường —
    trực quan hoá cùng lúc cả xu hướng theo lead-time (đường) LẪN độ ổn
    định giữa các seed (dải bóng), gần với phong cách Fig.6 của paper
    tham chiếu (nhiều đường + error band) hơn bản line-only hiện có.
    """
    models = _present_models(records)
    lead_times = sorted(set(r["lead_time"] for r in records))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric in zip(axes, ["ade", "ate", "cte"]):
        for model in models:
            means, stds = [], []
            for lt in lead_times:
                by_seed = defaultdict(list)
                for r in records:
                    if r["model"] == model and r["lead_time"] == lt and r.get(metric) is not None:
                        by_seed[r.get("seed", "unknown")].append(r[metric])
                seed_means = [np.mean(v) for v in by_seed.values() if v]
                means.append(np.mean(seed_means) if seed_means else np.nan)
                stds.append(np.std(seed_means) if len(seed_means) > 1 else 0.0)
            means = np.array(means); stds = np.array(stds)
            hours = np.array([lt * 6 for lt in lead_times])
            color = MODEL_COLORS.get(model, "#333")
            ax.plot(hours, means, "o-", color=color, label=model, linewidth=1.8, markersize=4)
            ax.fill_between(hours, means - stds, means + stds, color=color, alpha=0.15)
        ax.set_xlabel("Forecast Lead Time (h)", fontsize=10)
        ax.set_ylabel(f"{metric.upper()} (km)", fontsize=10)
        ax.set_title(metric.upper(), fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
    axes[0].legend(fontsize=9, framealpha=0.9)
    plt.suptitle("Track Forecast Errors Across Lead Time (±1 std qua seed)",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "error_vs_leadtime_band.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  FM-only: seed variance visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_seed_variance(records: List[Dict], output_dir: str, metric="ade"):
    """
    [EXTRA, FM-specific] Bar chart: mean±std ADE across seeds, per model.
    Directly visualizes what generate_paper_table.py's Table 1 reports
    as numbers — useful as a figure showing FM's seed-to-seed stability
    relative to the baselines (part of arguing FM's architecture is not
    just accurate but also STABLE across random initialization).
    """
    models = _present_models(records)
    means, stds = [], []
    for model in models:
        by_seed = defaultdict(list)
        for r in records:
            if r["model"] == model and r.get(metric) is not None:
                by_seed[r.get("seed", "unknown")].append(r[metric])
        # [FIX] r.get(metric) is not None lọc bỏ None (ate/cte tại
        # lead_time=1/6h) trước khi tích lũy theo seed; v (mỗi list
        # per-seed) giờ đảm bảo không rỗng do bộ lọc trên, nhưng vẫn
        # check `if v` để an toàn với model/seed không có bất kỳ record
        # hợp lệ nào (ví dụ seed đó chưa evaluate xong).
        seed_means = [np.mean(v) for v in by_seed.values() if v]
        means.append(np.mean(seed_means) if seed_means else np.nan)
        stds.append(np.std(seed_means) if seed_means else np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [MODEL_COLORS.get(m, "#888") for m in models]
    ax.bar(models, means, yerr=stds, capsize=5, color=colors, alpha=0.75,
          edgecolor="black", linewidth=0.6)
    ax.set_ylabel(f"{metric.upper()} (km)", fontsize=10)
    ax.set_title(f"{metric.upper()} Mean ± Std Across Seeds", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    out = os.path.join(output_dir, f"seed_variance_{metric}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  FM-only: ODE-steps (N) ablation — line + heatmap-style summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_ode_n_sweep(ode_sweep: Dict, output_dir: str):
    """
    [EXTRA, FM-specific] From ablation_runner.py's ode_steps_sweep()
    output: 4-panel figure (ADE, ATE, CTE, spread) vs N, all on the
    same x-axis, so the ADE/ATE/CTE-vs-spread TRADE-OFF (documented in
    the project's own notes: N=1->10 improves spread/CRPS but slightly
    worsens ADE/ATE/CTE, non-monotonic at N=3-4) is visible in one
    figure rather than requiring cross-referencing 2 separate tables.
    """
    ns = sorted(int(k) for k in ode_sweep.keys())
    ade = [ode_sweep[str(n)].get("ADE_mean", ode_sweep.get(n, {}).get("ADE_mean")) for n in ns]
    ate = [ode_sweep[str(n)].get("ATE_mean", ode_sweep.get(n, {}).get("ATE_mean")) for n in ns]
    cte = [ode_sweep[str(n)].get("CTE_mean", ode_sweep.get(n, {}).get("CTE_mean")) for n in ns]
    spread = [ode_sweep[str(n)].get("spread_mean", ode_sweep.get(n, {}).get("spread_mean")) for n in ns]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    titles = ["ADE (km)", "ATE (km)", "CTE (km)", "Ensemble Spread (km)"]
    series = [ade, ate, cte, spread]
    colors = ["#D62728", "#1F5FBF", "#2CA02C", "#9467BD"]

    for ax, title, vals, c in zip(axes, titles, series, colors):
        ax.plot(ns, vals, "o-", color=c, linewidth=2, markersize=6)
        ax.set_xlabel("ODE integration steps N", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xticks(ns)

    plt.suptitle("FM: Accuracy vs Ensemble Diversity Trade-off Across ODE Steps N",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "ode_n_sweep.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


def build_ode_n_table(ode_sweep: Dict) -> List[Dict]:
    """
    TABLE 4 (mới) — ADE/ATE/CTE/spread theo từng N (số bước tích phân
    ODE), đọc CÙNG file JSON với plot_ode_n_sweep() (từ
    ablation_runner.py --mode ode_steps --ode_steps_list ..., single-
    seed qua --checkpoint hoặc multi-seed qua --checkpoints — 2 luồng
    ra CÙNG SCHEMA "ADE_mean"/"ATE_mean"/"CTE_mean"/"spread_mean", chỉ
    khác multi-seed có thêm "n_seeds"/"*_std"/"by_lead_time" mà bảng
    này KHÔNG đọc — vẫn hiển thị đúng số liệu chính, không lỗi).
    Không phải mean±std theo seed CHO BẢNG NÀY (dù input JSON có thể đã
    là kết quả gộp qua seed) — bảng chỉ là số trực tiếp từ ode_sweep,
    kèm delta so với N nhỏ nhất (mốc tham chiếu) để thấy rõ trade-off
    ADE/ATE/CTE tăng nhẹ nhưng spread tăng mạnh khi N lớn.
    """
    ns = sorted(int(k) for k in ode_sweep.keys())
    ref = ode_sweep.get(str(ns[0]), ode_sweep.get(ns[0], {})) if ns else {}
    ref_ade = ref.get("ADE_mean", float("nan"))

    rows = []
    for n in ns:
        entry = ode_sweep.get(str(n), ode_sweep.get(n, {}))
        ade = entry.get("ADE_mean", float("nan"))
        rows.append({
            "n_steps":      n,
            "ade_mean":     ade,
            "ate_mean":     entry.get("ATE_mean", float("nan")),
            "cte_mean":     entry.get("CTE_mean", float("nan")),
            "spread_mean":  entry.get("spread_mean", float("nan")),
            "delta_ade_vs_n1": ade - ref_ade if not np.isnan(ade) and not np.isnan(ref_ade) else float("nan"),
            "time_s":       entry.get("time_s", float("nan")),
            "n_storms":     entry.get("n_storms", 0),
        })
    return rows


def print_ode_n_table(rows: List[Dict]):
    print(f"\n  {'='*100}")
    print(f"  TABLE 4 — ODE STEPS (N) SWEEP: ADE/ATE/CTE + Ensemble Spread")
    print(f"  {'='*100}")
    print(f"  {'N':>4} {'ADE(km)':>10} {'ATE(km)':>10} {'CTE(km)':>10} "
          f"{'Spread(km)':>12} {'dADE vs N=1':>13} {'Time(s)':>9} {'n':>6}")
    print(f"  {'-'*100}")
    for r in rows:
        print(f"  {r['n_steps']:>4} {r['ade_mean']:>10.2f} {r['ate_mean']:>10.2f} "
              f"{r['cte_mean']:>10.2f} {r['spread_mean']:>12.2f} "
              f"{r['delta_ade_vs_n1']:>+13.2f} {r['time_s']:>9.1f} {r['n_storms']:>6}")
    print(f"  {'='*100}")
    print(f"  Ghi chú: spread tăng mạnh khi N tăng thường đi kèm ADE/ATE/CTE nhích tệ nhẹ")
    print(f"  (trade-off đã quan sát trong log dự án: N=1->10, spread 4km->55km, CRPS -9.5%,")
    print(f"  ADE/ATE/CTE +1.7%/+1.9%/+7.2%). N=3-4 có thể KÉM HƠN cả N=1 và N=8-10 do lỗi")
    print(f"  rời rạc hóa Euler ở N thấp — không đơn điệu, đọc kỹ bảng thay vì suy diễn tuyến tính.\n")

# ─────────────────────────────────────────────────────────────────────────────
#  FM-only: ablation bar chart (loss components / architecture pieces)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation_bars(ablation_dir: str, output_dir: str, metric="ADE"):
    """
    Reads all <variant>.json files written by
    ablation_runner.py --mode eval_variant and draws a horizontal bar
    chart of `metric` per variant, sorted worst->best, with the "full"
    model highlighted — the standard "what breaks if you remove X" plot
    for an ablation table, visualized rather than just tabulated.
    """
    jsons = [f for f in os.listdir(ablation_dir) if f.endswith(".json")]
    if not jsons:
        print(f"  ⚠ No ablation JSONs found in {ablation_dir}")
        return None

    rows = []
    for jf in jsons:
        try:
            d = load_json(os.path.join(ablation_dir, jf))
        except Exception as e:
            print(f"  ⚠ Skipping {jf}: {e}")
            continue
        if metric not in d:
            continue
        name = d.get("variant", jf.replace(".json", ""))
        rows.append((name, d[metric]))

    if not rows:
        print(f"  ⚠ No variant JSONs contain metric={metric}")
        return None

    rows.sort(key=lambda r: r[1], reverse=True)  # worst (highest error) first
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = ["#D62728" if n == "full" else "#7F9BB5" for n in names]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(names))))
    ax.barh(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(f"{metric} (km)", fontsize=10)
    ax.set_title(f"Ablation Study — {metric} per Variant\n"
                "(red = full model, blue = ablated variant)",
                fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    out = os.path.join(output_dir, f"ablation_bars_{metric}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  FM-only: sigma sensitivity heatmap-style plot (from evaluate_full.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sigma_sensitivity(eval_full_json: Dict, output_dir: str):
    """
    From evaluate_full.py's --sigma_sensitivity block: line plot of
    ADE/CTE vs sigma_inference. Not a 2D heatmap (unlike the reference
    Fig.7, which sweeps 2 hyperparameters lambda_speed x lambda_accel) —
    the project's own sigma_sensitivity() only sweeps 1 hyperparameter
    (sigma_inference) at fixed training config, so a 1D line plot is the
    faithful representation; forcing a 2D heatmap here would require a
    2nd swept axis that doesn't exist in the current sigma_sensitivity()
    implementation.
    """
    block = eval_full_json.get("sigma_sensitivity")
    if not block:
        print("  ⚠ No 'sigma_sensitivity' block in eval_full_json — skip")
        return None

    sigmas = sorted(float(k) for k in block.keys())
    ade = [block[str(s)]["ADE"] if str(s) in block else block[s]["ADE"] for s in sigmas]
    cte = [block[str(s)]["CTE"] if str(s) in block else block[s]["CTE"] for s in sigmas]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(sigmas, ade, "o-", color="#D62728", label="ADE", linewidth=2)
    ax1.set_xlabel("sigma_inference", fontsize=10)
    ax1.set_ylabel("ADE (km)", color="#D62728", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#D62728")

    ax2 = ax1.twinx()
    ax2.plot(sigmas, cte, "s--", color="#1F5FBF", label="CTE", linewidth=2)
    ax2.set_ylabel("CTE (km)", color="#1F5FBF", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#1F5FBF")

    ax1.set_title("Sensitivity to sigma_inference", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    out = os.path.join(output_dir, "sigma_sensitivity.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


def plot_ensemble_size_ablation(eval_full_json: Dict, output_dir: str):
    """
    From evaluate_full.py's --ensemble_ablation block: ADE/ATE/CTE +
    spread vs K, 4-panel layout GIỐNG HỆT plot_ode_n_sweep() để 2 hình
    có thể đặt cạnh nhau trong paper, cho thấy trực quan sự khác biệt
    cốt lõi: spread ở đây (theo K) gần như PHẲNG, trong khi spread ở
    ode_n_sweep (theo N) tăng RÕ RỆT — bằng chứng trực tiếp cho việc N
    mới là tham số giải quyết co cụm, không phải K (xem giải thích đầy
    đủ trong ensemble_size_eval()'s docstring, evaluate_full.py).
    """
    block = eval_full_json.get("ensemble_ablation")
    if not block:
        print("  ⚠ No 'ensemble_ablation' block in eval_full_json — skip")
        return None

    ks = sorted(int(k) for k in block.keys())
    def _get(k, key):
        entry = block.get(str(k), block.get(k, {}))
        return entry.get(key, float("nan"))

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    titles = ["ADE (km)", "ATE (km)", "CTE (km)", "Ensemble Spread (km)"]
    keys   = ["ADE", "ATE", "CTE", "spread"]
    colors = ["#D62728", "#1F5FBF", "#2CA02C", "#9467BD"]

    for ax, title, key, c in zip(axes, titles, keys, colors):
        vals = [_get(k, key) for k in ks]
        ax.plot(ks, vals, "o-", color=c, linewidth=2, markersize=6)
        ax.set_xlabel("Ensemble size K", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xticks(ks)

    plt.suptitle("FM: Accuracy vs Ensemble Diversity Trade-off Across Ensemble Size K "
                "(so sánh với ODE N-sweep — spread ở đây nên PHẲNG hơn hẳn)",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "ensemble_size_ablation.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


def build_ensemble_k_table(eval_full_json: Dict) -> List[Dict]:
    """
    TABLE 5 (mới) — ADE/ATE/CTE/spread theo từng K (ensemble size),
    CÙNG CẤU TRÚC với Table 4 (ODE N-sweep) để so sánh trực tiếp 2 bảng
    cạnh nhau — đây là bằng chứng bằng số cho câu hỏi "K có xử lý được
    co cụm không" (câu trả lời: không, xem cột spread gần như không đổi
    theo K, khác hẳn Table 4 nơi spread tăng rõ theo N).
    """
    block = eval_full_json.get("ensemble_ablation")
    if not block:
        return []
    ks = sorted(int(k) for k in block.keys())
    ref = block.get(str(ks[0]), block.get(ks[0], {})) if ks else {}
    ref_ade = ref.get("ADE", float("nan"))

    rows = []
    for k in ks:
        entry = block.get(str(k), block.get(k, {}))
        ade = entry.get("ADE", float("nan"))
        rows.append({
            "K":            k,
            "ade_mean":     ade,
            "ate_mean":     entry.get("ATE", float("nan")),
            "cte_mean":     entry.get("CTE", float("nan")),
            "spread_mean":  entry.get("spread", float("nan")),
            "delta_ade_vs_k_min": ade - ref_ade if not np.isnan(ade) and not np.isnan(ref_ade) else float("nan"),
            "time_s":       entry.get("time_s", float("nan")),
            "n":            entry.get("n", 0),
        })
    return rows


def print_ensemble_k_table(rows: List[Dict]):
    print(f"\n  {'='*100}")
    print(f"  TABLE 5 — ENSEMBLE SIZE (K) SWEEP: ADE/ATE/CTE + Ensemble Spread")
    print(f"  {'='*100}")
    print(f"  {'K':>4} {'ADE(km)':>10} {'ATE(km)':>10} {'CTE(km)':>10} "
          f"{'Spread(km)':>12} {'dADE vs K_min':>14} {'Time(s)':>9} {'n':>6}")
    print(f"  {'-'*100}")
    for r in rows:
        print(f"  {r['K']:>4} {r['ade_mean']:>10.2f} {r['ate_mean']:>10.2f} "
              f"{r['cte_mean']:>10.2f} {r['spread_mean']:>12.2f} "
              f"{r['delta_ade_vs_k_min']:>+14.2f} {r['time_s']:>9.1f} {r['n']:>6}")
    print(f"  {'='*100}")
    print(f"  Ghi chú: spread ở bảng này nên gần như KHÔNG ĐỔI theo K (khác Table 4 — ODE")
    print(f"  N-sweep — nơi spread tăng rõ theo N). K chỉ giúp ước lượng phân phối mượt hơn")
    print(f"  trong vùng đã bị N quyết định trước, KHÔNG mở rộng spread. Nếu bảng này cho")
    print(f"  thấy spread tăng đáng kể theo K, cần xem lại giả thuyết trên bằng thực nghiệm.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  FM-only: per-storm CTE ranking (uses evaluate_full.py's per_storm_*.json)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_storm_cte(per_storm_json_path: str, output_dir: str, top_n=20):
    """
    [EXTRA] From evaluate_full.py's --per_storm output
    (per_storm_<split>_ep<N>.json): horizontal bar of CTE per storm,
    worst N first — visual version of print_per_storm_breakdown's table,
    useful for showing reviewers which specific storms drive aggregate
    CTE rather than just an averaged number.
    """
    d = load_json(per_storm_json_path)
    rows = sorted(d.items(), key=lambda kv: kv[1]["cte"], reverse=True)[:top_n]
    names = [r[0] for r in rows]
    ctes = [r[1]["cte"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(names))))
    ax.barh(names, ctes, color="#D62728", alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("CTE (km)", fontsize=10)
    ax.set_title(f"Worst {top_n} Storms by CTE", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    out = os.path.join(output_dir, "per_storm_cte_worst.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True,
                   help="Path to multi_model_<split>.json from evaluate_multi_model.py")
    p.add_argument("--baseline_model", default="FM")
    p.add_argument("--compare_against", nargs="+",
                   default=["ST-Trans", "RNN", "GRU", "LSTM"])
    p.add_argument("--metrics", nargs="+", default=["ade", "ate", "cte"],
                   choices=["ade", "ate", "cte"])
    p.add_argument("--output_dir", default="eval_multi",
                   help="Bảng (paper_tables.json) và hình (*.png) đều lưu vào đây")
    p.add_argument("--latex", action="store_true")
    p.add_argument("--lead_time_zero_indexed", action="store_true",
                   help="Pass if evaluate_multi_model.py's 'lead_time' field "
                        "is 0-indexed (0=6h) rather than 1-indexed (1=6h, "
                        "matching evaluate_full.py's HORIZONS convention).")
    p.add_argument("--tables_only", action="store_true",
                   help="Chỉ chạy 3 bảng, bỏ qua toàn bộ phần plot")
    p.add_argument("--plots_only", action="store_true",
                   help="Chỉ chạy plot, bỏ qua 3 bảng")
    # Optional plot inputs (không truyền thì tự bỏ qua đúng plot cần nó)
    p.add_argument("--ode_sweep", default=None,
                   help="ode_steps_sweep.json từ ablation_runner.py --mode ode_steps")
    p.add_argument("--ablation_dir", default=None,
                   help="Thư mục chứa <variant>.json từ ablation_runner.py --mode eval_variant")
    p.add_argument("--eval_full_json", default=None,
                   help="eval_<split>_ep<N>.json từ evaluate_full.py "
                        "(cho sigma_sensitivity / ensemble_ablation)")
    p.add_argument("--per_storm_json", default=None,
                   help="per_storm_<split>_ep<N>.json từ evaluate_full.py --per_storm")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = load_records(args.records)
    present_models = sorted(set(r["model"] for r in records))
    print(f"  Models present in records: {present_models}")
    if args.baseline_model not in present_models:
        print(f"  ❌ --baseline_model '{args.baseline_model}' not found. "
              f"Available: {present_models}")
        return

    # ═══════════════════════════════════════════════════════════════════
    #  PHẦN 1: 3 BẢNG (main / significance / per-horizon)
    # ═══════════════════════════════════════════════════════════════════
    if not args.plots_only:
        models_for_main = [m for m in ALL_MODELS if m in present_models]
        if not models_for_main:
            models_for_main = present_models

        main_rows = build_main_table(records, models_for_main)
        print_main_table(main_rows)
        if args.latex:
            print_main_table_latex(main_rows)

        compare_against = [m for m in args.compare_against if m in present_models]
        missing = set(args.compare_against) - set(compare_against)
        if missing:
            print(f"  ⚠ Requested comparison models not found, skipping: {sorted(missing)}")

        sig_results = {}
        for metric in args.metrics:
            rows = build_significance_table(records, args.baseline_model,
                                             compare_against, metric)
            if not rows:
                print(f"  ⚠ No valid comparisons for metric={metric}")
                continue
            print_significance_table(rows, metric, args.baseline_model)
            if args.latex:
                print_significance_table_latex(rows, metric, args.baseline_model)
            sig_results[metric] = rows

        horizon_rows = build_per_horizon_table(
            records, models_for_main, zero_indexed=args.lead_time_zero_indexed)
        for metric in args.metrics:
            print_per_horizon_table(horizon_rows, models_for_main, metric=metric)

        ode_n_rows = []
        if args.ode_sweep:
            ode_sweep_data = load_json(args.ode_sweep)
            ode_n_rows = build_ode_n_table(ode_sweep_data)
            print_ode_n_table(ode_n_rows)
        else:
            print(f"\n  ℹ TABLE 4 (ODE N-sweep) bị bỏ qua — thiếu --ode_sweep. "
                  f"Chạy trước: ablation_runner.py --mode ode_steps --output_dir "
                  f"<dir>, rồi truyền --ode_sweep <dir>/ode_steps_sweep.json vào lệnh này.\n")

        ensemble_k_rows = []
        if args.eval_full_json:
            eval_full_data = load_json(args.eval_full_json)
            ensemble_k_rows = build_ensemble_k_table(eval_full_data)
            if ensemble_k_rows:
                print_ensemble_k_table(ensemble_k_rows)
            else:
                print(f"\n  ⚠ TABLE 5 (Ensemble K-sweep) bị bỏ qua — file "
                      f"{args.eval_full_json} không có key 'ensemble_ablation'. "
                      f"Chạy trước: evaluate_full.py --ensemble_ablation (hoặc "
                      f"--checkpoints ... --ensemble_ablation cho multi-seed).\n")
        else:
            print(f"\n  ℹ TABLE 5 (Ensemble K-sweep) bị bỏ qua — thiếu "
                  f"--eval_full_json. Chạy trước: evaluate_full.py "
                  f"--ensemble_ablation, rồi truyền --eval_full_json <file .json> "
                  f"vào lệnh này.\n")

        out = {
            "main_table":        main_rows,
            "significance_table": sig_results,
            "per_horizon_table": {
                "models": models_for_main,
                "rows":   horizon_rows,
            },
            "ode_n_sweep_table":  ode_n_rows,
            "ensemble_k_sweep_table": ensemble_k_rows,
        }
        out_path = os.path.join(args.output_dir, "paper_tables.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved all 3 tables → {out_path}")

    # ═══════════════════════════════════════════════════════════════════
    #  PHẦN 2: PLOTS
    # ═══════════════════════════════════════════════════════════════════
    if not args.tables_only:
        print(f"\n  {'='*70}\n  Generating plots...\n  {'='*70}")
        saved = []
        saved += plot_error_vs_leadtime(records, args.output_dir)
        saved.append(plot_error_vs_leadtime_grid(records, args.output_dir))
        saved.append(plot_error_vs_leadtime_with_band(records, args.output_dir))
        saved.append(plot_error_boxplots(records, args.output_dir))
        saved.append(plot_error_violin(records, args.output_dir))
        saved.append(plot_boxplot_by_horizon(records, args.output_dir))
        saved.append(plot_speed_vs_error(records, args.output_dir, metric="ade"))
        saved.append(plot_seed_variance(records, args.output_dir, metric="ade"))
        saved.append(plot_seed_variance(records, args.output_dir, metric="ate"))
        saved.append(plot_seed_variance(records, args.output_dir, metric="cte"))

        # Trực quan hoá thống kê (Table 2) — diff histogram cho từng metric
        compare_against_plot = [m for m in args.compare_against if m in present_models]
        for metric in args.metrics:
            saved.append(plot_significance_diff_hist(
                records, args.baseline_model, compare_against_plot,
                metric, args.output_dir))

        if args.ode_sweep:
            ode_sweep = load_json(args.ode_sweep)
            saved.append(plot_ode_n_sweep(ode_sweep, args.output_dir))

        if args.ablation_dir:
            saved.append(plot_ablation_bars(args.ablation_dir, args.output_dir, metric="ADE"))
            saved.append(plot_ablation_bars(args.ablation_dir, args.output_dir, metric="CTE"))

        if args.eval_full_json:
            ej = load_json(args.eval_full_json)
            saved.append(plot_sigma_sensitivity(ej, args.output_dir))
            saved.append(plot_ensemble_size_ablation(ej, args.output_dir))

        if args.per_storm_json:
            saved.append(plot_per_storm_cte(args.per_storm_json, args.output_dir))

        saved = [s for s in saved if s]
        print(f"\n  Done — {len(saved)} figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()