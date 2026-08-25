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
  - error_vs_leadtime_{ade,ate,cte}.pdf + error_vs_leadtime_grid.pdf
  - error_boxplots.pdf, boxplot_by_horizon_ade.pdf
  - seed_variance_{ade,cte}.pdf
  - ode_n_sweep.pdf (nếu --ode_sweep)
  - ablation_bars_{ADE,CTE}.pdf (nếu --ablation_dir)
  - sigma_sensitivity.pdf, ensemble_size_ablation.pdf (nếu --eval_full_json)
  - per_storm_cte_worst.pdf (nếu --per_storm_json)

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

ALL_MODELS = ["FM", "ST-Trans", "LSTM", "GRU", "RNN", "MMSTN", "Phys-Diff", "TC-Diffuser"]

# ─── Constants dùng riêng cho phần PLOT (giữ tách khỏi HORIZON_LEAD_TIMES
# ở trên, vốn chỉ có 5 mốc cho bảng per-horizon — phần plot cần đủ 12 mốc
# 6h→72h cho boxplot_by_horizon và error_vs_leadtime) ───────────────────
MODEL_COLORS = {
    "FM":        "#D62728",
    "ST-Trans":  "#FF7F0E",
    "LSTM":      "#2CA02C",
    "GRU":       "#9467BD",
    "RNN":       "#8C564B",
    # MMSTN (Social-GAN style, Faiaz lineage), Phys-Diff (PIGA-augmented
    # latent DDPM), and TC-Diffuser (velocity-space DDPM, Trajectron++
    # lineage) — 3 additional generative baselines, all sharing PaperEncoder
    # (FNO3D+Mamba+Env_net) with the deterministic baselines above.
    "MMSTN":       "#17BECF",   # cyan — GAN family
    "Phys-Diff":   "#E377C2",   # pink — diffusion family (latent)
    "TC-Diffuser": "#BCBD22",   # olive — diffusion family (velocity-space)
}
# yếu->mạnh theo trực giác kiến trúc (RNN-family < Transformer < generative
# models); THỨ TỰ TƯƠNG ĐỐI GIỮA MMSTN/Phys-Diff/TC-Diffuser/FM (4 model
# generative) nên được XÁC NHẬN LẠI dựa trên ADE thực đo được sau khi train
# xong 3 seed — thứ tự dưới đây chỉ là placeholder hợp lý ban đầu, không
# phải khẳng định trước kết quả.
MODEL_PLOT_ORDER = ["RNN", "LSTM", "GRU", "ST-Trans", "MMSTN", "Phys-Diff", "TC-Diffuser", "FM"]

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

    [BỔ SUNG RMSE] Thêm cột RMSE (root-mean-square error) theo yêu cầu
    bổ sung vào Main Table -- KHÁC hẳn cách tính ADE/ATE/CTE (trung
    bình trị TUYỆT ĐỐI, "d.mean()" mỗi record đã là 1 khoảng cách
    haversine không âm) vì RMSE cần bình phương lỗi TRƯỚC KHI trung
    bình rồi mới lấy căn bậc hai -- 2 phép tính không giao hoán với
    nhau (mean(sqrt(x^2)) != sqrt(mean(x^2)) trừ phi mọi x giống hệt
    nhau), nên KHÔNG thể suy RMSE từ ade_mean đã có sẵn, phải tính
    riêng từ record thô (mỗi record["ade"] CHÍNH LÀ khoảng cách
    haversine tại 1 (storm, window, lead_time), giá trị không âm, nên
    RMSE = sqrt(mean(ade^2)) là định nghĩa đúng và duy nhất hợp lý ở
    đây -- không có "RMSE có dấu" vì haversine distance vốn luôn >= 0).
    Dùng đúng cùng logic seed-mean-rồi-mean/std như ADE/ATE/CTE để nhất
    quán trong toàn bảng: RMSE tính cho TỪNG SEED trước (sqrt(mean của
    ade^2 trong đúng seed đó)), rồi mean±std của 3 giá trị RMSE-theo-seed
    đó -- không phải sqrt(mean(ade^2)) gộp thẳng qua mọi seed cùng lúc
    (2 cách này cũng KHÁC nhau, cùng lý do non-giao-hoán ở trên).
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
        seed_rmse = []        # [NEW] RMSE tính riêng theo seed (overall)
        seed_rmse_final = []  # [NEW] RMSE tính riêng theo seed (chỉ 72h)
        n_seeds = 0
        for seed, vals in sorted(by_seed.items()):
            n_seeds += 1
            for m in ("ade", "ate", "cte"):
                if vals[m]:
                    seed_means[m].append(float(np.mean(vals[m])))
            if vals["ade"]:
                seed_rmse.append(float(np.sqrt(np.mean(np.square(vals["ade"])))))
        for seed, vals in sorted(by_seed_final.items()):
            for m in ("ade", "ate", "cte"):
                if vals[m]:
                    seed_means_final[m].append(float(np.mean(vals[m])))
            if vals["ade"]:
                seed_rmse_final.append(float(np.sqrt(np.mean(np.square(vals["ade"])))))

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

        # [NEW] RMSE columns, cùng quy ước tên field "{metric}_mean"/"_std"
        # như ADE/ATE/CTE ở trên để plot_ablation_bars() và các hàm khác
        # đọc field theo pattern có thể tái sử dụng nếu cần mà không phải
        # sửa logic đọc field riêng cho RMSE.
        row["rmse_mean"] = float(np.mean(seed_rmse)) if seed_rmse else float("nan")
        row["rmse_std"]  = float(np.std(seed_rmse))  if seed_rmse else float("nan")
        row["rmse_per_seed"] = seed_rmse
        row["rmse_final_mean"] = float(np.mean(seed_rmse_final)) if seed_rmse_final else float("nan")
        row["rmse_final_std"]  = float(np.std(seed_rmse_final))  if seed_rmse_final else float("nan")

        rows.append(row)
    return rows


def print_main_table(rows: List[Dict]):
    print(f"\n  {'='*140}")
    print(f"  TABLE 1 — MAIN RESULTS (mean ± std across seeds) — overall vs final step (72h)")
    print(f"  {'='*140}")
    print(f"  {'Model':<12} {'#seeds':>7} "
          f"{'ADE overall':>16} {'ADE@72h':>16} "
          f"{'RMSE overall':>16} {'RMSE@72h':>16} "
          f"{'ATE overall':>16} {'ATE@72h':>16} "
          f"{'CTE overall':>16} {'CTE@72h':>16}")
    print(f"  {'-'*140}")
    for r in rows:
        print(f"  {r['model']:<12} {r['n_seeds']:>7} "
              f"{r['ade_mean']:>9.2f}±{r['ade_std']:<5.2f} "
              f"{r['ade_final_mean']:>9.2f}±{r['ade_final_std']:<5.2f} "
              f"{r['rmse_mean']:>9.2f}±{r['rmse_std']:<5.2f} "
              f"{r['rmse_final_mean']:>9.2f}±{r['rmse_final_std']:<5.2f} "
              f"{r['ate_mean']:>9.2f}±{r['ate_std']:<5.2f} "
              f"{r['ate_final_mean']:>9.2f}±{r['ate_final_std']:<5.2f} "
              f"{r['cte_mean']:>9.2f}±{r['cte_std']:<5.2f} "
              f"{r['cte_final_mean']:>9.2f}±{r['cte_final_std']:<5.2f}")
    print(f"  {'='*140}")
    print(f"  'overall' = mean qua mọi lead_time (6h-72h) | '@72h' = chỉ final step "
          f"(horizon dự báo xa nhất, thường dùng làm tiêu chí so sánh chính)")
    print(f"  RMSE = sqrt(mean(ADE^2)), tính riêng theo TỪNG SEED rồi mean±std qua seed "
          f"(KHÔNG suy từ ADE_mean -- xem docstring build_main_table())\n")


def print_main_table_latex(rows: List[Dict]):
    print(r"  \begin{table}")
    print(r"  \caption{Main results: ADE/RMSE/ATE/CTE (km), mean $\pm$ std across seeds.}")
    print(r"  \begin{tabular}{lcccc}")
    print(r"  \hline")
    print(r"  Model & ADE (km) & RMSE (km) & ATE (km) & CTE (km) \\")
    print(r"  \hline")
    for r in rows:
        print(f"  {r['model']} & "
              f"{r['ade_mean']:.2f} $\\pm$ {r['ade_std']:.2f} & "
              f"{r['rmse_mean']:.2f} $\\pm$ {r['rmse_std']:.2f} & "
              f"{r['ate_mean']:.2f} $\\pm$ {r['ate_std']:.2f} & "
              f"{r['cte_mean']:.2f} $\\pm$ {r['cte_std']:.2f} \\\\")
    print(r"  \hline")
    print(r"  \end{tabular}")
    print(r"  \end{table}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Table 2: pooled significance table (FM vs each baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_storm_level_paired_arrays(records: List[Dict], model_a: str, model_b: str,
                                     metric: str):
    """
    [ADDED — fixes a pseudo-replication issue found during the XAI
    analysis of this same test set] build_pooled_paired_arrays above
    pairs at (seed, storm, window, lead_time) granularity, treating each
    sliding-window instance as an independent observation. Verified on
    the actual test-set data used by this project: 449 "sequences" (the
    granularity build_pooled_paired_arrays pairs at) are NOT 449
    independent storms — they are sliding windows drawn from only 12
    distinct named storms (e.g. RITA_1975, WAYNE_1986, ...), unevenly
    represented (from 9 to 70 windows per storm). Windows from the same
    storm share most of their observed trajectory (consecutive windows
    differ by only a few hours), so they are highly correlated, not
    independent draws — running a paired test at the window level
    inflates the effective sample size by roughly the average number of
    windows per storm, which inflates statistical significance in a way
    that does not reflect genuine evidence about how many DIFFERENT
    storms the improvement generalizes across.

    This function instead pairs at the STORM level: for each storm,
    average model_a's and model_b's per-window errors first (one number
    per storm per model, per seed), THEN builds paired arrays across
    storms x seeds. This makes the true sample size explicit (n storms
    x n seeds, not n windows), which is what a reviewer would actually
    want to know when judging how many independent tropical cyclone
    tracks a claimed improvement has been demonstrated on.

    Use this ALONGSIDE, not instead of, build_pooled_paired_arrays: the
    window-level test still has a legitimate purpose (it answers "does
    this hold at the level of individual forecasts", useful for e.g.
    operational deployment framing), but a paper claiming FM "performs
    significantly better" should report the storm-level test as the
    primary evidence for generalization, with the window-level result
    labeled explicitly as a secondary, higher-power-but-lower-independence
    check — not presented as if it were n=thousands of independent storms.
    """
    has_seed = any("seed" in r for r in records)
    if not has_seed:
        print("  ⚠ Records have no 'seed' field — storm-level pooling "
              "will not distinguish seeds.")

    by_storm_seed_a = defaultdict(list)
    by_storm_seed_b = defaultdict(list)
    for r in records:
        if r.get(metric) is None:
            continue
        key = (r.get("seed", "unknown"), r["storm"])
        if r["model"] == model_a:
            by_storm_seed_a[key].append(r[metric])
        elif r["model"] == model_b:
            by_storm_seed_b[key].append(r[metric])

    common_keys = sorted(set(by_storm_seed_a) & set(by_storm_seed_b), key=lambda k: str(k))
    n_unique_storms = len({k[1] for k in common_keys})
    if len(common_keys) < 5:
        return None, None, len(common_keys), n_unique_storms

    x = np.array([float(np.mean(by_storm_seed_a[k])) for k in common_keys])
    y = np.array([float(np.mean(by_storm_seed_b[k])) for k in common_keys])
    return x, y, len(common_keys), n_unique_storms


def leave_one_storm_out_check(records: List[Dict], model_a: str, model_b: str,
                               metric: str) -> Dict:
    """
    [ADDED, MANDATORY companion to any storm-level significance claim]
    Verified necessary on this project's real XAI attention data: an
    overall storm-level test can look significant (p<0.05) while being
    driven almost entirely by 1-2 storms — removing any single one of
    several storms individually was enough to lose significance in that
    earlier check. The same risk applies here: with only a handful of
    distinct storms, one unusually easy or hard storm for model_a vs
    model_b can dominate a pooled mean difference. This refits the
    storm-level paired test once per storm, with that storm held out,
    and reports how many of those n-1 fits still reach p<0.05 -- a
    result that only holds when EVERY storm is included, but flips as
    soon as any one storm is removed, should not be reported as a robust
    finding without this caveat stated explicitly.
    """
    by_storm_seed_a = defaultdict(list)
    by_storm_seed_b = defaultdict(list)
    for r in records:
        if r.get(metric) is None:
            continue
        key = (r.get("seed", "unknown"), r["storm"])
        if r["model"] == model_a:
            by_storm_seed_a[key].append(r[metric])
        elif r["model"] == model_b:
            by_storm_seed_b[key].append(r[metric])
    common_keys = sorted(set(by_storm_seed_a) & set(by_storm_seed_b), key=lambda k: str(k))
    all_storms = sorted({k[1] for k in common_keys})

    if len(all_storms) < 4:
        return {"n_storms": len(all_storms), "loo_results": [],
                "robust": None, "note": "Too few storms for a leave-one-out check."}

    loo_results = []
    for held_out in all_storms:
        keys_sub = [k for k in common_keys if k[1] != held_out]
        if len(keys_sub) < 5:
            continue
        x = np.array([float(np.mean(by_storm_seed_a[k])) for k in keys_sub])
        y = np.array([float(np.mean(by_storm_seed_b[k])) for k in keys_sub])
        wt = wilcoxon_test(x, y, alternative="less")
        loo_results.append({"storm_removed": held_out, "p_value": wt["p_value"]})

    n_flip_to_ns = sum(1 for r in loo_results
                        if not np.isnan(r["p_value"]) and r["p_value"] >= 0.05)
    return {
        "n_storms": len(all_storms),
        "loo_results": loo_results,
        "n_storms_whose_removal_loses_significance": n_flip_to_ns,
        "robust": n_flip_to_ns == 0,
    }


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

    [IMPORTANT CAVEAT, added after auditing the actual test set] This
    function's "n_pairs" is the number of matched WINDOWS, not
    independent storms. Verified: this project's test set has only 12
    distinct storms behind its ~449 sliding-window sequences. Treating
    n_pairs (which can run into the thousands once multiplied across
    lead_time and seed) as the effective sample size for significance
    testing substantially overstates statistical power. Use
    build_storm_level_paired_arrays for the PRIMARY significance claim
    in a paper; this window-level function's output is a secondary,
    higher-power-but-lower-independence check, and should be labeled as
    such wherever it is reported (see build_significance_table below,
    which now reports both).
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
    [UPDATED] Now reports BOTH tests per comparison:
      - Storm-level (PRIMARY): pairs at (seed, storm) granularity, one
        number per storm per seed. This is the honest sample size for
        generalization claims -- see build_storm_level_paired_arrays's
        docstring for why the window-level test alone overstates power.
      - Window-level (SECONDARY): the original pooled test at (seed,
        storm, window, lead_time) granularity, kept for completeness and
        because it answers a genuinely different, still-useful question
        (per-forecast-instance behavior) -- but no longer the sole basis
        for "FM achieves" status.

    FM "achieves" a comparison per the project's agreed threshold,
    evaluated on the STORM-LEVEL test (the honest-sample-size one), ALL
    FOUR of:
      mean_diff < 0  (FM lower error)
      |Cohen's d| >= 0.2  (at least small effect)
      Wilcoxon p (Bonferroni) < 0.05
      leave-one-storm-out robust (removing no single storm alone loses significance)
    The window-level Wilcoxon/t-test p-values are still computed and
    reported, but are explicitly labeled as the lower-independence
    secondary check, not part of the "achieves" criterion.
    """
    rows = []
    p_wilcox, p_ttest = [], []
    p_wilcox_storm = []
    row_data = []
    for other in compare_against:
        x, y, n = build_pooled_paired_arrays(records, baseline_model, other, metric)
        xs, ys, n_storm_pairs, n_unique_storms = build_storm_level_paired_arrays(
            records, baseline_model, other, metric)
        if x is None and xs is None:
            print(f"  ⚠ {baseline_model} vs {other} ({metric}): insufficient "
                  f"matched data at both window and storm level — skipping.")
            continue

        mean_diff = float((x - y).mean()) if x is not None else float("nan")
        d = cohen_d(x, y) if x is not None else float("nan")
        wt = wilcoxon_test(x, y, alternative="less") if x is not None else {"p_value": float("nan")}
        tt = paired_ttest(x, y) if x is not None else {"p_value": float("nan")}

        if xs is not None:
            mean_diff_storm = float((xs - ys).mean())
            d_storm = cohen_d(xs, ys)
            wt_storm = wilcoxon_test(xs, ys, alternative="less")
            loo = leave_one_storm_out_check(records, baseline_model, other, metric)
        else:
            mean_diff_storm, d_storm = float("nan"), float("nan")
            wt_storm = {"p_value": float("nan")}
            loo = {"robust": None, "n_storms": n_unique_storms if xs is None else 0,
                   "n_storms_whose_removal_loses_significance": None}

        row_data.append((other, n, mean_diff, d, wt["p_value"], tt["p_value"],
                          n_storm_pairs, n_unique_storms, mean_diff_storm, d_storm,
                          wt_storm["p_value"], loo))
        p_wilcox.append(wt["p_value"])
        p_ttest.append(tt["p_value"])
        p_wilcox_storm.append(wt_storm["p_value"])

    if not row_data:
        return []

    bonf_w = bonferroni_correction(p_wilcox)
    bonf_t = bonferroni_correction(p_ttest)
    bonf_w_storm = bonferroni_correction(p_wilcox_storm)
    for i, (other, n, mean_diff, d, pw, pt, n_storm_pairs, n_unique_storms,
            mean_diff_storm, d_storm, pw_storm, loo) in enumerate(row_data):
        storm_p_ok = (not np.isnan(bonf_w_storm[i])) and bonf_w_storm[i] < 0.05
        achieved = (not np.isnan(mean_diff_storm) and mean_diff_storm < 0
                    and not np.isnan(d_storm) and abs(d_storm) >= 0.2
                    and storm_p_ok
                    and loo.get("robust") is True)
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
            # [ADDED] Storm-level (primary) statistics.
            "n_storm_pairs":       n_storm_pairs,
            "n_unique_storms":     n_unique_storms,
            "mean_diff_storm_km":  mean_diff_storm,
            "cohens_d_storm":      d_storm,
            "wilcoxon_p_storm":    pw_storm,
            "wilcoxon_p_storm_bonf": bonf_w_storm[i],
            "loo_n_storms":        loo.get("n_storms"),
            "loo_n_flip_to_ns":    loo.get("n_storms_whose_removal_loses_significance"),
            "loo_robust":          loo.get("robust"),
            "fm_achieves":     achieved,
        })
    return rows


def print_significance_table(rows: List[Dict], metric: str, baseline_model: str):
    print(f"\n  {'='*140}")
    print(f"  TABLE 2a — STORM-LEVEL SIGNIFICANCE (PRIMARY test, honest sample "
          f"size) for {metric.upper()} ({baseline_model} vs baselines)")
    print(f"  {'='*140}")
    print(f"  {'Comparison':<24} {'n storms':>9} {'Mean diff':>12} "
          f"{'Cohen d':>9} {'Wilcoxon p':>12} {'p(Bonf.)':>12} "
          f"{'LOO robust?':>12}  {'FM achieves?':>13}")
    print(f"  {'-'*140}")
    for r in rows:
        mark = "✓ YES" if r["fm_achieves"] else "✗ no"
        loo = r.get("loo_robust")
        loo_str = ("robust" if loo is True else
                    f"NOT robust ({r.get('loo_n_flip_to_ns','?')}/{r.get('loo_n_storms','?')})"
                    if loo is False else "n/a")
        print(f"  {r['comparison']:<24} {r.get('n_unique_storms','?'):>9} "
              f"{r['mean_diff_storm_km']:>12.4f} {r['cohens_d_storm']:>9.4f} "
              f"{r['wilcoxon_p_storm']:>12.2E} {r['wilcoxon_p_storm_bonf']:>12.2E} "
              f"{loo_str:>12}  {mark:>13}")
    print(f"  {'='*140}")
    print(f"  'FM achieves' requires ALL FOUR (storm-level, honest sample size): "
          f"mean_diff<0, |d|>=0.2, Wilcoxon p(Bonf.)<0.05, AND leave-one-storm-"
          f"out robust (no single storm's removal alone loses significance).")
    print(f"  'LOO robust' = leave-one-storm-out check: 'NOT robust (k/n)' means "
          f"removing any one of k out of n storms individually was enough to "
          f"lose significance — a finding this fragile should be reported as "
          f"preliminary, not as a confirmed dataset-wide pattern.\n")

    print(f"  {'-'*140}")
    print(f"  TABLE 2b — WINDOW-LEVEL SIGNIFICANCE (SECONDARY, higher power / "
          f"lower independence -- see build_pooled_paired_arrays docstring)")
    print(f"  {'-'*140}")
    print(f"  {'Comparison':<24} {'n (windows)':>12} {'Mean diff':>12} "
          f"{'Cohen d':>9} {'Wilcoxon p':>12} {'Wilcoxon p(Bonf.)':>18} "
          f"{'t-test p':>12} {'t-test p(Bonf.)':>16}")
    print(f"  {'-'*140}")
    for r in rows:
        print(f"  {r['comparison']:<24} {r['n_pairs']:>12} "
              f"{r['mean_diff_km']:>12.4f} {r['cohens_d']:>9.4f} "
              f"{r['wilcoxon_p']:>12.2E} {r['wilcoxon_p_bonf']:>18.2E} "
              f"{r['ttest_p']:>12.2E} {r['ttest_p_bonf']:>16.2E}")
    print(f"  {'='*140}")
    print(f"  ⚠ Table 2b's 'n (windows)' overstates independence: sliding-window "
          f"instances from the same storm are correlated, not independent "
          f"observations. Report Table 2a as the primary significance claim in "
          f"the paper; Table 2b may be reported as a secondary, per-forecast-"
          f"instance check, explicitly labeled as such.\n")


def print_significance_table_latex(rows: List[Dict], metric: str, baseline_model: str):
    """
    [UPDATED] Now exports the STORM-LEVEL (primary, honest-sample-size)
    results as the main LaTeX table, with the LOO-robustness column
    included -- this is what should go in the paper body. A second,
    clearly-labeled secondary table with the window-level numbers
    follows, for an appendix/supplementary table if wanted, but should
    not be presented as the primary significance evidence (see
    build_storm_level_paired_arrays' docstring for the full rationale).
    """
    print(r"  \begin{table}")
    print(f"  \\caption{{Storm-level significance tests for {metric.upper()} "
          f"({baseline_model} vs baselines). Paired at the storm level "
          f"(one value per storm per seed) rather than per sliding-window "
          f"instance, to avoid overstating sample size from correlated "
          f"windows drawn from the same storm. LOO = leave-one-storm-out "
          f"robustness check.}}")
    print(r"  \begin{tabular}{lrrrrl}")
    print(r"  \hline")
    print(r"  Comparison & $n$ storms & Mean diff (km) & Cohen's $d$ & "
          r"Wilcoxon $p$ (Bonf.) & LOO robust \\")
    print(r"  \hline")
    for r in rows:
        loo = r.get("loo_robust")
        loo_str = ("Yes" if loo is True else
                   f"No ({r.get('loo_n_flip_to_ns','?')}/{r.get('loo_n_storms','?')})"
                   if loo is False else "n/a")
        print(f"  {r['comparison']} & {r.get('n_unique_storms','?')} & "
              f"{r['mean_diff_storm_km']:.2f} & {r['cohens_d_storm']:.3f} & "
              f"{r['wilcoxon_p_storm_bonf']:.2E} & {loo_str} \\\\")
    print(r"  \hline")
    print(r"  \end{tabular}")
    print(r"  \end{table}")
    print()

    print(r"  % --- Secondary table (window-level, appendix/supplementary only) ---")
    print(r"  \begin{table}")
    print(f"  \\caption{{Window-level significance tests for {metric.upper()} "
          f"({baseline_model} vs baselines), reported as a secondary, "
          f"higher-power-but-lower-independence check (sliding-window "
          f"instances from the same storm are correlated, not "
          f"independent) -- see Table~\\ref{{tab:storm-level}} for the "
          f"primary result.}}")
    print(r"  \begin{tabular}{lrrrrrrr}")
    print(r"  \hline")
    print(r"  Comparison & n (windows) & Mean diff (km) & Cohen's d & "
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

    [BỔ SUNG RMSE] Cùng lý do và cùng cách tính (sqrt(mean(ade^2)) theo
    TỪNG SEED rồi mean±std qua seed -- xem build_main_table()'s
    docstring cho giải thích đầy đủ vì sao không thể suy RMSE từ
    ade_mean) như Table 1, áp dụng riêng cho từng horizon ở đây. Chỉ
    tính RMSE cho ADE (không có "RMSE của ATE/CTE" -- ATE/CTE vốn đã có
    dấu (signed), bình phương rồi căn bậc hai sẽ đổi ý nghĩa thành
    "magnitude trung bình" chứ không còn là phần bổ sung cho MAE như
    RMSE-của-ADE, nên cố tình không thêm để tránh gây hiểu lầm).
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
            seed_rmse = []  # [NEW]
            n_seeds = 0
            for seed, vals in sorted(by_seed.items()):
                n_seeds += 1
                for m in ("ade", "ate", "cte"):
                    if vals[m]:
                        seed_means[m].append(float(np.mean(vals[m])))
                if vals["ade"]:
                    seed_rmse.append(float(np.sqrt(np.mean(np.square(vals["ade"])))))

            for m in ("ade", "ate", "cte"):
                vals = seed_means[m]
                row[f"{model}_{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
                row[f"{model}_{m}_std"]  = float(np.std(vals))  if vals else float("nan")
            row[f"{model}_rmse_mean"] = float(np.mean(seed_rmse)) if seed_rmse else float("nan")  # [NEW]
            row[f"{model}_rmse_std"]  = float(np.std(seed_rmse))  if seed_rmse else float("nan")  # [NEW]
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
    one colored line per model, WITH a shaded ±1 std-across-seeds band.

    [UPGRADED] Previously plotted only the mean line with no indication
    of uncertainty -- a reviewer cannot tell from a bare mean-only line
    whether two models' curves crossing or converging near 72h is a
    genuine pattern or within noise. The band is computed the same way
    Table 1's "±" is computed (per-seed mean first, then std OF THOSE
    SEED MEANS across seeds) rather than std of raw per-window errors,
    so it reflects seed-to-seed training variance, not storm-to-storm
    forecast difficulty -- consistent with how every other "±" in this
    report's tables is defined, and directly comparable to them.
    """
    models = _present_models(records)
    lead_times = sorted(set(r["lead_time"] for r in records))
    saved = []

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for model in models:
            means, stds = [], []
            for lt in lead_times:
                # Per-seed mean first (matches Table 1's convention),
                # THEN mean/std across those seed-means -- not raw
                # per-window std, which would conflate storm-to-storm
                # difficulty with seed-to-seed training variance.
                by_seed = defaultdict(list)
                for r in records:
                    if (r["model"] == model and r["lead_time"] == lt
                            and r.get(metric) is not None):
                        by_seed[r.get("seed", "unknown")].append(r[metric])
                seed_means = [float(np.mean(v)) for v in by_seed.values() if v]
                means.append(np.mean(seed_means) if seed_means else np.nan)
                stds.append(np.std(seed_means) if len(seed_means) > 1 else 0.0)
            hours = [lt * 6 for lt in lead_times]
            means_arr, stds_arr = np.array(means), np.array(stds)
            color = MODEL_COLORS.get(model, "#333")
            ax.plot(hours, means_arr, "o-", color=color,
                    label=model, linewidth=2.0, markersize=4.5, zorder=3)
            ax.fill_between(hours, means_arr - stds_arr, means_arr + stds_arr,
                             color=color, alpha=0.15, linewidth=0, zorder=1)

        ax.set_xlabel("Forecast Lead Time (hours)", fontsize=11)
        ax.set_ylabel(f"{metric.upper()} Error (km)", fontsize=11)
        ax.set_title(f"{metric.upper()} vs Forecast Lead Time "
                     f"(shaded: ±1 std across seeds)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out = os.path.join(output_dir, f"error_vs_leadtime_{metric}.pdf")
        plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        saved.append(out)
        print(f"  Saved → {out}")
    return saved


def plot_error_vs_leadtime_grid(records: List[Dict], output_dir: str):
    """
    Combined 1x3 grid (ADE/ATE/CTE side by side) — single figure for
    the paper. [UPGRADED] Now includes ±1 std-across-seeds shaded band,
    same convention as plot_error_vs_leadtime -- see that function's
    docstring for why this matters and how the band is computed.
    """
    models = _present_models(records)
    lead_times = sorted(set(r["lead_time"] for r in records))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    for ax, metric in zip(axes, ["ade", "ate", "cte"]):
        for model in models:
            means, stds = [], []
            for lt in lead_times:
                by_seed = defaultdict(list)
                for r in records:
                    if (r["model"] == model and r["lead_time"] == lt
                            and r.get(metric) is not None):
                        by_seed[r.get("seed", "unknown")].append(r[metric])
                seed_means = [float(np.mean(v)) for v in by_seed.values() if v]
                means.append(np.mean(seed_means) if seed_means else np.nan)
                stds.append(np.std(seed_means) if len(seed_means) > 1 else 0.0)
            hours = [lt * 6 for lt in lead_times]
            means_arr, stds_arr = np.array(means), np.array(stds)
            color = MODEL_COLORS.get(model, "#333")
            ax.plot(hours, means_arr, "o-", color=color,
                    label=model, linewidth=2.0, markersize=4, zorder=3)
            ax.fill_between(hours, means_arr - stds_arr, means_arr + stds_arr,
                             color=color, alpha=0.15, linewidth=0, zorder=1)
        ax.set_xlabel("Forecast Lead Time (h)", fontsize=10)
        ax.set_ylabel(f"{metric.upper()} (km)", fontsize=10)
        ax.set_title(metric.upper(), fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
    axes[0].legend(fontsize=9, framealpha=0.9)
    plt.suptitle("Track Forecast Errors Across Lead Time", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "error_vs_leadtime_grid.pdf")
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
    out = os.path.join(output_dir, "error_boxplots.pdf")
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

    [UPGRADED] The raw-window boxplot alone makes every model's box look
    nearly identical (medians overlapping visually) because a single
    heavy-tailed distribution's box-and-whisker summary is dominated by
    its bulk, not by the km-scale differences in the mean that actually
    separate models -- verified visually on this project's own
    boxplot_by_horizon_ade.pdf. Two additions make the real, smaller-
    magnitude difference visible without changing what data is plotted:
      1. A diamond marker for the MEAN (not just the median line already
         inside each box) -- ADE distributions here are right-skewed
         (long tail from hard storms), so mean and median diverge, and
         the paper's headline numbers (Table 1) are means, not medians;
         showing both makes the boxplot consistent with the table.
      2. Text annotation of each box's mean value directly above it, and
         the underlying n (matched-window count) below the axis --
         letting the reader read off the actual numbers instead of
         squinting at near-overlapping boxes for a visual impression
         that the raw distribution's scale makes almost impossible to
         judge by eye alone.
    """
    models = _present_models(records)
    fig, axes = plt.subplots(1, len(horizons), figsize=(6.5 * len(horizons), 5.5), sharey=True)
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
                        showmeans=True, meanprops=dict(marker="D", markerfacecolor="black",
                                                        markeredgecolor="black", markersize=6),
                        flierprops=dict(marker=".", markersize=2, alpha=0.3))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        # Annotate each box's mean value and n directly, so the reader
        # can read the actual numbers instead of relying on visually
        # near-identical box shapes at this data scale.
        ymax = ax.get_ylim()[1]
        for i, d in enumerate(data):
            if not d:
                continue
            mean_v = np.mean(d)
            ax.annotate(f"{mean_v:.0f}", xy=(i + 1, mean_v),
                        xytext=(i + 1, ymax * 0.92), fontsize=8, ha="center",
                        color="black", fontweight="bold")
            ax.annotate(f"n={len(d)}", xy=(i + 1, 0), xytext=(i + 1, -ymax * 0.14),
                        fontsize=7, ha="center", color="#555", annotation_clip=False)
        ax.set_title(f"{metric.upper()} @ {hz}", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    axes[0].set_ylabel(f"{metric.upper()} (km)", fontsize=10)

    plt.suptitle(f"{metric.upper()} Distribution by Horizon "
                 f"(◆ = mean, orange line = median; numbers above = mean, "
                 f"n = matched windows)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, f"boxplot_by_horizon_{metric}.pdf")
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
    out = os.path.join(output_dir, "error_violin.pdf")
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
    out = os.path.join(output_dir, f"speed_vs_{metric}.pdf")
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
    [UPGRADED] Previously plotted only the window-level paired
    difference (n up to ~9960 per comparison), giving the visual
    impression of a very large, highly powered sample. Verified on this
    project's actual test set: those ~9960 "pairs" come from only 12
    distinct storms via sliding windows -- window-level n overstates
    independence (see build_pooled_paired_arrays' docstring for the full
    audit). Each panel now shows TWO histograms stacked:
      (top, filled)   storm-level paired differences -- the honest
                       sample size (n = number of storms x seeds), the
                       PRIMARY evidence for a generalization claim.
      (bottom, outline) window-level paired differences -- the original
                       higher-n but lower-independence view, kept for
                       transparency but visually de-emphasized (outline
                       only, no fill) so it cannot be mistaken for
                       n-storms-worth of independent evidence.
    Both are built via the SAME pairing functions used in Table 2a/2b, so
    numbers on this figure match the table exactly.
    """
    others = [m for m in compare_against if m != baseline_model]
    if not others:
        return None
    n_panels = len(others)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    saved_any = False
    for ax, other in zip(axes, others):
        xs, ys, n_storm_pairs, n_unique = build_storm_level_paired_arrays(
            records, baseline_model, other, metric)
        x, y, n_window = build_pooled_paired_arrays(records, baseline_model, other, metric)

        if xs is None and x is None:
            ax.text(0.5, 0.5, "Insufficient matched data",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{baseline_model} vs {other}", fontsize=11, fontweight="bold")
            continue

        if xs is not None:
            diff_storm = xs - ys
            ax.hist(diff_storm, bins=min(15, max(5, len(diff_storm) // 2)),
                    color=MODEL_COLORS.get(baseline_model, "#D62728"),
                    alpha=0.75, edgecolor="black", linewidth=0.5,
                    label=f"storm-level (n={n_storm_pairs}, {n_unique} storms) — primary")
            ax.axvline(diff_storm.mean(), color="black", linestyle="-", linewidth=1.8,
                       label=f"storm-level mean = {diff_storm.mean():.1f}")

        if x is not None:
            # Rescale window-level histogram to the SAME y-axis density
            # as the storm-level one (via density=True + a visible-only
            # outline) so a reader cannot misread its much larger raw
            # count as "much stronger evidence" at a glance.
            ax2 = ax.twinx()
            ax2.hist(x - y, bins=40, histtype="step", color="#555",
                     linewidth=1.2, density=True,
                     label=f"window-level (n={n_window}) — secondary, correlated")
            ax2.set_yticks([])

        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="diff = 0")
        ax.set_title(f"{baseline_model} vs {other}", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{metric.upper()} diff ({baseline_model} − {other}) [km]", fontsize=9)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = (ax2.get_legend_handles_labels() if x is not None else ([], []))
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--")
        saved_any = True

    if not saved_any:
        plt.close()
        print(f"  ⚠ plot_significance_diff_hist: không có so sánh nào đủ dữ liệu cho metric={metric}")
        return None

    axes[0].set_ylabel("Storm-level count (filled bars, primary)", fontsize=9)
    plt.suptitle(f"Paired difference distribution — {metric.upper()} "
                f"({baseline_model} vs baselines)\n"
                f"Filled = storm-level (honest n), outline = window-level "
                f"(secondary, correlated samples)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, f"significance_diff_hist_{metric}.pdf")
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
    out = os.path.join(output_dir, "error_vs_leadtime_band.pdf")
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
    out = os.path.join(output_dir, f"seed_variance_{metric}.pdf")
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

    [UPGRADED] Previously each panel auto-scaled its y-axis tightly
    around the data range -- for ADE this meant a y-range of roughly
    219.8-225.3 km displayed across the panel's FULL height, visually
    exaggerating a <3km real-world difference into what looks like a
    dramatic swing. A reader scanning the figure without reading tick
    labels carefully could easily overstate how much this ablation
    matters. Two changes fix this without hiding any real information:
      1. Each panel's y-axis now starts at 0 (or a value close to 0 for
         spread, which is on a different natural scale) EXCEPT where an
         inset zoom is added (see #2), and reports the actual range in
         km directly on the panel via a text annotation, so the
         magnitude of the effect is stated in absolute terms regardless
         of axis scaling.
      2. A small inset (or the original tight-scaled view, retained
         alongside the zoomed-out main panel) shows the fine-grained
         shape of the curve for readers who DO want to see the
         non-monotonic N=3-4 dip -- without that detail being the ONLY
         view presented, which is what previously risked overstating
         the effect's practical size.
    """
    ns = sorted(int(k) for k in ode_sweep.keys())
    ade = [ode_sweep[str(n)].get("ADE_mean", ode_sweep.get(n, {}).get("ADE_mean")) for n in ns]
    ate = [ode_sweep[str(n)].get("ATE_mean", ode_sweep.get(n, {}).get("ATE_mean")) for n in ns]
    cte = [ode_sweep[str(n)].get("CTE_mean", ode_sweep.get(n, {}).get("CTE_mean")) for n in ns]
    spread = [ode_sweep[str(n)].get("spread_mean", ode_sweep.get(n, {}).get("spread_mean")) for n in ns]

    fig, axes = plt.subplots(2, 4, figsize=(22, 9),
                             gridspec_kw={"height_ratios": [1, 1]})
    titles = ["ADE (km)", "ATE (km)", "CTE (km)", "Ensemble Spread (km)"]
    series = [ade, ate, cte, spread]
    colors = ["#D62728", "#1F5FBF", "#2CA02C", "#9467BD"]

    for col, (title, vals, c) in enumerate(zip(titles, series, colors)):
        vals_arr = np.array([v for v in vals if v is not None], dtype=float)
        data_range = vals_arr.max() - vals_arr.min() if len(vals_arr) else 0.0

        # Top row: zoomed-out, y-axis anchored at 0 -- shows the TRUE
        # magnitude of the effect relative to the metric's absolute scale.
        ax_top = axes[0, col]
        ax_top.plot(ns, vals, "o-", color=c, linewidth=2, markersize=6)
        ymax_data = vals_arr.max() if len(vals_arr) else 1.0
        ax_top.set_ylim(0, ymax_data * 1.15)
        ax_top.set_title(f"{title} — full scale", fontsize=10, fontweight="bold")
        ax_top.set_xlabel("ODE integration steps N", fontsize=9)
        ax_top.set_ylabel(title, fontsize=9)
        ax_top.grid(True, alpha=0.3, linestyle="--")
        ax_top.set_xticks(ns)
        ax_top.annotate(f"range: {data_range:.2f} ({100*data_range/max(ymax_data,1e-9):.2f}% of max)",
                        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8,
                        va="top", color="#333")

        # Bottom row: original tight-scaled zoom -- shows the fine-
        # grained shape (e.g. the N=3-4 dip) for readers who want it,
        # clearly labeled as a zoomed view so it isn't mistaken for the
        # only/primary view.
        ax_bot = axes[1, col]
        ax_bot.plot(ns, vals, "o-", color=c, linewidth=2, markersize=6)
        ax_bot.set_title(f"{title} — zoomed (shape detail)", fontsize=10, fontweight="bold")
        ax_bot.set_xlabel("ODE integration steps N", fontsize=9)
        ax_bot.set_ylabel(title, fontsize=9)
        ax_bot.grid(True, alpha=0.3, linestyle="--")
        ax_bot.set_xticks(ns)

    plt.suptitle("FM: Accuracy vs Ensemble Diversity Trade-off Across ODE Steps N\n"
                "(top row: true scale from 0; bottom row: zoomed to show shape detail)",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "ode_n_sweep.pdf")
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
    out = os.path.join(output_dir, f"ablation_bars_{metric}.pdf")
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
    out = os.path.join(output_dir, "sigma_sensitivity.pdf")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


def plot_ensemble_size_ablation(eval_full_json: Dict, output_dir: str):
    """
    From evaluate_full.py's --ensemble_ablation block: ADE/ATE/CTE +
    spread vs K, layout matching plot_ode_n_sweep() so the two figures
    sit side by side in the paper, showing the key contrast: spread vs K
    here is nearly FLAT, while spread vs N in ode_n_sweep rises clearly
    — direct evidence that N, not K, is the parameter that resolves
    ensemble collapse (see ensemble_size_eval()'s docstring in
    evaluate_full.py for the full explanation).

    [UPGRADED] Same fix as plot_ode_n_sweep: previously each panel's
    y-axis auto-scaled tightly to the data range, visually exaggerating
    small real-world differences (e.g. the ADE panel's ~219.8-221.2 km
    range filled the full panel height, making a <1.5km effect look like
    a dramatic swing). Now uses the identical two-row layout as
    plot_ode_n_sweep (top: true scale from 0, with the actual range
    annotated in km; bottom: zoomed shape detail) -- keeping the two
    figures visually consistent AND making both honest about effect
    magnitude, which matters here specifically because the flat-vs-rising
    spread comparison between this figure and ode_n_sweep is the
    argument being made -- both need the same, honest scaling
    convention for that comparison to be trustworthy.
    """
    block = eval_full_json.get("ensemble_ablation")
    if not block:
        print("  ⚠ No 'ensemble_ablation' block in eval_full_json — skip")
        return None

    ks = sorted(int(k) for k in block.keys())
    def _get(k, key):
        entry = block.get(str(k), block.get(k, {}))
        return entry.get(key, float("nan"))

    fig, axes = plt.subplots(2, 4, figsize=(22, 9),
                             gridspec_kw={"height_ratios": [1, 1]})
    titles = ["ADE (km)", "ATE (km)", "CTE (km)", "Ensemble Spread (km)"]
    keys   = ["ADE", "ATE", "CTE", "spread"]
    colors = ["#D62728", "#1F5FBF", "#2CA02C", "#9467BD"]

    for col, (title, key, c) in enumerate(zip(titles, keys, colors)):
        vals = [_get(k, key) for k in ks]
        vals_arr = np.array([v for v in vals if not np.isnan(v)], dtype=float)
        data_range = vals_arr.max() - vals_arr.min() if len(vals_arr) else 0.0
        ymax_data = vals_arr.max() if len(vals_arr) else 1.0

        ax_top = axes[0, col]
        ax_top.plot(ks, vals, "o-", color=c, linewidth=2, markersize=6)
        ax_top.set_ylim(0, ymax_data * 1.15)
        ax_top.set_title(f"{title} — full scale", fontsize=10, fontweight="bold")
        ax_top.set_xlabel("Ensemble size K", fontsize=9)
        ax_top.set_ylabel(title, fontsize=9)
        ax_top.grid(True, alpha=0.3, linestyle="--")
        ax_top.set_xticks(ks)
        ax_top.annotate(f"range: {data_range:.2f} ({100*data_range/max(ymax_data,1e-9):.2f}% of max)",
                        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8,
                        va="top", color="#333")

        ax_bot = axes[1, col]
        ax_bot.plot(ks, vals, "o-", color=c, linewidth=2, markersize=6)
        ax_bot.set_title(f"{title} — zoomed (shape detail)", fontsize=10, fontweight="bold")
        ax_bot.set_xlabel("Ensemble size K", fontsize=9)
        ax_bot.set_ylabel(title, fontsize=9)
        ax_bot.grid(True, alpha=0.3, linestyle="--")
        ax_bot.set_xticks(ks)

    plt.suptitle("FM: Accuracy vs Ensemble Diversity Trade-off Across Ensemble Size K\n"
                "(top row: true scale from 0; bottom row: zoomed — spread here stays "
                "flat, contrast with ode_n_sweep.pdf where spread rises with N)",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "ensemble_size_ablation.pdf")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


def build_k_n_table(eval_full_json: Dict) -> List[Dict]:
    """
    [NEW] TABLE 6 — flattens the "k_n_sweep" JSON block (from
    evaluate_full.py's --k_n_sweep, either single- or multi-checkpoint
    schema) into one row per (K,N) cell, alongside Tables 4 (N-only) and
    5 (K-only) -- this is the table that actually answers whether the
    K=20/N=default combination used for the paper's headline numbers is
    jointly optimal, or just a reasonable-looking pair of independently
    tuned defaults (Tables 4/5 each hold the OTHER axis fixed while
    sweeping one, so neither can show an interaction between K and N;
    this table can).

    Reads the "K,N"-string-keyed JSON produced by
    evaluate_full.py's k_n_sweep output (see run_k_n_joint_sweep_multi_seed
    and the single-checkpoint --k_n_sweep branch in evaluate_full.py's
    main(), both of which serialize (K,N) tuple keys as "K,N" strings
    since JSON keys must be strings).
    """
    block = eval_full_json.get("k_n_sweep")
    if not block:
        return []
    rows = []
    for key, entry in block.items():
        if isinstance(key, str) and "," in key:
            k_str, n_str = key.split(",", 1)
            k, n = int(k_str), int(n_str)
        else:
            k, n = entry.get("K"), entry.get("N")
        rows.append({
            "K": k, "N": n,
            "ade_mean": entry.get("ADE", float("nan")),
            "ate_mean": entry.get("ATE", float("nan")),
            "cte_mean": entry.get("CTE", float("nan")),
            "spread_mean": entry.get("spread", float("nan")),
            "ade_std": entry.get("ADE_std", float("nan")),
            "ate_std": entry.get("ATE_std", float("nan")),
            "cte_std": entry.get("CTE_std", float("nan")),
            "spread_std": entry.get("spread_std", float("nan")),
            "n_seeds": entry.get("n_seeds"),
            "time_s": entry.get("time_s", float("nan")),
            "n": entry.get("n", 0),
        })
    return sorted(rows, key=lambda r: (r["N"], r["K"]))


def print_k_n_table(rows: List[Dict]):
    """
    [NEW] Prints the K,N joint sweep as one wide table per metric (rows
    = N, cols = K), matching evaluate_full.py's own print_k_n_table()
    console format so the JSON-based version here (usable after the
    fact, without re-running the sweep) shows identical numbers.
    """
    if not rows:
        print("\n  ℹ TABLE 6 (K,N joint sweep) bị bỏ qua — không có "
              "'k_n_sweep' block trong eval_full_json. Chạy trước: "
              "evaluate_full.py --k_n_sweep (hoặc --checkpoints ... "
              "--k_n_sweep cho multi-seed), rồi truyền --eval_full_json "
              "<file .json> vào lệnh này.\n")
        return
    ks = sorted(set(r["K"] for r in rows))
    ns = sorted(set(r["N"] for r in rows))
    by_kn = {(r["K"], r["N"]): r for r in rows}
    has_std = any(not np.isnan(r.get("ade_std", float("nan"))) and r.get("ade_std", 0) > 0 for r in rows)

    for metric_key, metric_name in [("ade_mean", "ADE"), ("ate_mean", "ATE"),
                                     ("cte_mean", "CTE"), ("spread_mean", "Ensemble Spread")]:
        col_w = 16 if has_std else 12
        print(f"\n  {'='*(10 + col_w * len(ks))}")
        print(f"  TABLE 6 — {metric_name} (km) joint K,N sweep — "
              f"rows: ODE steps N, cols: ensemble size K"
              f"{' (mean±std across seeds)' if has_std else ''}")
        print(f"  {'='*(10 + col_w * len(ks))}")
        header = f"  {'N \\ K':<8}" + "".join(f"{k:>{col_w}}" for k in ks)
        print(header)
        print(f"  {'-'*(10 + col_w * len(ks))}")
        for n in ns:
            line = f"  {n:<8}"
            for k in ks:
                cell = by_kn.get((k, n))
                if cell is None or np.isnan(cell.get(metric_key, float("nan"))):
                    line += f"{'n/a':>{col_w}}"
                    continue
                val = cell[metric_key]
                std_key = metric_key.replace("_mean", "_std")
                std = cell.get(std_key, 0.0)
                if has_std and not np.isnan(std):
                    line += f"{f'{val:.1f}±{std:.1f}':>{col_w}}"
                else:
                    line += f"{val:>{col_w}.2f}"
            print(line)
        print(f"  {'='*(10 + col_w * len(ks))}")
    print(f"  Ghi chú: cells nơi K khớp Table 5's mặc định (N fixed) hoặc N khớp "
          f"Table 4's mặc định (K=20) NÊN cho số liệu gần giống 2 bảng đó -- "
          f"khác biệt lớn ở các cell trùng này là dấu hiệu cần kiểm tra lại "
          f"(ví dụ seed/data loader không nhất quán giữa các lần chạy).\n")


def plot_k_n_heatmap(eval_full_json: Dict, output_dir: str):
    """
    [NEW] Heatmap of ADE (and CTE, spread) over the (K,N) grid --
    4-panel figure matching the ode_n_sweep.pdf / ensemble_size_ablation.pdf
    visual style (full-scale top row not applicable here since a heatmap
    already shows absolute values via its colorbar, so this uses a
    single well-scaled panel per metric instead). This is the primary
    visual evidence for whether K and N interact: if the heatmap's
    "best" cell sits at a specific (K,N) combination rather than along
    a flat ridge independent of the other axis, K and N ARE interacting,
    and Tables 4/5's independently-swept "optimal" values may not
    actually be jointly optimal.
    """
    block = eval_full_json.get("k_n_sweep")
    if not block:
        print("  ⚠ plot_k_n_heatmap: no 'k_n_sweep' block in eval_full_json — skip")
        return None

    rows = build_k_n_table(eval_full_json)
    if not rows:
        return None
    ks = sorted(set(r["K"] for r in rows))
    ns = sorted(set(r["N"] for r in rows))
    by_kn = {(r["K"], r["N"]): r for r in rows}

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    metrics = [("ade_mean", "ADE (km)", "viridis_r"),
               ("ate_mean", "ATE (km)", "viridis_r"),
               ("cte_mean", "CTE (km)", "viridis_r"),
               ("spread_mean", "Ensemble Spread (km)", "viridis")]
    # *_r (reversed) colormaps for ADE/ATE/CTE so DARKER = BETTER (lower
    # error) consistently across all 3 error panels -- for spread, darker
    # = lower spread is not necessarily "better" (spread reflects
    # ensemble diversity, not accuracy), so its colormap is NOT reversed,
    # avoiding a misleading visual implication that low spread is good.

    for ax, (metric_key, title, cmap) in zip(axes, metrics):
        grid = np.full((len(ns), len(ks)), np.nan)
        for i, n in enumerate(ns):
            for j, k in enumerate(ks):
                cell = by_kn.get((k, n))
                if cell is not None:
                    grid[i, j] = cell[metric_key]

        im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks)
        ax.set_yticks(range(len(ns))); ax.set_yticklabels(ns)
        ax.set_xlabel("Ensemble size K", fontsize=10)
        ax.set_ylabel("ODE integration steps N", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        for i in range(len(ns)):
            for j in range(len(ks)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                            fontsize=7, color="white" if
                            (grid[i, j] - np.nanmin(grid)) / max(np.nanmax(grid) - np.nanmin(grid), 1e-9) > 0.5
                            else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Joint K (ensemble size) x N (ODE integration steps) sweep\n"
                "(darker = lower error for ADE/ATE/CTE; spread colormap NOT "
                "reversed since low spread is not inherently 'better')",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "k_n_heatmap.pdf")
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
    out = os.path.join(output_dir, "per_storm_cte_worst.pdf")
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
                   default=["ST-Trans", "RNN", "GRU", "LSTM", "MMSTN", "Phys-Diff", "TC-Diffuser"])
    p.add_argument("--metrics", nargs="+", default=["ade", "ate", "cte"],
                   choices=["ade", "ate", "cte"])
    p.add_argument("--output_dir", default="eval_multi",
                   help="Bảng (paper_tables.json) và hình (*.pdf) đều lưu vào đây")
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
        # [FIX RMSE scope] Cố tình dùng list RIÊNG (args.metrics + ["rmse"])
        # cho per-horizon table, KHÔNG sửa args.metrics dùng chung với
        # build_significance_table() ở trên -- "RMSE significance" không
        # phải khái niệm chuẩn cho paired Wilcoxon/t-test per-record (RMSE
        # là 1 số tổng hợp toàn tập, không phải giá trị per-record để so
        # cặp), nên KHÔNG tự động thêm rmse vào phần significance chỉ vì
        # thêm vào đây. Nếu muốn rmse cũng chạy qua significance test,
        # cần quyết định phương pháp luận riêng trước, không phải patch
        # ngầm bằng cách đổi args.metrics mặc định.
        per_horizon_metrics = list(args.metrics)
        if "rmse" not in per_horizon_metrics:
            per_horizon_metrics.append("rmse")
        for metric in per_horizon_metrics:
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

            k_n_rows = build_k_n_table(eval_full_data)
            print_k_n_table(k_n_rows)
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
            kn_heatmap = plot_k_n_heatmap(ej, args.output_dir)
            if kn_heatmap:
                saved.append(kn_heatmap)

        if args.per_storm_json:
            saved.append(plot_per_storm_cte(args.per_storm_json, args.output_dir))

        saved = [s for s in saved if s]
        print(f"\n  Done — {len(saved)} figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()