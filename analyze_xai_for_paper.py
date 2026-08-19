"""
analyze_xai_for_paper.py
=========================
Chạy SAU KHI đã có output từ analyze_xai_multi_seed.py (đọc trực tiếp
attn_by_horizon_per_seed.csv đã lưu). Mục đích: làm phân tích thống kê CHẶT
CHẼ HƠN để đưa vào paper, giải quyết đúng 1 vấn đề quan trọng đã phát hiện
khi kiểm tra dữ liệu thô:

VẤN ĐỀ ĐÃ PHÁT HIỆN: 449 "sequences" trong test set KHÔNG PHẢI 449 cơn bão
độc lập -- chúng là 449 sliding-window instances trích từ chỉ 12 cơn bão
thật (RITA_1975, WAYNE_1986, ..., EWINIAR_2024), phân bố rất lệch (từ 9 tới
70 window/storm). Các window từ CÙNG một storm chồng lấn dữ liệu quan sát
(obs+pred window chỉ trượt vài giờ mỗi lần) nên KHÔNG độc lập với nhau về
mặt thống kê -- coi 449 window là 449 mẫu độc lập (như một kiểm định Mann-
Whitney/t-test đơn giản ngầm giả định) SẼ PHÓNG ĐẠI độ tin cậy thống kê một
cách sai lệch (pseudo-replication), một lỗi phương pháp luận phổ biến khi
làm việc với time-series sliding-window data.

GIẢI PHÁP: linear mixed-effects model với "storm" là random effect --
đây là cách chuẩn trong thống kê để xử lý dữ liệu có cấu trúc lồng nhau
(nested/clustered), tận dụng được toàn bộ 449 window mà không giả định sai
về tính độc lập. Verified trên dữ liệu thật: cho kết quả p=0.033 cho hệ số
"group" (recurving vs straight) sau khi kiểm soát horizon và random effect
theo storm -- một kết quả đáng tin cậy hơn nhiều so với phép kiểm định đơn
giản trên 449 "mẫu" giả-độc-lập.

USAGE
-----
python analyze_xai_for_paper.py \
    --attn_csv /kaggle/working/xai_multi_seed/attn_by_horizon_per_seed.csv \
    --output_dir /kaggle/working/xai_paper_stats

Yêu cầu: pip install statsmodels (nếu chưa có)
"""
from __future__ import annotations
import os, argparse
import numpy as np
import pandas as pd


def run_mixed_effects_analysis(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Overall test: does attention to context vector genuinely differ
    between recurving and straight-moving storms, across the whole
    forecast horizon, after accounting for (a) horizon itself having an
    effect and (b) multiple windows from the same storm being correlated
    (not independent)?

    Model: attn_context ~ group + horizon_h, random intercept per storm.
    """
    import statsmodels.formula.api as smf

    # Average across seeds first -- 3 seeds of the SAME checkpoint
    # architecture evaluated on the SAME storm window are not 3
    # independent replicates of "how this storm behaves", they are 3
    # noisy estimates of one underlying quantity. Averaging them keeps
    # the per-storm/per-horizon record count honest (449 windows, not
    # 449*3 pseudo-observations).
    per_window = df.groupby(["storm", "horizon_h", "group"], as_index=False)["attn_context"].mean()
    per_window["group_bin"] = (per_window["group"] == "recurving").astype(int)

    model = smf.mixedlm("attn_context ~ group_bin + horizon_h",
                         per_window, groups=per_window["storm"])
    result = model.fit()

    summary_path = os.path.join(output_dir, "mixed_effects_summary.txt")
    with open(summary_path, "w") as f:
        f.write(str(result.summary()))
        f.write(f"\n\nNote: 'group_bin'=1 means recurving, 0 means straight.\n")
        f.write(f"Interpretation: the coefficient on group_bin is the average\n")
        f.write(f"difference in context-vector attention between recurving\n")
        f.write(f"and straight-moving storms, AFTER accounting for horizon\n")
        f.write(f"and for repeated/correlated windows from the same storm\n")
        f.write(f"(random intercept per storm, {per_window['storm'].nunique()} storms,\n")
        f.write(f"{len(per_window)} total storm-horizon windows).\n")

    print(f"  Mixed-effects model fit. Full summary saved to: {summary_path}")
    print(result.summary())

    coef_df = pd.DataFrame({
        "term": result.params.index,
        "coef": result.params.values,
        "std_err": result.bse.values,
        "p_value": result.pvalues.values,
    })
    coef_df.to_csv(os.path.join(output_dir, "mixed_effects_coefficients.csv"), index=False)
    return coef_df


def run_per_horizon_mixed_effects(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Complementary, finer-grained view: instead of one overall test with
    horizon as a linear covariate, fit a SEPARATE simple comparison at
    EACH horizon, still correctly averaging within-storm first so each
    storm contributes one number per horizon (12 storms, not up to 70
    windows for the most heavily-represented storm). This is what
    produces the per-horizon table/figure for the paper, honestly scaled
    to n=12 storms rather than n=449 windows.
    """
    per_window = df.groupby(["storm", "horizon_h", "group"], as_index=False)["attn_context"].mean()

    rows = []
    for h in sorted(per_window["horizon_h"].unique()):
        sub = per_window[per_window["horizon_h"] == h]
        r = sub[sub["group"] == "recurving"]["attn_context"].values
        s = sub[sub["group"] == "straight"]["attn_context"].values
        rows.append({
            "horizon_h": h,
            "n_recurving_storms": len(r),
            "n_straight_storms": len(s),
            "mean_recurving": r.mean() if len(r) else float("nan"),
            "mean_straight": s.mean() if len(s) else float("nan"),
        })
    per_h = pd.DataFrame(rows)
    per_h.to_csv(os.path.join(output_dir, "per_horizon_storm_level_means.csv"), index=False)
    print(f"\n  Per-horizon storm-level means "
          f"(honest n -- {per_window['storm'].nunique()} storms, not 449 windows):")
    print(per_h.to_string(index=False))
    print(f"\n  ⚠ n_recurving/n_straight per horizon are small (a handful of "
          f"storms each) -- this table is for VISUALIZATION (the trend "
          f"across horizon), not for horizon-by-horizon significance "
          f"testing. Use the overall mixed-effects result above for the "
          f"significance claim, and this table only to show WHERE across "
          f"the horizon that overall effect is concentrated.")
    return per_h


def make_storm_level_figure(df: pd.DataFrame, output_dir: str):
    """
    One line per storm (12 lines total) instead of an aggregate mean±std
    band -- lets a reader see directly whether the recurving-vs-straight
    difference holds up consistently across individual storms, or is
    driven by one or two storms with many more windows than the others
    (WAYNE_1986: 70 windows vs MOLAVE_2020: 9 windows in this dataset --
    an aggregate mean without this view could be quietly dominated by
    whichever storm happens to have the most windows).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipped storm-level figure.")
        return

    per_window = df.groupby(["storm", "horizon_h", "group"], as_index=False)["attn_context"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, grp in zip(axes, ["recurving", "straight"]):
        sub_grp = per_window[per_window["group"] == grp]
        for storm in sorted(sub_grp["storm"].unique()):
            sub = sub_grp[sub_grp["storm"] == storm].sort_values("horizon_h")
            ax.plot(sub["horizon_h"], sub["attn_context"], marker="o",
                    alpha=0.6, label=storm, linewidth=1.2)
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_title(f"{grp} storms (n={sub_grp['storm'].nunique()})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean cross-attention weight on context vector")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "attn_by_individual_storm.pdf")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"\n  Storm-level figure saved: {fig_path}")


def run_robustness_check(df: pd.DataFrame, output_dir: str, coef_df: pd.DataFrame):
    """
    [MANDATORY, NOT OPTIONAL] The overall mixed-effects test above can
    still be dominated by one or two storms with unusually high or low
    attention values -- a random intercept per storm controls for each
    storm having its OWN baseline level, but does not by itself prove
    the recurving-vs-straight EFFECT is consistent across storms rather
    than driven by a couple of outliers. Verified directly on real data:
    visually inspecting attn_by_individual_storm.pdf showed 2-3 storms
    (in one run: MOLAVE_2020, FLO_1993, HINNAMNOR_2022) sitting well
    above the rest of their group -- refitting the SAME model with those
    storms removed changed the group_bin p-value from 0.033 (apparently
    significant) to 0.541 (not significant at all), and shrank the
    coefficient by ~4x. This means the original "significant" result was
    NOT a robust, dataset-wide pattern -- it was substantially driven by
    a small number of storms. Reporting only the first fit without this
    check would have been a genuine overstatement of the finding's
    strength. This function makes that leave-some-storms-out check
    automatic and mandatory, rather than something a reader has to
    notice is missing.

    Method: leave-one-storm-out refits (drop each storm once, refit,
    record how much group_bin's p-value/coefficient moves) -- this
    directly shows the paper's claim's sensitivity to any single storm,
    which is the honest way to report a finding built on only ~12
    independent units.
    """
    import statsmodels.formula.api as smf

    per_window = df.groupby(["storm", "horizon_h", "group"], as_index=False)["attn_context"].mean()
    per_window["group_bin"] = (per_window["group"] == "recurving").astype(int)
    storms = sorted(per_window["storm"].unique())

    loo_results = []
    for held_out in storms:
        sub = per_window[per_window["storm"] != held_out]
        try:
            m = smf.mixedlm("attn_context ~ group_bin + horizon_h", sub, groups=sub["storm"])
            r = m.fit(reml=True)
            loo_results.append({
                "storm_removed": held_out,
                "group_bin_coef": r.params.get("group_bin", float("nan")),
                "group_bin_p": r.pvalues.get("group_bin", float("nan")),
            })
        except Exception as e:
            loo_results.append({"storm_removed": held_out,
                                 "group_bin_coef": float("nan"),
                                 "group_bin_p": float("nan")})

    loo_df = pd.DataFrame(loo_results)
    loo_df.to_csv(os.path.join(output_dir, "leave_one_storm_out.csv"), index=False)

    full_p = coef_df.loc[coef_df["term"] == "group_bin", "p_value"].values[0]
    n_flips = int((loo_df["group_bin_p"] > 0.05).sum())

    print(f"\n{'='*70}\n  MANDATORY robustness check: leave-one-storm-out\n{'='*70}")
    print(loo_df.to_string(index=False))
    print(f"\n  Full-data p-value: {full_p:.4f}")
    print(f"  Number of storms whose removal flips significance "
          f"(p goes from <0.05 to >0.05, or the reverse): check the "
          f"table above manually -- {n_flips}/{len(loo_df)} single-storm "
          f"removals alone give p>0.05.")
    if n_flips >= 1:
        print(f"\n  ⚠ WARNING: The overall significance result is NOT robust "
              f"to removing individual storms -- at least one storm's "
              f"removal alone is enough to lose significance. This "
              f"finding should be reported as PRELIMINARY / storm-driven,"
              f" not as a robust dataset-wide pattern, unless a larger, "
              f"more balanced set of storms is evaluated. Do not report "
              f"only the full-data p-value without this caveat.")
    return loo_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn_csv", required=True,
                    help="Path to attn_by_horizon_per_seed.csv produced by "
                         "analyze_xai_multi_seed.py")
    p.add_argument("--output_dir", default="xai_paper_stats")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.attn_csv)

    n_storms = df["storm"].nunique()
    print(f"  Loaded {len(df)} rows, {n_storms} unique storms, "
          f"{df['seed'].nunique()} seeds.")
    if n_storms < 20:
        print(f"\n  ⚠ Only {n_storms} unique storms in this dataset. This is "
              f"the TRUE sample size for any claim about 'storms behaving "
              f"differently' -- treat significance results accordingly, "
              f"and consider reporting this as a limitation (small basin-"
              f"specific test set) alongside any positive finding.")

    print(f"\n{'='*70}\n  Overall mixed-effects test (recurving vs straight, "
          f"controlling for horizon and storm)\n{'='*70}")
    try:
        coef_df = run_mixed_effects_analysis(df, args.output_dir)
    except ImportError:
        print("  statsmodels not installed -- run: pip install statsmodels")
        return

    print(f"\n{'='*70}\n  Per-horizon storm-level means (for figure/table)\n{'='*70}")
    run_per_horizon_mixed_effects(df, args.output_dir)

    make_storm_level_figure(df, args.output_dir)

    # [MANDATORY] Always run the robustness check -- see that function's
    # docstring for why this is not optional: the first fit's p-value
    # alone was verified to be misleading on real data from this project.
    run_robustness_check(df, args.output_dir, coef_df)

    print(f"\n  All outputs saved to: {args.output_dir}")
    print(f"\n  ── For the paper ──────────────────────────────────────────")
    print(f"  Report BOTH the overall mixed-effects p-value AND the leave-")
    print(f"  one-storm-out robustness check together -- reporting only the")
    print(f"  former, if it does not survive the latter, would overstate")
    print(f"  the finding. Use attn_by_individual_storm.pdf so a reader can")
    print(f"  see for themselves which storms drive the pattern, rather")
    print(f"  than taking a single aggregate p-value on faith.")


if __name__ == "__main__":
    main()