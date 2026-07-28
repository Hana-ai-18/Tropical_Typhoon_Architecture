#!/usr/bin/env python3
"""

# ─────────────────────────────────────────────────────────────────────────────
# FILE PLACEMENT:
#
#   SOURCE:        run_all.py
#   KAGGLE TARGET: /kaggle/working/run_all.py    (root)
#   LOCAL DEV:     run_all.py
#
#   Cần đặt CÁC FILE SAU trước khi chạy run_all.py:
#     /kaggle/working/
#     ├── Model/
#     │   └── flow_matching_model.py   ← flow_matching_model_v24_final.py
#     ├── train_fm.py                  ← train_fm_v24.py
#     ├── evaluate_full.py
#     ├── statistical_tests.py
#     ├── ablation_runner.py
#     └── run_all.py                   ← file này
#
#   1 lệnh chạy tất cả:
#   python run_all.py \
#     --dataset_root /kaggle/input/datasets/tc-ofm \
#     --output_dir   /kaggle/working/runs/eswa_full \
#     --num_epochs   150 \
#     --seeds        0 1 2 \
#     --gpu          0
# ─────────────────────────────────────────────────────────────────────────────

run_all.py — TC-FlowMatching ESWA: Chạy toàn bộ pipeline 1 lệnh
════════════════════════════════════════════════════════════════════════════════

Thực hiện:
  Phase 0: Kiểm tra môi trường
  Phase 1: Train chính (3 seeds cho ESWA mean±std)
  Phase 2: Ablation studies (no_L_heading, no_L_calib, cfm_only, no_aug_c)
  Phase 3: ODE steps sweep (1,2,5,10)
  Phase 4: Full evaluation (test) cho từng seed + ablation
  Phase 5: Statistical tests (FM vs ST-Trans)
  Phase 6: Tổng hợp bảng kết quả cho paper

Usage (tất cả, đầy đủ):
  python run_all.py \\
    --dataset_root /path/to/tc-ofm \\
    --output_dir   runs/eswa_full \\
    --num_epochs   150 \\
    --seeds        0 1 2 \\
    --gpu          0

Usage (chỉ eval từ checkpoint có sẵn):
  python run_all.py \\
    --mode eval_only \\
    --checkpoint   runs/best_model.pth \\
    --dataset_root /path/to/tc-ofm \\
    --output_dir   runs/eswa_full

Usage (chỉ ablation):
  python run_all.py \\
    --mode ablation_only \\
    --checkpoint   runs/best_model.pth \\
    --dataset_root /path/to/tc-ofm \\
    --output_dir   runs/eswa_full
"""
from __future__ import annotations
import os, sys, argparse, json, time, subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd: str, desc: str, check: bool = True) -> int:
    """Run a shell command with logging."""
    print(f"\n{'─'*70}")
    print(f"  ▶  {desc}")
    print(f"  CMD: {cmd}")
    print(f"{'─'*70}")
    t0 = time.time()
    ret = os.system(cmd)
    elapsed = time.time() - t0
    status  = "✅ OK" if ret == 0 else f"❌ FAILED (code {ret})"
    print(f"  {status}  ({elapsed:.1f}s)")
    if check and ret != 0:
        print(f"\n  ABORT: {desc} failed. Fix error above then re-run.")
        sys.exit(1)
    return ret


def py(script: str) -> str:
    """Build python command."""
    return f"python {BASE_DIR / script}"


def phase(n: int, name: str):
    print(f"\n{'='*70}")
    print(f"  PHASE {n}: {name}")
    print(f"{'='*70}")


def save_summary(results: dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 0: Environment check
# ─────────────────────────────────────────────────────────────────────────────

def phase0_check(args):
    phase(0, "ENVIRONMENT CHECK")
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {torch.cuda.is_available()} ({torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else 'N/A'})")
    print(f"  Dataset: {args.dataset_root}")
    assert os.path.exists(args.dataset_root), f"Dataset not found: {args.dataset_root}"

    try:
        from scipy import stats
        import numpy as np
        print(f"  scipy:   OK  |  numpy: {np.__version__}")
    except ImportError as e:
        print(f"  ❌ Missing dependency: {e}")
        print(f"  Run: pip install scipy")
        sys.exit(1)

    # Required scripts
    required = ["train_fm_v24.py", "evaluate_full.py",
                "statistical_tests.py", "ablation_runner.py",
                "flow_matching_model_v24.py"]
    for f in required:
        path = BASE_DIR / f
        # Also try Model/ subfolder for model file
        if not path.exists():
            path2 = BASE_DIR / "Model" / f
            exists = path2.exists()
        else:
            exists = True
        print(f"  {'✅' if exists else '❌'} {f}")
        if not exists:
            print(f"  ERROR: {f} not found")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"  Output: {args.output_dir}")
    print(f"  Seeds:  {args.seeds}")
    print(f"  Epochs: {args.num_epochs}")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1: Train (multi-seed)
# ─────────────────────────────────────────────────────────────────────────────

def phase1_train(args) -> dict:
    phase(1, "TRAINING (multi-seed)")
    seed_ckpts = {}

    for seed in args.seeds:
        seed_dir = os.path.join(args.output_dir, f"seed{seed}")
        best_ckpt = os.path.join(seed_dir, "best_model.pth")

        if os.path.exists(best_ckpt) and args.skip_existing:
            print(f"  Seed {seed}: checkpoint exists, skipping.")
            seed_ckpts[seed] = best_ckpt
            continue

        cmd = (
            f"{py('train_fm_v24.py')}"
            f" --dataset_root {args.dataset_root}"
            f" --output_dir   {seed_dir}"
            f" --num_epochs   {args.num_epochs}"
            f" --seed         {seed}"
            f" --gpu_num      {args.gpu}"
            f" --batch_size   {args.batch_size}"
            f" --val_freq     5"
            f" --test_at_end"
        )
        run(cmd, f"Train seed={seed}", check=True)
        seed_ckpts[seed] = best_ckpt

    return seed_ckpts


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2: Ablation training
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_FLAGS = {
    "no_L_heading": "--disable_l_heading",
    "no_L_calib":   "--disable_l_calib",
    "cfm_only":     "--disable_l_heading --disable_l_calib --disable_l_reg",
    "no_aug_c":     "--disable_aug_c",
    "no_kendall":   "--disable_learned_weights",
}


def phase2_ablation(args) -> dict:
    phase(2, "ABLATION STUDIES")
    abl_ckpts = {}

    for variant, flags in ABLATION_FLAGS.items():
        abl_dir   = os.path.join(args.output_dir, f"ablation_{variant}")
        best_ckpt = os.path.join(abl_dir, "best_model.pth")

        if os.path.exists(best_ckpt) and args.skip_existing:
            print(f"  Ablation {variant}: checkpoint exists, skipping.")
            abl_ckpts[variant] = best_ckpt
            continue

        # Train with seed=0 only for ablation (no need for multiple seeds)
        cmd = (
            f"{py('train_fm_v24.py')}"
            f" --dataset_root  {args.dataset_root}"
            f" --output_dir    {abl_dir}"
            f" --num_epochs    {args.num_epochs}"
            f" --seed          0"
            f" --gpu_num       {args.gpu}"
            f" --batch_size    {args.batch_size}"
            f" --ablation_name {variant}"
            f" {flags}"
            f" --test_at_end"
        )
        run(cmd, f"Ablation: {variant}", check=False)  # don't abort if ablation fails
        if os.path.exists(best_ckpt):
            abl_ckpts[variant] = best_ckpt
        else:
            print(f"  ⚠ No checkpoint for {variant}, skipping eval.")

    return abl_ckpts


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3: ODE steps sweep
# ─────────────────────────────────────────────────────────────────────────────

def phase3_ode_sweep(args, seed_ckpts: dict):
    phase(3, "ODE STEPS SWEEP")
    # Use best seed (seed 0 or first available)
    best_seed = min(seed_ckpts.keys())
    ckpt = seed_ckpts[best_seed]

    cmd = (
        f"{py('ablation_runner.py')}"
        f" --mode         ode_steps"
        f" --checkpoint   {ckpt}"
        f" --dataset_root {args.dataset_root}"
        f" --output_dir   {os.path.join(args.output_dir, 'ode_sweep')}"
        f" --gpu          {args.gpu}"
    )
    run(cmd, "ODE steps sweep (1,2,5,10)", check=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 4: Full evaluation
# ─────────────────────────────────────────────────────────────────────────────

def phase4_eval(args, seed_ckpts: dict, abl_ckpts: dict) -> dict:
    phase(4, "FULL EVALUATION")
    eval_dir = os.path.join(args.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    all_results = {"seeds": {}, "ablations": {}}

    # Eval main model (all seeds)
    for seed, ckpt in seed_ckpts.items():
        out_json = os.path.join(eval_dir, f"seed{seed}_test.json")
        cmd = (
            f"{py('evaluate_full.py')}"
            f" --checkpoint   {ckpt}"
            f" --dataset_root {args.dataset_root}"
            f" --split        test"
            f" --output_dir   {eval_dir}"
            f" --n_ensemble   20"
            f" --case_studies"
            f" --gpu          {args.gpu}"
        )
        run(cmd, f"Eval seed={seed}", check=False)
        if os.path.exists(out_json):
            with open(out_json) as f:
                all_results["seeds"][seed] = json.load(f)

    # Eval ablation variants
    for variant, ckpt in abl_ckpts.items():
        out_json = os.path.join(eval_dir, f"ablation_{variant}_test.json")
        cmd = (
            f"{py('evaluate_full.py')}"
            f" --checkpoint   {ckpt}"
            f" --dataset_root {args.dataset_root}"
            f" --split        test"
            f" --output_dir   {eval_dir}"
            f" --n_ensemble   20"
            f" --no_crps"
            f" --gpu          {args.gpu}"
        )
        run(cmd, f"Eval ablation {variant}", check=False)
        if os.path.exists(out_json):
            with open(out_json) as f:
                all_results["ablations"][variant] = json.load(f)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 5: Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def phase5_stats(args, all_results: dict):
    phase(5, "STATISTICAL TESTS")
    eval_dir  = os.path.join(args.output_dir, "eval")
    stats_dir = os.path.join(args.output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    # Use seed 0 as representative FM result (or aggregate if available)
    seeds = list(all_results["seeds"].keys())
    if not seeds:
        print("  No seed results, skipping statistical tests.")
        return

    # Find the JSON file for seed 0
    fm_json = os.path.join(eval_dir, f"seed{seeds[0]}_test.json")
    if not os.path.exists(fm_json):
        print(f"  FM result not found: {fm_json}")
        return

    cmd = (
        f"{py('statistical_tests.py')}"
        f" --fm_results       {fm_json}"
        f" --use_st_trans_ref"
        f" --fm_n_storms      420"
        f" --baseline_name    ST-Trans"
        f" --output_dir       {stats_dir}"
        f" --n_bootstrap      10000"
    )
    run(cmd, "Statistical tests: FM vs ST-Trans", check=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 6: Aggregate summary
# ─────────────────────────────────────────────────────────────────────────────

def phase6_summary(args, all_results: dict):
    phase(6, "PAPER TABLE SUMMARY")
    import numpy as np

    summary = {
        "main_model":   {},
        "ablations":    {},
        "st_trans_ref": {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4},
    }

    # Main model: aggregate across seeds
    seeds_data = all_results.get("seeds", {})
    if seeds_data:
        for metric in ["ADE", "ATE", "CTE", "RMSE"]:
            vals = [r[metric] for r in seeds_data.values()
                    if metric in r and r[metric] == r[metric]]  # exclude nan
            if vals:
                summary["main_model"][metric]          = float(np.mean(vals))
                summary["main_model"][f"{metric}_std"] = float(np.std(vals))
                summary["main_model"][f"{metric}_seeds"] = vals

    # Ablations
    for variant, r in all_results.get("ablations", {}).items():
        summary["ablations"][variant] = {
            k: r.get(k) for k in ["ADE", "ATE", "CTE"]
        }

    # Print table
    print(f"\n  {'─'*65}")
    print(f"  {'Model':<30} {'ADE':>8} {'ATE':>8} {'CTE':>8}  Notes")
    print(f"  {'─'*65}")

    ref = summary["st_trans_ref"]
    print(f"  {'ST-Trans (reference)':<30} {ref['ADE']:>8.1f} {ref['ATE']:>8.1f} {ref['CTE']:>8.1f}")

    mm = summary.get("main_model", {})
    if mm.get("ADE"):
        ade_str = f"{mm['ADE']:.1f}±{mm.get('ADE_std',0):.1f}"
        ate_str = f"{mm['ATE']:.1f}±{mm.get('ATE_std',0):.1f}"
        cte_str = f"{mm['CTE']:.1f}±{mm.get('CTE_std',0):.1f}"
        n_seeds = len(summary["main_model"].get("ADE_seeds", []))
        print(f"  {'FM v2.4 (ours)':<30} {ade_str:>8} {ate_str:>8} {cte_str:>8}  ({n_seeds} seeds)")

    for variant, r in summary["ablations"].items():
        ade = r.get("ADE", float("nan"))
        ate = r.get("ATE", float("nan"))
        cte = r.get("CTE", float("nan"))
        # Delta vs main
        d_ade = ade - mm.get("ADE", float("nan"))
        delta = f"ΔADE={d_ade:+.1f}" if abs(d_ade) < 999 else ""
        print(f"  {'  - '+variant:<30} {ade:>8.1f} {ate:>8.1f} {cte:>8.1f}  {delta}")

    print(f"  {'─'*65}")

    # LaTeX table
    print(f"\n  ── LaTeX ABLATION TABLE ──")
    print(r"  \begin{tabular}{lrrrl}")
    print(r"  Model & ADE↓ & ATE↓ & CTE↓ & Notes \\")
    print(r"  \hline")
    print(f"  ST-Trans & {ref['ADE']:.1f} & {ref['ATE']:.1f} & {ref['CTE']:.1f} & reference \\\\")
    if mm.get("ADE"):
        n_s = len(mm.get("ADE_seeds", [1]))
        print(f"  FM (ours) & ${mm['ADE']:.1f}_{{\\pm {mm.get('ADE_std',0):.1f}}}$ & "
              f"${mm['ATE']:.1f}_{{\\pm {mm.get('ATE_std',0):.1f}}}$ & "
              f"${mm['CTE']:.1f}_{{\\pm {mm.get('CTE_std',0):.1f}}}$ & {n_s} seeds \\\\")
    print(r"  \hline")
    for variant, r in summary["ablations"].items():
        ade = r.get("ADE", float("nan"))
        ate = r.get("ATE", float("nan"))
        cte = r.get("CTE", float("nan"))
        print(f"  w/o {variant.replace('_',' ')} & {ade:.1f} & {ate:.1f} & {cte:.1f} & ablation \\\\")
    print(r"  \end{tabular}")

    # Save
    summary_path = os.path.join(args.output_dir, "paper_summary.json")
    save_summary(summary, summary_path)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="TC-FlowMatching ESWA Full Pipeline")
    p.add_argument("--dataset_root",  required=True,
                   help="Path to TC-OFM dataset root")
    p.add_argument("--output_dir",    default="runs/eswa_full",
                   help="Root output directory")
    p.add_argument("--mode",
                   choices=["full", "eval_only", "ablation_only", "stats_only"],
                   default="full",
                   help="Which phases to run")
    # Training
    p.add_argument("--num_epochs",    type=int, default=150)
    p.add_argument("--seeds",         type=int, nargs="+", default=[0, 1, 2],
                   help="Seeds for multi-seed training (ESWA requires ≥3)")
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--gpu",           type=int, default=0)
    # Eval only
    p.add_argument("--checkpoint",    type=str, default=None,
                   help="For eval_only/ablation_only: path to checkpoint")
    # Flags
    p.add_argument("--skip_existing", action="store_true", default=True,
                   help="Skip phases where outputs already exist")
    p.add_argument("--no_ablation",   action="store_true", default=False,
                   help="Skip ablation training (saves time)")
    p.add_argument("--ablation_epochs", type=int, default=None,
                   help="Epochs for ablation (default: same as num_epochs)")
    return p.parse_args()


def main():
    args = get_args()
    if args.ablation_epochs is None:
        args.ablation_epochs = args.num_epochs

    wall_start = time.time()

    print(f"\n{'='*70}")
    print(f"  TC-FlowMatching ESWA Full Pipeline")
    print(f"  Mode: {args.mode} | Seeds: {args.seeds} | Epochs: {args.num_epochs}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}")

    # Phase 0: Environment check
    phase0_check(args)

    all_results = {"seeds": {}, "ablations": {}}

    if args.mode == "full":
        # Phase 1: Train
        seed_ckpts = phase1_train(args)

        # Phase 2: Ablation (optional)
        abl_ckpts = {}
        if not args.no_ablation:
            abl_ckpts = phase2_ablation(args)

        # Phase 3: ODE sweep
        if seed_ckpts:
            phase3_ode_sweep(args, seed_ckpts)

        # Phase 4: Evaluate
        all_results = phase4_eval(args, seed_ckpts, abl_ckpts)

        # Phase 5: Stats
        phase5_stats(args, all_results)

        # Phase 6: Summary
        phase6_summary(args, all_results)

    elif args.mode == "eval_only":
        if args.checkpoint is None:
            print("  ERROR: --checkpoint required for eval_only"); return
        seed_ckpts = {0: args.checkpoint}
        abl_ckpts  = {}
        all_results = phase4_eval(args, seed_ckpts, abl_ckpts)
        phase5_stats(args, all_results)
        phase6_summary(args, all_results)

    elif args.mode == "ablation_only":
        if args.checkpoint is None:
            print("  ERROR: --checkpoint required for ablation_only"); return
        seed_ckpts = {0: args.checkpoint}
        abl_ckpts  = phase2_ablation(args)
        phase3_ode_sweep(args, seed_ckpts)
        all_results = phase4_eval(args, seed_ckpts, abl_ckpts)
        phase6_summary(args, all_results)

    elif args.mode == "stats_only":
        if args.checkpoint is None:
            print("  ERROR: --checkpoint required for stats_only"); return
        seed_ckpts = {0: args.checkpoint}
        all_results["seeds"] = {0: {}}  # placeholder
        phase5_stats(args, all_results)

    # Final
    wall_total = time.time() - wall_start
    print(f"\n{'='*70}")
    print(f"  COMPLETE ✅  Total wall-clock: {wall_total/3600:.2f}h ({wall_total:.0f}s)")
    print(f"  Results in: {args.output_dir}/")
    print(f"  Key files:")
    print(f"    paper_summary.json    ← bảng kết quả cho paper")
    print(f"    eval/                 ← per-seed + ablation eval JSONs")
    print(f"    stats/                ← Wilcoxon + Bootstrap CI")
    print(f"    ode_sweep/            ← ODE steps trade-off table")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()