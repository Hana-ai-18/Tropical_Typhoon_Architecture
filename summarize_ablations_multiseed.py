"""
summarize_ablations_multiseed.py — Tổng hợp 9 ablation, mỗi cái 3 seed,
thành 1 bảng mean±std GIỐNG HỆT cách build_main_table() trong
generate_paper_report.py tính cho full model (mean-trong-seed-trước, rồi
mean/std-qua-seed-sau).

QUAN TRỌNG: dùng --mode multi_seed của ablation_runner.py (đánh giá TRỰC
TIẾP lên checkpoint đã thực sự train với flag --disable_*), KHÔNG dùng
--mode eval_variant (mode đó chỉ giả lập ablation trên 1 checkpoint FULL
model có sẵn, KHÔNG phản ánh đúng model đã thực sự huấn luyện lại với
thành phần bị bớt — sẽ cho kết quả sai nếu dùng nhầm cho trường hợp của
bạn, vì bạn đã train 9 checkpoint riêng thật sự).

ĐÃ SỬA (theo yêu cầu): KHÔNG còn dùng file --config JSON riêng. Đường
dẫn 9 checkpoint + FULL model được HARD-CODE ngay dưới đây trong biến
CHECKPOINT_PATTERNS, khớp đúng tên file thật đã xác nhận qua ảnh Kaggle
Datasets/new_checkpoint/ablation/ (9 ablation x 3 seed, tên dạng
best_model_no_<X>_seed<N>.pth).

⚠️ CHỈ CẦN SỬA 2 CHỖ TRƯỚC KHI CHẠY:
  1. Biến ABLATION_CKPT_ROOT bên dưới — đường dẫn gốc dataset Kaggle
     chứa checkpoint ablation (tôi suy ra từ ảnh + từ prefix đã dùng ở
     ablations_study.txt trước đó là "gmnguynhng/new-checkpoint", nhưng
     tên dataset trong ảnh hiển thị là "new_checkpoint" — gạch dưới,
     khác "new-checkpoint" gạch ngang trong log cũ. CẦN BẠN TỰ XÁC NHẬN
     tên chính xác, vì tôi không truy cập được Kaggle của bạn).
  2. Biến FULL_CKPT_PATTERN — đường dẫn 3 checkpoint FM chính (seed0/1/2)
     dùng làm hàng "FULL" trong bảng so sánh, KHÔNG train riêng cho
     ablation study (tái dùng checkpoint đã có).

Cách dùng (không cần --config nữa):
    python summarize_ablations_multiseed.py \
        --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
        --split test \
        --n_ensemble 20 \
        --ablation_runner /kaggle/working/.../ablation_runner.py \
        --output_dir /kaggle/working/ablation_summary
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# SỬA 2 BIẾN NÀY CHO KHỚP PATH THẬT TRÊN KAGGLE CỦA BẠN
# ═══════════════════════════════════════════════════════════════════════

# Đường dẫn gốc chứa 27 file .pth (9 ablation x 3 seed) như trong ảnh.
# {seed} và {name} sẽ được .format() điền tự động bên dưới.
ABLATION_CKPT_ROOT = "/kaggle/input/datasets/gmnguynhng/new-checkpoint/ablation"

# Đường dẫn 3 checkpoint FM chính (dùng làm FULL model, KHÔNG train
# riêng cho ablation study — tái dùng checkpoint chính đã có sẵn).
FULL_CKPT_PATTERN = "/kaggle/input/datasets/gmnguynhng/new-checkpoint/best_model_fm_seed{seed}_new_ver.pth"

# Ánh xạ tên hiển thị -> tên trong tên file checkpoint (khớp CHÍNH XÁC
# ảnh Kaggle: best_model_no_<X>_seed<N>.pth). Lưu ý tên file thật dùng
# "no_calib"/"no_heading"/"no_reg" (KHÔNG có "l_" ở giữa như tôi từng
# đoán sai ở bản config JSON trước đây).
CHECKPOINT_PATTERNS = {
    "FULL":            FULL_CKPT_PATTERN,
    "no_ot":           ABLATION_CKPT_ROOT + "/best_model_no_ot_seed{seed}.pth",
    "no_aug_c":        ABLATION_CKPT_ROOT + "/best_model_no_aug_c_seed{seed}.pth",
    "no_hard_reg":     ABLATION_CKPT_ROOT + "/best_model_no_hard_reg_seed{seed}.pth",
    "no_calib":        ABLATION_CKPT_ROOT + "/best_model_no_calib_seed{seed}.pth",
    "no_heading":      ABLATION_CKPT_ROOT + "/best_model_no_heading_seed{seed}.pth",
    "no_reg":          ABLATION_CKPT_ROOT + "/best_model_no_reg_seed{seed}.pth",
    "no_kendall":      ABLATION_CKPT_ROOT + "/best_model_no_kendall_seed{seed}.pth",
    "no_horizon_nll":  ABLATION_CKPT_ROOT + "/best_model_no_horizon_nll_seed{seed}.pth",
    "no_film":         ABLATION_CKPT_ROOT + "/best_model_no_film_seed{seed}.pth",
}
# ═══════════════════════════════════════════════════════════════════════


def run_one_ablation(ablation_runner_path: str, name: str, pattern: str,
                      seeds: list, dataset_root: str, split: str,
                      n_ensemble: int, output_dir: str,
                      use_tta: bool = False, n_tta: int = 5,
                      use_curvature_score: bool = False,
                      ddim_steps: int = None) -> dict:
    """Gọi ablation_runner.py --mode multi_seed cho 1 ablation, trả về
    dict aggregated (đã có sẵn cấu trúc mean/std đúng chuẩn từ
    run_multi_seed trong ablation_runner.py).

    [ADD-TTA] use_tta/n_tta/use_curvature_score/ddim_steps được truyền
    thẳng xuống ablation_runner.py's --mode multi_seed (đã patch để hỗ
    trợ đúng 4 flag này, cùng tên/mặc định với evaluate_multi_model.py
    -- xem ablation_runner.py's run_multi_seed() docstring). Trước khi
    có patch này, mọi lệnh gọi từ hàm này đều CHẠY THIẾU TTA so với
    bảng kết quả chính (multi_model_test.json, luôn được tạo bằng
    --use_tta trong evaluate_multi_model.py), khiến 9 con số ablation +
    dòng FULL không so sánh công bằng được với bảng chính -- cùng loại
    lỗi apples-to-oranges đã tìm thấy và sửa ở evaluate_full.py's
    k_n_joint_sweep(). Mặc định vẫn TẮT (use_tta=False) để không âm
    thầm đổi hành vi của bất kỳ lệnh gọi cũ nào chưa truyền các tham
    số này; bật qua --use_tta ở CLI của CHÍNH script này (xem main()
    bên dưới) khi muốn ablation table khớp cấu hình với bảng chính.
    """
    sub_out = os.path.join(output_dir, name)
    os.makedirs(sub_out, exist_ok=True)
    cmd = [
        sys.executable, ablation_runner_path,
        "--mode", "multi_seed",
        "--checkpoint_pattern", pattern,
        "--seeds", *[str(s) for s in seeds],
        "--dataset_root", dataset_root,
        "--split", split,
        "--n_ensemble", str(n_ensemble),
        "--output_dir", sub_out,
    ]
    if use_tta:
        cmd += ["--use_tta", "--n_tta", str(n_tta)]
    if use_curvature_score:
        cmd += ["--use_curvature_score"]
    if ddim_steps is not None:
        cmd += ["--ddim_steps", str(ddim_steps)]
    print(f"\n{'='*70}\nAblation: {name}\n{'='*70}")
    print(" ".join(cmd))
    ret = subprocess.run(cmd, capture_output=False)
    if ret.returncode != 0:
        print(f"  ⚠ Lệnh thất bại cho ablation '{name}' (returncode={ret.returncode})")
        return {}

    result_path = os.path.join(sub_out, "multi_seed_results.json")
    if not os.path.exists(result_path):
        print(f"  ⚠ Không tìm thấy {result_path}")
        return {}
    with open(result_path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_ensemble", type=int, default=20,
                     help="PHẢI khớp K* đã chọn trên validation (xem "
                          "select_kn.py) để nhất quán với bảng so sánh "
                          "chính -- không dùng mù mặc định 20 nếu K* "
                          "khác.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ablation_runner", required=True,
                     help="Đường dẫn tới ablation_runner.py")
    ap.add_argument("--output_dir", default="ablation_summary")
    # [ADD-TTA] Same flag names/defaults as evaluate_multi_model.py and
    # (now) ablation_runner.py's own CLI, passed straight through to
    # run_one_ablation() -> ablation_runner.py's --mode multi_seed for
    # every one of the 9 ablations + FULL. Default OFF (False/None) so
    # existing invocations without these flags reproduce the exact same
    # command lines as before this patch -- pass --use_tta here once to
    # make every ablation in this run use the identical inference-time
    # configuration as the main results table, rather than adding the
    # flag separately to each of the 10 subprocess calls by hand.
    ap.add_argument("--use_tta", action="store_true", default=False,
                     help="Enable test-time augmentation for every "
                          "ablation run (FM only), matching "
                          "evaluate_multi_model.py's --use_tta and the "
                          "main results table's configuration. Without "
                          "this, ablation ADE/ATE/CTE are NOT directly "
                          "comparable to a --use_tta main table.")
    ap.add_argument("--n_tta", type=int, default=5,
                     help="Number of TTA scales; only relevant with "
                          "--use_tta.")
    ap.add_argument("--use_curvature_score", action="store_true", default=False,
                     help="Enable the 5th physics-score re-ranking "
                          "component for every ablation run (FM only). "
                          "Pure inference-time change, no retraining "
                          "needed.")
    ap.add_argument("--ddim_steps", type=int, default=None,
                     help="Override ODE integration steps at inference "
                          "for every ablation run (FM only). None "
                          "(default) defers to each checkpoint's own "
                          "trained value.")
    ap.add_argument("--full_name", default="FULL",
                     help="Tên hiển thị cho full model trong bảng so sánh "
                          "cuối (thường dùng lại 3 checkpoint FM chính, "
                          "không train riêng cho ablation study).")
    ap.add_argument("--skip_missing", action="store_true",
                     help="Nếu bật: tự bỏ qua ablation nào có checkpoint "
                          "không tồn tại thay vì dừng chương trình. Mặc "
                          "định TẮT -- kiểm tra file thiếu ngay từ đầu để "
                          "tránh chạy dở dang rồi mới phát hiện lỗi path.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Kiểm tra sớm: mọi checkpoint trong CHECKPOINT_PATTERNS có tồn
    # tại trên đĩa không, TRƯỚC KHI chạy bất kỳ eval nào (mỗi lần eval
    # 3 seed tốn thời gian, không nên để phát hiện thiếu file giữa
    # chừng). In rõ danh sách thiếu để bạn biết đúng path nào cần sửa
    # trong CHECKPOINT_PATTERNS ở đầu file. ──────────────────────────
    missing = []
    for name, pattern in CHECKPOINT_PATTERNS.items():
        for seed in args.seeds:
            p = pattern.format(seed=seed)
            if not os.path.exists(p):
                missing.append((name, seed, p))
    if missing:
        print(f"  ⚠ THIẾU {len(missing)} checkpoint (kiểm tra lại "
              f"CHECKPOINT_PATTERNS ở đầu file):")
        for name, seed, p in missing:
            print(f"    - {name} seed={seed}: {p}")
        if not args.skip_missing:
            print("\n  Dừng lại (dùng --skip_missing nếu muốn tự bỏ qua "
                  "các ablation thiếu và chạy phần còn lại).")
            return
        print("  --skip_missing bật -> tiếp tục, các ablation thiếu sẽ "
              "báo lỗi ở bước chạy thật (ablation_runner.py tự in "
              "'Missing checkpoint') và bị loại khỏi bảng cuối.\n")

    all_results = {}
    for name, pattern in CHECKPOINT_PATTERNS.items():
        res = run_one_ablation(args.ablation_runner, name, pattern,
                                args.seeds, args.dataset_root, args.split,
                                args.n_ensemble, args.output_dir,
                                use_tta=args.use_tta, n_tta=args.n_tta,
                                use_curvature_score=args.use_curvature_score,
                                ddim_steps=args.ddim_steps)
        if res:
            all_results[name] = res

    # ── In bảng tổng hợp cuối, format giống hệt print_main_table() ──────
    print(f"\n\n{'='*100}")
    print(f"  BẢNG TỔNG HỢP ABLATION (mean ± std qua {len(args.seeds)} seed, "
          f"n_ensemble={args.n_ensemble})")
    print(f"{'='*100}")
    header = f"  {'Ablation':<20} {'n_seeds':>7} {'ADE':>18} {'ATE':>18} {'CTE':>18}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    summary_for_json = {}
    for name, res in all_results.items():
        agg = res.get("aggregated", {})
        n_seeds_ok = len(res.get("per_seed", []))
        row = (f"  {name:<20} {n_seeds_ok:>7} "
               f"{agg.get('ADE', float('nan')):>8.2f}±{agg.get('ADE_std', float('nan')):<7.2f} "
               f"{agg.get('ATE', float('nan')):>8.2f}±{agg.get('ATE_std', float('nan')):<7.2f} "
               f"{agg.get('CTE', float('nan')):>8.2f}±{agg.get('CTE_std', float('nan')):<7.2f}")
        print(row)
        summary_for_json[name] = agg

        if n_seeds_ok < len(args.seeds):
            print(f"    ⚠ Chỉ {n_seeds_ok}/{len(args.seeds)} seed thành công "
                  f"cho '{name}' -- kiểm tra checkpoint bị thiếu ở trên.")

    # ── So với FULL model (nếu có trong config) — đánh dấu hiệu ứng có
    # vượt qua 1x std của FULL không (ngưỡng thực dụng, KHÔNG thay thế
    # cho kiểm định thống kê đầy đủ — chỉ là cờ hiệu nhanh để biết ablation
    # nào đáng ưu tiên kiểm định kỹ hơn bằng paired test riêng) ──────────
    if args.full_name in all_results:
        full_agg = all_results[args.full_name].get("aggregated", {})
        full_ade_std = full_agg.get("ADE_std", None)
        if full_ade_std:
            print(f"\n  So với {args.full_name} "
                  f"(ADE={full_agg.get('ADE'):.2f}±{full_ade_std:.2f}):")
            for name, res in all_results.items():
                if name == args.full_name:
                    continue
                agg = res.get("aggregated", {})
                diff = agg.get("ADE", float("nan")) - full_agg.get("ADE", float("nan"))
                n_std = diff / full_ade_std if full_ade_std else float("nan")
                flag = ""
                if abs(n_std) >= 2.0:
                    flag = "  <-- lệch >=2x std FULL, đáng chú ý"
                print(f"    {name:<20} lệch={diff:+7.2f}km  ({n_std:+.2f}x std FULL){flag}")

    out_path = os.path.join(args.output_dir, "ablation_multiseed_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary_for_json, f, indent=2)
    print(f"\n  Đã lưu bảng tổng hợp → {out_path}")


if __name__ == "__main__":
    main()