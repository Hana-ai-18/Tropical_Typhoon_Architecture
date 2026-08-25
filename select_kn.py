"""
select_kn.py — Tự động chọn (K, N) tối ưu từ file sweep trên VALIDATION set.
Không được chạy trên test set — chỉ dùng validation, sau đó đóng băng (K*, N*)
để dùng CHO MỌI bảng trong paper (main comparison, ablation, XAI).

BẢN 3 — SỬA QUY TRÌNH: tách biệt 2 bước rõ ràng, minh bạch, dễ giải
trình với reviewer, thay vì gộp chung 1 công thức điểm số (bản 1, 2):

  Bước A — CHỌN N TRƯỚC, chỉ dựa vào calibration (spread_skill_ratio),
           KHÔNG nhìn ADE. Chọn N nhỏ nhất đạt ngưỡng calibration chấp
           nhận được (mặc định: spread_skill_ratio trung bình >= 0.5,
           tự chỉnh qua --min_skill_ratio). Lý do tách bước: đã xác
           nhận N=1 gây ensemble collapse rõ rệt (spread/ADE=0.066),
           không được chọn N chỉ vì nó cho ADE thấp nhất -- đây chính
           là bẫy dẫn tới co cụm.

  Bước B — SAU KHI có N cố định, mới chọn K theo ADE thấp nhất. Đã xác
           nhận qua sweep thực tế: K hầu như KHÔNG ảnh hưởng đến spread
           (so K=5 vs K=30 ở cùng N, spread lệch <1%), nên chọn K theo
           ADE không đánh đổi gì thêm về calibration -- bước này "miễn
           phí".

Cách dùng:
    python select_kn.py --sweep_json eval_full_VAL/k_n_sweep_multiseed_fm_val.json \
                         --min_skill_ratio 0.5 \
                         --out best_kn.json
"""
import json
import argparse
from collections import defaultdict


def select_n_by_calibration(sweep: dict, min_skill_ratio: float,
                             skill_field: str = "spread_skill_ratio") -> tuple:
    """
    Bước A: với mỗi N (gộp qua mọi K), lấy calibration trung bình.
    Trả về (N_chosen, bảng_chi_tiết_theo_N).

    [FIX NaN bug] Trước đây, khi v.get(skill_field) trả về NaN (không
    phải None -- xảy ra ở MỌI cell K=1, vì spread cần tối thiểu 2
    candidate để tính pairwise distance, K=1 chỉ có 1 candidate nên
    spread/spread_skill_ratio luôn là NaN theo đúng thiết kế, không
    phải lỗi), điều kiện "if ratio is None" KHÔNG kích hoạt (NaN is not
    None, đúng theo Python), nên giá trị NaN bị đẩy thẳng vào danh sách
    ratios của N đó mà không qua fallback. Bước lọc "if r is not None"
    sau đó cũng không loại được NaN vì lý do tương tự. Hậu quả: chỉ cần
    1/5 giá trị K của một N là NaN (luôn đúng tại K=1), sum(ratios) của
    N đó bị lan truyền thành NaN, xóa sạch 4 giá trị K hợp lệ còn lại
    -- khiến MỌI N đều báo mean_ratio=nan dù dữ liệu thật (K=5..30) đầy
    đủ và hợp lệ. Patch: dùng r == r (loại NaN, vì NaN != NaN là true
    duy nhất với NaN trong so sánh float) thay vì "r is not None" ở cả
    hai chỗ lọc.
    """
    has_skill = any(skill_field in v for v in sweep.values())
    by_n = defaultdict(list)
    for v in sweep.values():
        ratio = v.get(skill_field) if has_skill else None
        if ratio is None or ratio != ratio:  # None hoặc NaN -> thử fallback
            spread = v.get("spread")
            ade = v["ADE"]
            ratio = (spread / ade) if (spread is not None and spread == spread) else None
        by_n[v["N"]].append(ratio)

    n_table = []
    for n, ratios in sorted(by_n.items()):
        valid = [r for r in ratios if r is not None and r == r]  # loại cả None và NaN
        mean_ratio = sum(valid) / len(valid) if valid else None
        n_table.append({"N": n, "mean_ratio": mean_ratio,
                         "source": skill_field if has_skill else "spread/ADE (proxy thô)",
                         "n_valid_of_total": f"{len(valid)}/{len(ratios)}"})

    # chọn N nhỏ nhất đạt ngưỡng (N nhỏ hơn -> ít cost tính toán hơn khi
    # suy luận nhiều bước, ưu tiên nếu có nhiều N cùng đạt ngưỡng)
    candidates = [r for r in n_table if r["mean_ratio"] is not None
                  and r["mean_ratio"] >= min_skill_ratio]
    if candidates:
        chosen = min(candidates, key=lambda r: r["N"])
    else:
        # không N nào đạt ngưỡng -> lấy N có ratio cao nhất (gần ngưỡng nhất)
        chosen = max(n_table, key=lambda r: (r["mean_ratio"] or -1))
        print(f"  ⚠ CẢNH BÁO: không N nào đạt min_skill_ratio={min_skill_ratio}. "
              f"Lấy N={chosen['N']} (ratio cao nhất hiện có = {chosen['mean_ratio']:.3f}). "
              f"Cân nhắc hạ min_skill_ratio hoặc mở rộng --n_values lúc sweep.")
    return chosen["N"], n_table


def select_k_by_ade(sweep: dict, n_fixed: int) -> tuple:
    """Bước B: với N đã cố định, chọn K có ADE thấp nhất."""
    rows = [v for v in sweep.values() if v["N"] == n_fixed]
    rows.sort(key=lambda r: r["ADE"])
    return rows[0]["K"], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_json", required=True)
    ap.add_argument("--min_skill_ratio", type=float, default=0.5,
                     help="Ngưỡng calibration tối thiểu chấp nhận được. "
                          "1.0 = lý tưởng (chuẩn UQ). Nhiều paper thực tế "
                          "chấp nhận 0.5-1.3. TỰ QUYẾT ĐỊNH số này dựa "
                          "trên phân tích riêng, không dùng mù giá trị "
                          "mặc định.")
    ap.add_argument("--out", default="best_kn.json")
    args = ap.parse_args()

    d = json.load(open(args.sweep_json))
    sweep = d["k_n_sweep"] if "k_n_sweep" in d else d

    print("=" * 70)
    print("BƯỚC A — Chọn N theo calibration (KHÔNG nhìn ADE)")
    print("=" * 70)
    n_star, n_table = select_n_by_calibration(sweep, args.min_skill_ratio)
    print(f"{'N':>4} {'mean_ratio':>12} {'n_valid':>10} {'nguồn':>25}")
    for r in n_table:
        mark = " <== CHỌN" if r["N"] == n_star else ""
        ratio_str = f"{r['mean_ratio']:.3f}" if r["mean_ratio"] is not None else "N/A"
        print(f"{r['N']:>4} {ratio_str:>12} {r.get('n_valid_of_total',''):>10} {r['source']:>25}{mark}")
    print(f"\n>>> N* = {n_star} <<<")

    print("\n" + "=" * 70)
    print(f"BƯỚC B — Với N*={n_star} cố định, chọn K theo ADE thấp nhất")
    print("=" * 70)
    k_star, k_rows = select_k_by_ade(sweep, n_star)
    print(f"{'K':>4} {'ADE':>8}")
    for r in k_rows:
        mark = " <== CHỌN" if r["K"] == k_star else ""
        print(f"{r['K']:>4} {r['ADE']:>8.2f}{mark}")
    print(f"\n>>> K* = {k_star} <<<")

    result = {"N": n_star, "K": k_star,
              "ADE_at_KN": next(r["ADE"] for r in k_rows if r["K"] == k_star),
              "min_skill_ratio_used": args.min_skill_ratio}
    print(f"\n>>> KẾT QUẢ CUỐI: K*={k_star}, N*={n_star} <<<")
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"Đã lưu vào {args.out}")


if __name__ == "__main__":
    main()