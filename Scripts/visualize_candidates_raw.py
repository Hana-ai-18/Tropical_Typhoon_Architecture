"""
visualize_candidates_raw.py
════════════════════════════════════════════════════════════════════════════
Vẽ N_SHOW đường dự báo RIÊNG LẺ (mặc định 5) cho MỘT thời điểm cụ thể, LẤY
TRỰC TIẾP TỪ TOÀN BỘ K CANDIDATE model.sample() sinh ra TRƯỚC bước physics-
score re-ranking/top-3 weighted-average — tức đúng "raw" candidate, chưa
qua chọn lọc, khác với forecast_*.png thông thường (vốn chỉ vẽ 1 đường
pred_mean = kết quả SAU top-3).

NGUỒN GỐC DỮ LIỆU (đã verify trực tiếp từ flow_matching_model.py's
TCFlowMatching.sample()): trong vòng lặp `for _ in range(K): ... all_traj.
append(...)`, mỗi phần tử của `all_traj` là một candidate ĐỘC LẬP, tích hợp
qua đúng N bước Euler (giống hệt candidate cuối cùng), nhưng CHƯA được đưa
qua `_physics_score(...)` / `scores.topk(3, ...)` / weighted-average — đây
CHÍNH XÁC là "K candidates trước khi chọn top-3" cần vẽ ở đây. Hàm
`model.sample(..., num_ensemble=K)` trả về `(pred_mean, pred_Me, all_traj)`
với `all_traj` có shape [K, T, B, 2] — ta chỉ cần lấy K_SHOW phần tử ĐẦU
TIÊN của trục 0 (không cần re-implement lại logic sampling).

Vẫn vẽ Ground Truth (khác với visualize_no_gt.py) để so sánh trực quan
candidate nào gần đúng nhất so với những gì thực tế xảy ra.

CÁCH DÙNG (giống hệt convention của visual_evaluate_mode.py):
    python visualize_candidates_raw.py \
        --model_path /path/to/checkpoint.pth \
        --TC_data_path /path/to/dataset \
        --tc_name YANCY --tc_date 1990081412 \
        --dset_type test \
        --num_candidates_show 5 \
        --ode_steps 40 \
        --output_dir ./figs_candidates
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("  Warning: cartopy not found — using plain axes.")

# Toàn bộ hạ tầng dùng lại NGUYÊN VẸN từ file gốc: style, helper hình học,
# I/O dataset, load model, cách vẽ nền bản đồ — KHÔNG re-implement để tránh
# lệch hành vi/lỗi số học so với script chính đã kiểm chứng.
from visual_evaluate_mode import (
    STYLE, set_seed, move_batch, denorm_traj, to_deg, haversine_km,
    resolve_date, find_target, list_available, make_map_ax, _draw_geo_labels,
    _extract_seq, _extract_ens, load_model_and_data, seq_collate,
)


# ── Candidate colors (distinguishable, colorblind-friendlier palette) ──────
CANDIDATE_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#e377c2",  # pink
    "#8c564b",  # brown
    "#17becf",  # cyan
    "#bcbd22",  # olive
]


def run_inference_raw_candidates(model, target, device, ode_steps, num_candidates):
    """
    Giống hệt run_inference() của visual_evaluate_mode.py về phần
    obs/gt/error, nhưng thay vì chỉ giữ pred_mean (sau top-3), trả về
    TOÀN BỘ all_trajs (trước top-3) đã convert sang độ, cắt còn đúng
    num_candidates phần tử đầu tiên.
    """
    batch = move_batch(seq_collate([target]), device)
    with torch.no_grad():
        pred_mean, pred_Me, all_trajs = model.sample(
            batch, num_ensemble=num_candidates, ddim_steps=ode_steps
        )
    # all_trajs: [K, T, B, 2] — K candidate ĐỘC LẬP, TRƯỚC physics-score
    # re-ranking/top-3 (xem docstring đầu file để có bằng chứng từ code
    # gốc). Không cắt bớt ở đây: num_ensemble đã truyền đúng K mong muốn.

    obs_n = _extract_seq(batch[0])
    gt_n  = _extract_seq(batch[1])
    ens_n = _extract_ens(all_trajs)   # [K, T, 2]

    obs_abs_mean = np.abs(obs_n).mean()
    # Dùng candidate đầu tiên để auto-detect delta/absolute, giống logic gốc.
    pred_abs_mean = np.abs(ens_n[0]).mean()
    IS_DELTA = pred_abs_mean < obs_abs_mean * 0.15

    if IS_DELTA:
        ens_abs = obs_n[-1:] + np.cumsum(ens_n, axis=1)
    else:
        ens_abs = ens_n

    obs_deg = to_deg(denorm_traj(obs_n))
    gt_deg  = to_deg(denorm_traj(gt_n))
    ens_deg = to_deg(denorm_traj(ens_abs))   # [K, T, 2]

    if ens_deg.shape[1] != gt_deg.shape[0]:
        T_min = min(ens_deg.shape[1], gt_deg.shape[0])
        ens_deg = ens_deg[:, :T_min]
        gt_deg  = gt_deg[:T_min]

    # Sai số Haversine của TỪNG candidate riêng lẻ so với GT (để in ra
    # console, không bắt buộc phải hiển thị trên hình).
    errors_per_candidate = np.array([
        haversine_km(ens_deg[k], gt_deg) for k in range(ens_deg.shape[0])
    ])   # [K, T]

    return obs_deg, gt_deg, ens_deg, errors_per_candidate


def plot_raw_candidates(ax, lon_range, lat_range, obs_deg, gt_deg, cand_deg,
                          title="", dt_str=""):
    """
    Vẽ obs (đen), ground truth (đường chính, nổi bật), và cand_deg.shape[0]
    candidate riêng lẻ (mỗi đường 1 màu, mảnh hơn GT, KHÔNG có cone/vùng
    tô — đây là điểm khác biệt chính so với _plot_on_ax của file gốc, vốn
    chỉ vẽ pred_mean + cone tổng hợp).
    """
    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    outline   = [pe.withStroke(linewidth=2.5, foreground="white")]
    cur_pos   = obs_deg[-1]

    def _plot(x, y, fmt=None, **kw):
        args_ = [x, y] + ([fmt] if fmt is not None else [])
        if HAS_CARTOPY:
            ax.plot(*args_, transform=transform, **kw)
        else:
            ax.plot(*args_, **kw)

    def _scatter(x, y, **kw):
        if HAS_CARTOPY:
            ax.scatter(x, y, transform=transform, **kw)
        else:
            ax.scatter(x, y, **kw)

    _draw_geo_labels(ax, lon_range, lat_range, transform)

    # Observed track
    _plot(obs_deg[:, 0], obs_deg[:, 1], fmt="o-",
          color=STYLE["obs_color"], linewidth=STYLE["lw_thin"], markersize=5,
          markeredgecolor="white", markeredgewidth=0.8,
          zorder=7, path_effects=outline)

    # Ground truth
    gt_lon = np.concatenate([[cur_pos[0]], gt_deg[:, 0]])
    gt_lat = np.concatenate([[cur_pos[1]], gt_deg[:, 1]])
    _plot(gt_lon, gt_lat, fmt="o-",
          color=STYLE["gt_color"], linewidth=STYLE["lw_main"],
          markersize=STYLE["marker_size"],
          markeredgecolor="white", markeredgewidth=1.2,
          zorder=8, path_effects=outline)

    # K raw candidates — mỗi đường 1 màu riêng, mảnh, không markers dày
    # để không che lẫn nhau; alpha<1 để đường chồng lấn vẫn phân biệt được.
    K = cand_deg.shape[0]
    candidate_handles = []
    for k in range(K):
        color = CANDIDATE_COLORS[k % len(CANDIDATE_COLORS)]
        c_lon = np.concatenate([[cur_pos[0]], cand_deg[k, :, 0]])
        c_lat = np.concatenate([[cur_pos[1]], cand_deg[k, :, 1]])
        _plot(c_lon, c_lat, fmt="o-", color=color, linewidth=1.6,
              markersize=3.5, alpha=0.85, zorder=9 + k,
              markeredgecolor="white", markeredgewidth=0.5)
        candidate_handles.append(
            Line2D([0], [0], color=color, lw=1.8, label=f"Candidate {k+1}"))

    # NOW star
    _scatter([cur_pos[0]], [cur_pos[1]],
             s=350, marker="*", color="#FFD700",
             edgecolors="black", linewidths=1.5, zorder=25)

    # Legend
    track_handles = [
        Line2D([0], [0], color=STYLE["obs_color"], lw=2, label="Observed"),
        Line2D([0], [0], color=STYLE["gt_color"],  lw=2, label="Ground truth"),
    ] + candidate_handles
    ax.legend(handles=track_handles, loc="lower right", fontsize=7.5,
              facecolor="white", edgecolor=STYLE["panel_edge"],
              labelcolor=STYLE["text_color"], framealpha=0.92,
              title=f"Legend ({K} raw candidates, pre-selection)",
              title_fontsize=8)

    ax.set_title(
        f"{title}\n{dt_str}", color=STYLE["text_color"], fontsize=11,
        fontweight="bold", pad=STYLE["title_pad"],
        bbox=dict(fc="white", alpha=0.9, ec=STYLE["panel_edge"], lw=1.2),
    )
    ax.set_facecolor(STYLE["bg_color"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["panel_edge"])


def visualize_candidates(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_name              = args.tc_name.strip().upper()
    t_date, was_snapped = resolve_date(args.tc_date)

    print(f"{'=' * 65}")
    print(f"  Raw-candidate visualize  |  {t_name}  @  {t_date}  |  "
          f"showing {args.num_candidates_show} of {args.num_candidates_show} "
          f"candidates (pre top-3 selection)")
    print(f"{'=' * 65}\n")

    model, dset = load_model_and_data(args, device, args.dset_type)

    target, matched_obs_len, actual_date = find_target(
        dset, t_name, t_date, args.obs_len
    )
    if target is None:
        print(f"  '{t_name} @ {t_date}' not found.")
        list_available(dset, t_name, args.obs_len)
        return
    if actual_date != t_date:
        t_date = actual_date

    print(f"  Found: {t_name} @ {t_date}\n")

    obs_deg, gt_deg, cand_deg, errors_per_candidate = run_inference_raw_candidates(
        model, target, device, args.ode_steps, args.num_candidates_show
    )

    print("  Per-candidate mean track error (km), averaged over all lead times:")
    for k in range(cand_deg.shape[0]):
        print(f"    Candidate {k+1}: {errors_per_candidate[k].mean():.1f} km")
    print()

    all_deg = np.vstack([obs_deg, gt_deg, cand_deg.reshape(-1, 2)])
    lon_span = all_deg[:, 0].max() - all_deg[:, 0].min()
    lat_span = all_deg[:, 1].max() - all_deg[:, 1].min()
    margin_lon = float(np.clip(lon_span * 0.10, 1.0, 4.5))
    margin_lat = float(np.clip(lat_span * 0.10, 1.0, 4.5))
    extra_lon_widen = max(0.0, (lat_span - lon_span) * 0.35)
    margin_lon += extra_lon_widen
    lon_range = (all_deg[:, 0].min() - margin_lon, all_deg[:, 0].max() + margin_lon)
    lat_range = (all_deg[:, 1].min() - margin_lat, all_deg[:, 1].max() + margin_lat)

    # Khung cố định, chữ nhật đứng — cùng cách tiếp cận đã áp dụng ở
    # visualize_forecast() trong file gốc: mở rộng lon_range để khớp
    # đúng tỷ lệ khung thay vì để figsize co giãn theo track.
    FIG_W, FIG_H = 9.0, 12.0
    target_aspect = FIG_W / FIG_H
    lon_span_cur  = lon_range[1] - lon_range[0]
    lat_span_cur  = max(lat_range[1] - lat_range[0], 0.01)
    cur_aspect    = lon_span_cur / lat_span_cur
    if cur_aspect < target_aspect:
        wanted_lon_span = target_aspect * lat_span_cur
        extra = (wanted_lon_span - lon_span_cur) / 2.0
        lon_range = (lon_range[0] - extra, lon_range[1] + extra)
    else:
        wanted_lat_span = lon_span_cur / target_aspect
        extra = (wanted_lat_span - lat_span_cur) / 2.0
        lat_range = (lat_range[0] - extra, lat_range[1] + extra)

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=STYLE["bg_color"])
    gs  = fig.add_gridspec(1, 1)
    ax_map = make_map_ax(fig, gs[0, 0], lon_range, lat_range)

    dt_str = datetime.strptime(t_date, "%Y%m%d%H").strftime("%d %b %Y  %H:%M UTC")
    fh     = cand_deg.shape[1] * 6

    plot_raw_candidates(ax_map, lon_range, lat_range, obs_deg, gt_deg, cand_deg,
                          title=t_name, dt_str=dt_str)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir,
                        f"candidates_{fh}h_{t_name}_{t_date}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=STYLE["bg_color"])
    plt.close()
    print(f"  Saved → {out}\n")


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_path",           default=None)
    p.add_argument("--TC_data_path",         required=True)
    p.add_argument("--output_dir",           default="figs_candidates")
    p.add_argument("--dset_type",            default="test")
    p.add_argument("--tc_name",              required=True)
    p.add_argument("--tc_date",              required=True)
    p.add_argument("--obs_len",              type=int, default=8)
    p.add_argument("--pred_len",             type=int, default=12)
    p.add_argument("--ode_steps",            type=int, default=40)
    p.add_argument("--num_candidates_show",  type=int, default=5,
                    help="Số candidate riêng lẻ hiển thị (trước top-3 "
                         "selection). Mặc định 5 theo yêu cầu.")
    # Tham số bắt buộc bởi data_loader()/load_model_and_data(), giữ đúng
    # default như visual_evaluate_mode.py để hành vi load dữ liệu nhất quán.
    p.add_argument("--test_year",            type=int, default=None)
    p.add_argument("--num_workers",          type=int, default=0)
    p.add_argument("--delim",                default=" ")
    p.add_argument("--skip",                 type=int, default=1)
    p.add_argument("--min_ped",              type=int, default=1)
    p.add_argument("--other_modal",          default="gph")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    visualize_candidates(args)
