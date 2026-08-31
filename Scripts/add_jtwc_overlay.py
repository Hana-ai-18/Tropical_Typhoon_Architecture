"""
add_jtwc_overlay.py
════════════════════════════════════════════════════════════════════════════
Vẽ THÊM đường dự báo chính thức JTWC (nét đứt màu cam, marker tam giác rỗng)
lên các ảnh forecast đã có sẵn, theo đúng style quan sát được từ ảnh mẫu
"forecast_2009061818_with_jtwc.png" (LINFA):
    - Đường: nét đứt (--), màu cam đậm (#FF8C00), linewidth ~3
    - Marker: tam giác (^), rỗng/viền (không fill đặc), màu cam nhạt hơn
      đường một chút, size vừa phải
    - Chú thích: KHÔNG nằm trong bảng Legend chính (Observed/Ground truth/
      Predicted/cone...) — là 1 nhãn nổi riêng biệt, đặt ngay cạnh đường
      JTWC, nền trắng bo góc, chữ "JTWC official forecast", kèm 1 đoạn
      mẫu đường+marker nhỏ phía trước chữ.

QUAN TRỌNG: script này KHÔNG re-run model, KHÔNG vẽ lại toàn bộ bản đồ từ
đầu — nó dùng lại đúng 3 file .png forecast ĐÃ CÓ SẴN (do
visual_evaluate_mode.py hoặc visual_evaluate_mode_no_gt.py xuất ra), MỞ
LẠI figure bằng cách VẼ THÊM (overlay) trực tiếp lên toạ độ pixel/geo đã
biết trước của bản đồ gốc là không khả thi (matplotlib không cho "mở lại"
1 PNG tĩnh và vẽ thêm theo toạ độ geo). Do đó, cách làm ĐÚNG và AN TOÀN là
CHẠY LẠI đúng lệnh visualize gốc (visual_evaluate_mode.py /
visual_evaluate_mode_no_gt.py) cho 3 thời điểm này, nhưng với 1 tham số
MỚI (--jtwc_data_file) trỏ tới 1 file JSON chứa toạ độ JTWC — script này
CHỈ định nghĩa hàm vẽ overlay JTWC dùng CHUNG, để import và gọi thêm
1 dòng duy nhất ngay sau khi bản đồ chính (obs/GT/pred/cone) đã vẽ xong,
trước khi lưu file.

CÁCH TÍCH HỢP (đã áp dụng sẵn trong visual_evaluate_mode.py, xem
`--jtwc_data_file` trong get_args() và lời gọi `draw_jtwc_overlay(...)`
ngay trước `plt.savefig(...)` trong visualize_forecast()/
visualize_batch_all_storms()):

    from add_jtwc_overlay import draw_jtwc_overlay
    draw_jtwc_overlay(ax_map, jtwc_points, gt_deg, transform)

với `jtwc_points` là list các dict {"hour": int, "lat": float, "lon": float}
lấy từ JSON đã chuẩn bị sẵn cho từng storm (xem JTWC_DATA bên dưới).
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib.lines as mlines


# ── Dữ liệu JTWC cho 3 storm — ĐANG CHỜ BẠN XÁC NHẬN ĐỦ CẢ 3 ────────────────
# Mỗi storm: list các mốc {"hour": forecast_hour, "lat": ..., "lon": ...}.
# DANAS đã có (bạn gửi ở lượt trước), LINFA/EWINIAR còn thiếu — điền vào
# đúng format bên dưới rồi chạy lại là dùng được ngay, không cần sửa gì
# thêm trong hàm draw_jtwc_overlay().
JTWC_DATA = {
    "DANAS_2019071612": {
        "forecast_time": "2019-07-16 12:00",
        "points": [
            {"hour": 0,   "lat": 17.2, "lon": 123.7, "intensity_kt": 30},
            {"hour": 12,  "lat": 18.2, "lon": 122.6, "intensity_kt": 30},
            {"hour": 24,  "lat": 19.7, "lon": 122.1, "intensity_kt": 35},
            {"hour": 36,  "lat": 21.6, "lon": 121.8, "intensity_kt": 40},
            {"hour": 48,  "lat": 23.9, "lon": 121.7, "intensity_kt": 40},
            {"hour": 72,  "lat": 28.2, "lon": 121.4, "intensity_kt": 45},
            {"hour": 96,  "lat": 32.3, "lon": 122.2, "intensity_kt": 40},
            {"hour": 120, "lat": 36.5, "lon": 123.8, "intensity_kt": 35},
        ],
    },
    "LINFA_2009061818": {
        "forecast_time": "2009-06-18 12:00",
        "points": [
            {"hour": 0,  "lat": 17.5, "lon": 116.4},
            {"hour": 12, "lat": 17.9, "lon": 116.9},
            {"hour": 24, "lat": 18.6, "lon": 117.8},
            {"hour": 36, "lat": 19.6, "lon": 118.8},
            {"hour": 48, "lat": 20.7, "lon": 119.8},
            {"hour": 72, "lat": 23.6, "lon": 121.9},
        ],
    },
    "EWINIAR_2024052506": {
        "forecast_time": "2024-05-25 06:00",
        "points": [
            {"hour": 0,  "lat": 12.8, "lon": 123.0},
            {"hour": 12, "lat": 13.9, "lon": 121.7},
            {"hour": 24, "lat": 14.8, "lon": 121.5},
            {"hour": 36, "lat": 15.6, "lon": 122.2},
            {"hour": 48, "lat": 16.6, "lon": 123.3},
            {"hour": 72, "lat": 19.4, "lon": 126.2},
        ],
    },
}


# ── Style, khớp đúng ảnh mẫu ─────────────────────────────────────────────
JTWC_LINE_COLOR   = "#FF8C00"   # cam đậm (dark orange) — đường nét đứt
JTWC_MARKER_COLOR = "#FFA94D"   # cam nhạt hơn — viền tam giác
JTWC_LINEWIDTH    = 3.0
JTWC_MARKERSIZE   = 11


def haversine_km(p1, p2):
    """p1, p2: (lat, lon) tuples, độ. Trả về khoảng cách km."""
    R = 6371.0
    lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def draw_jtwc_overlay(ax, jtwc_points, transform=None, gt_deg=None,
                        label_hours=(24, 48, 72)):
    """
    Vẽ đường JTWC (nét đứt cam) + marker tam giác + nhãn nổi riêng lên
    axes bản đồ ĐANG MỞ (gọi hàm này TRƯỚC plt.savefig(), NGAY SAU khi
    obs/GT/pred/cone/legend chính đã vẽ xong, để JTWC nằm ở zorder cao,
    không bị che).

    jtwc_points : list[dict] — mỗi phần tử {"hour", "lat", "lon", ...},
                  ĐÃ SẮP XẾP tăng dần theo "hour".
    gt_deg      : np.ndarray [T,2] (lon,lat) ground truth, degrees — nếu
                  truyền vào, hàm tự tính khoảng cách JTWC-vs-GT tại các
                  mốc trong label_hours để hiển thị "Xh / Ykm" (giống
                  format đã dùng cho model prediction trong ảnh mẫu).
                  Nếu None (ví dụ file no_gt), chỉ hiển thị "Xh" không
                  kèm khoảng cách.
    label_hours : các mốc giờ cần gắn nhãn dọc đường (mặc định 24/48/72h,
                  giống ảnh mẫu).
    """
    if not jtwc_points:
        return

    lats = [p["lat"] for p in jtwc_points]
    lons = [p["lon"] for p in jtwc_points]

    plot_kwargs = dict(transform=transform) if transform is not None else {}

    # Đường nét đứt
    ax.plot(lons, lats, "--", color=JTWC_LINE_COLOR,
            linewidth=JTWC_LINEWIDTH, zorder=20, **plot_kwargs)
    # Marker tam giác rỗng tại mỗi forecast hour
    ax.plot(lons, lats, "^", color=JTWC_MARKER_COLOR,
            markersize=JTWC_MARKERSIZE, markeredgecolor=JTWC_LINE_COLOR,
            markeredgewidth=1.8, linestyle="None", zorder=21, **plot_kwargs)

    # Nhãn dọc đường tại các mốc quan tâm (24h/48h/72h mặc định)
    for p in jtwc_points:
        if p["hour"] not in label_hours:
            continue
        lbl = f"{p['hour']}h"
        if gt_deg is not None:
            # Xấp xỉ vị trí GT tại đúng mốc giờ này bằng nội suy tuyến
            # tính theo index (GT thường lấy mẫu 6h/step, T bước = 6*T h).
            gt_idx = p["hour"] // 6 - 1   # gt_deg[0] tương ứng +6h
            if 0 <= gt_idx < len(gt_deg):
                dist_km = haversine_km(
                    (p["lat"], p["lon"]), (gt_deg[gt_idx, 1], gt_deg[gt_idx, 0])
                )
                lbl += f"\n{dist_km:.0f}km"
        ax.annotate(
            lbl, xy=(p["lon"], p["lat"]), xytext=(6, 6),
            textcoords="offset points", fontsize=8, fontweight="bold",
            color=JTWC_LINE_COLOR, zorder=22,
            **({"transform": transform} if transform is not None else {}),
        )

    # "JTWC official forecast" giờ được thêm vào bảng Legend chính thay vì
    # nổi giữa bản đồ (trước đây dùng ax.annotate/ax.text ở axes fraction
    # (0.24–0.32, 0.865), tức góc trên-trái, gây đè lên track/cone). Lấy
    # lại legend đang có sẵn trên axes (đã được _plot_on_ax() vẽ trước đó),
    # gộp thêm 1 handle JTWC, rồi vẽ lại đúng 1 legend duy nhất — giữ
    # nguyên vị trí/style (loc, title, fontsize...) của legend gốc.
    jtwc_handle = mlines.Line2D(
        [], [], color=JTWC_LINE_COLOR, linestyle="--",
        linewidth=JTWC_LINEWIDTH, marker="^",
        markerfacecolor=JTWC_MARKER_COLOR, markeredgecolor=JTWC_LINE_COLOR,
        markeredgewidth=1.8, markersize=JTWC_MARKERSIZE * 0.7,
        label="JTWC official forecast",
    )

    existing_legend = ax.get_legend()
    if existing_legend is not None:
        # Ghi nhớ style của legend hiện có trước khi nó bị legend mới ghi đè
        loc          = getattr(existing_legend, "_loc", "lower right")
        title_obj    = existing_legend.get_title()
        title        = title_obj.get_text() if title_obj else None
        fontsize     = existing_legend.get_texts()[0].get_fontsize() if existing_legend.get_texts() else 7.5
        framealpha   = existing_legend.get_frame().get_alpha()
        handles      = [h for h in existing_legend.legend_handles] if hasattr(existing_legend, "legend_handles") \
                       else existing_legend.legendHandles
        labels       = [t.get_text() for t in existing_legend.get_texts()]
        existing_legend.remove()
    else:
        loc, title, fontsize, framealpha = "lower right", None, 7.5, 0.92
        handles, labels = [], []

    handles = list(handles) + [jtwc_handle]
    labels  = list(labels) + ["JTWC official forecast"]

    new_legend = ax.legend(
        handles=handles, labels=labels, loc=loc, fontsize=fontsize,
        framealpha=framealpha if framealpha is not None else 0.92,
        title=title, title_fontsize=8, ncol=1,
    )
    ax.add_artist(new_legend)


def load_jtwc_points(storm_key: str, json_path: str | None = None):
    """
    Lấy list điểm JTWC cho 1 storm, ưu tiên đọc từ file JSON ngoài (nếu
    truyền json_path) để không phải sửa code mỗi lần thêm storm mới;
    fallback về JTWC_DATA hard-code ở trên nếu không truyền/không có key.
    """
    if json_path:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if storm_key in data:
            return data[storm_key]["points"]
    if storm_key in JTWC_DATA:
        return JTWC_DATA[storm_key]["points"]
    return []