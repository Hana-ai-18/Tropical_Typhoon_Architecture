from __future__ import annotations

import os
import sys
import json
import re
import random
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from scipy.stats import chi2

try:
    from shapely.geometry import Point, Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("  Warning: cartopy not found — using plain axes.")

from Model.flow_matching_model import TCFlowMatching
from Model.paper_baseline_model import PaperBaseline
from Model.st_trans_model import STTrans
from Model.data.loader import data_loader
from Model.data.trajectoriesWithMe_unet_training import seq_collate


# ── Styling (paper/white, thay cho dark neon của v13) ───────────────────────
STYLE = dict(
    obs_color     = "#000000",   # đen — track quan sát
    gt_color      = "#1F5FBF",   # xanh dương — Actual Track (khớp ảnh mẫu)
    pred_color    = "#D62728",   # đỏ — Predicted (khớp ảnh mẫu)
    ens_color     = "#D62728",
    ens_alpha     = 0.05,
    marker_size   = 6,
    lw_main       = 2.0,
    lw_thin       = 1.3,
    bg_color      = "#FFFFFF",
    land_color    = "#FFFFFF",
    ocean_color   = "#EAF3FB",
    border_color  = "#BBBBBB",
    grid_color    = "#CCCCCC",
    grid_alpha    = 0.5,
    error_color   = "#B8860B",   # dark goldenrod — đọc được trên nền trắng
    title_pad     = 14,
    # [RESTYLE — NCHMF cone colors] Đổi từ đỏ/xanh dương (dễ nhầm với
    # pred_color/gt_color đang dùng đúng 2 màu này cho TRACK, gây trùng
    # màu track/cone) sang đúng tông tím/xanh lá của bản đồ "TIN BAO
    # KHAN CAP" chuẩn NCHMF: vùng ngoài (90%, "gió mạnh có thể xảy ra")
    # màu tím nhạt, vùng trong (50%, "tâm bão/ATNĐ có thể đi qua") màu
    # xanh lá -- đúng thứ tự lồng nhau (90% bao ngoài 50%) như ảnh mẫu.
    cone_50_fill  = "#5FA85F",   # xanh lá — vùng tâm bão/ATNĐ có thể qua (50%)
    cone_90_fill  = "#B08FD0",   # tím nhạt — vùng có thể có gió mạnh (90%)
    cone_50_alpha = 0.45,
    cone_90_alpha = 0.35,
    cone_edge_lw  = 0.0,         # [RESTYLE] bỏ viền đứt nét quanh cone --
                                  # ảnh mẫu NCHMF dùng vùng tô mượt, không
                                  # có đường biên rời rạc; viền đứt nét
                                  # trước đây là nguồn gây "rối mắt" đã
                                  # phản hồi, nay bỏ luôn cả 2 lớp 50%/90%.
    text_color    = "#000000",
    panel_edge    = "#888888",
    info_box_edge = "#2C4A7C",   # [NEW] viền khung info box kiểu NCHMF
    info_box_title_bg = "#EAF0F8",  # [NEW] nền dòng tiêu đề info box
)

# Màu riêng cho từng model khi vẽ nhiều model trên cùng bản đồ
# (--mode multi_model, xem plot_multi_model_comparison bên dưới).
MODEL_COLORS = {
    "FM":       "#D62728",
    "ST-Trans": "#FF7F0E",
    "LSTM":     "#2CA02C",
    "GRU":      "#9467BD",
    "RNN":      "#8C564B",
}

_CHI2_50 = chi2.ppf(0.50, df=2)
_CHI2_90 = chi2.ppf(0.90, df=2)

INTENSITY = [
    (0,   34,  "TD",       "#6699CC"),
    (34,  48,  "TS",       "#33AA33"),
    (48,  64,  "TY",       "#CCAA00"),
    (64,  84,  "Sev.TY",   "#FF8C00"),
    (84,  115, "Vis.TY",   "#E03C00"),
    (115, 999, "Super TY", "#B000B0"),
]


def wind_intensity(wind_kt):
    for lo, hi, name, color in INTENSITY:
        if lo <= wind_kt < hi:
            return name, color
    return "Super TY", "#FF00FF"


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def move_batch(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in x.items()}
    return tuple(out)


def denorm_traj(n):
    """Inverse-normalise trajectory array (any shape ending in 2)."""
    r = np.zeros_like(n)
    r[..., 0] = n[..., 0] * 50.0 + 1800.0
    r[..., 1] = n[..., 1] * 50.0
    return r


def to_deg(pts_01):
    return pts_01 / 10.0


def denorm_wind(wind_norm):
    return wind_norm * 25.0 + 40.0


def haversine_km(p1_deg, p2_deg):
    """
    Haversine distance (km) between two arrays of (lon, lat) points.
    Both arrays must have identical shape (..., 2).
    """
    # [FIX-6] Explicit shape check to catch silent broadcast errors early
    p1_deg = np.asarray(p1_deg)
    p2_deg = np.asarray(p2_deg)
    if p1_deg.shape != p2_deg.shape:
        raise ValueError(
            f"haversine_km: shape mismatch {p1_deg.shape} vs {p2_deg.shape}"
        )
    lat1 = np.deg2rad(p1_deg[..., 1])
    lat2 = np.deg2rad(p2_deg[..., 1])
    dlat = np.deg2rad(p2_deg[..., 1] - p1_deg[..., 1])
    dlon = np.deg2rad(p2_deg[..., 0] - p1_deg[..., 0])
    a    = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2.0 * 6371.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def detect_pred_len(ckpt_path):
    """
    Infer pred_len from the checkpoint — prefer model_cfg (authoritative,
    matches the exact args used at train time), fallback to
    velocity.pos_emb's shape if model_cfg is absent.

    [FIX-7] Wrapped in try/except so unusual checkpoint layouts don't crash.
    [FIX-8] Checkpoint thật lưu weights dưới key "model" (xác nhận qua
    evaluate_multi_model.py's load_fm() và train_flowmatching.py's
    _save()), KHÔNG PHẢI "model_state_dict"/"model_state".
    [FIX-12, quan trọng] Bug thật đã tìm và sửa: pattern tìm kiếm cũ
    (`"pos_enc" in k`) khớp NHẦM layer "encoder.env_enc.pos_enc_env"
    (positional encoding của ENVIRONMENT ENCODER — dữ liệu khí tượng,
    hoàn toàn không liên quan đến số bước dự báo), có shape (1,8,64) vì
    lý do riêng của feature map môi trường. Hàm cũ trả về 8 từ layer
    SAI này, trong khi layer THẬT quyết định pred_len là
    "velocity.pos_emb" (shape (1,12,256) — xác nhận qua kiểm tra trực
    tiếp checkpoint thật) và "velocity.step_emb.weight" (shape
    (12,256)) — cả 2 đều cho pred_len=12, KHỚP ĐÚNG với model_cfg's
    pred_len=12. Sự kiện này gây crash "shape mismatch (12,2) vs (8,2)"
    và khiến forecast bị cắt nhầm còn 48h thay vì đúng 72h.
    Giờ ưu tiên model_cfg (nguồn đáng tin cậy nhất, ghi trực tiếp từ
    args lúc train), chỉ dùng velocity.pos_emb làm fallback nếu
    checkpoint không có model_cfg.
    """
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_cfg = ck.get("model_cfg")
        if model_cfg and "pred_len" in model_cfg:
            return model_cfg["pred_len"]

        sd = ck.get("model", ck.get("model_state_dict", ck.get("model_state", ck)))
        for key in ["velocity.pos_emb", "net.pos_emb", "pos_emb"]:
            if key in sd:
                return sd[key].shape[1]
        for key in ["velocity.step_emb.weight", "step_emb.weight"]:
            if key in sd:
                return sd[key].shape[0]
    except Exception as e:
        print(f"  [WARN] detect_pred_len failed ({e}); defaulting to 12")
    return 12


# ── Snap & Search ──────────────────────────────────────────────────────────────

def snap_to_6h(date_str: str) -> str:
    """Floor YYYYMMDDHH to the nearest prior 6-hour mark."""
    s  = str(date_str).strip()[:10]
    dt = datetime.strptime(s, "%Y%m%d%H")
    dt = dt.replace(hour=(dt.hour // 6) * 6, minute=0, second=0)
    return dt.strftime("%Y%m%d%H")


def resolve_date(raw_date: str) -> tuple[str, bool]:
    original    = str(raw_date).strip()[:10]
    snapped     = snap_to_6h(original)
    was_snapped = snapped != original
    if was_snapped:
        print(f"  [SNAP] {original} → {snapped}  "
              f"(làm tròn về mốc 6h gần nhất trước đó)")
    return snapped, was_snapped


def _search_one_date(dset, t_name: str, t_date: str, obs_len: int):
    """
    Scan the entire dataset for a sample matching t_name + t_date.

    Priority:
      0  — date falls exactly at index obs_len  (ideal)
      N  — date falls at index obs_len+N        (later window, still usable)

    Returns (item, matched_idx) or (None, None).

    [FIX-2] best_pri initialised to float("inf") instead of hardcoded 99,
            so datasets with obs_len > 99 are handled correctly.
    """
    best_item = None
    best_idx  = None
    best_pri  = float("inf")   # FIX-2

    for i in range(len(dset)):
        item = dset[i]
        info = item[-1]
        if t_name not in str(info["old"][1]).strip().upper():
            continue
        for idx, td in enumerate(info["tydate"]):
            if str(td).strip() != t_date:
                continue
            if idx < obs_len:
                continue
            pri = 0 if idx == obs_len else (idx - obs_len + 1)
            if pri < best_pri:
                best_item, best_idx, best_pri = item, idx, pri

    return best_item, best_idx


def find_target(
    dset,
    t_name: str,
    t_date: str,
    obs_len: int,
    max_forward_steps: int = 20,
):
    """
    Flexible sample search:
      1. Try t_date (already snapped to 6h).
      2. If not found (e.g. TC track too short at that timestamp),
         advance by 6h up to max_forward_steps times.

    Returns (item, matched_idx, actual_date) or (None, None, None).

    [FIX-1] Removed the dead no-op `dt.replace(hour=dt.hour)`.
            `timedelta` import is now at module level (not inside the loop).
    """
    dt = datetime.strptime(t_date, "%Y%m%d%H")

    for step in range(max_forward_steps + 1):
        candidate = dt.strftime("%Y%m%d%H")
        item, idx = _search_one_date(dset, t_name, candidate, obs_len)
        if item is not None:
            if step > 0:
                print(
                    f"  [AUTO-FORWARD] {t_date} không có dữ liệu → "
                    f"dùng mốc tiếp theo: {candidate}  (+{step * 6}h)"
                )
            return item, idx, candidate
        dt = dt + timedelta(hours=6)   # FIX-1: no dead replace(); timedelta at top

    return None, None, None


def list_available(dset, t_name: str, obs_len: int, limit: int = 30):
    """Print available timestamps for TC t_name in the dataset."""
    shown = 0
    seen  = set()
    for i in range(len(dset)):
        info = dset[i][-1]
        name = str(info["old"][1]).strip().upper()
        if t_name not in name:
            continue
        td = str(info["tydate"][obs_len]).strip()
        if td in seen:
            continue
        seen.add(td)
        print(f"    {name:<15s}  @  {td}")
        shown += 1
        if shown >= limit:
            break

    if shown == 0:
        print(f"  (Không tìm thấy TC '{t_name}' trong dataset)")
        print("  Một số TC có sẵn:")
        seen_names: set[str] = set()
        for i in range(len(dset)):
            info = dset[i][-1]
            n = str(info["old"][1]).strip().upper()
            if n in seen_names:
                continue
            seen_names.add(n)
            print(f"    {n}")
            if len(seen_names) >= 15:
                break


# ── NHC-style smooth probability cone ─────────────────────────────────────────

def _gaussian_cone_boundary(pts_deg, chi2_thresh):
    """
    Fit a 2-D Gaussian to pts_deg and return the chi2 confidence ellipse.

    [FIX-3] np.cov with N==1 returns a scalar (not a 2×2 matrix), which
            crashes np.linalg.eigh.  Guard raised to N >= 3 to also ensure
            the covariance estimate is meaningful.
    """
    if len(pts_deg) < 3:          # FIX-3: was `< 3` in name only; enforce here
        return None
    mu               = pts_deg.mean(axis=0)
    cov              = np.cov(pts_deg.T)
    if cov.ndim < 2:               # FIX-3: scalar guard for N==1 or N==2 edge case
        return None
    cov             += np.eye(2) * 1e-8
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals          = np.maximum(eigvals, 1e-8)
    a                = np.sqrt(chi2_thresh * eigvals[-1])
    b                = np.sqrt(chi2_thresh * eigvals[0])
    theta            = np.linspace(0, 2 * np.pi, 64)
    ell              = np.stack([a * np.cos(theta), b * np.sin(theta)], axis=1)
    return ell @ eigvecs.T + mu


def draw_smooth_cone(ax, ens_deg, cur_pos_deg, transform=None, pred_deg=None):
    """
    [RESTYLE — NCHMF-style rounded cone] Thay hoàn toàn cách vẽ cone cũ
    (nối 2 đường biên trái/phải của ellipse tại mỗi mốc thời gian, tạo
    hình thuôn nhọn ở đầu NOW và cuối 72h) bằng cách hợp nhất (union)
    TOÀN BỘ ellipse xác suất tại từng mốc thời gian thành một vùng mượt
    duy nhất -- đúng kiểu "TIN BAO KHAN CAP" của NCHMF (ảnh mẫu): mỗi
    mốc là một hình tròn/ellipse riêng, và vùng tô là hợp bao mượt của
    tất cả các hình tròn liên tiếp đó. Vì bản thân mỗi ellipse đã tròn,
    vùng hợp nhất tự động tròn ở đầu (quanh NOW) và tròn ở cuối (quanh
    72h) mà không cần xử lý riêng — khác hẳn cách nối trái/phải cũ vốn
    luôn cho ra đầu nhọn tại hai đầu mút.

    Cần thư viện `shapely` để union nhiều ellipse thành 1 polygon mượt.
    Nếu môi trường không có shapely (HAS_SHAPELY=False), lùi về đúng
    cách vẽ cũ (nối trái/phải) để không làm hỏng toàn bộ script.
    """
    S, T, _ = ens_deg.shape
    if S < 3:
        return

    def _fill_polygon(poly, color, alpha, zo):
        """Vẽ 1 shapely Polygon (có thể có lỗ trong, hoặc là MultiPolygon)
        lên ax, kèm 1 viền TRẮNG mảnh quanh biên ngoài -- đúng chi tiết
        "đường viền trắng" thấy trong ảnh mẫu NCHMF, giúp vùng màu nổi
        bật, tách bạch khỏi nền bản đồ thay vì tô liền không viền."""
        polys = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
        for p in polys:
            xs, ys = p.exterior.xy
            kw = dict(color=color, alpha=alpha, zorder=zo, linewidth=0)
            if HAS_CARTOPY and transform is not None:
                ax.fill(xs, ys, transform=transform, **kw)
                ax.plot(xs, ys, color="white", linewidth=1.1, alpha=0.9,
                        zorder=zo + 0.5, transform=transform)
            else:
                ax.fill(xs, ys, **kw)
                ax.plot(xs, ys, color="white", linewidth=1.1, alpha=0.9,
                        zorder=zo + 0.5)

    def _fill_legacy(verts, color, alpha, zo):
        v = np.vstack([verts, verts[0]])
        kw = dict(color=color, alpha=alpha, zorder=zo, linewidth=0)
        if HAS_CARTOPY and transform is not None:
            ax.fill(v[:, 0], v[:, 1], transform=transform, **kw)
        else:
            ax.fill(v[:, 0], v[:, 1], **kw)

    def _cone_union(chi2_thresh):
        """Trả về 1 shapely (Multi)Polygon = union của ellipse xác suất
        tại từng mốc thời gian (bao gồm cả mốc NOW, xem như 1 điểm/
        ellipse suy biến rất nhỏ, để vùng cone bắt đầu đúng từ vị trí
        hiện tại của bão thay vì chỉ từ mốc dự báo đầu tiên).

        [FIX — vùng liền mạch] Nếu khoảng cách tâm giữa 2 mốc liên tiếp
        lớn hơn tổng bán kính "hiệu dụng" của 2 ellipse đó, 2 hình tròn
        sẽ KHÔNG chạm nhau và union() cho ra nhiều mảnh rời rạc (khác
        ảnh mẫu NCHMF, luôn là 1 dải liền từ NOW tới 72h). Để đảm bảo
        liền mạch, chèn thêm các ellipse nội suy TUYẾN TÍNH (tâm + kích
        thước) giữa mỗi cặp mốc liên tiếp, đủ dày để hình tròn kế cận
        luôn overlap nhau bất kể ensemble spread giãn nhanh cỡ nào.
        """
        def _ellipse_at(mu, a, b, theta_vec=None, eigvecs=None):
            theta = np.linspace(0, 2 * np.pi, 48)
            ell = np.stack([a * np.cos(theta), b * np.sin(theta)], axis=1)
            if eigvecs is not None:
                ell = ell @ eigvecs.T
            return ell + mu

        # Thu thập tham số ellipse (mu, a, b, eigvecs) tại từng mốc T,
        # cộng thêm mốc "NOW" suy biến rất nhỏ ở đầu.
        # [FIX — đối xứng quanh track đỏ] mu ở đây LUÔN lấy đúng điểm
        # trên đường dự báo trung bình (pred_deg[t], chính là đường đỏ
        # đang vẽ), KHÔNG lấy lại mean riêng của ensemble tại mốc đó.
        # Trước đây "mu = pts.mean(axis=0)" có thể lệch khỏi pred_deg[t]
        # do sai số làm tròn / cách tính mean khác nhau giữa 2 nơi, khiến
        # tâm ellipse xê dịch khỏi đường đỏ -> cone nhìn "lệch tâm". Giờ
        # ellipse luôn tâm đúng tại điểm trên đường đỏ, hình dạng (a, b,
        # eigvecs) vẫn phản ánh độ phân tán thật của ensemble quanh điểm
        # đó, nhưng vị trí tâm thì khớp 100% với track đỏ.
        params = [(cur_pos_deg, 1e-3, 1e-3, np.eye(2))]
        for t in range(T):
            pts = ens_deg[:, t, :]
            if len(pts) < 3:
                continue
            mu_pred = pred_deg[t] if pred_deg is not None else pts.mean(axis=0)
            mu_ens = pts.mean(axis=0)
            cov = np.cov((pts - mu_ens).T)  # lệch tâm quanh mean ensemble thật
            if cov.ndim < 2:
                continue
            cov = cov + np.eye(2) * 1e-8
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-8)
            a = np.sqrt(chi2_thresh * eigvals[-1])
            b = np.sqrt(chi2_thresh * eigvals[0])
            params.append((mu_pred, a, b, eigvecs))

        if len(params) < 2:
            return None

        polys = []
        N_INTERP = 6  # số bước nội suy chèn giữa mỗi cặp mốc liên tiếp
        for k in range(len(params) - 1):
            mu0, a0, b0, ev0 = params[k]
            mu1, a1, b1, ev1 = params[k + 1]
            for s in range(N_INTERP + 1):
                f = s / N_INTERP
                mu = mu0 * (1 - f) + mu1 * f
                a = a0 * (1 - f) + a1 * f
                b = b0 * (1 - f) + b1 * f
                ev = ev0 if f < 0.5 else ev1  # tránh nội suy ma trận xoay phức tạp
                ring = _ellipse_at(mu, max(a, 1e-3), max(b, 1e-3), eigvecs=ev)
                ring = np.vstack([ring, ring[0]])
                try:
                    poly = ShapelyPolygon(ring)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    polys.append(poly)
                except Exception:
                    continue

        if len(polys) < 2:
            return None
        merged = unary_union(polys)
        # An toàn: nếu vẫn còn bị tách mảnh (trường hợp cực hiếm), lấy
        # bao lồi tổng hợp của tất cả polygon để đảm bảo 1 khối liền —
        # đúng tinh thần "cone liền mạch" của ảnh mẫu NCHMF.
        if merged.geom_type == "MultiPolygon" and len(merged.geoms) > 1:
            merged = merged.convex_hull
        return merged

    def _cone_edges_legacy(chi2_thresh):
        """Cách vẽ cũ (nối trái/phải) — chỉ dùng khi thiếu shapely."""
        means     = np.array([ens_deg[:, t, :].mean(axis=0) for t in range(T)])
        track_pts = np.vstack([cur_pos_deg, means])

        def _perp(p1, p2):
            d = p2 - p1
            n = np.linalg.norm(d)
            if n < 1e-10:
                return np.array([0.0, 1.0])
            d /= n
            return np.array([-d[1], d[0]])

        left  = [cur_pos_deg.copy()]
        right = [cur_pos_deg.copy()]
        for t in range(T):
            b = _gaussian_cone_boundary(ens_deg[:, t, :], chi2_thresh)
            if b is None:
                left.append(means[t])
                right.append(means[t])
                continue
            perp = (
                _perp(track_pts[t], track_pts[t + 1])
                if t + 1 < len(track_pts)
                else _perp(track_pts[t - 1], track_pts[t])
            )
            proj = (b - means[t]) @ perp
            left.append(b[proj.argmax()])
            right.append(b[proj.argmin()])
        return np.array(left), np.array(right)

    if HAS_SHAPELY:
        u90 = _cone_union(_CHI2_90)
        if u90 is not None:
            _fill_polygon(u90, STYLE["cone_90_fill"], STYLE["cone_90_alpha"], 3)
        u50 = _cone_union(_CHI2_50)
        if u50 is not None:
            _fill_polygon(u50, STYLE["cone_50_fill"], STYLE["cone_50_alpha"], 5)
    else:
        l90, r90 = _cone_edges_legacy(_CHI2_90)
        _fill_legacy(np.vstack([l90, r90[::-1]]), STYLE["cone_90_fill"], STYLE["cone_90_alpha"], 3)
        l50, r50 = _cone_edges_legacy(_CHI2_50)
        _fill_legacy(np.vstack([l50, r50[::-1]]), STYLE["cone_50_fill"], STYLE["cone_50_alpha"], 5)

    # [FIX-16] Trước đây vẽ THÊM từng ensemble member riêng lẻ (S đường
    # mờ, alpha=0.05) đè lên cone — đây chính là nguồn "rối" đã phản
    # hồi (nhìn như 1 mớ chỉ rối ở giữa track). Cone (fill 50%/90%) đã
    # đủ thể hiện độ phân tán về mặt hình học; vẽ thêm từng đường không
    # tăng thông tin đáng kể mà chỉ gây nhiễu mắt, đặc biệt khi K lớn.
    # Bỏ hẳn — khớp đúng phong cách track-forecast-cone chuẩn (NHC/
    # JTWC không vẽ spaghetti plot riêng khi đã có cone).


# ── Spread panel ───────────────────────────────────────────────────────────────

def plot_spread_over_time(ax, ens_deg, errors_km, cliper_err_km, t_name):
    """
    Left axis  (ax)       : ensemble spread 1σ  [km]
    Right axis (ax_twin)  : track error & CLIPER [km]

    [FIX-5] Error fill_between was incorrectly called on `ax` (spread axis)
            instead of `ax_twin` (error axis), causing it to be drawn against
            the wrong Y scale and potentially hidden under the spread fill.
            (Fix preserved in this white-style version — unchanged logic.)
    """
    S, T, _ = ens_deg.shape
    lead_h   = np.arange(1, T + 1) * 6

    spreads_km = []
    for t in range(T):
        pts        = ens_deg[:, t, :]
        mean_lat   = pts[:, 1].mean()
        std_lon_km = pts[:, 0].std() * 111.32 * np.cos(np.deg2rad(mean_lat))
        std_lat_km = pts[:, 1].std() * 110.57
        spreads_km.append(np.sqrt(std_lon_km ** 2 + std_lat_km ** 2))
    spreads_km = np.array(spreads_km)

    spread_color = "#1F77B4"  # xanh dương — spread (khác pred_color để không trùng)

    ax.set_facecolor(STYLE["bg_color"])
    ax.fill_between(lead_h, 0, spreads_km, alpha=0.18,
                    color=spread_color, label="Ensemble spread (1σ)")
    ax.plot(lead_h, spreads_km, "-", color=spread_color, lw=2.2, zorder=5)

    ax_twin = ax.twinx()
    ax_twin.set_facecolor(STYLE["bg_color"])
    ax_twin.plot(lead_h, errors_km, "o-", color=STYLE["pred_color"],
                 lw=2.5, ms=5, label="FM ADE", zorder=6)
    # [FIX-5] Use ax_twin (not ax) so the fill is scaled to the error Y-axis
    ax_twin.fill_between(lead_h, 0, errors_km, alpha=0.10, color=STYLE["pred_color"])

    if cliper_err_km is not None:
        ax_twin.plot(lead_h, cliper_err_km[:T], "s--",
                     color="#666666", lw=2, ms=4, label="CLIPER", zorder=4)

    for xm in [24, 48, 72]:
        ax.axvline(xm, color=STYLE["error_color"], alpha=0.3, lw=0.7, ls=":")

    ax.set_xlabel("Lead time (h)", color=STYLE["text_color"], fontsize=8)
    ax.set_ylabel("Spread 1σ (km)", color=spread_color, fontsize=8)
    ax_twin.set_ylabel("Track error (km)", color=STYLE["pred_color"], fontsize=8)
    ax.set_title(f"Spread vs Error — {t_name}", color=STYLE["text_color"],
                 fontsize=9, fontweight="bold")

    lines1, lbs1 = ax.get_legend_handles_labels()
    lines2, lbs2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbs1 + lbs2, fontsize=7.5,
              facecolor="white", edgecolor=STYLE["panel_edge"],
              labelcolor=STYLE["text_color"], loc="upper left", framealpha=0.92)

    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["panel_edge"])
    ax.tick_params(colors=STYLE["text_color"], labelsize=7)
    ax_twin.tick_params(colors=STYLE["text_color"], labelsize=7)
    ax_twin.yaxis.label.set_color(STYLE["pred_color"])
    ax.yaxis.label.set_color(spread_color)


# ── Map setup ──────────────────────────────────────────────────────────────────

def make_map_ax(fig, subplot_spec, lon_range, lat_range, use_satellite_bg=True):
    if HAS_CARTOPY:
        ax = fig.add_subplot(
            subplot_spec,
            projection=ccrs.PlateCarree(central_longitude=0),
        )
        ax.set_extent(
            [lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
            crs=ccrs.PlateCarree(),
        )
        # [FIX-13, quan trọng] Bug thật đã tìm ra: ax.add_feature(...)
        # chỉ ĐĂNG KÝ feature vào axes — việc TẢI shapefile Natural
        # Earth qua mạng (cho "10m"/"50m", không phải built-in bundle)
        # chỉ thực sự xảy ra lúc RENDER (matplotlib gọi draw() khi
        # savefig()), tức NẰM NGOÀI try/except cũ (bọc quanh
        # add_feature()) — nếu Kaggle không có mạng lúc đó, add_feature()
        # "thành công" (không raise), drew_bg=True, nhưng feature KHÔNG
        # VẼ ĐƯỢC GÌ khi render → nền trắng/trơn hoàn toàn như đã quan
        # sát, không phải lỗi code hiển thị mà là lỗi tải bị nuốt mất.
        # Giờ ép fig.canvas.draw() NGAY sau add_feature() để bắt lỗi
        # tải thật tại đúng lúc nó xảy ra, in rõ nguyên nhân, rồi mới
        # quyết định fallback — không còn đoán mò "chắc là do mạng".
        drew_bg = False
        last_err = None
        if use_satellite_bg:
            for scale in ("10m", "50m"):
                try:
                    land_feat  = cfeature.NaturalEarthFeature(
                        "physical", "land", scale,
                        facecolor="#E8E4D8", edgecolor="none")
                    ocean_feat = cfeature.NaturalEarthFeature(
                        "physical", "ocean", scale,
                        facecolor="#C8DCF0", edgecolor="none")
                    ax.add_feature(land_feat, zorder=1)
                    ax.add_feature(ocean_feat, zorder=0)
                    fig.canvas.draw()   # ép render NGAY để lộ lỗi tải mạng thật
                    drew_bg = True
                    print(f"  [make_map_ax] Nền bản đồ scale={scale}: OK")
                    break
                except Exception as e:
                    last_err = e
                    for coll in list(ax.collections):
                        coll.remove()
                    print(f"  [make_map_ax] Nền bản đồ scale={scale} LỖI: "
                          f"{type(e).__name__}: {e}")
                    continue
        if not drew_bg:
            # [FIX-13] Fallback CUỐI thật sự offline-safe: cfeature.LAND/
            # OCEAN KHÔNG gọi .with_scale() dùng data 110m BUNDLE SẴN
            # trong gói cartopy (không cần mạng) — khác hẳn
            # .with_scale("50m") (vẫn cần tải). Đây là fallback duy
            # nhất chắc chắn hoạt động khi Kaggle không có mạng.
            if last_err is not None:
                print(f"  [make_map_ax] Dùng fallback offline-safe "
                      f"(cfeature.LAND/OCEAN 110m bundle). Lỗi gốc: {last_err}")
            try:
                ax.add_feature(cfeature.OCEAN, facecolor=STYLE["ocean_color"], zorder=0)
                ax.add_feature(cfeature.LAND, facecolor=STYLE["land_color"], zorder=1, alpha=0.9)
                fig.canvas.draw()
                print("  [make_map_ax] Fallback 110m bundle: OK")
            except Exception as e2:
                print(f"  [make_map_ax] ❌ Fallback 110m CŨNG LỖI "
                      f"({type(e2).__name__}: {e2}) — map sẽ chỉ có màu nền phẳng, "
                      f"không coastline. Kiểm tra lại cài đặt cartopy.")
                ax.set_facecolor(STYLE["ocean_color"])

        try:
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                           edgecolor="#4D4D4D", linewidth=0.8, zorder=2)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                           edgecolor=STYLE["border_color"],
                           linewidth=0.4, linestyle=":", zorder=2)
            fig.canvas.draw()
        except Exception as e:
            print(f"  [make_map_ax] Coastline/Borders 50m LỖI ({e}) — "
                  f"dùng fallback 110m bundle.")
            ax.add_feature(cfeature.COASTLINE, edgecolor="#4D4D4D", linewidth=0.8, zorder=2)
            ax.add_feature(cfeature.BORDERS, edgecolor=STYLE["border_color"],
                           linewidth=0.4, linestyle=":", zorder=2)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True,
            linewidth=0.5, color=STYLE["grid_color"],
            alpha=STYLE["grid_alpha"], linestyle="--",
        )
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = dict(color=STYLE["text_color"], fontsize=7)
        gl.ylabel_style = dict(color=STYLE["text_color"], fontsize=7)
    else:
        ax = fig.add_subplot(subplot_spec)
        ax.set_facecolor(STYLE["bg_color"])
        ax.set_xlim(*lon_range)
        ax.set_ylim(*lat_range)
        for lon in np.arange(np.ceil(lon_range[0] / 5) * 5, lon_range[1], 5):
            ax.axvline(lon, color=STYLE["grid_color"], alpha=STYLE["grid_alpha"], lw=0.5)
        for lat in np.arange(np.ceil(lat_range[0] / 5) * 5, lat_range[1], 5):
            ax.axhline(lat, color=STYLE["grid_color"], alpha=STYLE["grid_alpha"], lw=0.5)
        ax.set_xlabel("Longitude (°E)", color=STYLE["text_color"], fontsize=8)
        ax.set_ylabel("Latitude (°N)",  color=STYLE["text_color"], fontsize=8)
        ax.tick_params(colors=STYLE["text_color"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["panel_edge"])
    return ax


def _inset_range(pred_deg, ens_deg):
    """
    [MỚI] Tính (lon_range, lat_range) hẹp, zoom RIÊNG vào vùng dự báo
    (pred_deg + toàn bộ ensemble members) — dùng cho inset zoom, khác
    hẳn cách tính margin của map chính (vốn phải bao trùm cả obs/gt để
    không cắt mất track thật, nên margin lớn khi sai số dự báo lớn).
    Margin ở đây CỐ ĐỊNH theo % độ trải dài của chính vùng dự báo (30%
    mỗi chiều), với sàn 0.3° — đủ nhỏ để ensemble spread luôn chiếm
    phần lớn khung nhìn, bất kể map chính phải zoom xa cỡ nào.
    """
    pts = pred_deg if ens_deg is None else np.vstack([pred_deg, ens_deg.reshape(-1, 2)])
    lon_span = pts[:, 0].max() - pts[:, 0].min()
    lat_span = pts[:, 1].max() - pts[:, 1].min()
    margin_lon = max(lon_span * 0.30, 0.3)
    margin_lat = max(lat_span * 0.30, 0.3)
    lon_range = (pts[:, 0].min() - margin_lon, pts[:, 0].max() + margin_lon)
    lat_range = (pts[:, 1].min() - margin_lat, pts[:, 1].max() + margin_lat)
    return lon_range, lat_range


# [NEW] Nhãn địa lý cố định (tiếng Anh) kiểu bản đồ NCHMF: thành phố mốc
# + quần đảo + tên biển. (lon, lat, label, style) — style "city" = chấm
# nhỏ + tên; style "sea" = chữ nghiêng lớn không có chấm (tên vùng biển).
GEO_LABELS = [
    (105.85, 21.03, "Ha Noi",      "city"),
    (106.70, 10.78, "Ho Chi Minh City", "city"),
    (111.9,  16.5,  "Hoang Sa", "sea_small"),   # Hoàng Sa
    (114.3,  10.5,  "Truong Sa", "sea_small"),   # Trường Sa
    (112.5,  14.5,  "East Sea",    "sea"),          # Biển Đông
]


def _draw_geo_labels(ax, lon_range, lat_range, transform):
    """Vẽ các nhãn địa lý cố định (thành phố, quần đảo, tên biển) bằng
    tiếng Anh nếu tọa độ nằm trong phạm vi hiện tại của bản đồ — tránh
    in nhãn ngoài khung khi zoom hẹp (ví dụ inset track dự báo)."""
    outline = [pe.withStroke(linewidth=2.2, foreground="white")]
    for lon, lat, label, kind in GEO_LABELS:
        if not (lon_range[0] <= lon <= lon_range[1] and
                lat_range[0] <= lat <= lat_range[1]):
            continue
        if kind == "city":
            kw = dict(color="#222222", fontsize=8, fontweight="bold",
                      ha="left", va="center", zorder=15, path_effects=outline)
            if HAS_CARTOPY:
                ax.plot(lon, lat, marker="o", markersize=4,
                        markerfacecolor="black", markeredgecolor="white",
                        markeredgewidth=0.6, transform=transform, zorder=15)
                ax.text(lon + 0.15, lat, label, transform=transform, **kw)
            else:
                ax.plot(lon, lat, marker="o", markersize=4,
                        markerfacecolor="black", markeredgecolor="white",
                        markeredgewidth=0.6, zorder=15)
                ax.text(lon + 0.15, lat, label, **kw)
        elif kind == "sea":
            kw = dict(color="#3B6EA5", fontsize=11, fontstyle="italic",
                      fontweight="bold", ha="center", va="center",
                      alpha=0.85, zorder=4, path_effects=outline)
            if HAS_CARTOPY:
                ax.text(lon, lat, label, transform=transform, **kw)
            else:
                ax.text(lon, lat, label, **kw)
        else:  # sea_small — quần đảo
            kw = dict(color="#555555", fontsize=7.5, fontstyle="italic",
                      ha="center", va="center", zorder=4, path_effects=outline)
            if HAS_CARTOPY:
                ax.text(lon, lat, label, transform=transform, **kw)
            else:
                ax.text(lon, lat, label, **kw)


def _plot_on_ax(
    ax, lon_range, lat_range,
    obs_deg, gt_deg, pred_deg, pred_Me_deg,
    all_trajs_deg=None, errors_km=None,
    title="", dt_str="", pred_label="FM (mean)",
    ref_spread_km=None,
):
    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    # Viền TRẮNG quanh chữ/marker (đảo ngược so với bản dark, vốn viền
    # ĐEN quanh chữ sáng để nổi trên nền tối) — cho chữ đọc được trên nền
    # trắng/xanh nhạt của bản đồ.
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

    def _text(x, y, s, **kw):
        if HAS_CARTOPY:
            ax.text(x, y, s, transform=transform, **kw)
        else:
            ax.text(x, y, s, **kw)

    # 1. Probability cone
    if all_trajs_deg is not None and all_trajs_deg.shape[0] >= 3:
        draw_smooth_cone(ax, all_trajs_deg, cur_pos, transform, pred_deg=pred_deg)

    # 1b. Geographic reference labels (English): Ha Noi, Ho Chi Minh City,
    # Paracel/Spratly Islands, East Sea — drawn under the track so the
    # red/blue/black lines and dots stay clearly on top.
    _draw_geo_labels(ax, lon_range, lat_range, transform)

    # 2. Observed track
    _plot(obs_deg[:, 0], obs_deg[:, 1], fmt="o-",
          color=STYLE["obs_color"], linewidth=STYLE["lw_thin"], markersize=5,
          markeredgecolor="white", markeredgewidth=0.8,
          zorder=7, path_effects=outline)

    # 3. Ground truth
    gt_lon = np.concatenate([[cur_pos[0]], gt_deg[:, 0]])
    gt_lat = np.concatenate([[cur_pos[1]], gt_deg[:, 1]])
    _plot(gt_lon, gt_lat, fmt="o-",
          color=STYLE["gt_color"], linewidth=STYLE["lw_main"],
          markersize=STYLE["marker_size"],
          markeredgecolor="white", markeredgewidth=1.2,
          zorder=8, path_effects=outline)

    # 4. Predicted track (ensemble mean) — red line stays exactly centered
    # through every forecast dot (NCHMF style): the growing white-ringed
    # circles below are drawn AROUND each red dot, never offsetting the
    # line itself.
    pred_lon = np.concatenate([[cur_pos[0]], pred_deg[:, 0]])
    pred_lat = np.concatenate([[cur_pos[1]], pred_deg[:, 1]])

    _plot(pred_lon, pred_lat, fmt="o-",
          color=STYLE["pred_color"], linewidth=STYLE["lw_main"],
          markersize=STYLE["marker_size"],
          markeredgecolor="white", markeredgewidth=1.0,
          zorder=9, path_effects=outline)

    # 5. Wind intensity markers
    if pred_Me_deg is not None:
        for i in range(len(pred_deg)):
            wnd_kt  = denorm_wind(float(pred_Me_deg[i, 1]))
            _, wcolor = wind_intensity(wnd_kt)
            _scatter([pred_deg[i, 0]], [pred_deg[i, 1]],
                     s=70, color=wcolor,
                     edgecolors="black", linewidths=0.6, zorder=11)

    # 6. Error connectors at 24/48/72h
    if errors_km is not None:
        for si, lbl in {3: "24h", 7: "48h", 11: "72h"}.items():
            if si < len(gt_deg) and si < len(pred_deg):
                gx, gy = gt_deg[si, 0], gt_deg[si, 1]
                px, py = pred_deg[si, 0], pred_deg[si, 1]
                if HAS_CARTOPY:
                    ax.plot([gx, px], [gy, py], "--",
                            color=STYLE["error_color"], linewidth=1.2,
                            alpha=0.8, transform=transform, zorder=7)
                else:
                    ax.plot([gx, px], [gy, py], "--",
                            color=STYLE["error_color"], linewidth=1.2,
                            alpha=0.8, zorder=7)
                _text(
                    (gx + px) / 2, (gy + py) / 2,
                    f" {lbl}\n{errors_km[si]:.0f}km",
                    fontsize=7, color=STYLE["error_color"],
                    ha="center", va="bottom", zorder=14,
                    path_effects=outline,
                )

    # 7. Lead-time labels every 24h for both GT and Pred
    for i in range(len(pred_lon)):
        h   = i * 6
        if h % 24 == 0:
            lbl = "NOW" if i == 0 else f"+{h}h"
            _text(pred_lon[i], pred_lat[i] + 0.5, lbl,
                  color=STYLE["pred_color"], fontweight="bold", fontsize=7.5,
                  path_effects=outline)
            if i < len(gt_lon):
                _text(gt_lon[i], gt_lat[i] - 0.7, lbl,
                      color=STYLE["gt_color"], fontsize=6, alpha=0.9,
                      path_effects=outline)

    # 8. NOW star
    _scatter([cur_pos[0]], [cur_pos[1]],
             s=350, marker="*", color="#FFD700",
             edgecolors="black", linewidths=1.5, zorder=20)

    # 9. [RESTYLE — NCHMF-style info box] Thay hoàn toàn text box cũ
    # (monospace, góc dưới trái, chỉ có ADE+spread rời rạc) bằng bảng
    # dạng ax.table() ở góc phải trên, giống bố cục "TIN BAO KHAN CAP"
    # trong ảnh mẫu NCHMF: mỗi hàng = 1 lead time, các cột = lat/lon dự
    # báo, lat/lon thực tế, ADE. KHÔNG có cột cấp gió/Vmax/Pmin -- bài
    # này chỉ dự báo track, không dự báo cường độ, nên cố tình bỏ các
    # cột đó thay vì để trống/N-A gây hiểu lầm là có nhưng thiếu dữ
    # liệu. Toàn bộ nhãn bằng tiếng Anh theo yêu cầu.
    if errors_km is not None:
        n = len(errors_km)
        # [UPDATE] Bổ sung +6h và +12h để bảng đủ mọi mốc chuẩn (6/12/24/48/72h),
        # không chỉ 24/48/72h như trước.
        # [FIX] Sửa lỗi lệch 1 bước có sẵn trong code gốc: bảng cũ dùng
        # pred_deg[si-1] trong khi si đã được đặt tên theo đúng mốc giờ
        # thật (si=3 nghĩ là "+24h" nhưng pred_deg[2] thực chất là 18h,
        # do pred_deg là 0-based với index 0 = bước 6h đầu tiên). Ở đây
        # đổi sang index 0-based trực tiếp (0=6h, 1=12h, 3=24h, 7=48h,
        # 11=72h), khớp đúng với "Error connectors" (mục 6 phía trên,
        # dùng gt_deg[si]/pred_deg[si] không trừ 1) để 2 nơi nhất quán.
        lead_times = [(0, "NOW"), (0, "+6h"), (1, "+12h"),
                      (3, "+24h"), (7, "+48h"), (11, "+72h")]
        table_rows = []
        for si, lbl in lead_times:
            if lbl == "NOW":
                plat, plon = cur_pos[1], cur_pos[0]
                glat, glon = cur_pos[1], cur_pos[0]
                ade_str = "-"
            elif si < len(pred_deg) and si < len(gt_deg):
                plat, plon = pred_deg[si, 1], pred_deg[si, 0]
                glat, glon = gt_deg[si, 1], gt_deg[si, 0]
                ade_str = f"{errors_km[si]:.0f}" if si < n else "-"
            else:
                continue
            table_rows.append([lbl, f"{plat:.1f}N", f"{plon:.1f}E",
                               f"{glat:.1f}N", f"{glon:.1f}E", ade_str])

        col_labels = ["Time", "Pred.\nLat", "Pred.\nLon",
                     "Actual\nLat", "Actual\nLon", "ADE\n(km)"]

        # [SHRINK] Bảng thu nhỏ (bbox height 0.26->0.20, width 0.39->0.33) và
        # đặt cao hơn (y0 0.72->0.78) để không che vùng cone/track phía dưới
        # bên trong bản đồ; đồng thời bỏ dòng title "Track Forecast Summary"
        # và dòng "Ensemble spread" phía trên/dưới bảng theo yêu cầu, nên bảng
        # giờ là thành phần duy nhất trong góc phải trên, không cần chừa thêm
        # khoảng trống cho 2 dòng text đó nữa.
        tbl = ax.table(
            cellText=table_rows, colLabels=col_labels,
            cellLoc="center", colLoc="center",
            bbox=[0.66, 0.78, 0.33, 0.20],   # [x0, y0, width, height] trong axes-fraction, góc phải trên
            zorder=25,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(5.5)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor(STYLE["info_box_edge"])
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_facecolor(STYLE["info_box_title_bg"])
                cell.set_text_props(fontweight="bold", color=STYLE["info_box_edge"])
            else:
                cell.set_facecolor("white")


    # 10. Legends
    track_handles = [
        Line2D([0], [0], color=STYLE["obs_color"],  lw=2,   label="Observed"),
        Line2D([0], [0], color=STYLE["gt_color"],   lw=2,   label="Ground truth"),
        Line2D([0], [0], color=STYLE["pred_color"], lw=2.5, label=f"Predicted ({pred_label})"),
        mpatches.Patch(facecolor=STYLE["cone_50_fill"], alpha=0.5,
                       label="50% region (multiple trajectory predictions)"),
        mpatches.Patch(facecolor=STYLE["cone_90_fill"], alpha=0.35,
                       label="90% region (multiple trajectory predictions)"),
    ]
    ax.legend(handles=track_handles, loc="lower right", fontsize=7.5,
              facecolor="white", edgecolor=STYLE["panel_edge"],
              labelcolor=STYLE["text_color"], framealpha=0.92,
              title="Legend", title_fontsize=8)

    # [RESTYLE] Wind-intensity legend chỉ hiện khi thực sự có wind
    # marker được vẽ (pred_Me_deg is not None) -- trước đây luôn vẽ dù
    # không có marker nào, gây khung "Wind (kt)" trống vô nghĩa. Đồng
    # thời dời sang "lower left" thay vì "upper right" để không đè lên
    # bảng thông tin NCHMF-style mới đặt ở góc phải trên.
    if pred_Me_deg is not None:
        wind_handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=c, markersize=7,
                   markeredgecolor="black", markeredgewidth=0.5,
                   label=f"{nm} ({lo}–{hi}kt)")
            for lo, hi, nm, c in INTENSITY
        ]
        leg2 = ax.legend(
            handles=wind_handles, loc="lower left", fontsize=6.5,
            facecolor="white", edgecolor=STYLE["panel_edge"],
            labelcolor=STYLE["text_color"], title="Wind intensity (kt)",
            title_fontsize=7, ncol=2, framealpha=0.92,
        )
        ax.add_artist(leg2)

    ax.set_title(
        f"{title}\n{dt_str}", color=STYLE["text_color"], fontsize=10,
        fontweight="bold", pad=STYLE["title_pad"],
        bbox=dict(fc="white", alpha=0.9, ec=STYLE["panel_edge"], lw=1.2),
    )
    ax.set_facecolor(STYLE["bg_color"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["panel_edge"])


def _plot_multi_seed_on_ax(
    ax, lon_range, lat_range,
    obs_deg, gt_deg, preds_by_seed, errors_by_seed,
    all_trajs_by_seed=None,
    title="", dt_str="",
    ref_spread_km=None,
):
    """
    [MỚI] Bản multi-seed của _plot_on_ax() — GIỮ NGUYÊN toàn bộ chi tiết
    (cone xác suất 50/90%, error connector 24/48/72h, error+spread
    summary box, lead-time label, NOW star, legend track) nhưng:
      - KHÔNG có wind-intensity markers/legend (theo yêu cầu bỏ wind)
      - Predicted track giờ là NHIỀU đường (1/seed), màu theo XẾP HẠNG
        chất lượng: seed có ADE thấp nhất (tốt nhất) tô ĐẬM/dày, các
        seed còn lại tô NHẠT/mảnh hơn — thay vì mỗi seed 1 màu riêng
        như plot_multi_seed_comparison() cũ.
      - [FIX-14] Cone xác suất giờ vẽ TỪ ENSEMBLE CỦA SEED TỐT NHẤT
        (không phải gộp cả 3 seed như bản trước) — gộp 3 seed khiến
        cone phồng to bất thường (che gần hết map, xem ảnh phản hồi)
        vì cộng dồn cả model uncertainty lẫn bias khác nhau giữa các
        seed, không còn giống ý nghĩa "cone dự báo của 1 model" nữa.
        Quyết định: ưu tiên cone gọn, đúng ý nghĩa gốc (giống ảnh mẫu
        ban đầu), chấp nhận cone chỉ đại diện đúng 1 seed thay vì cả 3.

    all_trajs_by_seed: dict seed_label -> ens_deg [K,T,2] (optional).
    """
    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    outline   = [pe.withStroke(linewidth=2.5, foreground="white")]
    cur_pos   = obs_deg[-1]

    # Xếp hạng seed theo ADE — tính SỚM (đầu hàm) vì cả cone (bước 1)
    # lẫn predicted track (bước 4) đều cần biết seed nào tốt nhất.
    ranked = sorted(errors_by_seed.items(), key=lambda kv: kv[1].mean())
    best_seed_label = ranked[0][0]

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

    def _text(x, y, s, **kw):
        if HAS_CARTOPY:
            ax.text(x, y, s, transform=transform, **kw)
        else:
            ax.text(x, y, s, **kw)

    # 1. Probability cone — [FIX-14] CHỈ từ ensemble của seed TỐT NHẤT
    # (không gộp cả 3 seed nữa — xem giải thích trong docstring).
    all_trajs_deg = all_trajs_by_seed.get(best_seed_label) if all_trajs_by_seed else None
    if all_trajs_deg is not None and all_trajs_deg.shape[0] >= 3:
        draw_smooth_cone(ax, all_trajs_deg, cur_pos, transform,
                          pred_deg=preds_by_seed.get(best_seed_label))

    # 2. Observed track
    _plot(obs_deg[:, 0], obs_deg[:, 1], fmt="o-",
          color=STYLE["obs_color"], linewidth=STYLE["lw_thin"], markersize=5,
          markeredgecolor="white", markeredgewidth=0.8,
          zorder=7, path_effects=outline)

    # 3. Ground truth
    gt_lon = np.concatenate([[cur_pos[0]], gt_deg[:, 0]])
    gt_lat = np.concatenate([[cur_pos[1]], gt_deg[:, 1]])
    _plot(gt_lon, gt_lat, fmt="o-",
          color=STYLE["gt_color"], linewidth=STYLE["lw_main"],
          markersize=STYLE["marker_size"],
          markeredgecolor="white", markeredgewidth=1.2,
          zorder=8, path_effects=outline)

    # 4. Predicted track — NHIỀU seed, màu/độ đậm theo xếp hạng ADE.
    # Seed tốt nhất (ADE thấp nhất): pred_color đậm, nét dày, zorder cao
    # nhất (vẽ đè lên trên). Seed còn lại: cùng màu nhưng alpha thấp
    # hơn, nét mảnh hơn, zorder thấp hơn (vẽ dưới) — tạo cảm giác "mờ
    # dần" cho seed kém hơn, đúng yêu cầu "tốt nhất đậm, tệ nhạt".
    # (ranked/best_seed_label đã tính ở đầu hàm)
    n_seeds = len(ranked)
    all_pred_lon_last, all_pred_lat_last = None, None
    for rank, (seed_label, err) in enumerate(ranked):
        pred_deg = preds_by_seed[seed_label]
        pred_lon = np.concatenate([[cur_pos[0]], pred_deg[:, 0]])
        pred_lat = np.concatenate([[cur_pos[1]], pred_deg[:, 1]])
        is_best  = (rank == 0)
        alpha    = 1.0 if is_best else max(0.35, 0.75 - 0.25 * rank)
        lw       = STYLE["lw_main"] * (1.15 if is_best else 0.75)
        ms       = STYLE["marker_size"] * (1.0 if is_best else 0.75)
        zo       = 9 + (n_seeds - rank)   # best seed vẽ sau cùng => đè lên trên
        _plot(pred_lon, pred_lat, fmt="o-",
              color=STYLE["pred_color"], linewidth=lw, alpha=alpha,
              markersize=ms, markeredgecolor="white",
              markeredgewidth=1.0 if is_best else 0.6,
              zorder=zo, path_effects=(outline if is_best else None))
        if is_best:
            all_pred_lon_last, all_pred_lat_last = pred_lon, pred_lat

    # Dùng track của seed TỐT NHẤT làm mốc cho label/error-connector bên
    # dưới (đại diện, tránh chồng chéo label nếu vẽ cho cả 3 seed).
    best_seed_label, best_err = ranked[0]
    pred_deg_best = preds_by_seed[best_seed_label]
    pred_lon, pred_lat = all_pred_lon_last, all_pred_lat_last
    errors_km = best_err

    # 5. (Wind markers — ĐÃ BỎ theo yêu cầu)

    # 6. Error connectors at 24/48/72h — dùng seed tốt nhất
    for si, lbl in {3: "24h", 7: "48h", 11: "72h"}.items():
        if si < len(gt_deg) and si < len(pred_deg_best):
            gx, gy = gt_deg[si, 0], gt_deg[si, 1]
            px, py = pred_deg_best[si, 0], pred_deg_best[si, 1]
            if HAS_CARTOPY:
                ax.plot([gx, px], [gy, py], "--",
                        color=STYLE["error_color"], linewidth=1.2,
                        alpha=0.8, transform=transform, zorder=7)
            else:
                ax.plot([gx, px], [gy, py], "--",
                        color=STYLE["error_color"], linewidth=1.2,
                        alpha=0.8, zorder=7)
            _text(
                (gx + px) / 2, (gy + py) / 2,
                f" {lbl}\n{errors_km[si]:.0f}km",
                fontsize=7, color=STYLE["error_color"],
                ha="center", va="bottom", zorder=14,
                path_effects=outline,
            )

    # 7. Lead-time labels every 24h
    for i in range(len(pred_lon)):
        h = i * 6
        if h % 24 == 0:
            lbl = "NOW" if i == 0 else f"+{h}h"
            _text(pred_lon[i], pred_lat[i] + 0.5, lbl,
                  color=STYLE["pred_color"], fontweight="bold", fontsize=7.5,
                  path_effects=outline)
            if i < len(gt_lon):
                _text(gt_lon[i], gt_lat[i] - 0.7, lbl,
                      color=STYLE["gt_color"], fontsize=6, alpha=0.9,
                      path_effects=outline)

    # 8. NOW star
    _scatter([cur_pos[0]], [cur_pos[1]],
             s=350, marker="*", color="#FFD700",
             edgecolors="black", linewidths=1.5, zorder=20)

    # 9. Error + Spread summary box — ADE mỗi seed + spread gộp
    n = len(errors_km)
    lines = ["ADE by seed (km):"]
    for seed_label, err in ranked:
        tag = " ★best" if seed_label == best_seed_label else ""
        lines.append(f" seed={seed_label}: {err.mean():.0f}{tag}")
    lines.append("")
    lines.append(f"Best seed @ 24/48/72h:")
    for si, lh in [(3, 24), (7, 48), (11, 72)]:
        if si < n:
            lines.append(f" {lh}h: {errors_km[si]:.0f} km")

    if all_trajs_deg is not None and all_trajs_deg.shape[0] >= 3:
        lines.append("")
        lines.append("Spread (1σ, gộp seed):")
        for si, lh in [(3, 24), (7, 48), (11, 72)]:
            if si < all_trajs_deg.shape[1]:
                members_at_t = all_trajs_deg[:, si, :]
                mean_at_t    = members_at_t.mean(axis=0, keepdims=True)
                d_to_mean    = haversine_km(members_at_t, np.repeat(mean_at_t, members_at_t.shape[0], axis=0))
                this_spread  = d_to_mean.std()
                ref_str = ""
                if ref_spread_km and lh in ref_spread_km:
                    ref_str = f" (ref: {ref_spread_km[lh]:.0f})"
                lines.append(f" {lh}h: {this_spread:.0f} km{ref_str}")

    ax.text(
        0.02, 0.03, "\n".join(lines),
        transform=ax.transAxes, fontsize=8, va="bottom",
        color=STYLE["text_color"], family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  alpha=0.9, ec=STYLE["panel_edge"], lw=0.8),
        zorder=16,
    )

    # 10. Legend (không có wind legend)
    track_handles = [
        Line2D([0], [0], color=STYLE["obs_color"],  lw=2,   label="Observed"),
        Line2D([0], [0], color=STYLE["gt_color"],   lw=2,   label="Ground truth"),
        Line2D([0], [0], color=STYLE["pred_color"], lw=2.5, alpha=1.0,
               label=f"Predicted (seed={best_seed_label}, best)"),
        Line2D([0], [0], color=STYLE["pred_color"], lw=1.5, alpha=0.5,
               label="Predicted (other seeds)"),
        mpatches.Patch(facecolor=STYLE["cone_50_fill"], alpha=0.5,
                       label="50% region (multiple trajectory predictions)"),
        mpatches.Patch(facecolor=STYLE["cone_90_fill"], alpha=0.35,
                       label="90% region (multiple trajectory predictions)"),
    ]
    ax.legend(handles=track_handles, loc="lower right", fontsize=7.5,
              facecolor="white", edgecolor=STYLE["panel_edge"],
              labelcolor=STYLE["text_color"], framealpha=0.92)

    ax.set_title(
        f"{title}\n{dt_str}", color=STYLE["text_color"], fontsize=10,
        fontweight="bold", pad=STYLE["title_pad"],
        bbox=dict(fc="white", alpha=0.9, ec=STYLE["panel_edge"], lw=1.2),
    )
    ax.set_facecolor(STYLE["bg_color"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["panel_edge"])


# ── Run inference ──────────────────────────────────────────────────────────────

def _extract_seq(tensor, batch_idx=0):
    """
    Extract trajectory for one sample → [T, F].

    seq_collate (traj_TBC) produces [T, B, F].
    model.sample typically produces [B, T, F].

    Decision rule (unambiguous given our domain):
      - If d1 == 1  → time-first [T, 1, F],  take [:, 0, :]
      - If d0 == 1  → batch-first [1, T, F], take [0, :, :]
      - If d0 > d1  → time-first  [T, B, F], take [:, batch_idx, :]
      - If d1 > d0  → batch-first [B, T, F], take [batch_idx, :, :]
      - If d0 == d1 → ambiguous; default to time-first (seq_collate convention)
    """
    t = tensor.cpu()
    if t.dim() != 3:
        raise ValueError(f"_extract_seq: expected 3-D tensor, got shape {t.shape}")
    d0, d1, _ = t.shape
    if d1 == 1:
        return t[:, batch_idx, :].numpy()    # [T, 1, F] → [T, F]
    if d0 == 1:
        return t[batch_idx, :, :].numpy()    # [1, T, F] → [T, F]
    if d0 > d1:
        return t[:, batch_idx, :].numpy()    # [T, B, F]
    if d1 > d0:
        return t[batch_idx, :, :].numpy()    # [B, T, F]
    # d0 == d1: default to seq_collate convention (time-first)
    return t[:, batch_idx, :].numpy()


def _extract_ens(all_trajs, batch_idx=0):
    """
    Extract ensemble trajectories for one sample.
    Expected shape: [S, B, T, F]  →  returns [S, T, F]
    Also handles [S, T, B, F] just in case.
    """
    t = all_trajs.cpu()
    if t.dim() == 4:
        S, d1, d2, F = t.shape
        if d1 == 1 or d2 > d1:
            # [S, B, T, F]
            return t[:, batch_idx, :, :].numpy()   # → [S, T, F]
        else:
            # [S, T, B, F]
            return t[:, :, batch_idx, :].numpy()   # → [S, T, F]
    raise ValueError(f"_extract_ens: unexpected tensor dim {t.dim()}, shape {t.shape}")


def run_inference(model, target, device, ode_steps, num_ensemble):
    batch = move_batch(seq_collate([target]), device)
    with torch.no_grad():
        pred_mean, pred_Me, all_trajs = model.sample(
            batch, num_ensemble=num_ensemble, ddim_steps=ode_steps
        )

    # ── Shapes ────────────────────────────────────────────────────────────
    print(f"  [shape] batch[0] (obs_traj) : {tuple(batch[0].shape)}")
    print(f"  [shape] batch[1] (pred_traj): {tuple(batch[1].shape)}")
    print(f"  [shape] pred_mean           : {tuple(pred_mean.shape)}")
    print(f"  [shape] pred_Me             : {tuple(pred_Me.shape)}")
    print(f"  [shape] all_trajs           : {tuple(all_trajs.shape)}")

    # ── Extract: seq_collate → [T, B, F]; model.sample → depends on impl ──
    # batch tensors are [T, B, F] (time-first, from traj_TBC in seq_collate)
    # model output convention must be checked via shape
    obs_n     = _extract_seq(batch[0])    # [T_obs,  2]  absolute, normalised
    gt_n      = _extract_seq(batch[1])    # [T_pred, 2]  absolute, normalised
    pred_n    = _extract_seq(pred_mean)   # [T_pred, 2]  — unknown space
    pred_Me_n = _extract_seq(pred_Me)     # [T_pred, F_me]
    ens_n     = _extract_ens(all_trajs)   # [S, T_pred, 2]

    # ── Auto-detect: does model output absolute coords or relative deltas? ──
    # obs_n values are normalised absolute coords, typically in range [-1, 1]
    # or similar (e.g. lon_norm ~ -0.3..0.3, lat_norm ~ -0.5..0.5).
    # If pred_n has much smaller magnitude than obs_n → it's delta (relative).
    obs_abs_mean  = np.abs(obs_n).mean()
    pred_abs_mean = np.abs(pred_n).mean()

    print(f"\n  [raw] obs_n  (all rows):\n{obs_n}")
    print(f"\n  [raw] gt_n   (all rows):\n{gt_n}")
    print(f"\n  [raw] pred_n (all rows):\n{pred_n}")
    print(f"\n  obs |mean|={obs_abs_mean:.4f}  pred |mean|={pred_abs_mean:.4f}")

    IS_DELTA = pred_abs_mean < obs_abs_mean * 0.15   # heuristic: delta << absolute

    if IS_DELTA:
        print("  [AUTO] pred looks like DELTA (relative) → cumsum + obs[-1]")
        # cumulative sum of deltas, starting from last observed position
        pred_n_abs  = obs_n[-1:] + np.cumsum(pred_n, axis=0)
        ens_abs     = obs_n[-1:] + np.cumsum(ens_n, axis=1)
    else:
        print("  [AUTO] pred looks like ABSOLUTE → use directly")
        pred_n_abs = pred_n
        ens_abs    = ens_n

    print(f"\n  [raw] pred_n_abs (first/last):\n{pred_n_abs[0]}  …  {pred_n_abs[-1]}")
    print(f"  [raw] gt_n       (first/last):\n{gt_n[0]}  …  {gt_n[-1]}\n")

    obs_deg  = to_deg(denorm_traj(obs_n))
    gt_deg   = to_deg(denorm_traj(gt_n))
    pred_deg = to_deg(denorm_traj(pred_n_abs))
    ens_deg  = to_deg(denorm_traj(ens_abs))

    print(f"  [deg] obs_deg  (last)  : {obs_deg[-1]}")
    print(f"  [deg] gt_deg   (first) : {gt_deg[0]}")
    print(f"  [deg] pred_deg (first) : {pred_deg[0]}")
    print(f"  expected: lon 100-180°E, lat 0-60°N\n")

    # [FIX-11] pred_deg/gt_deg có thể lệch độ dài nếu model_cfg's
    # pred_len (quyết định T của model.sample()) khác T thật của ground
    # truth trong dataset (xem cảnh báo "XUNG ĐỘT pred_len" ở
    # load_model_and_data nếu có) — trước đây haversine_km() crash
    # cứng ValueError. Giờ cắt về min(T) chung, kèm cảnh báo rõ ràng.
    if pred_deg.shape[0] != gt_deg.shape[0]:
        T_min = min(pred_deg.shape[0], gt_deg.shape[0])
        print(f"  ⚠ LỆCH ĐỘ DÀI: pred_deg có {pred_deg.shape[0]} bước, "
              f"gt_deg có {gt_deg.shape[0]} bước — model_cfg's pred_len "
              f"không khớp T thật của ground truth. Cắt cả 2 về {T_min} "
              f"bước ({T_min * 6}h). Cần xác nhận lại đúng pred_len "
              f"checkpoint thật được train với — kết quả sau đây chỉ "
              f"phản ánh {T_min * 6}h đầu, KHÔNG phải toàn bộ horizon gốc.")
        pred_deg = pred_deg[:T_min]
        gt_deg   = gt_deg[:T_min]
        if ens_deg is not None and ens_deg.shape[1] != T_min:
            ens_deg = ens_deg[:, :T_min]

    errors_km = haversine_km(pred_deg, gt_deg)

    # CLIPER: constant-velocity extrapolation from last two observed points
    if len(obs_deg) >= 2:
        v_deg            = obs_deg[-1] - obs_deg[-2]
        cliper_preds_deg = np.array(
            [obs_deg[-1] + (k + 1) * v_deg for k in range(len(gt_deg))]
        )
    else:
        cliper_preds_deg = np.tile(obs_deg[-1], (len(gt_deg), 1))

    cliper_err = haversine_km(cliper_preds_deg, gt_deg)

    return obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg, errors_km, cliper_err


def run_inference_generic(model, target, device, model_type: str,
                           ode_steps: int = 10, num_ensemble: int = 1):
    """
    [MERGE, từ plot_track_paper_style.py] Bản tổng quát của run_inference()
    ở trên, dùng được cho CẢ FM lẫn RNN/GRU/LSTM/ST-Trans (không truyền
    ddim_steps cho baseline vì các model đó không nhận tham số này).
    Cùng logic auto-detect delta-vs-absolute và denorm — không đổi gì so
    với run_inference() gốc, chỉ tổng quát hoá phần gọi model.sample().
    Không in log chi tiết từng bước như run_inference() (dùng cho single
    mode, cần debug kỹ) — bản này dùng cho multi_model mode, cần gọn.

    [BỔ SUNG] Giờ trả thêm wind_pred_kt (từ pred_Me, model output — cột
    index 1 = WND, xác nhận qua trajectoriesWithMe_unet_training.py's
    dòng "wind_norm = float(obs_Me[1, t])") và wind_gt_kt (ground-truth
    wind thật, đọc từ batch[8] = pred_Me_out theo đúng thứ tự trả về
    của seq_collate() — KHÔNG PHẢI đoán, đã xác nhận trực tiếp từ
    source seq_collate: return (obs_traj, pred_traj, obs_rel, pred_rel,
    nlp, mask, seq_start_end, obs_Me, pred_Me, pred_Me_rel, ...) — index
    8 = pred_Me, đúng là Me (PRES,WND) của khoảng GT/prediction target).
    """
    batch = move_batch(seq_collate([target]), device)
    is_fm = (model_type == "fm")

    with torch.no_grad():
        if is_fm:
            pred_mean, pred_Me, all_trajs = model.sample(
                batch, num_ensemble=max(num_ensemble, 1), ddim_steps=ode_steps)
        else:
            out = model.sample(batch, num_ensemble=1)
            if isinstance(out, tuple) and len(out) == 3:
                pred_mean, pred_Me, all_trajs = out
            else:
                pred_mean, pred_Me, all_trajs = out, None, None

    obs_n  = _extract_seq(batch[0])
    gt_n   = _extract_seq(batch[1])
    pred_n = _extract_seq(pred_mean)
    ens_n  = (_extract_ens(all_trajs)
              if (all_trajs is not None and torch.is_tensor(all_trajs)
                  and all_trajs.dim() == 4) else None)

    # [BỔ SUNG] wind: pred_Me[:, 1] denorm bằng denorm_wind(); gt wind
    # thật đọc từ batch[8] (pred_Me_out, đã xác nhận đúng index qua
    # seq_collate() source thật, không phải đoán).
    wind_pred_kt = None
    if pred_Me is not None and torch.is_tensor(pred_Me):
        pred_Me_n = _extract_seq(pred_Me)   # [T_pred, F_me], cột 1 = WND
        if pred_Me_n.shape[-1] >= 2:
            wind_pred_kt = denorm_wind(pred_Me_n[:, 1])

    wind_gt_kt = None
    if len(batch) > 8 and torch.is_tensor(batch[8]):
        gt_Me_n = _extract_seq(batch[8])    # [T_pred, F_me], cột 1 = WND
        if gt_Me_n.shape[-1] >= 2:
            wind_gt_kt = denorm_wind(gt_Me_n[:, 1])

    obs_abs_mean  = np.abs(obs_n).mean()
    pred_abs_mean = np.abs(pred_n).mean()
    is_delta = pred_abs_mean < obs_abs_mean * 0.15

    if is_delta:
        pred_n_abs = obs_n[-1:] + np.cumsum(pred_n, axis=0)
        ens_abs = (obs_n[-1:] + np.cumsum(ens_n, axis=1)) if ens_n is not None else None
    else:
        pred_n_abs = pred_n
        ens_abs = ens_n

    obs_deg  = to_deg(denorm_traj(obs_n))
    gt_deg   = to_deg(denorm_traj(gt_n))
    pred_deg = to_deg(denorm_traj(pred_n_abs))
    ens_deg  = to_deg(denorm_traj(ens_abs)) if ens_abs is not None else None

    # [FIX-11] Cùng vấn đề với run_inference() — model_cfg's pred_len có
    # thể lệch T thật của ground truth. Cắt an toàn thay vì crash.
    if pred_deg.shape[0] != gt_deg.shape[0]:
        T_min = min(pred_deg.shape[0], gt_deg.shape[0])
        print(f"  ⚠ LỆCH ĐỘ DÀI ({model_type}): pred={pred_deg.shape[0]} "
              f"vs gt={gt_deg.shape[0]} bước — cắt về {T_min} bước.")
        pred_deg = pred_deg[:T_min]
        gt_deg   = gt_deg[:T_min]
        if ens_deg is not None and ens_deg.shape[1] != T_min:
            ens_deg = ens_deg[:, :T_min]
        if wind_pred_kt is not None and wind_pred_kt.shape[0] != T_min:
            wind_pred_kt = wind_pred_kt[:T_min]
        if wind_gt_kt is not None and wind_gt_kt.shape[0] != T_min:
            wind_gt_kt = wind_gt_kt[:T_min]

    errors_km = haversine_km(pred_deg, gt_deg)
    return obs_deg, gt_deg, pred_deg, ens_deg, errors_km, wind_pred_kt, wind_gt_kt


def load_model_generic(model_path: str, model_type: str, device,
                        obs_len: int = 8, pred_len: int = 12):
    """
    [MERGE, từ plot_track_paper_style.py] Load 1 trong 5 kiến trúc
    (fm/st_trans/lstm/gru/rnn) từ checkpoint, dùng model_cfg đã lưu nếu
    có (khớp cách evaluate_multi_model.py load model), fallback về
    default constructor nếu checkpoint cũ không có model_cfg.
    """
    ck = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = ck.get("model_cfg") or {}

    if model_type == "fm":
        model = TCFlowMatching(**(model_cfg or dict(pred_len=pred_len, obs_len=obs_len))).to(device)
        # [FIX] "model" là key thật (xác nhận qua evaluate_multi_model.py
        # và train_flowmatching.py) — đặt đầu tiên cho rõ ràng, dù về
        # mặt chức năng thứ tự cũ vẫn ra đúng kết quả (2 key đầu luôn
        # miss nên tự rơi xuống "model").
        state = ck.get("model", ck.get("model_state_dict", ck.get("model_state", ck)))
    elif model_type == "st_trans":
        if model_cfg:
            model = STTrans(**model_cfg).to(device)
        else:
            model = STTrans(obs_len=obs_len, pred_len=pred_len).to(device)
        state = ck.get("model_state", ck.get("model"))
    else:  # lstm/gru/rnn
        if model_cfg:
            model = PaperBaseline(**model_cfg).to(device)
        else:
            model = PaperBaseline(model_type=model_type, obs_len=obs_len,
                                   pred_len=pred_len).to(device)
        state = ck.get("model_state", ck.get("model"))

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def plot_multi_model_comparison(obs_deg, gt_deg, preds_by_model, errors_by_model,
                                 t_name: str, output_path: str):
    """
    [MERGE, từ plot_track_paper_style.py] Vẽ nhiều model (FM + baselines)
    trên CÙNG 1 bản đồ, so với 1 ground truth chung — mỗi model 1 màu
    (MODEL_COLORS), legend ghi kèm ADE của từng model. Dùng make_map_ax
    (đã đổi sang style trắng ở trên) để bản đồ nhất quán với single/
    case_study mode.
    """
    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    cur_pos = obs_deg[-1]

    all_pts = [obs_deg, gt_deg] + list(preds_by_model.values())
    all_deg = np.vstack(all_pts)

    # [FIX] Cùng bug với plot_multi_seed_comparison — margin cố định
    # 3.0° gây map bị co lệch khi track dài (aspect ratio PlateCarree
    # giữ 1:1 độ kinh/vĩ). Dùng margin động theo độ trải dài thật.
    lon_span = all_deg[:, 0].max() - all_deg[:, 0].min()
    lat_span = all_deg[:, 1].max() - all_deg[:, 1].min()
    margin_lon = float(np.clip(lon_span * 0.10, 1.0, 4.5))
    margin_lat = float(np.clip(lat_span * 0.10, 1.0, 4.5))
    extra_lon_widen = max(0.0, (lat_span - lon_span) * 0.35)
    margin_lon += extra_lon_widen
    lon_range = (all_deg[:, 0].min() - margin_lon, all_deg[:, 0].max() + margin_lon)
    lat_range = (all_deg[:, 1].min() - margin_lat, all_deg[:, 1].max() + margin_lat)

    map_aspect = (lon_range[1] - lon_range[0]) / max(lat_range[1] - lat_range[0], 0.01)
    fig_h = 10.0
    fig_w_map = float(np.clip(fig_h * map_aspect, 5.0, 13.0))
    fig = plt.figure(figsize=(fig_w_map, fig_h), facecolor=STYLE["bg_color"])
    ax = make_map_ax(fig, 111, lon_range, lat_range)

    def _plot(x, y, **kw):
        if HAS_CARTOPY:
            ax.plot(x, y, transform=transform, **kw)
        else:
            ax.plot(x, y, **kw)

    _plot(obs_deg[:, 0], obs_deg[:, 1], marker="o",
          color=STYLE["obs_color"], linewidth=STYLE["lw_thin"],
          markersize=STYLE["marker_size"], zorder=6, label="Observed")

    gt_lon = np.concatenate([[cur_pos[0]], gt_deg[:, 0]])
    gt_lat = np.concatenate([[cur_pos[1]], gt_deg[:, 1]])
    _plot(gt_lon, gt_lat, marker="o",
          color=STYLE["gt_color"], linewidth=2.2,
          markersize=STYLE["marker_size"] + 1, zorder=10, label="Actual Track")

    handles = [
        Line2D([0], [0], color=STYLE["obs_color"], marker="o", lw=1.2, label="Observed"),
        Line2D([0], [0], color=STYLE["gt_color"], marker="o", lw=2.2, label="Actual Track"),
    ]

    for model_name, pred_deg in preds_by_model.items():
        color = MODEL_COLORS.get(model_name, "#333333")
        pred_lon = np.concatenate([[cur_pos[0]], pred_deg[:, 0]])
        pred_lat = np.concatenate([[cur_pos[1]], pred_deg[:, 1]])
        _plot(pred_lon, pred_lat, marker="o",
              color=color, linewidth=STYLE["lw_main"],
              markersize=STYLE["marker_size"] - 1, zorder=9, alpha=0.9)
        ade = errors_by_model[model_name].mean()
        handles.append(Line2D([0], [0], color=color, marker="o", lw=1.6,
                              label=f"{model_name} (ADE={ade:.0f}km)"))

    ax.set_title(f"{t_name} — Model Comparison", fontsize=13,
                fontweight="bold", color=STYLE["text_color"])
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {output_path}")


# Bảng màu riêng cho multi_seed (khác MODEL_COLORS — ở đây chỉ có 1
# kiến trúc (FM), mỗi màu là 1 SEED, không phải 1 model).
SEED_COLORS = {
    "0": "#D62728", "1": "#1F77B4", "2": "#2CA02C",
    "3": "#9467BD", "4": "#FF7F0E", "5": "#8C564B",
}
_SEED_COLOR_FALLBACK = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD",
                        "#FF7F0E", "#8C564B", "#17BECF", "#BCBD22"]


def _seed_color(seed_label: str, idx: int) -> str:
    return SEED_COLORS.get(str(seed_label),
                           _SEED_COLOR_FALLBACK[idx % len(_SEED_COLOR_FALLBACK)])


def plot_multi_seed_comparison(obs_deg, gt_deg, preds_by_seed, errors_by_seed,
                                t_name: str, output_path: str,
                                winds_pred_by_seed: dict = None,
                                wind_gt=None):
    """
    Vẽ nhiều SEED của CÙNG 1 kiến trúc (mặc định FM, nhưng tổng quát cho
    bất kỳ model nào truyền vào) trên cùng 1 bản đồ, so với 1 ground
    truth chung — mỗi seed 1 màu (SEED_COLORS), legend ghi kèm ADE từng
    seed. Khác plot_multi_model_comparison() ở chỗ trục "nhiều đường" là
    SEED thay vì MODEL — dùng để minh hoạ độ ổn định của 1 kiến trúc qua
    random init, không phải so sánh kiến trúc với nhau.

    [BỔ SUNG] winds_pred_by_seed (dict seed_label -> wind_kt array) và
    wind_gt (ground-truth wind, chung cho mọi seed) — nếu được truyền,
    thêm panel wind theo lead-time bên phải map (giống layout Spread vs
    Error trước đây) + box thống kê MAE wind. Optional, default None
    giữ nguyên hành vi cũ (chỉ map, không panel wind) cho tương thích
    ngược với mọi lời gọi cũ chưa có wind data.
    """
    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    cur_pos = obs_deg[-1]

    all_pts = [obs_deg, gt_deg] + list(preds_by_seed.values())
    all_deg = np.vstack(all_pts)

    # [FIX, quan trọng] Trước đây margin=3.0 CỐ ĐỊNH — cùng lỗi đã tìm
    # và sửa ở visualize_forecast(): với track dài (lệch hướng nhiều,
    # trải rộng theo vĩ độ), khung map thật sự sẽ CAO-HẸP vì PlateCarree
    # giữ đúng tỷ lệ 1:1 độ kinh/vĩ, nhưng figsize CỐ ĐỊNH (14,9) hay
    # (8,9) không khớp tỷ lệ đó — matplotlib tự co map lại theo chiều
    # ngang để giữ đúng aspect ratio, để lại khoảng trắng lớn 2 bên
    # (đúng hiện tượng quan sát ở ảnh CONSON: map dồn lệch, viền trắng
    # rất to). Sửa: margin tỷ lệ theo độ trải dài track thật (10% mỗi
    # chiều, sàn 1.0°, trần 4.5°) — CÙNG công thức với visualize_forecast.
    lon_span = all_deg[:, 0].max() - all_deg[:, 0].min()
    lat_span = all_deg[:, 1].max() - all_deg[:, 1].min()
    margin_lon = float(np.clip(lon_span * 0.10, 1.0, 4.5))
    margin_lat = float(np.clip(lat_span * 0.10, 1.0, 4.5))
    extra_lon_widen = max(0.0, (lat_span - lon_span) * 0.35)
    margin_lon += extra_lon_widen
    lon_range = (all_deg[:, 0].min() - margin_lon, all_deg[:, 0].max() + margin_lon)
    lat_range = (all_deg[:, 1].min() - margin_lat, all_deg[:, 1].max() + margin_lat)

    # [FIX] figsize giờ tính ĐỘNG theo đúng tỷ lệ lon_range/lat_range
    # thật của map — không còn hardcode (14,9)/(8,9). Panel wind (nếu
    # có) giữ độ rộng cố định hợp lý bên cạnh, không phụ thuộc aspect
    # ratio địa lý (nó là biểu đồ thường, không phải bản đồ).
    map_aspect = (lon_range[1] - lon_range[0]) / max(lat_range[1] - lat_range[0], 0.01)
    fig_h = 10.0
    fig_w_map = float(np.clip(fig_h * map_aspect, 5.0, 13.0))

    has_wind = bool(winds_pred_by_seed) and wind_gt is not None
    if has_wind:
        fig_w_wind = 6.0
        fig = plt.figure(figsize=(fig_w_map + fig_w_wind + 1.0, fig_h),
                         facecolor=STYLE["bg_color"])
        gs  = fig.add_gridspec(1, 2, width_ratios=[fig_w_map, fig_w_wind], wspace=0.15)
        ax  = make_map_ax(fig, gs[0, 0], lon_range, lat_range)
        ax_wind = fig.add_subplot(gs[0, 1])
        ax_wind.set_facecolor(STYLE["bg_color"])
    else:
        fig = plt.figure(figsize=(fig_w_map, fig_h), facecolor=STYLE["bg_color"])
        ax  = make_map_ax(fig, 111, lon_range, lat_range)

    def _plot(x, y, **kw):
        if HAS_CARTOPY:
            ax.plot(x, y, transform=transform, **kw)
        else:
            ax.plot(x, y, **kw)

    _plot(obs_deg[:, 0], obs_deg[:, 1], marker="o",
          color=STYLE["obs_color"], linewidth=STYLE["lw_thin"],
          markersize=STYLE["marker_size"], zorder=6, label="Observed")

    gt_lon = np.concatenate([[cur_pos[0]], gt_deg[:, 0]])
    gt_lat = np.concatenate([[cur_pos[1]], gt_deg[:, 1]])
    _plot(gt_lon, gt_lat, marker="o",
          color=STYLE["gt_color"], linewidth=2.2,
          markersize=STYLE["marker_size"] + 1, zorder=10, label="Actual Track")

    handles = [
        Line2D([0], [0], color=STYLE["obs_color"], marker="o", lw=1.2, label="Observed"),
        Line2D([0], [0], color=STYLE["gt_color"], marker="o", lw=2.2, label="Actual Track"),
    ]

    wind_lines = ["Wind MAE (kt):"] if has_wind else []
    for idx, (seed_label, pred_deg) in enumerate(preds_by_seed.items()):
        color = _seed_color(seed_label, idx)
        pred_lon = np.concatenate([[cur_pos[0]], pred_deg[:, 0]])
        pred_lat = np.concatenate([[cur_pos[1]], pred_deg[:, 1]])
        _plot(pred_lon, pred_lat, marker="o",
              color=color, linewidth=STYLE["lw_main"],
              markersize=STYLE["marker_size"] - 1, zorder=9, alpha=0.9)
        ade = errors_by_seed[seed_label].mean()
        handles.append(Line2D([0], [0], color=color, marker="o", lw=1.6,
                              label=f"seed={seed_label} (ADE={ade:.0f}km)"))

        # [BỔ SUNG] Panel wind theo lead-time, 1 đường/seed + 1 đường GT
        if has_wind:
            wpred = winds_pred_by_seed.get(seed_label)
            if wpred is not None:
                T = min(len(wpred), len(wind_gt))
                hours = np.arange(1, T + 1) * 6
                ax_wind.plot(hours, wpred[:T], "o-", color=color,
                            linewidth=1.6, markersize=3.5,
                            label=f"seed={seed_label}")
                mae = float(np.abs(wpred[:T] - wind_gt[:T]).mean())
                wind_lines.append(f" seed={seed_label}: {mae:.1f} kt")

    if has_wind:
        T_gt = len(wind_gt)
        hours_gt = np.arange(1, T_gt + 1) * 6
        ax_wind.plot(hours_gt, wind_gt, "o-", color=STYLE["gt_color"],
                    linewidth=2.2, markersize=4, label="Actual", zorder=10)
        ax_wind.set_xlabel("Forecast Lead Time (h)", fontsize=9)
        ax_wind.set_ylabel("Wind Speed (kt)", fontsize=9)
        ax_wind.set_title("Wind Speed Comparison", fontsize=11, fontweight="bold")
        ax_wind.legend(fontsize=7.5, framealpha=0.9)
        ax_wind.grid(True, alpha=0.3, linestyle="--")

        ax.text(
            0.02, 0.03, "\n".join(wind_lines),
            transform=ax.transAxes, fontsize=8, va="bottom",
            color=STYLE["text_color"], family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      alpha=0.9, ec=STYLE["panel_edge"], lw=0.8),
            zorder=16,
        )

    ax.set_title(f"{t_name} — Seed Comparison", fontsize=13,
                fontweight="bold", color=STYLE["text_color"])
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {output_path}")


def plot_multi_seed_forecast(obs_deg, gt_deg, preds_by_seed, errors_by_seed,
                              t_name: str, dt_str: str, output_path: str,
                              all_trajs_by_seed: dict = None,
                              ref_spread_km=None):
    """
    [MỚI] Layout ĐẦY ĐỦ CHI TIẾT giống visualize_forecast() (cone xác
    suất 50/90%, error connector, error+spread summary box) nhưng cho
    NHIỀU SEED của 1 kiến trúc thay vì 1 model đơn — dùng
    _plot_multi_seed_on_ax() (không có wind, màu theo xếp hạng: seed
    tốt nhất đậm, seed còn lại nhạt, cone chỉ từ seed tốt nhất).

    [FIX-15] Theo yêu cầu: BỎ inset zoom — chỉ 1 map duy nhất, không
    còn panel phụ bên phải.

    Đây là hàm THAY THẾ plot_multi_seed_comparison() cho mục đích cần
    độ chi tiết cao (batch_all mode) — plot_multi_seed_comparison() cũ
    vẫn giữ nguyên, dùng cho trường hợp cần cả wind panel.
    """
    all_pts = [obs_deg, gt_deg] + list(preds_by_seed.values())
    if all_trajs_by_seed:
        for v in all_trajs_by_seed.values():
            if v is not None:
                all_pts.append(v.reshape(-1, 2))
    all_deg = np.vstack(all_pts)

    # Margin động theo track thật — cùng công thức đã dùng ở
    # visualize_forecast()/plot_multi_seed_comparison() đã sửa.
    lon_span = all_deg[:, 0].max() - all_deg[:, 0].min()
    lat_span = all_deg[:, 1].max() - all_deg[:, 1].min()
    margin_lon = float(np.clip(lon_span * 0.10, 1.0, 4.5))
    margin_lat = float(np.clip(lat_span * 0.10, 1.0, 4.5))
    extra_lon_widen = max(0.0, (lat_span - lon_span) * 0.35)
    margin_lon += extra_lon_widen
    lon_range = (all_deg[:, 0].min() - margin_lon, all_deg[:, 0].max() + margin_lon)
    lat_range = (all_deg[:, 1].min() - margin_lat, all_deg[:, 1].max() + margin_lat)

    # Figsize động theo đúng tỷ lệ khung hình thật (cùng công thức đã
    # dùng ở visualize_forecast()) — giờ chỉ 1 map, không cộng thêm
    # width cho inset nữa.
    map_aspect = (lon_range[1] - lon_range[0]) / max(lat_range[1] - lat_range[0], 0.01)
    fig_h = 11.0
    fig_w_map = float(np.clip(fig_h * map_aspect, 5.0, 14.0))

    fig = plt.figure(figsize=(fig_w_map, fig_h), facecolor=STYLE["bg_color"])
    ax_map = make_map_ax(fig, 111, lon_range, lat_range)

    _plot_multi_seed_on_ax(
        ax_map, lon_range, lat_range,
        obs_deg, gt_deg, preds_by_seed, errors_by_seed,
        all_trajs_by_seed=all_trajs_by_seed,
        title=t_name, dt_str=dt_str,
        ref_spread_km=ref_spread_km,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=STYLE["bg_color"])
    plt.close()
    print(f"  Saved → {output_path}")


# ── Load model & dataset ───────────────────────────────────────────────────────

def load_model_and_data(args, device, dset_type="test"):
    detected = detect_pred_len(args.model_path)
    if args.pred_len != detected:
        print(f"  pred_len: {args.pred_len} → {detected}")
        args.pred_len = detected

    ck = torch.load(args.model_path, map_location=device, weights_only=False)

    # [FIX-8, FIX-9] 2 bug đã tìm và sửa (xem chi tiết trong
    # visual_evaluate_mode_ME.py's cùng hàm — copy nguyên văn fix sang
    # đây để 2 file nhất quán):
    # (1) model_cfg từ checkpoint bị bỏ qua hoàn toàn trước đây.
    # (2) key "model_state_dict"/"model_state" không tồn tại trong
    #     checkpoint thật (key đúng là "model") — khiến state_dict gần
    #     như không load được tensor nào, model chạy random-init.
    model_cfg = ck.get("model_cfg") or {}
    if not model_cfg:
        print("  ⚠ Checkpoint không có model_cfg — dùng constructor "
              "DEFAULTS + pred_len/obs_len từ CLI. Chỉ đúng nếu checkpoint "
              "train với kiến trúc mặc định.")
        model = TCFlowMatching(pred_len=args.pred_len, obs_len=args.obs_len).to(device)
    else:
        # [FIX-10] model_cfg["pred_len"] có thể XUNG ĐỘT với
        # detect_pred_len()'s kết quả (dựa vào pos_enc.shape[1]) — đây
        # là nguyên nhân thật của lỗi "shape mismatch" giữa pred_deg và
        # gt_deg quan sát được (model.sample() ra T theo model_cfg,
        # nhưng ground truth trong dataset có T khác). model_cfg được
        # ưu tiên (đại diện đúng architecture checkpoint thật), nhưng
        # cảnh báo rõ để biết đây là vấn đề cần xác nhận lại, không
        # phải lỗi code — run_inference() sẽ tự cắt về min(T) an toàn
        # nếu vẫn lệch sau bước này.
        cfg_pred_len = model_cfg.get("pred_len")
        if cfg_pred_len is not None and cfg_pred_len != detected:
            print(f"  ⚠ XUNG ĐỘT pred_len: model_cfg ghi {cfg_pred_len}, "
                  f"detect_pred_len() phát hiện {detected}. Dùng "
                  f"model_cfg's {cfg_pred_len} — có thể gây lệch shape "
                  f"với ground truth, run_inference() sẽ tự cắt an toàn "
                  f"nếu cần nhưng CẦN XÁC NHẬN LẠI đúng pred_len thật.")
        model = TCFlowMatching(**model_cfg).to(device)

    sd = ck.get("model", ck.get("model_state_dict", ck.get("model_state", ck)))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_total = sum(1 for _ in model.state_dict())
    print(f"  load_state_dict: {n_total - len(missing)}/{n_total} tensors khớp "
          f"| {len(missing)} missing | {len(unexpected)} unexpected")
    if len(missing) > n_total * 0.5:
        print(f"  ❌ CẢNH BÁO: hơn 50% tensor KHÔNG load được — model gần như "
              f"chắc chắn đang chạy random-init, không phải checkpoint thật.")
    model.eval()
    print("  Model loaded\n")

    dset, _ = data_loader(
        args,
        {"root": args.TC_data_path, "type": dset_type},
        test=True,
        test_year=args.test_year,
    )
    print(f"  Dataset: {len(dset)} samples\n")
    return model, dset


# ── Single mode ────────────────────────────────────────────────────────────────

def visualize_forecast(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_name              = args.tc_name.strip().upper()
    t_date, was_snapped = resolve_date(args.tc_date)

    print(f"{'=' * 65}")
    print(f"  TC-FM Visualize (paper style)  |  {t_name}  @  {t_date}")
    print(f"{'=' * 65}\n")

    # [MỚI, optional] Đọc ode_steps_sweep.json nếu được truyền, lấy
    # spread tại đúng N khớp với --ode_steps đang dùng — để không so
    # sánh nhầm N khác nhau (spread phụ thuộc RẤT NHIỀU vào N, xem
    # Table 4). Giờ lấy ĐỦ CẢ 3 MỐC (24h/48h/72h) từ by_lead_time's
    # "spread" field (đã bổ sung — trước đây chỉ có tại bước cuối
    # cùng). Fallback về spread_mean (chỉ 72h) nếu file JSON cũ chưa
    # có "spread" trong by_lead_time (tương thích ngược với file sinh
    # ra trước khi có fix per-lead-time spread).
    ref_spread_km = None
    if args.ode_sweep_json:
        try:
            with open(args.ode_sweep_json) as f:
                sweep_data = json.load(f)
            entry = sweep_data.get(str(args.ode_steps), sweep_data.get(args.ode_steps))
            if entry:
                ref_spread_km = {}
                by_lt = entry.get("by_lead_time", {})
                for si, lh in [(3, 24), (7, 48), (11, 72)]:
                    lt_key = si + 1  # 0-indexed step -> 1-indexed lead_time
                    lt_entry = by_lt.get(str(lt_key), by_lt.get(lt_key))
                    if lt_entry and lt_entry.get("spread") is not None:
                        val = lt_entry["spread"]
                        if not (isinstance(val, float) and val != val):  # NaN check
                            ref_spread_km[lh] = val
                # Fallback: file JSON cũ chưa có "spread" trong
                # by_lead_time — dùng spread_mean (chỉ 72h) như trước.
                if not ref_spread_km and entry.get("spread_mean") is not None:
                    val = entry["spread_mean"]
                    if not (isinstance(val, float) and val != val):
                        fh_ref = args.pred_len * 6
                        ref_spread_km = {fh_ref: val}
                        print(f"  ⚠ ode_sweep_json chưa có spread per-lead-time "
                              f"(file cũ) — chỉ dùng spread_mean tại {fh_ref}h.\n")
                if ref_spread_km:
                    print(f"  [ref] spread trung bình test set tại N={args.ode_steps}: "
                          f"{ref_spread_km} (n={entry.get('n_storms', '?')} storm-window)\n")
                else:
                    print(f"  ⚠ ode_sweep_json không có spread hợp lệ cho N={args.ode_steps}\n")
            else:
                print(f"  ⚠ ode_sweep_json không có entry cho N={args.ode_steps}\n")
        except Exception as e:
            print(f"  ⚠ Không đọc được --ode_sweep_json: {e}\n")

    model, dset = load_model_and_data(args, device, args.dset_type)

    target, matched_obs_len, actual_date = find_target(
        dset, t_name, t_date, args.obs_len
    )

    if target is None:
        print(f"  '{t_name} @ {t_date}' not found (kể cả sau khi thử tiến 20 mốc).")
        print(f"\n  Các thời điểm có sẵn của '{t_name}':")
        list_available(dset, t_name, args.obs_len)
        return

    if actual_date != t_date:
        t_date = actual_date

    if matched_obs_len != args.obs_len:
        print(
            f"  [INFO] Dùng tydate[{matched_obs_len}] thay vì [{args.obs_len}] "
            f"(date khớp ở window khác)\n"
        )

    print(f"  Found: {t_name} @ {t_date}\n")

    (obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg,
     errors_km, cliper_err) = run_inference(
        model, target, device, args.ode_steps, args.num_ensemble
    )

    print("  Track errors (km):")
    for i, e in enumerate(errors_km):
        mark = "  ◀" if (i + 1) in [4, 8, 12] else ""
        print(f"    +{(i + 1) * 6:3d}h : {e:6.1f} km{mark}")
    print(f"    Mean  : {errors_km.mean():.1f} km\n")

    all_deg = np.vstack([obs_deg, gt_deg, pred_deg, ens_deg.reshape(-1, 2)])

    # [FIX, quan trọng] Margin cố định 4.5° (~500km) trước đây khiến map
    # LUÔN rộng hơn track rất nhiều — với track dài (ví dụ 724km ở 72h),
    # tổng khung nhìn ra >1700km trong khi ensemble spread thật chỉ
    # 27-34km, khiến ensemble "biến mất" trực quan dù dữ liệu HOÀN TOÀN
    # KHÔNG co cụm (đã xác nhận bằng panel Spread vs Error trước khi bị
    # bỏ theo yêu cầu). Đây là vấn đề TỶ LỆ HIỂN THỊ, không phải model.
    # Sửa: margin giờ tỷ lệ theo độ trải dài thật của track (10% mỗi
    # chiều), với sàn 1.0° (đủ hiển thị coastline quanh 1 điểm nếu track
    # rất ngắn) và trần 4.5° (giữ hành vi cũ cho track thật sự dài, tránh
    # zoom quá sát mất context địa lý).
    lon_span = all_deg[:, 0].max() - all_deg[:, 0].min()
    lat_span = all_deg[:, 1].max() - all_deg[:, 1].min()
    margin_lon = float(np.clip(lon_span * 0.10, 1.0, 4.5))
    margin_lat = float(np.clip(lat_span * 0.10, 1.0, 4.5))

    # [BỔ SUNG] Theo yêu cầu "kéo bề ngang to ra": track RITA trải dài
    # chủ yếu theo VĨ ĐỘ (Bắc-Nam), lon span tự nhiên hẹp hơn nhiều so
    # với lat span -> nếu chỉ dùng margin tỷ lệ nhỏ, map ra hình rất
    # cao-hẹp. Mở rộng thêm khoảng ngang (lon) để map có tỷ lệ gần
    # vuông/ngang hơn, cho thấy nhiều bối cảnh địa lý xung quanh hơn —
    # không đổi lat_range (giữ đúng độ dài track thật theo chiều dọc).
    extra_lon_widen = max(0.0, (lat_span - lon_span) * 0.35)
    margin_lon += extra_lon_widen

    lon_range = (all_deg[:, 0].min() - margin_lon, all_deg[:, 0].max() + margin_lon)
    lat_range = (all_deg[:, 1].min() - margin_lat, all_deg[:, 1].max() + margin_lat)

    # [FIX, cố định khung hình] Trước đây figsize được TÍNH ĐỘNG theo tỷ
    # lệ lon_range/lat_range thật của track (map_aspect), để tránh
    # khoảng trắng 2 bên khi cartopy PlateCarree tự giữ đúng tỷ lệ 1:1
    # kinh/vĩ. Nhưng cách đó khiến MỖI ảnh xuất ra có kích thước khác
    # nhau tùy theo storm — không so sánh được cạnh nhau dễ dàng.
    #
    # Giờ đảo ngược cách tiếp cận: CỐ ĐỊNH khung hình (figsize không đổi,
    # chữ nhật đứng — portrait), rồi thay vào đó MỞ RỘNG lon_range (giữ
    # nguyên lat_range, vốn đã phản ánh đúng độ dài track theo chiều
    # dọc) sao cho tỷ lệ lon_range/lat_range KHỚP ĐÚNG tỷ lệ khung hình
    # cố định. Nhờ vậy map luôn lấp đầy toàn bộ khung (không còn khoảng
    # trắng 2 bên do mismatch aspect ratio), và track luôn được vẽ to
    # hết mức có thể trong khung đó — đúng yêu cầu "phóng to/thu nhỏ
    # để bão được vẽ lên rõ trong khung", thay vì đổi khung theo track.
    FIG_W, FIG_H = 9.0, 12.0   # khung chữ nhật đứng cố định cho MỌI storm
    target_aspect = FIG_W / FIG_H   # tỷ lệ lon_range/lat_range cần đạt
    lon_span_cur  = lon_range[1] - lon_range[0]
    lat_span_cur  = max(lat_range[1] - lat_range[0], 0.01)
    cur_aspect    = lon_span_cur / lat_span_cur

    if cur_aspect < target_aspect:
        # Track hẹp hơn khung theo chiều ngang -> mở rộng lon_range,
        # giữ nguyên tâm và giữ nguyên lat_range.
        wanted_lon_span = target_aspect * lat_span_cur
        extra = (wanted_lon_span - lon_span_cur) / 2.0
        lon_range = (lon_range[0] - extra, lon_range[1] + extra)
    else:
        # Track rộng hơn khung theo chiều ngang (hiếm, track gần theo
        # hướng Đông-Tây) -> mở rộng lat_range thay vì cắt bớt lon_range,
        # để không làm mất một phần track khỏi khung nhìn.
        wanted_lat_span = lon_span_cur / target_aspect
        extra = (wanted_lat_span - lat_span_cur) / 2.0
        lat_range = (lat_range[0] - extra, lat_range[1] + extra)

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=STYLE["bg_color"])
    gs  = fig.add_gridspec(1, 1)
    ax_map = make_map_ax(fig, gs[0, 0], lon_range, lat_range)

    dt_str    = datetime.strptime(t_date, "%Y%m%d%H").strftime("%d %b %Y  %H:%M UTC")
    fh        = args.pred_len * 6

    _plot_on_ax(
        ax_map, lon_range, lat_range,
        obs_deg, gt_deg, pred_deg, pred_Me_n,
        all_trajs_deg=ens_deg if args.num_ensemble >= 3 else None,
        errors_km=errors_km,
        title=t_name,
        dt_str=dt_str,
        ref_spread_km=ref_spread_km,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"forecast_{fh}h_{t_name}_{t_date}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=STYLE["bg_color"])
    plt.close()
    print(f"  Saved → {out}\n")


# ── Case-study grid ────────────────────────────────────────────────────────────

def visualize_case_study(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, dset = load_model_and_data(args, device, "test")

    cases = [
        {"name": args.straight1_name, "date": args.straight1_date,
         "label": "Straight-track 1"},
        {"name": args.straight2_name, "date": args.straight2_date,
         "label": "Straight-track 2"},
        {"name": "WIPHA",             "date": args.recurv_date,
         "label": "Recurvature — WIPHA"},
    ]

    fig = plt.figure(figsize=(22, 8 * len(cases)), facecolor=STYLE["bg_color"])
    gs  = fig.add_gridspec(len(cases), 3, wspace=0.10, hspace=0.28)

    for row, case in enumerate(cases):
        t_name              = case["name"].strip().upper()
        t_date, was_snapped = resolve_date(case["date"])
        label               = case.get("label", t_name)

        target, matched_obs_len, actual_date = find_target(
            dset, t_name, t_date, args.obs_len
        )

        if target is None:
            print(f"  ⚠  {t_name} @ {t_date} — not found")
            # [FIX-4] Use a single spanning subplot instead of 3 separate ones
            # that would conflict with the 2-column map layout.
            ax_nf = fig.add_subplot(gs[row, :])
            ax_nf.set_facecolor(STYLE["bg_color"])
            ax_nf.text(
                0.5, 0.5, f"NOT FOUND\n{t_name}",
                ha="center", va="center", color="red",
                fontsize=14, transform=ax_nf.transAxes,
            )
            ax_nf.axis("off")
            continue

        if actual_date != t_date:
            t_date = actual_date

        if matched_obs_len != args.obs_len:
            print(
                f"  [INFO] {t_name}: dùng tydate[{matched_obs_len}] "
                f"thay vì [{args.obs_len}]"
            )

        (obs_deg, gt_deg, pred_deg, pred_Me_n, ens_deg,
         errors_km, cliper_err) = run_inference(
            model, target, device, args.ode_steps, args.num_ensemble
        )

        all_deg   = np.vstack([obs_deg, gt_deg, pred_deg, ens_deg.reshape(-1, 2)])
        margin    = 4.5
        lon_range = (all_deg[:, 0].min() - margin, all_deg[:, 0].max() + margin)
        lat_range = (all_deg[:, 1].min() - margin, all_deg[:, 1].max() + margin)

        dt_str    = datetime.strptime(t_date, "%Y%m%d%H").strftime("%d %b %Y %H:%M UTC")
        snap_note = f" [snapped from {case['date']}]" if was_snapped else ""
        ax_map    = make_map_ax(fig, gs[row, :2], lon_range, lat_range)
        ax_err    = fig.add_subplot(gs[row, 2])
        ax_err.set_facecolor(STYLE["bg_color"])

        _plot_on_ax(
            ax_map, lon_range, lat_range,
            obs_deg, gt_deg, pred_deg, pred_Me_n,
            all_trajs_deg=ens_deg if args.num_ensemble >= 3 else None,
            errors_km=errors_km,
            title=f"[{label}]  {t_name}  (ode_steps={args.ode_steps}){snap_note}",
            dt_str=dt_str,
        )
        plot_spread_over_time(ax_err, ens_deg, errors_km, cliper_err, t_name)

        ade = errors_km.mean()
        e72 = errors_km[11] if len(errors_km) > 11 else float("nan")
        print(f"  [{label}] ADE={ade:.1f} km  72h={e72:.1f} km")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "case_study_grid_v13.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=STYLE["bg_color"])
    plt.close()
    print(f"\n  Saved → {out}")


# ── Multi-model mode (MERGE, từ plot_track_paper_style.py) ─────────────────────

def visualize_multi_model(args):
    """
    So sánh FM + tối đa 4 baseline (ST-Trans/LSTM/GRU/RNN) trên CÙNG 1
    storm/window — mỗi checkpoint CLI arg là optional, chỉ vẽ model nào
    được truyền checkpoint.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_name              = args.tc_name.strip().upper()
    t_date, was_snapped = resolve_date(args.tc_date)

    print(f"{'=' * 65}")
    print(f"  TC-FM Visualize — Multi-model comparison  |  {t_name}  @  {t_date}")
    print(f"{'=' * 65}\n")

    # Dataset load 1 lần, KHÔNG qua load_model_and_data (hàm đó gắn với
    # riêng TCFlowMatching) — tự load dataset trực tiếp để dùng chung
    # cho mọi model.
    dset, _ = data_loader(
        args, {"root": args.TC_data_path, "type": args.dset_type},
        test=True, test_year=args.test_year,
    )
    print(f"  Dataset: {len(dset)} samples\n")

    target, matched_obs_len, actual_date = find_target(dset, t_name, t_date, args.obs_len)
    if target is None:
        print(f"  '{t_name} @ {t_date}' not found.")
        list_available(dset, t_name, args.obs_len)
        return
    if actual_date != t_date:
        t_date = actual_date
    print(f"  Found: {t_name} @ {t_date}\n")

    jobs = [
        ("FM",       "fm",       args.fm_checkpoint),
        ("ST-Trans", "st_trans", args.st_trans_checkpoint),
        ("LSTM",     "lstm",     args.lstm_checkpoint),
        ("GRU",      "gru",      args.gru_checkpoint),
        ("RNN",      "rnn",      args.rnn_checkpoint),
    ]
    jobs = [(n, k, p) for n, k, p in jobs if p]
    if not jobs:
        print("  ERROR: cần ít nhất 1 checkpoint (--fm_checkpoint / "
              "--st_trans_checkpoint / --lstm_checkpoint / --gru_checkpoint / "
              "--rnn_checkpoint)")
        return

    preds_by_model, errors_by_model = {}, {}
    winds_pred_by_model, wind_gt = {}, None
    obs_deg = gt_deg = None
    for name, kind, ckpt in jobs:
        print(f"  Loading {name}: {ckpt}")
        model = load_model_generic(ckpt, kind, device,
                                   obs_len=args.obs_len, pred_len=args.pred_len)
        od, gd, pd_, ens, err, wpred, wgt = run_inference_generic(
            model, target, device, kind,
            ode_steps=args.ode_steps,
            num_ensemble=(args.num_ensemble if kind == "fm" else 1))
        obs_deg, gt_deg = od, gd
        preds_by_model[name] = pd_
        errors_by_model[name] = err
        winds_pred_by_model[name] = wpred
        if wind_gt is None:
            wind_gt = wgt  # giống nhau cho mọi model (cùng ground truth)
        print(f"    {name}: ADE={err.mean():.1f}km")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"track_multi_{t_name}_{t_date}.png")
    plot_multi_model_comparison(obs_deg, gt_deg, preds_by_model, errors_by_model,
                                t_name, out)


# ── Batch mode: mọi storm x mọi timestep x 5 model x 3 seed ─────────────────

def iterate_all_storms(dset, obs_len: int):
    """
    [MỚI] Liệt kê TOÀN BỘ storm trong dataset + mọi index (mỗi index =
    1 sample = 1 mốc dự báo cụ thể, dataset đã tự chia sẵn sliding
    window lúc build — xác nhận qua trajectoriesWithMe_unet_training.py:
    self.tyID.append({"old": [year, name, start_idx], "tydate": [...]})
    với mỗi (start_idx, start_idx+seq_len) là 1 cửa sổ cố định, KHÔNG
    cần tự trượt cửa sổ thủ công ở đây).

    Trả về dict {storm_name: [(dataset_idx, forecast_date_str), ...]},
    sắp theo đúng thứ tự thời gian trong dataset (đã là thứ tự sliding
    window tăng dần vì cách self.tyID được append tuần tự lúc build).
    """
    storms = defaultdict(list)
    seen_per_storm = defaultdict(set)  # tránh trùng lặp (storm, date) nếu dataset có augment
    for i in range(len(dset)):
        info = dset[i][-1]
        name = str(info["old"][1]).strip().upper()
        if obs_len >= len(info["tydate"]):
            continue
        fdate = str(info["tydate"][obs_len]).strip()
        key = (name, fdate)
        if key in seen_per_storm[name]:
            continue
        seen_per_storm[name].add(key)
        storms[name].append((i, fdate))
    return storms


def visualize_batch_all_storms(args):
    """
    [MỚI] Chạy visualize cho MỌI storm x MỌI timestep có sẵn trong
    dataset x 5 model (FM/ST-Trans/LSTM/GRU/RNN). Với mỗi model, chạy
    đủ 3 seed để XÁC ĐỊNH seed nào có ADE thấp nhất (tốt nhất), nhưng
    CHỈ VẼ seed tốt nhất đó — dùng thẳng _plot_on_ax() (giống hệt
    visualize_forecast(), full map không inset, đúng style/nền đã xác
    nhận đúng), KHÔNG còn hiển thị multi-seed đậm/nhạt như bản trước.
    Không có wind panel/marker. Mỗi model dự báo đủ args.pred_len bước
    (mặc định 12 = 72h).

    Output: <output_dir>/<Storm>/<Model>/forecast_<date>.png

    Checkpoint cho mỗi model: dùng --fm_checkpoints/--st_trans_checkpoints/
    --lstm_checkpoints/--gru_checkpoints/--rnn_checkpoints (mỗi flag
    nhận nhiều path, 1/seed) — model nào KHÔNG được truyền checkpoint sẽ
    tự động BỎ QUA (không lỗi), để bạn có thể chạy batch chỉ với vài
    model nếu chưa có đủ 5x3=15 checkpoint.

    CẢNH BÁO SỐ LƯỢNG: với dataset nhiều storm dài, số hình sinh ra có
    thể RẤT LỚN (mỗi storm N mốc x 5 model = 5N hình). Dùng --storm_filter
    để giới hạn 1 vài storm cụ thể nếu chỉ muốn test trước khi chạy full.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'=' * 65}")
    print(f"  TC-FM Visualize — BATCH mode (mọi storm x mọi timestep)")
    print(f"{'=' * 65}\n")

    model_groups = [
        ("FM",       "fm",       args.fm_checkpoints),
        ("ST-Trans", "st_trans", args.st_trans_checkpoints),
        ("LSTM",     "lstm",     args.lstm_checkpoints),
        ("GRU",      "gru",      args.gru_checkpoints),
        ("RNN",      "rnn",      args.rnn_checkpoints),
    ]
    model_groups = [(name, kind, ckpts) for name, kind, ckpts in model_groups if ckpts]
    if not model_groups:
        print("  ERROR: cần ít nhất 1 trong --fm_checkpoints / "
              "--st_trans_checkpoints / --lstm_checkpoints / "
              "--gru_checkpoints / --rnn_checkpoints (mỗi flag nhận "
              "nhiều path, 1/seed)")
        return
    print(f"  Models sẽ chạy: {[name for name, _, _ in model_groups]}\n")

    # Dataset load 1 lần dùng chung cho mọi model/seed (giống
    # visualize_multi_model — tự load trực tiếp, không qua
    # load_model_and_data() vốn gắn với 1 checkpoint FM cụ thể).
    dset, _ = data_loader(
        args, {"root": args.TC_data_path, "type": args.dset_type},
        test=True, test_year=args.test_year,
    )
    print(f"  Dataset: {len(dset)} samples\n")

    storms = iterate_all_storms(dset, args.obs_len)
    if args.storm_filter:
        wanted = {s.strip().upper() for s in args.storm_filter}
        storms = {k: v for k, v in storms.items() if k in wanted}
        print(f"  --storm_filter áp dụng: {sorted(storms.keys())}")

    total_windows = sum(len(v) for v in storms.values())
    print(f"  Tổng: {len(storms)} storm, {total_windows} mốc dự báo, "
          f"{len(model_groups)} model → tối đa {total_windows * len(model_groups)} hình\n")

    n_done, n_skipped = 0, 0
    for storm_name, windows in storms.items():
        for dataset_idx, fdate in windows:
            target = dset[dataset_idx]

            for model_name, model_kind, checkpoints in model_groups:
                out_dir = os.path.join(args.output_dir, storm_name, model_name)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"forecast_{fdate}.png")

                if args.skip_existing and os.path.exists(out_path):
                    n_skipped += 1
                    continue

                preds_by_seed, errors_by_seed, ens_by_seed = {}, {}, {}
                obs_deg = gt_deg = None
                ok = True
                for seed_idx, ckpt in enumerate(checkpoints):
                    try:
                        model = load_model_generic(
                            ckpt, model_kind, device,
                            obs_len=args.obs_len, pred_len=args.pred_len)
                        od, gd, pd_, ens, err, wpred, wgt = run_inference_generic(
                            model, target, device, model_kind,
                            ode_steps=args.ode_steps,
                            num_ensemble=(args.num_ensemble if model_kind == "fm" else 1))
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"  ⚠ Lỗi {storm_name}/{model_name}/seed{seed_idx} "
                              f"@ {fdate}: {e}")
                        ok = False
                        continue
                    obs_deg, gt_deg = od, gd
                    seed_label = str(seed_idx)
                    m = re.search(r"seed[_-]?(\d+)", ckpt)
                    if m:
                        seed_label = m.group(1)
                    preds_by_seed[seed_label] = pd_
                    errors_by_seed[seed_label] = err
                    ens_by_seed[seed_label] = ens   # [K,T,2] ensemble của seed này (None nếu model không phải FM/không có ensemble)

                if not ok or not preds_by_seed:
                    n_skipped += 1
                    continue

                # [FIX-17] Theo yêu cầu: KHÔNG còn hiển thị multi-seed
                # (đậm/nhạt) nữa — chỉ chọn seed có ADE thấp nhất (tốt
                # nhất) rồi vẽ y hệt visualize_forecast() (dùng thẳng
                # _plot_on_ax(), full map không inset, đúng style/nền
                # đã xác nhận đúng ở visualize_forecast()).
                try:
                    best_seed_label = min(errors_by_seed, key=lambda k: errors_by_seed[k].mean())
                    pred_deg  = preds_by_seed[best_seed_label]
                    ens_deg   = ens_by_seed.get(best_seed_label)
                    errors_km = errors_by_seed[best_seed_label]

                    all_pts = [obs_deg, gt_deg, pred_deg]
                    if ens_deg is not None:
                        all_pts.append(ens_deg.reshape(-1, 2))
                    all_deg = np.vstack(all_pts)

                    # Margin động theo track thật — CÙNG công thức đã
                    # dùng ở visualize_forecast() (đây chính là style
                    # bạn xác nhận đúng ở ảnh mẫu).
                    lon_span = all_deg[:, 0].max() - all_deg[:, 0].min()
                    lat_span = all_deg[:, 1].max() - all_deg[:, 1].min()
                    margin_lon = float(np.clip(lon_span * 0.10, 1.0, 4.5))
                    margin_lat = float(np.clip(lat_span * 0.10, 1.0, 4.5))
                    extra_lon_widen = max(0.0, (lat_span - lon_span) * 0.35)
                    margin_lon += extra_lon_widen
                    lon_range = (all_deg[:, 0].min() - margin_lon, all_deg[:, 0].max() + margin_lon)
                    lat_range = (all_deg[:, 1].min() - margin_lat, all_deg[:, 1].max() + margin_lat)

                    # [FIX, cố định khung hình] Cùng cách tiếp cận đã áp
                    # dụng ở visualize_forecast(): khung chữ nhật đứng
                    # CỐ ĐỊNH cho MỌI storm/model (so sánh cạnh nhau dễ
                    # dàng), mở rộng lon_range để khớp đúng tỷ lệ khung
                    # thay vì để figsize co giãn theo từng track.
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
                    ax_map = make_map_ax(fig, 111, lon_range, lat_range)

                    dt_str = datetime.strptime(fdate, "%Y%m%d%H").strftime("%d %b %Y  %H:%M UTC")
                    fh = args.pred_len * 6
                    _plot_on_ax(
                        ax_map, lon_range, lat_range,
                        obs_deg, gt_deg, pred_deg, None,   # pred_Me_deg=None -> không có wind marker
                        all_trajs_deg=ens_deg if (ens_deg is not None and ens_deg.shape[0] >= 3) else None,
                        errors_km=errors_km,
                        title=storm_name,
                        dt_str=dt_str,
                        pred_label=f"{model_name} seed={best_seed_label}",
                    )

                    os.makedirs(out_dir, exist_ok=True)
                    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=STYLE["bg_color"])
                    plt.close()
                    print(f"  Saved → {out_path}")
                    n_done += 1
                except Exception as e:
                    print(f"  ⚠ Lỗi vẽ {storm_name}/{model_name} @ {fdate}: {e}")
                    n_skipped += 1

    print(f"\n  Hoàn tất: {n_done} hình đã lưu, {n_skipped} bị bỏ qua "
          f"(lỗi hoặc đã tồn tại nếu --skip_existing).")


# ── Multi-seed mode (chỉ cho 1 kiến trúc, mặc định FM) ──────────────────────

def visualize_multi_seed(args):
    """
    So sánh nhiều SEED của CÙNG 1 kiến trúc (mặc định --model_type fm)
    trên CÙNG 1 storm/window — mỗi checkpoint trong --seed_checkpoints là
    1 seed, tất cả cùng 1 kiến trúc (không trộn FM với baseline khác;
    dùng --mode multi_model cho việc đó). Dùng để minh hoạ độ ổn định
    của kiến trúc qua random init, KHÔNG dùng để thay thế bảng thống kê
    mean±std theo seed (generate_paper_report.py) — bản đồ chỉ minh hoạ
    1 lần dự báo cụ thể, không phải đại lượng thống kê.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_name              = args.tc_name.strip().upper()
    t_date, was_snapped = resolve_date(args.tc_date)

    print(f"{'=' * 65}")
    print(f"  TC-FM Visualize — Multi-seed comparison ({args.model_type.upper()})"
          f"  |  {t_name}  @  {t_date}")
    print(f"{'=' * 65}\n")

    if not args.seed_checkpoints:
        print("  ERROR: --seed_checkpoints cần ít nhất 1 checkpoint "
              "(khuyến nghị >=2 để so sánh có ý nghĩa)")
        return

    dset, _ = data_loader(
        args, {"root": args.TC_data_path, "type": args.dset_type},
        test=True, test_year=args.test_year,
    )
    print(f"  Dataset: {len(dset)} samples\n")

    target, matched_obs_len, actual_date = find_target(dset, t_name, t_date, args.obs_len)
    if target is None:
        print(f"  '{t_name} @ {t_date}' not found.")
        list_available(dset, t_name, args.obs_len)
        return
    if actual_date != t_date:
        t_date = actual_date
    print(f"  Found: {t_name} @ {t_date}\n")

    # Suy ra nhãn seed từ checkpoint: ưu tiên đọc field "seed" trong
    # checkpoint (khớp cách evaluate_multi_model.py làm), fallback parse
    # "seed<N>" từ đường dẫn, cuối cùng dùng số thứ tự nếu không tìm được.
    import re
    def _infer_seed_label(ckpt_path: str, idx: int) -> str:
        try:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ck, dict) and "seed" in ck:
                return str(ck["seed"])
        except Exception:
            pass
        m = re.search(r"seed[_-]?(\d+)", ckpt_path)
        if m:
            return m.group(1)
        return str(idx)

    preds_by_seed, errors_by_seed = {}, {}
    winds_pred_by_seed, wind_gt = {}, None
    obs_deg = gt_deg = None
    for idx, ckpt in enumerate(args.seed_checkpoints):
        seed_label = _infer_seed_label(ckpt, idx)
        print(f"  Loading seed={seed_label}: {ckpt}")
        model = load_model_generic(ckpt, args.model_type, device,
                                   obs_len=args.obs_len, pred_len=args.pred_len)
        od, gd, pd_, ens, err, wpred, wgt = run_inference_generic(
            model, target, device, args.model_type,
            ode_steps=args.ode_steps,
            num_ensemble=(args.num_ensemble if args.model_type == "fm" else 1))
        obs_deg, gt_deg = od, gd
        preds_by_seed[seed_label] = pd_
        errors_by_seed[seed_label] = err
        winds_pred_by_seed[seed_label] = wpred
        if wind_gt is None:
            wind_gt = wgt  # giống nhau cho mọi seed (cùng ground truth)
        print(f"    seed={seed_label}: ADE={err.mean():.1f}km")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir,
                       f"track_multiseed_{args.model_type}_{t_name}_{t_date}.png")
    plot_multi_seed_comparison(obs_deg, gt_deg, preds_by_seed, errors_by_seed,
                               f"{t_name} ({args.model_type.upper()})", out,
                               winds_pred_by_seed=winds_pred_by_seed,
                               wind_gt=wind_gt)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",     default=None,
                   help="Checkpoint FM (bắt buộc cho --mode single/case_study)")
    p.add_argument("--TC_data_path",   required=True)
    p.add_argument("--output_dir",     default="outputs")
    p.add_argument("--mode",           default="single",
                   choices=["single", "case_study", "multi_model", "multi_seed", "batch_all"])
    p.add_argument("--tc_name",        default="WIPHA")
    p.add_argument("--tc_date",        default="2019073106")
    p.add_argument("--dset_type",      default="test")
    p.add_argument("--straight1_name", default="BEBINCA")
    p.add_argument("--straight1_date", default="2018090806")
    p.add_argument("--straight2_name", default="MANGKHUT")
    p.add_argument("--straight2_date", default="2018091312")
    p.add_argument("--recurv_date",    default="2019073106")
    p.add_argument("--test_year",      type=int,   default=None,
                   help="[FIX] Trước đây default=2019 -- data_loader() "
                        "chỉ load file .txt có '2019' trong TÊN FILE "
                        "(xác nhận qua trajectoriesWithMe_unet_training.py: "
                        "'test_year is None or str(test_year) in f'), khiến "
                        "MỌI storm khác năm 2019 bị loại bỏ hoàn toàn (đây "
                        "là nguyên nhân RITA 1975 báo 'not found' trước đây). "
                        "Dataset test ở đây có 1 file/storm trải dài "
                        "1975-2024 (RITA/WAYNE/YANCY/FLO/TERESA/LINFA/"
                        "DANAS/MOLAVE/CONSON/SURIGAE/HINNAMNOR/EWINIAR) -- "
                        "default=None (không lọc gì, load HẾT mọi storm) "
                        "là đúng cho cấu trúc dataset này. Chỉ truyền "
                        "--test_year <năm> nếu bạn CHỦ ĐỘNG muốn giới hạn "
                        "1 năm cụ thể.")
    # [FIX-DATA-30] Region filter — must match whatever was used during
    # training for consistent evaluation. If a checkpoint was trained
    # with --filter_region, pass the SAME flag here too, or the
    # visualized dataset composition won't match what the model saw
    # during training/validation.
    p.add_argument("--filter_region",  action="store_true", default=False,
                   help="Keep only storms whose track substantially enters "
                        "the South China Sea / Vietnam region. Should match "
                        "the setting used when training the checkpoint(s) "
                        "being visualized here.")
    p.add_argument("--min_pct_in_scs", type=float, default=15.0,
                   help="Minimum %% of track points inside the SCS/Vietnam "
                        "box required to keep a storm when --filter_region.")
    p.add_argument("--obs_len",        type=int,   default=8)
    p.add_argument("--pred_len",       type=int,   default=12)
    p.add_argument("--ode_steps",      type=int,   default=10)
    p.add_argument("--num_ensemble",   type=int,   default=20)
    p.add_argument("--batch_size",     type=int,   default=1)
    p.add_argument("--num_workers",    type=int,   default=0)
    p.add_argument("--delim",          default=" ")
    p.add_argument("--skip",           type=int,   default=1)
    p.add_argument("--min_ped",        type=int,   default=1)
    p.add_argument("--threshold",      type=float, default=0.002)
    p.add_argument("--other_modal",    default="gph")
    p.add_argument("--ode_sweep_json", default=None,
                   help="[MỚI, optional] Đường dẫn ode_steps_sweep.json từ "
                        "ablation_runner.py --mode ode_steps — nếu truyền, "
                        "box 'Spread (1σ)' trên map sẽ in kèm số tham chiếu "
                        "(trung bình toàn test set, ~420 storm-window) bên "
                        "cạnh spread của riêng storm đang xem, để biết storm "
                        "này có bất thường so với trung bình hay không. "
                        "Không truyền thì giữ nguyên hành vi cũ (chỉ 1 số).")

    # --mode multi_model: mỗi checkpoint optional, chỉ vẽ model được truyền
    p.add_argument("--fm_checkpoint",       default=None)
    p.add_argument("--st_trans_checkpoint", default=None)
    p.add_argument("--lstm_checkpoint",     default=None)
    p.add_argument("--gru_checkpoint",      default=None)
    p.add_argument("--rnn_checkpoint",      default=None)

    # --mode multi_seed: nhiều checkpoint CÙNG 1 kiến trúc (mặc định FM)
    p.add_argument("--seed_checkpoints",    nargs="+", default=None,
                   help="Danh sách checkpoint, mỗi cái 1 seed, CÙNG 1 "
                        "kiến trúc (xem --model_type). Ví dụ: "
                        "--seed_checkpoints runs/fm_seed0/best_model.pth "
                        "runs/fm_seed1/best_model.pth runs/fm_seed3/best_model.pth")
    p.add_argument("--model_type",          default="fm",
                   choices=["fm", "st_trans", "lstm", "gru", "rnn"],
                   help="Kiến trúc dùng cho --mode multi_seed (mặc định fm)")

    # --mode batch_all: MỖI model nhận NHIỀU checkpoint (nhiều seed) —
    # khác --fm_checkpoint (số ít, multi_model) và --seed_checkpoints
    # (chỉ 1 model, multi_seed). Model nào không truyền checkpoints sẽ
    # tự động bỏ qua khi chạy batch.
    p.add_argument("--fm_checkpoints",       nargs="+", default=None,
                   help="[batch_all] Nhiều checkpoint FM, 1/seed")
    p.add_argument("--st_trans_checkpoints", nargs="+", default=None,
                   help="[batch_all] Nhiều checkpoint ST-Trans, 1/seed")
    p.add_argument("--lstm_checkpoints",     nargs="+", default=None,
                   help="[batch_all] Nhiều checkpoint LSTM, 1/seed")
    p.add_argument("--gru_checkpoints",      nargs="+", default=None,
                   help="[batch_all] Nhiều checkpoint GRU, 1/seed")
    p.add_argument("--rnn_checkpoints",      nargs="+", default=None,
                   help="[batch_all] Nhiều checkpoint RNN, 1/seed")
    p.add_argument("--storm_filter",         nargs="+", default=None,
                   help="[batch_all, optional] Chỉ chạy các storm này "
                        "(tên viết hoa, ví dụ RITA WIPHA) — không truyền "
                        "thì chạy TOÀN BỘ storm trong dataset.")
    p.add_argument("--skip_existing",        action="store_true", default=False,
                   help="[batch_all] Bỏ qua nếu file output đã tồn tại "
                        "(hữu ích khi chạy lại sau khi bị gián đoạn).")

    args = p.parse_args()
    if args.mode == "single":
        if not args.model_path:
            print("  ERROR: --model_path required for --mode single"); sys.exit(1)
        visualize_forecast(args)
    elif args.mode == "case_study":
        if not args.model_path:
            print("  ERROR: --model_path required for --mode case_study"); sys.exit(1)
        visualize_case_study(args)
    elif args.mode == "multi_model":
        visualize_multi_model(args)
    elif args.mode == "multi_seed":
        visualize_multi_seed(args)
    elif args.mode == "batch_all":
        visualize_batch_all_storms(args)