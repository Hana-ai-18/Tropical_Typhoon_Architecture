"""
prepare_dataset.py
===================
Công cụ kiểm tra + lọc outlier + chia train/val/test cho bộ dữ liệu bão
(Data1d / Data3d / Env_data), khớp với TrajectoryDataset trong
Model/data/trajectoriesWithMe_unet_training.py (v25 patch).

CẤU TRÚC INPUT MONG ĐỢI (root):
    root/
      Data1d/            <-- các file "<year>_<name>.txt" nằm PHẲNG ở đây
          1970_0003.txt
          1975_0012.txt
          ...
      Data3d/
          <year>/<name>/WP<year><name>_<timestamp>.npy
      Env_data/                (tên thư mục không phân biệt hoa/thường,
                                 khớp logic auto-detect trong dataset gốc)
          <year>/<name>/<timestamp>.npy   (hoặc WP<year><name>_<timestamp>.npy)

FORMAT 1 DÒNG TRONG FILE .txt (đúng như code gốc _read_file):
    ID  LONG(norm)  LAT(norm)  PRES(norm)  WND(norm)  YYYYMMDDHH  Name
Cột LONG/LAT là dạng "normalized" (không phải độ thật), quy đổi ra độ:
    lon_deg = (lon_norm * 50 + 1800) / 10
    lat_deg = (lat_norm * 50) / 10
Đây CHÍNH XÁC là công thức dùng trong toàn bộ pipeline train hiện tại,
nên script này dùng lại y hệt để đảm bảo tính nhất quán.

CÔNG VIỆC SCRIPT NÀY LÀM:
  1. QUÉT & KHỚP: với mỗi file bão trong Data1d, kiểm tra:
       - file .txt parse được, đủ số dòng tối thiểu (obs_len+pred_len)
       - có ít nhất 1 timestamp có file Data3d tương ứng
       - có ít nhất 1 timestamp có file Env_data tương ứng
     In báo cáo chi tiết các bão bị thiếu Data3d/Env_data toàn bộ hoặc
     một phần, để bạn biết cần tải bổ sung gì trước khi train.

  1b. ĐỐI CHIẾU CHÉO CHẶT (file-level, cả 2 chiều):
       - Chiều Data1d -> Data3d/Env: mỗi timestamp trong file .txt phải
         có ĐÚNG 1 file khớp tên convention "WP<year><name>_<ts>.npy"
         (Data3d) và "<ts>.npy" hoặc "WP<year><name>_<ts>.npy" (Env).
       - Chiều NGƯỢC LẠI: quét toàn bộ thư mục con trong Data3d/<year>/
         <name>/ và Env_data/<year>/<name>/, tìm những file KHÔNG khớp
         với bất kỳ bão nào có trong Data1d (dữ liệu rác/thừa, thư mục
         mồ côi — ví dụ thư mục Data3d cho 1 bão đã bị xoá khỏi Data1d,
         hoặc gõ sai year/name) -> ghi report riêng để bạn dọn dẹp.
       - Kiểm tra thử load 1 file .npy Data3d/năm/bão để bắt lỗi shape
         sớm (không đúng 81x81x13, hoặc file hỏng không đọc được).

  2. LỌC OUTLIER:
       a. Độ dài quỹ đạo < (obs_len + pred_len) điểm  -> loại
       b. Toạ độ ngoài vùng hợp lệ toàn cầu (giống _LON_VALID_MIN/MAX,
          _LAT_VALID_MAX trong dataset gốc) -> loại
       c. Giá trị PRES/WND normalized bất thường (NaN/Inf, hoặc vượt
          ngưỡng tuyệt đối --max_pres_abs/--max_wnd_abs) -> loại
       LƯU Ý: KHÔNG còn lọc "bước nhảy toạ độ bất thường" ở đây nữa —
       việc phát hiện/cắt điểm gãy quỹ đạo (bão bị ghép nhầm 2 cơn khác
       nhau) đã tách sang script riêng fix_discontinuity_and_sync.py,
       dùng DUY NHẤT tiêu chí khoảng cách THỜI GIAN giữa 2 timestep
       (không dùng bước nhảy toạ độ). Chạy script đó TRƯỚC (sửa tận gốc
       Data1d/Data3d/Env_data) rồi mới chạy prepare_dataset.py này.

  3. GẮN NHÃN "vào Biển Đông" / "đi vào Việt Nam" (2 nhãn TÁCH BIỆT):
       a. is_scs_storm: dùng đúng logic _storm_touches_scs_vietnam trong
          code gốc — lon 99-121E, lat 0-23N, giữ nếu >= min_pct% số điểm
          quỹ đạo nằm trong vùng đó. Đây là nhãn "quanh Biển Đông" rộng,
          KHÔNG bắt buộc bão phải áp sát Việt Nam.
       b. touches_vn_coast: nhãn HẸP HƠN, riêng cho "bão thực sự đi vào
          Việt Nam" (đất liền hoặc vùng biển sát bờ), dùng bounding box
          _VN_LON_MIN/MAX, _VN_LAT_MIN/MAX (mặc định 102-110E, 7-23N).
          Nhãn NÀY dùng để đảm bảo >= 4 bão trong tập test đúng nghĩa
          "đi vào Việt Nam" theo yêu cầu của bạn — không lẫn với bão chỉ
          lượn quanh Biển Đông mà không áp sát VN.

  4. CHIA TRAIN/VAL/TEST theo STORM (không theo năm):
       - Tập TEST: mặc định CỐ ĐỊNH 10 bão (--test_min_storms/--test_max_storms,
         mặc định đều =10), CHỈ được chọn từ các bão có TÊN CHỮ thật
         (vd JOAN, RITA — has_real_name=True); bão chỉ có MÃ SỐ (vd
         0019, 0019_2) bị loại khỏi việc được chọn vào test (vẫn nằm
         trong train/val bình thường). Trong phạm vi bão tên chữ đó:
           (a) BẮT BUỘC >= 4 bão có touches_vn_coast=True (đi vào VN)
           (b) BẮT BUỘC >= 5 bão có difficulty_tier="de" (ít đổi hướng,
               đi theo 1 hướng tương đối ổn định)
           (c) Phần còn lại lấy từ các bão is_scs_storm (quanh Biển
               Đông, không nhất thiết chạm VN) — CHO PHÉP
         difficulty_tier dùng NGƯỠNG TUYỆT ĐỐI trên góc đổi hướng trung
         bình (curvature_score, xem assign_difficulty_tiers): "de" =
         đi theo 1 hướng ổn định, "kho" = có đổi hướng/chuyển hướng (ví
         dụ đang đi Tây Bắc rồi ngoặt xuống Tây Nam — recurve).
       - Tập TRAIN/VAL: phần còn lại (gồm cả bão mã số) được chia theo
         tỷ lệ 80/20 (--val_ratio mặc định 0.20 -> 20% vào val, 80% vào
         train), tính TRÊN PHẦN CÒN LẠI sau khi đã tách 10 bão test.
       Toàn bộ các bão bị loại ở bước 2 (outlier, theo đúng ngưỡng từ
       code gốc) sẽ KHÔNG được đưa vào bất kỳ tập nào.

  5. GHI KẾT QUẢ:
       - Copy (hoặc symlink) các file .txt hợp lệ vào
         root/Data1d/train/, root/Data1d/val/, root/Data1d/test/
         (đúng cấu trúc mà TrajectoryDataset(data_dir=...) cần)
       - Xuất report CSV: dataset_report.csv (toàn bộ bão + trạng thái),
         outliers_report.csv (bão bị loại + lý do),
         split_report.csv (bão nào vào train/val/test + nhãn dễ/khó/VN)

  6. KIỂM TRA "CÓ KHỚP VỚI CODE BẠN CUNG CẤP KHÔNG" (runtime smoke test):
       Sau khi tạo xong Data1d/train,val,test, script THỬ IMPORT VÀ GỌI
       THẬT class TrajectoryDataset + seq_collate từ chính file
       trajectoriesWithMe_unet_training.py của bạn (không phải suy đoán
       format) để phát hiện sớm các lỗi runtime mà chỉ đối chiếu file
       tĩnh không thấy được, ví dụ: lỗi shape Data3d, .npy Env hỏng khi
       np.load, key bị thiếu trong ENV_FEATURE_DIMS, seq_collate lỗi
       stack tensor, v.v. Dùng --check_with_real_code --project_root
       <đường dẫn tới thư mục cha của Model/> để bật kiểm tra này.
       Nếu không truyền --project_root, script tự dò lên các thư mục
       cha để tìm gói "Model" chứa trajectoriesWithMe_unet_training.py.

CÁCH DÙNG:
    python prepare_dataset.py \
        --root /path/to/TCND_vn \
        --obs_len 8 --pred_len 12 \
        --test_min_storms 10 --test_max_storms 10 --test_min_vn 4 \
        --val_ratio 0.20 \
        --min_pct_in_scs 15.0 \
        --check_with_real_code \
        --apply

    (Lưu ý: nên chạy fix_discontinuity_and_sync.py TRƯỚC script này để
    làm sạch Data1d/Data3d/Env_data ở gốc, vì prepare_dataset.py không
    còn tự lọc bước nhảy toạ độ nữa.)

    Bỏ --apply để chỉ chạy ở chế độ DRY-RUN (chỉ in báo cáo, không copy
    file / không tạo thư mục train,val,test). Khuyến nghị chạy dry-run
    trước để xem báo cáo, rồi mới --apply.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass, field

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Hằng số — LẤY ĐÚNG TỪ trajectoriesWithMe_unet_training.py (phần code active,
# không phải phần comment ở đầu file) để đảm bảo nhất quán với lúc train.
# ─────────────────────────────────────────────────────────────────────────────
DATA3D_H, DATA3D_W, DATA3D_CH = 81, 81, 13

_LON_VALID_MIN = 73.00
_LON_VALID_MAX = 186.40
_LAT_VALID_MAX = 64.00
_LAT_VALID_MIN = -90.00   # code gốc không chặn min lat -> chỉ chặn max

# Vùng Biển Đông / Việt Nam dùng để gắn nhãn (giống _storm_touches_scs_vietnam)
_SCS_LON_MIN, _SCS_LON_MAX = 99.0, 121.0
_SCS_LAT_MIN, _SCS_LAT_MAX = 0.0, 23.0

# Dải "bão ĐI VÀO Việt Nam" (đất liền hoặc vùng biển sát bờ) — dùng RIÊNG
# để chọn bão cho tập test theo đúng yêu cầu "ít nhất 4 bão đi vào Việt
# Nam". Thu hẹp hơn hẳn vùng Biển Đông nói chung (99-121E, 0-23N) để
# không lẫn các bão chỉ đi ngang/quanh Biển Đông mà không thực sự áp sát
# hoặc đổ bộ Việt Nam.
#   Kinh độ 102-110E : dải bờ biển Việt Nam (từ Vịnh Thái Lan tới Vịnh
#                       Bắc Bộ) cộng thêm vùng biển sát bờ phía đông.
#   Vĩ độ    7-23N    : từ mũi Cà Mau (~8.5N) tới biên giới phía Bắc
#                       (~23N), nới nhẹ xuống 7N để không bỏ sót các bão
#                       đi vào cực Nam.
# Có thể chỉnh 4 hằng số này nếu bạn có định nghĩa chính xác hơn (vd EEZ).
_VN_LON_MIN, _VN_LON_MAX = 102.0, 110.0
_VN_LAT_MIN, _VN_LAT_MAX = 7.0, 23.0

# Có thể chỉnh 4 hằng số này nếu bạn có định nghĩa chính xác hơn (vd EEZ).
_VN_LON_MIN, _VN_LON_MAX = 102.0, 110.0
_VN_LAT_MIN, _VN_LAT_MAX = 7.0, 23.0

# LUU Y: da BO hoan toan bo loc "buoc nhay toa do bat thuong" (truoc day
# la DEFAULT_MAX_STEP_DEG). Viec phat hien diem gay quy dao gio dung
# DUY NHAT tieu chi khoang cach thoi gian, thuc hien boi script rieng
# fix_discontinuity_and_sync.py (chay TRUOC script nay, sua tan goc
# Data1d/Data3d/Env_data), khong con o day nua.


def lonlat_from_norm(lon_norm: float, lat_norm: float) -> tuple[float, float]:
    """Công thức y hệt code gốc (denorm_traj / _storm_touches_scs_vietnam)."""
    lon_deg = (lon_norm * 50.0 + 1800.0) / 10.0
    lat_deg = (lat_norm * 50.0) / 10.0
    return lon_deg, lat_deg


def is_numeric_code(name: str) -> bool:
    """True neu `name` chi la MA SO (vd '0019', '0001'), False neu la
    TEN CHU that (vd 'JOAN', 'RITA'). Bo hau to '_<so>' o cuoi truoc khi
    kiem tra (vd '0019_2' -> '0019' -> van la ma so), vi hau to nay chi
    danh dau bao thu N trung ma so trong cung nam (xem FIX-DATA-31),
    khong lam no thanh "co ten"."""
    base = re.sub(r"_\d+$", "", name)
    return base.isdigit()


# ─────────────────────────────────────────────────────────────────────────────
# Đọc file Data1d .txt — PARSE Y HỆT _read_file() trong code gốc
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StormRecord:
    path: str
    year: str
    name: str
    n_points: int = 0
    dates: list = field(default_factory=list)
    lon_deg: np.ndarray = None
    lat_deg: np.ndarray = None
    pres_norm: np.ndarray = None
    wnd_norm: np.ndarray = None

    # Kết quả kiểm tra khớp dữ liệu 3D / env
    n_data3d_found: int = 0
    n_env_found: int = 0
    data3d_missing_dates: list = field(default_factory=list)
    env_missing_dates: list = field(default_factory=list)

    # Kết quả lọc outlier
    is_outlier: bool = False
    outlier_reasons: list = field(default_factory=list)

    # Nhãn vùng
    pct_in_scs: float = 0.0
    touches_vn_coast: bool = False
    is_scs_storm: bool = False

    # Độ khó (độ cong quỹ đạo tích luỹ, chuẩn hoá theo độ dài)
    curvature_score: float = 0.0
    max_turn_angle: float = 0.0        # goc re lon nhat tai 1 diem (radian)
    net_bearing_change: float = 0.0    # do lech huong tong the dau vs cuoi (radian)
    difficulty_tier: str = ""   # "de" | "kho" (it doi huong vs co doi huong, gan boi assign_difficulty_tiers)

    # Ket qua cat bot theo timestep thieu Data3d/Env o giua quy dao
    was_trimmed: bool = False
    trim_start_idx: int = 0          # index (trong rec.dates goc) bat dau doan giu lai
    trim_end_idx: int = -1           # index (trong rec.dates goc) ket thuc doan giu lai (inclusive)
    trim_reason: str = ""
    n_points_after_trim: int = 0
    dropped_after_trim: bool = False  # True neu sau khi cat khong con du diem de train

    # True neu ten bao la TEN CHU that (vd JOAN, RITA), False neu chi la
    # MA SO (vd 0019, 0019_2 - ke ca co hau to "_<so>" o cuoi)
    has_real_name: bool = False


def parse_data1d_file(path: str) -> StormRecord | None:
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    year = parts[0] if parts else "unknown"
    # FIX QUAN TRONG: lay TOAN BO phan con lai sau year (noi lai bang "_"),
    # KHONG chi lay parts[1]. Ly do: ten file co the co dang
    # "<year>_<name>_<so_thu_tu>.txt" (vd "2000_0019_2.txt" cho con bao
    # thu 2 trung ma so/ten "0019" trong cung nam 2000). Neu chi lay
    # parts[1], "2000_0019_2.txt" va "2000_0019_7.txt" se DEU bi gan
    # name="0019", khien 7 con bao khac nhau bi GOM CHUNG 1 khoa (year,
    # name) -> ghi de/tron lan nhau trong moi buoc xu ly phia sau (loc
    # outlier, doi chieu Data3d/Env, gan nhan vung, chia split). Lay
    # toan bo phan con lai giu dung "0019_2", "0019_7" rieng biet, khop
    # voi cach Data3d/Env_data da dat ten thu muc (WP20000019_2_... v.v).
    name = "_".join(parts[1:]) if len(parts) > 1 else base

    rec = StormRecord(path=path, year=year, name=name)
    rec.has_real_name = not is_numeric_code(name)

    rows = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            raw_lines = f.readlines()
    except Exception as e:
        rec.is_outlier = True
        rec.outlier_reasons.append(f"khong_doc_duoc_file: {e}")
        return rec

    for line in raw_lines:
        line = line.strip()
        if not line or line.startswith(("#", "//", "-", "=")):
            continue
        toks = line.split()
        if len(toks) < 7:
            continue
        try:
            int(toks[0])
        except ValueError:
            continue
        try:
            lon_norm = float(toks[1])
            lat_norm = float(toks[2])
            pres_norm = float(toks[3])
            wnd_norm = float(toks[4])
            date = toks[5]
        except (ValueError, IndexError):
            continue
        rows.append((date, lon_norm, lat_norm, pres_norm, wnd_norm))

    if not rows:
        rec.is_outlier = True
        rec.outlier_reasons.append("file_rong_hoac_khong_parse_duoc_dong_nao")
        return rec

    dates = [r[0] for r in rows]
    lon_norm_arr = np.array([r[1] for r in rows], dtype=np.float64)
    lat_norm_arr = np.array([r[2] for r in rows], dtype=np.float64)
    pres_arr = np.array([r[3] for r in rows], dtype=np.float64)
    wnd_arr = np.array([r[4] for r in rows], dtype=np.float64)

    lon_deg, lat_deg = lonlat_from_norm(lon_norm_arr, lat_norm_arr)

    rec.n_points = len(rows)
    rec.dates = dates
    rec.lon_deg = lon_deg
    rec.lat_deg = lat_deg
    rec.pres_norm = pres_arr
    rec.wnd_norm = wnd_arr
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Kiểm tra khớp Data3d / Env_data
# ─────────────────────────────────────────────────────────────────────────────

def find_env_root(root: str) -> str:
    for cand_name in ("Env_data", "ENV_DATA", "env_data", "Env_Data"):
        cand = os.path.join(root, cand_name)
        if os.path.isdir(cand):
            return cand
    return os.path.join(root, "Env_data")


def check_data3d_env_match(rec: StormRecord, data3d_root: str, env_root: str) -> None:
    d3d_folder = os.path.join(data3d_root, rec.year, rec.name)
    env_folder = os.path.join(env_root, rec.year, rec.name)

    d3d_files = set()
    if os.path.isdir(d3d_folder):
        d3d_files = set(os.listdir(d3d_folder))

    env_files = set()
    if os.path.isdir(env_folder):
        env_files = set(os.listdir(env_folder))

    for ts in rec.dates:
        prefix = f"WP{rec.year}{rec.name}_{ts}"
        found_3d = any(f.startswith(prefix) and f.endswith((".npy", ".nc")) for f in d3d_files) \
            or any(ts in f and f.endswith((".npy", ".nc")) for f in d3d_files)
        if found_3d:
            rec.n_data3d_found += 1
        else:
            rec.data3d_missing_dates.append(ts)

        found_env = (f"WP{rec.year}{rec.name}_{ts}.npy" in env_files) \
            or (f"{ts}.npy" in env_files) \
            or any(ts in f and f.endswith(".npy") for f in env_files)
        if found_env:
            rec.n_env_found += 1
        else:
            rec.env_missing_dates.append(ts)


def write_timestep_detail_report(out_path: str, all_recs: list) -> None:
    """
    Ghi report CHI TIET TUNG TIMESTEP: moi dong = 1 (bao, timestep), voi
    trang thai khop/thieu rieng cho Data1d/Data3d/Env_data. Day la muc
    kiem tra sau nhat - giup phat hien chinh xac timestep nao bi thieu,
    thay vi chi biet "bao X thieu 5 diem" ma khong biet la diem nao.
    """
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "name", "timestep_index", "timestamp",
                    "co_trong_data1d", "co_data3d", "co_env_data", "trang_thai"])
        for rec in all_recs:
            if rec.n_points == 0:
                continue
            missing_3d = set(rec.data3d_missing_dates)
            missing_env = set(rec.env_missing_dates)
            for idx, ts in enumerate(rec.dates):
                has_3d = ts not in missing_3d
                has_env = ts not in missing_env
                if has_3d and has_env:
                    trang_thai = "du"
                elif not has_3d and not has_env:
                    trang_thai = "thieu_ca_2"
                elif not has_3d:
                    trang_thai = "thieu_data3d"
                else:
                    trang_thai = "thieu_env"
                w.writerow([rec.year, rec.name, idx, ts, True, has_3d, has_env, trang_thai])



def try_load_one_data3d(rec: StormRecord, data3d_root: str) -> str | None:
    """Thử load 1 file .npy Data3d đầu tiên tìm được để kiểm tra shape khớp
    (81,81,13) hoặc (13,81,81) như code gốc mong đợi. Trả về None nếu OK,
    hoặc chuỗi mô tả lỗi nếu shape sai."""
    d3d_folder = os.path.join(data3d_root, rec.year, rec.name)
    if not os.path.isdir(d3d_folder):
        return None
    for ts in rec.dates:
        prefix = f"WP{rec.year}{rec.name}_{ts}"
        for f in os.listdir(d3d_folder):
            if f.startswith(prefix) and f.endswith(".npy"):
                p = os.path.join(d3d_folder, f)
                try:
                    arr = np.load(p)
                except Exception as e:
                    return f"loi_doc_npy {f}: {e}"
                if arr.ndim == 2:
                    return None  # sẽ được unsqueeze, chấp nhận được
                if arr.ndim != 3:
                    return f"shape_khong_hop_le {f}: ndim={arr.ndim}"
                H, W, C = arr.shape if arr.shape[0] != DATA3D_CH else (arr.shape[1], arr.shape[2], arr.shape[0])
                if (H, W) != (DATA3D_H, DATA3D_W):
                    return f"shape_khong_khop {f}: got HxW=({H},{W}), expect ({DATA3D_H},{DATA3D_W})"
                return None
    return None


def try_load_one_env(rec: StormRecord, env_root: str) -> str | None:
    """Thử load 1 file .npy Env đầu tiên tìm được, kiểm tra parse được
    thành dict (allow_pickle) và có ít nhất 1 key mong đợi. Trả về None
    nếu OK, hoặc chuỗi mô tả lỗi nếu không đọc/parse được."""
    env_folder = os.path.join(env_root, rec.year, rec.name)
    if not os.path.isdir(env_folder):
        return None
    for ts in rec.dates:
        for fname in (f"WP{rec.year}{rec.name}_{ts}.npy", f"{ts}.npy"):
            p = os.path.join(env_folder, fname)
            if os.path.exists(p):
                try:
                    raw = np.load(p, allow_pickle=True).item()
                except Exception as e:
                    return f"loi_doc_npy {fname}: {e}"
                if not isinstance(raw, dict):
                    return f"noi_dung_khong_phai_dict {fname}: type={type(raw)}"
                return None
    return None


def find_orphan_files(all_recs: list, data3d_root: str, env_root: str) -> tuple[list, list]:
    """Đối chiếu CHIỀU NGƯỢC: quét toàn bộ thư mục con trong Data3d/ và
    Env_data/, tìm những thư mục <year>/<name> KHÔNG khớp bất kỳ bão
    nào có trong Data1d (dữ liệu rác/mồ côi - có thể do bão đã bị xoá
    khỏi Data1d, gõ sai year/name khi build, hoặc thư mục thử nghiệm
    còn sót lại). Trả về (orphan_data3d_dirs, orphan_env_dirs), mỗi
    phần tử là (year, name, so_file_ben_trong, duong_dan_day_du)."""
    known = {(r.year, r.name) for r in all_recs}

    def scan(root):
        orphans = []
        if not os.path.isdir(root):
            return orphans
        for year in os.listdir(root):
            year_path = os.path.join(root, year)
            if not os.path.isdir(year_path):
                continue
            for name in os.listdir(year_path):
                name_path = os.path.join(year_path, name)
                if not os.path.isdir(name_path):
                    continue
                if (year, name) not in known:
                    n_files = len(os.listdir(name_path))
                    orphans.append((year, name, n_files, name_path))
        return orphans

    return scan(data3d_root), scan(env_root)


# ─────────────────────────────────────────────────────────────────────────────
# Cat bot quy dao khi thieu Data3d/Env o GIUA (khong phai dau/cuoi)
# ─────────────────────────────────────────────────────────────────────────────

def compute_trim_range(rec: StormRecord, min_len: int) -> None:
    """
    Voi moi bao, xac dinh danh sach timestep "sach" (co CA Data3d va Env).
    Neu co khoang trong o GIUA quy dao (mot hoac nhieu diem thieu nam
    giua 2 diem sach), so sanh do dai doan LIEN TUC SACH truoc diem hong
    DAU TIEN va doan LIEN TUC SACH sau diem hong CUOI CUNG - giu doan
    DAI HON, cat bo doan con lai. Doan giu lai PHAI sach 100% (khong
    thieu, khong thua timestep so voi Data3d/Env) va PHAI co it nhat
    min_len diem, neu khong se bi loai (dropped_after_trim=True).

    Ghi ket qua vao rec.trim_start_idx / trim_end_idx (index trong
    rec.dates GOC, inclusive ca 2 dau) va rec.was_trimmed.
    """
    n = rec.n_points
    if n == 0:
        return

    missing = set(rec.data3d_missing_dates) | set(rec.env_missing_dates)
    is_clean = [rec.dates[i] not in missing for i in range(n)]

    if all(is_clean):
        # Khong thieu gi ca - giu nguyen toan bo, khong can cat
        rec.trim_start_idx = 0
        rec.trim_end_idx = n - 1
        rec.n_points_after_trim = n
        rec.was_trimmed = False
        rec.dropped_after_trim = rec.n_points_after_trim < min_len
        if rec.dropped_after_trim:
            rec.trim_reason = (f"khong_thieu_gi_nhung_qua_ngan: {n} diem < {min_len}")
        return

    # Tim vi tri diem thieu DAU TIEN va CUOI CUNG
    missing_idxs = [i for i, c in enumerate(is_clean) if not c]
    first_missing = missing_idxs[0]
    last_missing = missing_idxs[-1]

    # Doan TRUOC diem hong dau tien: [0, first_missing - 1], phai kiem
    # tra bam sat tu dau (index 0) den first_missing-1 co LIEN TUC sach
    # khong (khong duoc co 1 diem hong khac xen giua truoc do - nhung vi
    # first_missing la diem hong DAU TIEN nen mac dinh [0, first_missing-1]
    # da la lien tuc sach).
    before_len = first_missing  # so diem sach lien tuc [0 .. first_missing-1]
    before_start, before_end = 0, first_missing - 1

    # Doan SAU diem hong cuoi cung: [last_missing + 1, n-1], tuong tu
    # da mac dinh lien tuc sach vi last_missing la diem hong CUOI CUNG.
    after_len = n - 1 - last_missing  # so diem sach lien tuc [last_missing+1 .. n-1]
    after_start, after_end = last_missing + 1, n - 1

    if before_len >= after_len:
        chosen_start, chosen_end, chosen_len = before_start, before_end, before_len
        chosen_side = "truoc"
    else:
        chosen_start, chosen_end, chosen_len = after_start, after_end, after_len
        chosen_side = "sau"

    rec.was_trimmed = True
    rec.trim_start_idx = chosen_start
    rec.trim_end_idx = chosen_end
    rec.n_points_after_trim = max(0, chosen_len)
    rec.trim_reason = (
        f"co {len(missing_idxs)} diem thieu Data3d/Env o giua quy dao "
        f"(dau tien tai idx={first_missing}, cuoi cung tai idx={last_missing}). "
        f"Doan truoc dai {before_len} diem, doan sau dai {after_len} diem -> "
        f"giu doan {chosen_side} ({chosen_len} diem)."
    )
    rec.dropped_after_trim = rec.n_points_after_trim < min_len
    if rec.dropped_after_trim:
        rec.trim_reason += f" Sau khi cat chi con {rec.n_points_after_trim} diem < {min_len} -> LOAI BAO NAY."


def rewrite_trimmed_file(rec: StormRecord, target_path: str) -> None:
    """Ghi de file .txt tai target_path (duong dan trong Data1d/<split>/,
    KHONG PHAI rec.path la file GOC PHANG trong Data1d/), chi giu lai
    cac dong thuoc doan [trim_start_idx, trim_end_idx] (inclusive, index
    trong rec.dates GOC truoc khi cat). Doc noi dung tu target_path (ban
    da duoc materialize_split copy sang) de dam bao khong dung nham
    duong dan, va khong bao gio dung tu rec.path o day - vi rec.path
    tro toi file GOC trong Data1d/ (chua chia split), ghi de no se lam
    hong du lieu goc ngay ca khi ban chi muon sua ban sao trong split."""
    with open(target_path, encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    data_lines = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", "-", "=")):
            continue
        toks = stripped.split()
        if len(toks) < 7:
            continue
        try:
            int(toks[0])
        except ValueError:
            continue
        data_lines.append(line.rstrip("\n"))

    kept_lines = data_lines[rec.trim_start_idx: rec.trim_end_idx + 1]
    with open(target_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept_lines))


# ─────────────────────────────────────────────────────────────────────────────
# Lọc outlier
# ─────────────────────────────────────────────────────────────────────────────

def apply_outlier_filters(
    rec: StormRecord,
    min_len: int,
    max_pres_abs: float,
    max_wnd_abs: float,
) -> None:
    if rec.is_outlier:
        return  # đã bị đánh dấu lỗi từ bước parse

    if rec.n_points < min_len:
        rec.is_outlier = True
        rec.outlier_reasons.append(
            f"qua_ngan: {rec.n_points} diem < obs_len+pred_len={min_len}")
        return

    if not np.isfinite(rec.lon_deg).all() or not np.isfinite(rec.lat_deg).all():
        rec.is_outlier = True
        rec.outlier_reasons.append("toa_do_chua_NaN_hoac_Inf")
        return

    if not np.isfinite(rec.pres_norm).all() or not np.isfinite(rec.wnd_norm).all():
        rec.is_outlier = True
        rec.outlier_reasons.append("pres_wnd_chua_NaN_hoac_Inf")
        return

    if rec.lon_deg.min() < _LON_VALID_MIN or rec.lon_deg.max() > _LON_VALID_MAX:
        rec.is_outlier = True
        rec.outlier_reasons.append(
            f"kinh_do_ngoai_vung_hop_le: [{rec.lon_deg.min():.2f}, "
            f"{rec.lon_deg.max():.2f}] khong nam trong "
            f"[{_LON_VALID_MIN}, {_LON_VALID_MAX}]")

    if rec.lat_deg.max() > _LAT_VALID_MAX or rec.lat_deg.min() < _LAT_VALID_MIN:
        rec.is_outlier = True
        rec.outlier_reasons.append(
            f"vi_do_ngoai_vung_hop_le: [{rec.lat_deg.min():.2f}, "
            f"{rec.lat_deg.max():.2f}] khong nam trong "
            f"[{_LAT_VALID_MIN}, {_LAT_VALID_MAX}]")

    if abs(rec.pres_norm).max() > max_pres_abs:
        rec.is_outlier = True
        rec.outlier_reasons.append(
            f"pres_norm_bat_thuong: max|.|={abs(rec.pres_norm).max():.3f} "
            f"> nguong {max_pres_abs}")

    if abs(rec.wnd_norm).max() > max_wnd_abs:
        rec.is_outlier = True
        rec.outlier_reasons.append(
            f"wnd_norm_bat_thuong: max|.|={abs(rec.wnd_norm).max():.3f} "
            f"> nguong {max_wnd_abs}")

    # LUU Y: KHONG con loc "buoc nhay toa do bat thuong" o day nua. Viec
    # phat hien/cat diem gay quy dao (bao bi ghep nham) da duoc tach
    # rieng sang script fix_discontinuity_and_sync.py, dung DUY NHAT
    # tieu chi khoang cach THOI GIAN giua 2 timestep (khong dung buoc
    # nhay toa do - xem docstring cua script do de biet ly do). Script
    # nay (prepare_dataset.py) gia dinh Data1d da duoc lam sach truoc
    # boi fix_discontinuity_and_sync.py, nen khong loc lai theo toa do
    # de tranh trung lap/mau thuan tieu chi.


def compute_scs_vn_labels(rec: StormRecord, min_pct_in_scs: float) -> None:
    in_scs = (
        (rec.lon_deg >= _SCS_LON_MIN) & (rec.lon_deg <= _SCS_LON_MAX) &
        (rec.lat_deg >= _SCS_LAT_MIN) & (rec.lat_deg <= _SCS_LAT_MAX)
    )
    rec.pct_in_scs = 100.0 * in_scs.sum() / max(1, len(in_scs))
    rec.is_scs_storm = rec.pct_in_scs >= min_pct_in_scs

    in_vn = (
        (rec.lon_deg >= _VN_LON_MIN) & (rec.lon_deg <= _VN_LON_MAX) &
        (rec.lat_deg >= _VN_LAT_MIN) & (rec.lat_deg <= _VN_LAT_MAX)
    )
    rec.touches_vn_coast = bool(in_vn.any())


# Nguong goc (radian) de phan loai do kho, ap dung cho curvature_score =
# max(max_turn_angle, net_bearing_change) - xem compute_curvature(). Vi
# day la MAX chu khong phai trung binh, nguong duoc dat o muc "re dang
# ke" that su, khong bi pha loang boi cac doan di thang con lai:
#   ~1.05 rad (60 do)  -> mot cu re/doi huong ro rang tai 1 diem, HOAC
#                          huong di tong the nua sau lech >=60 do so voi
#                          nua dau (vd dang Tay Bac doi sang Tay Nam).
#   < nguong           -> "de" (di theo 1 huong on dinh, khong co re
#                          gap nao va khong doi huong tong the dang ke)
_CURVATURE_EASY_THRESHOLD = 1.05  # ~60 do


def assign_difficulty_tiers(recs: list[StormRecord],
                             easy_threshold: float = _CURVATURE_EASY_THRESHOLD) -> None:
    """Gan difficulty_tier NHI PHAN theo nguong tuyet doi cua curvature_score:
    "de" (it doi huong, curvature < easy_threshold) hoac "kho" (co doi
    huong/chuyen huong, curvature >= easy_threshold). Khac ban cu (tam
    phan vi tuong doi trong pool) - o day mot bao chi la "de" khi THAT
    SU di theo 1 huong on dinh, khong phu thuoc cac bao khac trong pool."""
    for r in recs:
        r.difficulty_tier = "de" if r.curvature_score < easy_threshold else "kho"


def compute_curvature(rec: StormRecord) -> None:
    """
    Do do KHO cua quy dao bang 2 chi so bo sung nhau (luu vao
    rec.curvature_score = MAX cua 2 chi so, dung lam gia tri chinh cho
    assign_difficulty_tiers):

      1. max_turn_angle: goc RE LON NHAT tai bat ky 1 diem nao tren quy
         dao (khong phai trung binh ca quy dao). Bat dung kieu bao "di
         thang 1 mach roi ngoat GAP" - vi trung binh tren toan bo diem
         se pha loang 1 cu re du lon o giua, khien no bi tinh nham la
         "de" neu chi dung trung binh.
      2. net_bearing_change: do lech GOC HUONG DI tong the giua nua dau
         va nua sau quy dao (vd dang di Tay Bac ma sau do doi han sang
         Tay Nam - dung chinh vi du ban dua ra). Bat dung kieu doi
         huong "tu tu" (khong co 1 diem re gap nao) nhung tong the van
         chuyen huong lon.

    Bao duoc coi la "KHO" neu MOT TRONG HAI chi so tren vuot nguong
    (xem assign_difficulty_tiers), tuc la co re gap HOAC co chuyen
    huong tong the lon, khong can ca 2.
    """
    if rec.n_points < 3:
        rec.curvature_score = 0.0
        rec.max_turn_angle = 0.0
        rec.net_bearing_change = 0.0
        return
    lon, lat = rec.lon_deg, rec.lat_deg
    v = np.stack([np.diff(lon), np.diff(lat)], axis=1)
    norms = np.linalg.norm(v, axis=1)
    valid = norms > 1e-6
    if valid.sum() < 2:
        rec.curvature_score = 0.0
        rec.max_turn_angle = 0.0
        rec.net_bearing_change = 0.0
        return
    v_valid = v[valid] / norms[valid, None]
    cos_theta = np.clip((v_valid[:-1] * v_valid[1:]).sum(axis=1), -1.0, 1.0)
    turn_angle = np.arccos(cos_theta)  # radian tai moi diem giua, 0 = di thang

    max_turn = float(turn_angle.max()) if len(turn_angle) > 0 else 0.0

    # Huong di trung binh cua nua DAU va nua SAU quy dao (vector tong,
    # khong phai trung binh tung buoc, de on dinh hon voi nhieu nho).
    mid = len(v) // 2
    if mid >= 1 and len(v) - mid >= 1:
        v_first = v[:mid].sum(axis=0)
        v_second = v[mid:].sum(axis=0)
        n1, n2 = np.linalg.norm(v_first), np.linalg.norm(v_second)
        if n1 > 1e-6 and n2 > 1e-6:
            cos_net = np.clip(np.dot(v_first, v_second) / (n1 * n2), -1.0, 1.0)
            net_change = float(np.arccos(cos_net))
        else:
            net_change = 0.0
    else:
        net_change = 0.0

    rec.max_turn_angle = max_turn
    rec.net_bearing_change = net_change
    rec.curvature_score = max(max_turn, net_change)


# ─────────────────────────────────────────────────────────────────────────────
# Chia train/val/test
# ─────────────────────────────────────────────────────────────────────────────


def split_storms(
    valid_recs: list[StormRecord],
    test_min: int,
    test_max: int,
    test_min_vn: int,
    test_min_easy: int,
    val_ratio: float,
    seed: int,
) -> tuple[list[StormRecord], list[StormRecord], list[StormRecord]]:
    """
    Logic chia test set (theo yeu cau moi nhat):
      0. Tap test CHI duoc chon tu cac bao co TEN CHU that (has_real_name
         =True, vd JOAN, RITA) - LOAI HAN bao chi co ma so (vd 0019,
         0019_2) khoi viec duoc chon vao test. Bao ma so van duoc dua
         vao train/val binh thuong, chi khong duoc vao test.
      1. Bat buoc lay >= test_min_vn bao co touches_vn_coast=True (di
         vao Viet Nam that su) - dieu kien CUNG. UU TIEN CHON BAO CO
         NAM GAN DAY NHAT truoc (vd YAGI 2024 duoc uu tien hon bao tu
         thap nien 1970-1990), vi bao gan day thuong phan anh dung dieu
         kien khi hau/quan trac hien tai hon.
      2. Bat buoc lay >= test_min_easy bao "DE" (difficulty_tier="de":
         it doi huong / di theo 1 huong) - dieu kien CUNG rieng, doc
         lap voi dieu kien VN o buoc 1 (1 bao co the vua la VN vua la
         de, khong tinh trung 2 lan).
      3. Phan con lai de du test_min..test_max bao duoc lay tu TOAN BO
         pool con lai (uu tien is_scs_storm=True truoc - "quanh Bien
         Dong cung duoc"), khong phan biet de/kho, mien la van la bao
         co ten chu.
      4. difficulty_tier: "de" = it doi huong (di theo 1 huong tuong
         doi thang), "kho" = co doi huong / chuyen huong (vd Tay Bac
         doi xuong Tay Nam) - xem assign_difficulty_tiers().
    """
    rng = random.Random(seed)
    recs = list(valid_recs)
    rng.shuffle(recs)

    # Luu y: difficulty_tier va has_real_name da duoc gan tu truoc khi
    # goi ham nay (xem assign_difficulty_tiers() va parse_data1d_file()
    # trong main()) - khong goi lai o day de tranh trung lap vo ich.

    # Buoc 0: tap test CHI duoc chon tu bao co ten chu that.
    named_recs = [r for r in recs if r.has_real_name]
    rng.shuffle(named_recs)

    if len(named_recs) < test_min:
        print(f"[CANH BAO] Chi co {len(named_recs)} bao co TEN CHU (khong phai ma "
              f"so) trong toan bo dataset hop le, it hon yeu cau toi thieu {test_min} "
              f"cho tap test. Test set se nho hon mong muon.")

    # Buoc 1: bat buoc >= test_min_vn bao "di vao Viet Nam" (touches_vn_coast),
    # CHI trong pho bao co ten chu. UU TIEN BAO NAM GAN DAY NHAT (moi
    # nhat truoc, vd YAGI 2024) thay vi chon ngau nhien hoan toan - vi
    # bao gan day thuong co du lieu ve tinh/quan trac chinh xac hon va
    # phan anh dung dieu kien khi hau hien tai hon bao cu. Trong cung 1
    # nam (neu co nhieu bao trung nam), xao ngau nhien de cong bang.
    vn_pool = [r for r in named_recs if r.touches_vn_coast]
    rng.shuffle(vn_pool)  # xao truoc de tie-break ngau nhien trong cung nam
    vn_pool.sort(key=lambda r: int(r.year), reverse=True)  # moi nhat len dau
    if len(vn_pool) < test_min_vn:
        print(f"[CANH BAO] Chi tim thay {len(vn_pool)} bao 'di vao Viet Nam' CO TEN "
              f"CHU, it hon yeu cau toi thieu {test_min_vn}. Se lay toi da co the.")
    test_vn = vn_pool[:test_min_vn]
    test_names = {(r.year, r.name) for r in test_vn}

    # Buoc 2: bat buoc >= test_min_easy bao "de" (it doi huong), CHI
    # trong pho bao co ten chu, TRU cac bao da lay o buoc 1 (tranh
    # trung, nhung 1 bao vua VN vua de van hop le duoc tinh 1 lan).
    easy_pool = [r for r in named_recs
                 if r.difficulty_tier == "de" and (r.year, r.name) not in test_names]
    rng.shuffle(easy_pool)
    n_easy_already = sum(1 for r in test_vn if r.difficulty_tier == "de")
    n_easy_more_needed = max(0, test_min_easy - n_easy_already)
    test_easy_extra = easy_pool[:n_easy_more_needed]
    if n_easy_already + len(test_easy_extra) < test_min_easy:
        n_have = n_easy_already + len(test_easy_extra)
        print(f"[CANH BAO] Chi co {n_have} bao 'de' (it doi huong) CO TEN CHU co the "
              f"dua vao test, it hon yeu cau toi thieu {test_min_easy}.")
    test_names.update((r.year, r.name) for r in test_easy_extra)

    # Buoc 3: phan con lai de du test_min..test_max, uu tien SCS truoc,
    # van CHI trong pho bao co ten chu.
    remaining_named = [r for r in named_recs if (r.year, r.name) not in test_names]
    scs_pool = [r for r in remaining_named if r.is_scs_storm]
    non_scs_pool = [r for r in remaining_named if not r.is_scs_storm]
    rng.shuffle(scs_pool)
    rng.shuffle(non_scs_pool)

    n_selected_so_far = len(test_vn) + len(test_easy_extra)
    target_test_total = max(test_min, min(test_max, n_selected_so_far + len(remaining_named)))
    n_more_needed = max(0, target_test_total - n_selected_so_far)
    n_more_needed = min(n_more_needed, test_max - n_selected_so_far, len(remaining_named))
    n_more_needed = max(0, n_more_needed)

    extra_test = (scs_pool + non_scs_pool)[:n_more_needed]

    test_set = test_vn + test_easy_extra + extra_test
    final_test_names = {(r.year, r.name) for r in test_set}

    remaining = [r for r in recs if (r.year, r.name) not in final_test_names]
    rng.shuffle(remaining)
    n_val = max(1, int(round(len(remaining) * val_ratio))) if remaining else 0
    val_set = remaining[:n_val]
    train_set = remaining[n_val:]

    return train_set, val_set, test_set


# ─────────────────────────────────────────────────────────────────────────────
# Ghi kết quả
# ─────────────────────────────────────────────────────────────────────────────

def write_reports(
    out_dir: str,
    all_recs: list[StormRecord],
    outliers: list[StormRecord],
    train_set: list[StormRecord],
    val_set: list[StormRecord],
    test_set: list[StormRecord],
    orphan_data3d: list | None = None,
    orphan_env: list | None = None,
    duplicate_keys: dict | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "duplicate_keys_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "name", "n_files_trung", "duong_dan_file"])
        for (yr, nm), paths in (duplicate_keys or {}).items():
            for p in paths:
                w.writerow([yr, nm, len(paths), p])

    with open(os.path.join(out_dir, "dataset_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "name", "has_real_name", "n_points", "n_data3d_found", "n_env_found",
                    "n_data3d_missing", "n_env_missing", "is_outlier",
                    "pct_in_scs", "touches_vn_coast", "max_turn_angle_deg", "net_bearing_change_deg",
                    "curvature_score", "difficulty_tier"])
        for r in all_recs:
            w.writerow([r.year, r.name, r.has_real_name, r.n_points, r.n_data3d_found, r.n_env_found,
                        len(r.data3d_missing_dates), len(r.env_missing_dates),
                        r.is_outlier, f"{r.pct_in_scs:.1f}", r.touches_vn_coast,
                        f"{np.degrees(r.max_turn_angle):.1f}", f"{np.degrees(r.net_bearing_change):.1f}",
                        f"{r.curvature_score:.4f}", r.difficulty_tier])

    with open(os.path.join(out_dir, "outliers_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "name", "n_points", "reasons"])
        for r in outliers:
            w.writerow([r.year, r.name, r.n_points, " | ".join(r.outlier_reasons)])

    with open(os.path.join(out_dir, "split_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "year", "name", "has_real_name", "n_points", "touches_vn_coast",
                    "is_scs_storm", "pct_in_scs", "curvature_score", "difficulty_tier"])
        for split_name, group in (("train", train_set), ("val", val_set), ("test", test_set)):
            for r in group:
                w.writerow([split_name, r.year, r.name, r.has_real_name, r.n_points, r.touches_vn_coast,
                            r.is_scs_storm, f"{r.pct_in_scs:.1f}",
                            f"{r.curvature_score:.4f}", r.difficulty_tier])

    with open(os.path.join(out_dir, "cross_check_orphans.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "year", "name", "n_files", "path"])
        for year, name, n_files, path in (orphan_data3d or []):
            w.writerow(["Data3d", year, name, n_files, path])
        for year, name, n_files, path in (orphan_env or []):
            w.writerow(["Env_data", year, name, n_files, path])

    trimmed_recs = [r for r in all_recs if r.was_trimmed]
    with open(os.path.join(out_dir, "trim_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "name", "n_points_goc", "trim_start_idx", "trim_end_idx",
                    "n_points_sau_khi_cat", "bi_loai_sau_khi_cat", "ly_do"])
        for r in trimmed_recs:
            w.writerow([r.year, r.name, r.n_points, r.trim_start_idx, r.trim_end_idx,
                        r.n_points_after_trim, r.dropped_after_trim, r.trim_reason])

    print(f"\nDa ghi report vao: {out_dir}")
    print(f"  - duplicate_keys_report.csv ({len(duplicate_keys or {})} cap (year,name) trung tu nhieu file .txt)")
    print(f"  - dataset_report.csv     ({len(all_recs)} bao)")
    print(f"  - outliers_report.csv    ({len(outliers)} bao bi loai)")
    print(f"  - split_report.csv       (train={len(train_set)}, val={len(val_set)}, test={len(test_set)})")
    n_orphan = len(orphan_data3d or []) + len(orphan_env or [])
    print(f"  - cross_check_orphans.csv ({n_orphan} thu muc Data3d/Env_data mo coi, "
          f"khong khop bao nao trong Data1d)")
    print(f"  - trim_report.csv        ({len(trimmed_recs)} bao bi cat do thieu Data3d/Env o giua)")


def _find_project_root_with_model(start: str) -> str | None:
    """Tự dò lên các thư mục cha để tìm thư mục chứa gói 'Model/' có
    file trajectoriesWithMe_unet_training.py bên trong Model/data/."""
    cur = os.path.abspath(start)
    for _ in range(8):
        candidate = os.path.join(cur, "Model", "data", "trajectoriesWithMe_unet_training.py")
        if os.path.isfile(candidate):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def run_runtime_smoke_test(
    root: str,
    project_root: str | None,
    obs_len: int,
    pred_len: int,
    split_names: list[str],
) -> None:
    """
    Kiểm tra "CÓ KHỚP VỚI CODE BẠN CUNG CẤP KHÔNG" bằng cách import và
    GỌI THẬT class TrajectoryDataset + seq_collate từ chính file
    Model/data/trajectoriesWithMe_unet_training.py của bạn (yêu cầu
    project_root là thư mục CHỨA thư mục Model/, không phải Model/ hay
    Model/data/). Đây là kiểm tra mạnh hơn hẳn so với chỉ đối chiếu tên
    file tĩnh: bắt được lỗi shape thật, lỗi .npy hỏng, lỗi thiếu key mà
    build_env_features_one_step cần, lỗi seq_collate khi stack tensor.

    Nếu import lỗi (thiếu torch, thiếu Model/env_net_transformer_gphsplit.py,
    sai cấu trúc package, ...), in rõ lý do và HƯỚNG DẪN cách sửa thay vì
    crash toàn bộ script chuẩn bị dữ liệu.
    """
    print("=== KIEM TRA KHOP VOI CODE THAT (TrajectoryDataset runtime smoke test) ===")

    if project_root is None:
        project_root = _find_project_root_with_model(root) or _find_project_root_with_model(os.getcwd())
    if project_root is None:
        print("  [BO QUA] Khong tim thay thu muc chua goi 'Model/' (can co file")
        print("  Model/data/trajectoriesWithMe_unet_training.py va Model/__init__.py,")
        print("  Model/data/__init__.py). Truyen --project_root <duong_dan> de chi ro,")
        print("  hoac dat --check_with_real_code o gan dung cay thu muc du an.")
        print()
        return

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from Model.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate  # type: ignore
    except Exception as e:
        print(f"  [LOI IMPORT] Khong import duoc TrajectoryDataset tu "
              f"{project_root}/Model/data/trajectoriesWithMe_unet_training.py")
        print(f"    Chi tiet: {type(e).__name__}: {e}")
        print("    Kiem tra: da cai torch/numpy chua? co file Model/__init__.py va")
        print("    Model/data/__init__.py (de Model la package hop le) khong? co file")
        print("    Model/env_net_transformer_gphsplit.py (module ma file dataset import) khong?")
        print()
        return

    print(f"  Import OK tu: {project_root}")

    any_split_ok = False
    for split_name in split_names:
        split_dir = os.path.join(root, "Data1d", split_name)
        if not os.path.isdir(split_dir) or not os.listdir(split_dir):
            print(f"  [{split_name}] thu muc rong hoac khong ton tai, bo qua: {split_dir}")
            continue
        try:
            ds = TrajectoryDataset(
                data_dir=split_dir,
                obs_len=obs_len, pred_len=pred_len, skip=1,
                threshold=0.002, min_ped=1, delim=" ", other_modal="gph",
                split=split_name, is_test=(split_name != "train"),
            )
        except Exception as e:
            print(f"  [{split_name}] [LOI KHOI TAO TrajectoryDataset]: {type(e).__name__}: {e}")
            continue

        n = len(ds)
        print(f"  [{split_name}] TrajectoryDataset khoi tao OK, so sequence: {n}")
        if n == 0:
            print(f"    [CANH BAO] {split_name} co 0 sequence sau khi TrajectoryDataset tu loc "
                  f"(vd toa do ngoai _LON_VALID_MIN/MAX cua chinh dataset, hoac qua ngan). "
                  f"Kiem tra log INFO/ERROR o tren.")
            continue

        # Thử load thật vài sample + seq_collate, bắt lỗi runtime cụ thể.
        n_try = min(3, n)
        try:
            samples = [ds[i] for i in range(n_try)]
            batch = seq_collate(samples)
        except Exception as e:
            print(f"    [LOI KHI LOAD SAMPLE / seq_collate]: {type(e).__name__}: {e}")
            continue

        obs_traj_out, pred_traj_out = batch[0], batch[1]
        img_obs_out, img_pred_out = batch[11], batch[12]
        env_out = batch[13]
        print(f"    Load thu {n_try} sample OK.")
        print(f"    obs_traj shape : {tuple(obs_traj_out.shape)}  (ky vong ~[obs_len, N, 2])")
        print(f"    pred_traj shape: {tuple(pred_traj_out.shape)} (ky vong ~[pred_len, N, 2])")
        print(f"    img_obs shape  : {tuple(img_obs_out.shape)}  "
              f"(ky vong [N, {DATA3D_CH}, obs_len, {DATA3D_H}, {DATA3D_W}])")
        if img_obs_out.shape[-2:] != (DATA3D_H, DATA3D_W) or img_obs_out.shape[1] != DATA3D_CH:
            print(f"    [CANH BAO] img_obs shape KHONG khop ky vong cua model "
                  f"(FNO3DEncoder / Unet3D can input channel={DATA3D_CH}, "
                  f"HxW=({DATA3D_H},{DATA3D_W})).")
        if env_out is None:
            print(f"    [CANH BAO] env_out la None - toan bo Env_data khong load duoc "
                  f"cho {n_try} sample dau tien cua split nay.")
        else:
            n_zero_u500 = 0
            if "u500_mean" in env_out:
                u500 = env_out["u500_mean"]
                n_zero_u500 = int((u500 == 0).float().mean().item() * 100) if u500.numel() else 0
            print(f"    env_out co {len(env_out)} keys. u500_mean zero-rate: ~{n_zero_u500}% "
                  f"(neu gan 100% nghia la Env_data khong duoc doc dung, xem lai FIX-DATA-28/29 "
                  f"trong file dataset goc).")
        any_split_ok = True

    if not any_split_ok:
        print("  [CANH BAO TONG] Khong co split nao load duoc thanh cong bang TrajectoryDataset "
              "that. Neu ban da chay --apply de tao Data1d/train,val,test, hay kiem tra lai "
              "duong dan --root va thu chay lai voi --check_with_real_code sau khi --apply.")
    print()


def materialize_split(root: str, split_name: str, recs: list[StormRecord], mode: str) -> None:
    out_dir = os.path.join(root, "Data1d", split_name)
    os.makedirs(out_dir, exist_ok=True)
    for r in recs:
        dst = os.path.join(out_dir, os.path.basename(r.path))
        if os.path.exists(dst):
            continue
        if mode == "symlink":
            os.symlink(os.path.abspath(r.path), dst)
        else:
            shutil.copy2(r.path, dst)


def apply_trim_to_split(root: str, split_name: str, recs: list[StormRecord]) -> int:
    """Voi moi bao trong `recs` co was_trimmed=True (va khong bi loai),
    ghi de ban copy trong Data1d/<split_name>/ (KHONG dung rec.path -
    xem ghi chu trong rewrite_trimmed_file). Tra ve so file da ghi de."""
    out_dir = os.path.join(root, "Data1d", split_name)
    n_done = 0
    for r in recs:
        if not r.was_trimmed or r.dropped_after_trim:
            continue
        target_path = os.path.join(out_dir, os.path.basename(r.path))
        if not os.path.exists(target_path):
            continue
        rewrite_trimmed_file(r, target_path)
        n_done += 1
    return n_done


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Thu muc goc chua Data1d/Data3d/Env_data")
    ap.add_argument("--obs_len", type=int, default=8)
    ap.add_argument("--pred_len", type=int, default=12)

    ap.add_argument("--max_pres_abs", type=float, default=10.0,
                     help="Nguong |pres_norm| toi da truoc khi coi la outlier")
    ap.add_argument("--max_wnd_abs", type=float, default=10.0,
                     help="Nguong |wnd_norm| toi da truoc khi coi la outlier")

    ap.add_argument("--min_pct_in_scs", type=float, default=15.0,
                     help="Phan tram diem quy dao toi thieu trong vung Bien Dong de gan nhan is_scs_storm")

    ap.add_argument("--test_min_storms", type=int, default=10,
                     help="So bao TOI THIEU cho tap test (mac dinh co dinh 10)")
    ap.add_argument("--test_max_storms", type=int, default=10,
                     help="So bao TOI DA cho tap test (mac dinh co dinh 10)")
    ap.add_argument("--test_min_vn", type=int, default=4,
                     help="So bao toi thieu 'di vao Viet Nam' bat buoc trong tap test "
                          "(chi tinh trong pho bao co ten chu)")
    ap.add_argument("--test_min_easy", type=int, default=5,
                     help="So bao toi thieu 'de' (it doi huong) bat buoc trong tap test "
                          "(chi tinh trong pho bao co ten chu)")
    ap.add_argument("--val_ratio", type=float, default=0.20,
                     help="Ty le bao (tren phan con lai SAU KHI tach test) dua vao val. "
                          "Mac dinh 0.20 -> tren phan con lai chia train/val theo ty le 80/20.")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--materialize_mode", choices=["copy", "symlink"], default="copy",
                     help="Copy file that hay tao symlink khi ghi vao Data1d/train,val,test")
    ap.add_argument("--check_data3d_shape", action="store_true",
                     help="Doc thu 1 file .npy Data3d moi bao de kiem tra shape (cham hon)")
    ap.add_argument("--check_env_content", action="store_true",
                     help="Doc thu 1 file .npy Env_data moi bao de kiem tra parse duoc dict (cham hon)")
    ap.add_argument("--trim_missing_middle", action="store_true",
                     help="Neu bao thieu Data3d/Env O GIUA quy dao (khong phai dau/cuoi): so sanh "
                          "doan lien tuc sach truoc vs sau diem thieu, GIU doan dai hon, CAT bo phan "
                          "con lai. Doan giu lai phai sach 100%% va >= obs_len+pred_len diem, neu "
                          "khong du se bi loai. Voi --apply, GHI DE truc tiep file .txt trong "
                          "Data1d/train,val,test (khong co ban sao du phong).")
    ap.add_argument("--check_with_real_code", action="store_true",
                     help="Sau khi --apply, IMPORT VA GOI THAT TrajectoryDataset/seq_collate tu "
                          "file trajectoriesWithMe_unet_training.py cua ban de kiem tra khop runtime "
                          "(can torch da cai va dung cau truc goi Model/)")
    ap.add_argument("--project_root", default=None,
                     help="Duong dan toi thu muc CHUA thu muc Model/ (vd: thu muc chua Model/data/"
                          "trajectoriesWithMe_unet_training.py). Neu bo qua, script tu do len cac "
                          "thu muc cha de tim.")
    ap.add_argument("--apply", action="store_true",
                     help="Neu KHONG bat co nay: chi chay dry-run, chi in bao cao, khong ghi file nao")
    ap.add_argument("--out_dir", default=None,
                     help="Thu muc ghi report CSV (mac dinh: <root>/_prepare_reports)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    data1d_dir = os.path.join(root, "Data1d")
    data3d_dir = os.path.join(root, "Data3d")
    env_dir = find_env_root(root)

    if not os.path.isdir(data1d_dir):
        print(f"LOI: khong tim thay {data1d_dir}", file=sys.stderr)
        sys.exit(1)

    txt_files = sorted(
        os.path.join(data1d_dir, f) for f in os.listdir(data1d_dir)
        if f.endswith(".txt") and os.path.isfile(os.path.join(data1d_dir, f))
    )
    if not txt_files:
        print(f"LOI: khong co file .txt nao trong {data1d_dir} (co the da chia san "
              f"train/val/test roi? script nay yeu cau Data1d PHANG, chua chia).",
              file=sys.stderr)
        sys.exit(1)

    print(f"Tim thay {len(txt_files)} file bao trong {data1d_dir}")
    print(f"Data3d root : {data3d_dir}  (ton tai: {os.path.isdir(data3d_dir)})")
    print(f"Env_data root: {env_dir}  (ton tai: {os.path.isdir(env_dir)})")
    print()

    min_len = args.obs_len + args.pred_len

    all_recs: list[StormRecord] = []
    for path in txt_files:
        rec = parse_data1d_file(path)
        if rec is None:
            continue
        all_recs.append(rec)

    # [QUAN TRONG] Phat hien (year, name) TRUNG LAP giua nhieu FILE .txt
    # KHAC NHAU. Day khong phai loi script - la dau hieu du lieu that co
    # 2 con bao khac nhau nhung parse ra cung ten (vd file dat ten khong
    # dong nhat, hoac co "_2"/"(1)" trong ten file khien parts[1] bi cat
    # sai). Khi bi trung, doi chieu Data3d/Env_data "mo coi" o buoc 1b co
    # the BI SAI (bi bao nham la mo coi du thuc ra khop voi 1 trong 2 file).
    from collections import defaultdict
    key_to_paths = defaultdict(list)
    for rec in all_recs:
        key_to_paths[(rec.year, rec.name)].append(rec.path)
    duplicate_keys = {k: v for k, v in key_to_paths.items() if len(v) > 1}
    if duplicate_keys:
        print(f"[CANH BAO QUAN TRONG] Phat hien {len(duplicate_keys)} cap (year, name) "
              f"TRUNG LAP tu NHIEU FILE .txt KHAC NHAU trong Data1d:")
        for (yr, nm), paths in list(duplicate_keys.items())[:20]:
            print(f"    {yr}_{nm}: {len(paths)} file -> {[os.path.basename(p) for p in paths]}")
        if len(duplicate_keys) > 20:
            print(f"    ... va {len(duplicate_keys) - 20} cap khac, xem duplicate_keys_report.csv")
        print("  Nguyen nhan thuong gap: 2 con bao khac nhau trong cung 1 nam duoc dat")
        print("  ten trung nhau trong Data1d (vd ca 2 deu ten '<year>_JOAN.txt' o thu muc")
        print("  khac nhau, hoac file bi doi ten mat phan hau to '_2'). Data3d/Env_data")
        print("  thuong da tach dung (vd JOAN, JOAN_2) nen se bi bao 'mo coi' o muc 1b duoi")
        print("  day dù thuc ra co the khop voi file thu 2. HAY KIEM TRA TAY cac cap nay.")
        print()

    # 1) Kiem tra khop Data3d / Env_data
    n_no_data3d_at_all = 0
    n_no_env_at_all = 0
    n_partial_data3d = 0
    n_partial_env = 0
    for rec in all_recs:
        if rec.n_points == 0:
            continue
        check_data3d_env_match(rec, data3d_dir, env_dir)
        if rec.n_data3d_found == 0:
            n_no_data3d_at_all += 1
        elif rec.n_data3d_found < rec.n_points:
            n_partial_data3d += 1
        if rec.n_env_found == 0:
            n_no_env_at_all += 1
        elif rec.n_env_found < rec.n_points:
            n_partial_env += 1
        if args.check_data3d_shape:
            err = try_load_one_data3d(rec, data3d_dir)
            if err:
                rec.outlier_reasons.append(f"data3d_shape_loi: {err}")
        if args.check_env_content:
            err = try_load_one_env(rec, env_dir)
            if err:
                rec.outlier_reasons.append(f"env_content_loi: {err}")

    # 1b) Doi chieu chieu NGUOC: tim file/thu muc Data3d, Env_data mo coi
    # (khong khop bao nao trong Data1d) - vi du bao da bi xoa khoi Data1d,
    # go sai year/name, hoac du lieu thu nghiem con sot lai.
    orphan_data3d, orphan_env = find_orphan_files(all_recs, data3d_dir, env_dir)

    print("=== BAO CAO KHOP DU LIEU (Data1d vs Data3d vs Env_data) ===")
    print(f"  Bao KHONG co file Data3d nao khop      : {n_no_data3d_at_all}")
    print(f"  Bao co Data3d nhung THIEU MOT PHAN      : {n_partial_data3d}")
    print(f"  Bao KHONG co file Env_data nao khop      : {n_no_env_at_all}")
    print(f"  Bao co Env_data nhung THIEU MOT PHAN     : {n_partial_env}")
    print(f"  Thu muc Data3d MO COI (khong khop Data1d)  : {len(orphan_data3d)}")
    print(f"  Thu muc Env_data MO COI (khong khop Data1d): {len(orphan_env)}")
    if orphan_data3d or orphan_env:
        print("  -> Xem chi tiet trong cross_check_orphans.csv. Day co the la du lieu")
        print("     rac (bao da bi xoa/doi ten trong Data1d) hoac loi go year/name khi build.")
    if n_no_data3d_at_all or n_no_env_at_all:
        print("  -> Cac bao thieu HOAN TOAN Data3d/Env se van duoc train (code goc")
        print("     tu dong fallback ve zeros / None cho cac truong hop nay), nhung")
        print("     ban nen kiem tra lai xem co bi thieu file do quen tai khong.")
    print()

    # 1c) [CHI khi --trim_missing_middle] Xac dinh doan quy dao "sach" dai
    # nhat khi co diem thieu Data3d/Env o GIUA quy dao (khong phai dau/
    # cuoi): so sanh doan lien tuc sach TRUOC diem hong dau tien vs doan
    # lien tuc sach SAU diem hong cuoi cung, giu doan DAI HON. Neu doan
    # giu lai khong du min_len diem, bao se bi loai (outlier).
    n_will_trim = 0
    n_will_drop_after_trim = 0
    if args.trim_missing_middle:
        for rec in all_recs:
            if rec.n_points == 0:
                continue
            compute_trim_range(rec, min_len)
            if rec.was_trimmed:
                n_will_trim += 1
                if rec.dropped_after_trim:
                    n_will_drop_after_trim += 1
                    rec.is_outlier = True
                    rec.outlier_reasons.append(f"cat_giua_qua_ngan: {rec.trim_reason}")

        print("=== BAO CAO CAT QUY DAO DO THIEU DATA3D/ENV O GIUA ===")
        print(f"  So bao co diem thieu O GIUA quy dao (can cat)  : {n_will_trim}")
        print(f"  Trong do, sau khi cat khong du diem -> bi loai : {n_will_drop_after_trim}")
        if n_will_trim:
            print("  -> Xem chi tiet tung bao trong trim_report.csv. Neu dung --apply,")
            print("     cac file .txt tuong ung trong Data1d/train,val,test se bi GHI DE")
            print("     de chi con lai doan da chon (theo dung index trim_start_idx/")
            print("     trim_end_idx trong trim_report.csv).")
        print()

    # 2) Loc outlier
    for rec in all_recs:
        apply_outlier_filters(rec, min_len, args.max_pres_abs, args.max_wnd_abs)

    outliers = [r for r in all_recs if r.is_outlier]
    valid_recs = [r for r in all_recs if not r.is_outlier]

    print("=== BAO CAO LOC OUTLIER ===")
    print(f"  Tong so bao          : {len(all_recs)}")
    print(f"  Bi loai (outlier)    : {len(outliers)}")
    print(f"  Con lai (hop le)     : {len(valid_recs)}")
    if outliers:
        print("  Vi du 5 bao bi loai dau tien:")
        for r in outliers[:5]:
            print(f"    - {r.year}_{r.name}: {'; '.join(r.outlier_reasons)}")
    print()

    # 3) Gan nhan SCS / Viet Nam + do kho
    for rec in valid_recs:
        compute_scs_vn_labels(rec, args.min_pct_in_scs)
        compute_curvature(rec)
    assign_difficulty_tiers(valid_recs)

    n_scs = sum(1 for r in valid_recs if r.is_scs_storm)
    n_vn = sum(1 for r in valid_recs if r.touches_vn_coast)
    n_named = sum(1 for r in valid_recs if r.has_real_name)
    n_named_vn = sum(1 for r in valid_recs if r.has_real_name and r.touches_vn_coast)
    n_named_easy = sum(1 for r in valid_recs if r.has_real_name and r.difficulty_tier == "de")
    print("=== BAO CAO GAN NHAN VUNG ===")
    print(f"  Bao vao Bien Dong (is_scs_storm, >= {args.min_pct_in_scs}% diem)  : {n_scs}/{len(valid_recs)}")
    print(f"  Bao cham vung sat Viet Nam (touches_vn_coast)                     : {n_vn}/{len(valid_recs)}")
    print(f"  Bao co TEN CHU that (khong phai ma so)                            : {n_named}/{len(valid_recs)}")
    print(f"    trong do di vao Viet Nam                                        : {n_named_vn}")
    print(f"    trong do 'de' (it doi huong)                                    : {n_named_easy}")
    print()

    if args.test_min_vn > n_named_vn:
        print(f"[CANH BAO QUAN TRONG] Yeu cau toi thieu {args.test_min_vn} bao 'di vao Viet Nam' "
              f"CO TEN CHU cho tap test, nhung toan bo dataset (sau loc outlier) chi co "
              f"{n_named_vn} bao thoa dieu kien nay. Hay kiem tra lai bounding box "
              f"_VN_LON_MIN/MAX, _VN_LAT_MIN/MAX o dau file, hoac bo sung du lieu.")
    if args.test_min_easy > n_named_easy:
        print(f"[CANH BAO QUAN TRONG] Yeu cau toi thieu {args.test_min_easy} bao 'de' (it doi "
              f"huong) CO TEN CHU cho tap test, nhung toan bo dataset chi co {n_named_easy} "
              f"bao thoa dieu kien nay. Co the chinh --test_min_easy hoac nguong "
              f"_CURVATURE_EASY_THRESHOLD o dau file.")

    # 4) Chia train/val/test - tap test CHI chon tu bao co TEN CHU that,
    # bat buoc >= test_min_vn bao vao VN va >= test_min_easy bao "de" (it
    # doi huong), phan con lai uu tien bao "quanh Bien Dong" (is_scs_storm).
    # Xem chi tiet trong split_storms().
    train_set, val_set, test_set = split_storms(
        valid_recs, args.test_min_storms, args.test_max_storms, args.test_min_vn,
        args.test_min_easy, args.val_ratio, args.seed,
    )

    n_vn_in_test = sum(1 for r in test_set if r.touches_vn_coast)
    n_scs_in_test = sum(1 for r in test_set if r.is_scs_storm)
    n_easy_in_test = sum(1 for r in test_set if r.difficulty_tier == "de")
    n_hard_in_test = sum(1 for r in test_set if r.difficulty_tier == "kho")
    n_not_named_in_test = sum(1 for r in test_set if not r.has_real_name)
    print("=== BAO CAO CHIA TRAIN/VAL/TEST (theo tung bao) ===")
    print(f"  Train : {len(train_set)} bao")
    print(f"  Val   : {len(val_set)} bao")
    print(f"  Test  : {len(test_set)} bao")
    print(f"    - Ten: {[f'{r.year}_{r.name}' for r in test_set]}")
    print(f"    - Di vao Viet Nam (touches_vn_coast) : {n_vn_in_test}")
    print(f"    - Quanh Bien Dong (is_scs_storm)      : {n_scs_in_test}")
    print(f"    - De (it doi huong) / Kho (co doi huong): {n_easy_in_test} / {n_hard_in_test}")
    if n_not_named_in_test:
        print(f"    - [LOI LOGIC] {n_not_named_in_test} bao KHONG co ten chu lot vao test "
              f"- bao loi cho nguoi phat trien, khong nen xay ra.")
    if n_vn_in_test < args.test_min_vn:
        print(f"  [CANH BAO] Test set chi co {n_vn_in_test}/{args.test_min_vn} bao di vao VN yeu cau.")
    if n_easy_in_test < args.test_min_easy:
        print(f"  [CANH BAO] Test set chi co {n_easy_in_test}/{args.test_min_easy} bao 'de' yeu cau.")
    if not (args.test_min_storms <= len(test_set) <= args.test_max_storms):
        print(f"  [CANH BAO] So bao test ({len(test_set)}) nam ngoai khoang mong muon "
              f"[{args.test_min_storms}, {args.test_max_storms}] - co the dataset khong du bao hop le.")
    print()

    out_dir = args.out_dir or os.path.join(root, "_prepare_reports")
    write_reports(out_dir, all_recs, outliers, train_set, val_set, test_set,
                  orphan_data3d=orphan_data3d, orphan_env=orphan_env,
                  duplicate_keys=duplicate_keys)

    timestep_report_path = os.path.join(out_dir, "timestep_match_detail.csv")
    write_timestep_detail_report(timestep_report_path, all_recs)
    n_timestep_rows = sum(r.n_points for r in all_recs if r.n_points > 0)
    n_missing_rows = sum(
        1 for r in all_recs if r.n_points > 0
        for ts in r.dates if ts in set(r.data3d_missing_dates) | set(r.env_missing_dates)
    )
    print(f"  - timestep_match_detail.csv ({n_timestep_rows} dong, moi dong = 1 timestep cua "
          f"1 bao; {n_missing_rows} timestep bi thieu Data3d va/hoac Env_data)")

    if args.apply:
        print(f"\n--apply duoc bat: dang ghi file vao {data1d_dir}/{{train,val,test}} "
              f"(mode={args.materialize_mode}) ...")
        materialize_split(root, "train", train_set, args.materialize_mode)
        materialize_split(root, "val", val_set, args.materialize_mode)
        materialize_split(root, "test", test_set, args.materialize_mode)
        print("Da tao xong Data1d/train, Data1d/val, Data1d/test.")
        print("Ban co the goi TrajectoryDataset(data_dir=root_hoac_train_path, ...) nhu binh thuong.")

        if args.trim_missing_middle:
            n_trimmed_files = 0
            n_trimmed_files += apply_trim_to_split(root, "train", train_set)
            n_trimmed_files += apply_trim_to_split(root, "val", val_set)
            n_trimmed_files += apply_trim_to_split(root, "test", test_set)
            print(f"Da GHI DE {n_trimmed_files} file .txt trong Data1d/train,val,test "
                  f"(chi giu doan quy dao sach dai hon, xem trim_report.csv de biet chi tiet).")

        if args.check_with_real_code:
            print()
            run_runtime_smoke_test(
                root=root,
                project_root=args.project_root,
                obs_len=args.obs_len,
                pred_len=args.pred_len,
                split_names=["train", "val", "test"],
            )
    else:
        print("\n[DRY-RUN] Chua ghi file nao. Xem lai split_report.csv / outliers_report.csv,")
        print("neu on thi chay lai voi co --apply de tao thuc su Data1d/train,val,test.")
        if args.check_with_real_code:
            print("[LUU Y] --check_with_real_code chi chay SAU khi Data1d/train,val,test da")
            print("duoc tao (can --apply). Hay chay lai voi ca --apply va --check_with_real_code.")


if __name__ == "__main__":
    main()