# """
# Model/data/trajectoriesWithMe_unet_training.py  ── v25 patch
# =============================================================
# FIXES vs v24:

#   FIX-DATA-28  [P0-CRITICAL] _build_csv_env_lookup: normalize u/v500 đúng.

#                build_env_data_scs_v10.py dùng U500_NORM=30.0:
#                  "u500_mean_n": clip(u_mean / 30.0, -1.0, 1.0)

#                → .npy lưu u500_mean trong [-1.0, 1.0].
#                → CSV có d3d_u500_mean_raw (raw m²/s² ~5843).
#                → Cần: clip(d3d_u500_mean_raw / 30.0, -1.0, 1.0)
#                  để consistent với .npy format.
#                → Lưu vào key "u500_mean" (không phải "u500_raw_mean")
#                  để build_env_features_one_step đọc đúng.

#   FIX-DATA-29  [P1] _load_env_npy: thêm flag gph500_already_normed=True
#                khi đọc từ .npy mới (không có "_n" suffix).
#                .npy mới (build_env_data_scs_v10) lưu:
#                  "gph500_mean": (gph - 5900) / 200  → ~[-0.3, 0.3]
#                Đây là đã normalized. Cần set flag để không z-score lại.

#   FIX-DATA-30  [P0] check_env_features_in_batch(): thêm diagnostic
#                function để verify u500/v500 != 0 trong batch đầu tiên.
#                Gọi trong _check_gph500() của train script.

# Kept from v24:
#   FIX-DATA-24..27 (synthetic data, coord filter, clip norm)
# """
# from __future__ import annotations

# import logging
# import math
# import os

# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset

# try:
#     import cv2
#     HAS_CV2 = True
# except ImportError:
#     HAS_CV2 = False

# try:
#     import netCDF4 as nc
#     HAS_NC = True
# except ImportError:
#     HAS_NC = False

# from Model.env_net_transformer_gphsplit import (
#     bearing_to_scs_center_onehot, dist_to_scs_boundary_onehot,
#     delta_velocity_onehot, intensity_class_onehot,
#     build_env_features_one_step, feat_to_tensor, ENV_FEATURE_DIMS,
# )

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# DATA3D_H  = 81
# DATA3D_W  = 81
# DATA3D_CH = 13

# DATA3D_MEAN = np.array([
#     12444.037,  # CH0: GPH_200  — from compute_data3d_stats_v2
#      5844.935,  # CH1: GPH_500
#      1482.430,  # CH2: GPH_850
#       752.466,  # CH3: GPH_925
#        -1.052,  # CH4: U_200
#        -0.188,  # CH5: U_500
#        -0.407,  # CH6: U_850
#        -0.888,  # CH7: U_925
#        -0.055,  # CH8: V_200
#         1.655,  # CH9: V_500
#         1.343,  # CH10: V_850
#         1.018,  # CH11: V_925
#       301.133,  # CH12: SST
# ], dtype=np.float32)
 
# DATA3D_STD = np.array([
#     113.583,  # CH0: GPH_200
#      57.135,  # CH1: GPH_500
#      37.852,  # CH2: GPH_850
#      37.274,  # CH3: GPH_925
#      13.315,  # CH4: U_200
#       8.042,  # CH5: U_500
#       7.911,  # CH6: U_850
#       7.494,  # CH7: U_925
#       8.377,  # CH8: V_200
#       5.999,  # CH9: V_500
#       6.203,  # CH10: V_850
#       6.555,  # CH11: V_925
#       3.023,  # CH12: SST
# ], dtype=np.float32)

# _DATA3D_SENTINEL_LARGE         = 20000.0
# _DATA3D_SENTINEL_ZERO_CHANNELS = {0}
# _DATA3D_GPH_VALID_MIN          = 25.0
# _DATA3D_GPH_VALID_MAX          = 95.0
# _DATA3D_SST_CHANNEL            = 12
# _SST_VALID_MIN                 = 270.0
# _SST_FILL_K                    = 298.0
# _MOVE_VEL_NORM                 = 150.0

# # FIX-DATA-25: coordinate filter bounds
# _LON_VALID_MIN  = 100.0
# _LON_VALID_MAX  = 185.0
# _LAT_VALID_MAX  = 50.0
# _LON_NORM_MIN   = -9.0
# _LON_NORM_MAX   =  2.0
# _LAT_NORM_MIN   =  0.0
# _LAT_NORM_MAX   = 10.0

# # FIX-DATA-28: U/V500 normalization constant (same as build_env_data_scs_v10.py)
# _UV500_NORM = 30.0

# _NPY_KEY_REMAP = {
#     "gph500_mean_n"   : "gph500_mean",
#     "gph500_center_n" : "gph500_center",
# }
# _NPY_U500_KEY_REMAP = {
#     "u500_mean_n"   : "u500_mean",
#     "u500_center_n" : "u500_center",
#     "v500_mean_n"   : "v500_mean",
#     "v500_center_n" : "v500_center",
# }


# # ── FIX-DATA-30: Diagnostic function ─────────────────────────────────────────

# def check_env_features_in_batch(bl, tag: str = "") -> None:
#     """
#     Verify env features (u500, v500, gph500) != 0 trong batch.
#     Gọi từ train script tại epoch 0, batch 0.
#     """
#     env_data = bl[13] if len(bl) > 13 else None
#     if env_data is None:
#         print(f"  ⚠️  [{tag}] env_data is None!")
#         return

#     sep = "!" * 60
#     ok  = "✅"
#     warn = "⚠️ "

#     for key in ["u500_mean", "v500_mean", "gph500_mean",
#                 "u500_center", "v500_center"]:
#         if key not in env_data:
#             print(f"  {warn} [{tag}] key '{key}' MISSING from env_data!")
#             continue
#         v = env_data[key]
#         if torch.is_tensor(v):
#             v_np = v.detach().cpu().float().numpy().flatten()
#         else:
#             v_np = np.array(v, dtype=float).flatten()

#         mn   = float(v_np.mean())
#         std  = float(v_np.std())
#         zr   = float((v_np == 0.0).mean() * 100)
#         vmin = float(v_np.min())
#         vmax = float(v_np.max())

#         if zr > 90.0:
#             print(f"\n{sep}")
#             print(f"  {warn} {key} = 0 cho {zr:.1f}% samples!")
#             print(f"     mean={mn:.4f}, std={std:.4f}")
#             print(f"     FIX-DATA-28 chưa được apply hoặc CSV không có d3d_u500_mean_raw")
#             print(f"{sep}\n")
#         else:
#             print(f"  {ok}  [{tag}] {key:15}: mean={mn:+.4f}  std={std:.4f}  "
#                   f"zero={zr:.1f}%  range=[{vmin:.3f}, {vmax:.3f}]")


# # ── CSV fallback builder ──────────────────────────────────────────────────────

# def _build_csv_env_lookup(csv_path: str) -> dict:
#     """
#     FIX-DATA-28: normalize d3d_u500_mean_raw → clip(raw/30.0, -1, 1).
#                  Lưu vào key "u500_mean" để consistent với .npy format.
#                  Không dùng env_u500_mean (boolean flag).
#     """
#     try:
#         import pandas as pd
#     except ImportError:
#         logger.warning("pandas không có → CSV fallback không khả dụng")
#         return {}

#     if not os.path.exists(csv_path):
#         logger.warning(f"CSV fallback không tìm thấy: {csv_path}")
#         return {}

#     logger.info(f"Loading CSV env fallback: {csv_path}")
#     try:
#         df = pd.read_csv(csv_path, dtype={"storm_name": str, "dt": str})
#     except Exception as e:
#         logger.warning(f"CSV load failed: {e}")
#         return {}

#     lookup: dict = {}
#     for _, row in df.iterrows():
#         yr         = str(int(float(row["year"])))
#         name_raw   = str(row["storm_name"])
#         name_strip = name_raw.lstrip("0") or "0"
#         ts         = str(row["dt"])

#         dir12   = [float(row.get(f"env_dir12_{i}",  0.0)) for i in range(8)]
#         dir24   = [float(row.get(f"env_dir24_{i}",  0.0)) for i in range(8)]
#         inten24 = [float(row.get(f"env_inten24_{i}", 0.0)) for i in range(4)]

#         gph_mean   = float(row.get("env_gph500_mean",   -29.5))
#         gph_center = float(row.get("env_gph500_center", -29.5))

#         mv_norm = float(row.get("env_move_velocity", 0.0))
#         mv_raw  = mv_norm * _MOVE_VEL_NORM

#         # FIX-DATA-28: normalize u/v500 raw → [-1,1] consistent với .npy
#         def _norm_uv(col_name):
#             raw = float(row.get(col_name, 0.0))
#             if raw <= 0.0 or raw > 50000.0:
#                 return 0.0
#             return float(np.clip(raw / _UV500_NORM, -1.0, 1.0))

#         u500_mean   = _norm_uv("d3d_u500_mean_raw")
#         u500_center = _norm_uv("d3d_u500_center_raw") if "d3d_u500_center_raw" in df.columns else u500_mean
#         v500_mean   = _norm_uv("d3d_v500_mean_raw")
#         v500_center = _norm_uv("d3d_v500_center_raw") if "d3d_v500_center_raw" in df.columns else v500_mean

#         d = {
#             "gph500_mean"           : gph_mean,
#             "gph500_center"         : gph_center,
#             "gph500_already_normed" : True,  # CSV: raw dam → cần z-score
#             # FIX-DATA-28: normalized [-1,1], key consistent với .npy
#             "u500_mean"             : u500_mean,
#             "u500_center"           : u500_center,
#             "v500_mean"             : v500_mean,
#             "v500_center"           : v500_center,
#             "has_data3d"            : (u500_mean != 0.0),
#             "move_velocity"         : mv_raw,
#             "history_direction12"   : dir12,
#             "history_direction24"   : dir24,
#             "history_inte_change24" : inten24,
#         }

#         for name_try in (name_raw, name_strip, name_raw.zfill(4), name_raw.zfill(2)):
#             lookup[(yr, name_try, ts)] = d

#     n_storms = df['storm_name'].nunique()
#     logger.info(f"CSV env lookup built: {len(lookup)} entries ({n_storms} storms)")

#     # Quick sanity check u500
#     sample_u = [v["u500_mean"] for v in list(lookup.values())[:100]]
#     nonzero = sum(1 for x in sample_u if x != 0.0)
#     logger.info(f"CSV u500_mean: {nonzero}/100 non-zero in first 100 entries "
#                 f"(mean={np.mean(sample_u):.4f})")

#     return lookup


# def _auto_discover_csv(root_path: str) -> str | None:
#     candidates = [
#         os.path.join(root_path, "all_storms_final.csv"),
#         os.path.join(os.path.dirname(root_path), "all_storms_final.csv"),
#         os.path.join(os.path.dirname(os.path.dirname(root_path)), "all_storms_final.csv"),
#         "/kaggle/input/datasets/gmnguynhng/data-tc-finall/all_storms_final.csv",
#         "/kaggle/working/all_storms_final.csv",
#     ]
#     for p in candidates:
#         if os.path.exists(p):
#             logger.info(f"Auto-discovered CSV: {p}")
#             return p
#     return None


# def _auto_discover_synthetic_csv(root_path: str) -> str | None:
#     candidates = [
#         os.path.join(root_path, "all_storms_synthetic.csv"),
#         os.path.join(os.path.dirname(root_path), "all_storms_synthetic.csv"),
#         os.path.join(os.path.dirname(os.path.dirname(root_path)), "all_storms_synthetic.csv"),
#         "/kaggle/working/all_storms_synthetic.csv",
#         "/kaggle/input/datasets/gmnguynhng/data-tc-finall/all_storms_synthetic.csv",
#     ]
#     for p in candidates:
#         if os.path.exists(p):
#             logger.info(f"Auto-discovered synthetic CSV: {p}")
#             return p
#     return None


# def env_data_processing(env_dict: dict) -> dict:
#     if not isinstance(env_dict, dict):
#         return {}
#     already_normed = bool(env_dict.get("gph500_already_normed", False))
#     cleaned = {"gph500_already_normed": already_normed}
#     for k, v in env_dict.items():
#         if k in ("gph500_already_normed", "has_data3d",
#                  "gph500_mean_already_normed", "gph500_center_already_normed"):
#             continue
#         if isinstance(v, (list, np.ndarray)):
#             cleaned[k] = v
#         elif isinstance(v, bool):
#             cleaned[k] = v
#         elif v == -1:
#             cleaned[k] = 0.0
#         else:
#             cleaned[k] = v
#     # Preserve has_data3d
#     cleaned["has_data3d"] = env_dict.get("has_data3d", True)
#     for sst_key in ("sst_mean", "sst_center", "sst"):
#         if sst_key in cleaned:
#             val = cleaned[sst_key]
#             if val is None or val == 0 or (isinstance(val, float) and val < _SST_VALID_MIN):
#                 cleaned[sst_key] = _SST_FILL_K
#     if not already_normed:
#         for gph_key in ("gph500_mean", "gph500_center"):
#             if gph_key in cleaned:
#                 val = cleaned[gph_key]
#                 if val is not None and isinstance(val, (int, float)):
#                     if val < _DATA3D_GPH_VALID_MIN or val > _DATA3D_GPH_VALID_MAX:
#                         cleaned[gph_key] = None
#     return cleaned


# # ── seq_collate ───────────────────────────────────────────────────────────────

# def seq_collate(data):
#     (obs_traj, pred_traj, obs_rel, pred_rel,
#      nlp, mask, obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
#      obs_date, pred_date, img_obs, img_pred, env_data_raw, tyID) = zip(*data)

#     def traj_TBC(lst):
#         cat = torch.cat(lst, dim=0)
#         return cat.permute(2, 0, 1)

#     obs_traj_out    = traj_TBC(obs_traj)
#     pred_traj_out   = traj_TBC(pred_traj)
#     obs_rel_out     = traj_TBC(obs_rel)
#     pred_rel_out    = traj_TBC(pred_rel)
#     obs_Me_out      = traj_TBC(obs_Me)
#     pred_Me_out     = traj_TBC(pred_Me)
#     obs_Me_rel_out  = traj_TBC(obs_Me_rel)
#     pred_Me_rel_out = traj_TBC(pred_Me_rel)

#     nlp_out = torch.tensor(
#         [v for sl in nlp for v in (sl if hasattr(sl, "__iter__") else [sl])],
#         dtype=torch.float,
#     )
#     mask_out = torch.cat(list(mask), dim=0).permute(1, 0)

#     counts        = torch.tensor([t.shape[0] for t in obs_traj])
#     cum           = torch.cumsum(counts, dim=0)
#     starts        = torch.cat([torch.tensor([0]), cum[:-1]])
#     seq_start_end = torch.stack([starts, cum], dim=1)

#     img_obs_out  = torch.stack(list(img_obs),  dim=0).permute(0, 4, 1, 2, 3).float()
#     img_pred_out = torch.stack(list(img_pred), dim=0).permute(0, 4, 1, 2, 3).float()

#     env_out    = None
#     valid_envs = [d for d in env_data_raw if isinstance(d, dict)]
#     if valid_envs:
#         env_out  = {}
#         all_keys = set()
#         for d in valid_envs:
#             all_keys.update(d.keys())

#         _skip_keys = {
#             "gph500_already_normed", "has_data3d",
#             "gph500_mean_already_normed", "gph500_center_already_normed",
#             "history_direction12_valid", "history_direction24_valid",
#             "history_inte_change24_valid",
#         }
#         all_keys -= _skip_keys

#         # sorted(): set iteration order is not stable across processes
#         # (hash randomization), which would give env_out a non-deterministic
#         # key order. Sorting fixes the order without changing any values.
#         for key in sorted(all_keys):
#             vals = []
#             for d in env_data_raw:
#                 if isinstance(d, dict) and key in d:
#                     v = d[key]
#                     v = torch.tensor(v, dtype=torch.float) if not torch.is_tensor(v) else v.float()
#                     vals.append(v)
#                 else:
#                     ref = next((d[key] for d in valid_envs if key in d), None)
#                     if ref is not None:
#                         rt = torch.tensor(ref, dtype=torch.float) if not torch.is_tensor(ref) else ref.float()
#                         vals.append(torch.zeros_like(rt))
#                     else:
#                         vals.append(torch.zeros(1))
#             try:
#                 env_out[key] = torch.stack(vals, dim=0)
#             except Exception:
#                 try:
#                     mx     = max(v.numel() for v in vals)
#                     padded = [F.pad(v.flatten(), (0, mx - v.numel())) for v in vals]
#                     env_out[key] = torch.stack(padded, dim=0)
#                 except Exception:
#                     pass

#     return (
#         obs_traj_out, pred_traj_out, obs_rel_out, pred_rel_out,
#         nlp_out, mask_out, seq_start_end,
#         obs_Me_out, pred_Me_out, obs_Me_rel_out, pred_Me_rel_out,
#         img_obs_out, img_pred_out, env_out, None, list(tyID),
#     )


# # ── TrajectoryDataset ─────────────────────────────────────────────────────────

# class TrajectoryDataset(Dataset):

#     def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1,
#                  threshold=0.002, min_ped=1, delim=" ", other_modal="gph",
#                  test_year=None, type="train", split=None, is_test=False,
#                  csv_env_path=None, synthetic_csv_path=None,
#                  **kwargs):
#         super().__init__()

#         dtype = split if split is not None else type

#         if isinstance(data_dir, dict):
#             root  = data_dir["root"]
#             dtype = data_dir.get("type", dtype)
#         else:
#             root = data_dir
#         if is_test and dtype not in ("val", "test"):
#             dtype = "test"

#         root = os.path.abspath(root)
#         bn   = os.path.basename(root)
#         if bn in ("train", "test", "val"):
#             self.root_path = os.path.dirname(os.path.dirname(root))
#         elif bn == "Data1d":
#             self.root_path = os.path.dirname(root)
#         else:
#             self.root_path = root

#         self.data1d_path = os.path.join(self.root_path, "Data1d", dtype)
#         self.data3d_path = os.path.join(self.root_path, "Data3d")
#         for env_name in ("Env_data", "ENV_DATA", "env_data", "Env_Data", "Env_Data"):
#             candidate = os.path.join(self.root_path, env_name)
#             if os.path.exists(candidate):
#                 self.env_path = candidate
#                 break
#         else:
#             self.env_path = os.path.join(self.root_path, "Env_data")

#         self._is_train = (dtype == "train")

#         self._csv_env_lookup: dict  = {}
#         self._env_path_missing      = not os.path.exists(self.env_path)

#         if self._env_path_missing or csv_env_path is not None:
#             if csv_env_path is None:
#                 csv_env_path = _auto_discover_csv(self.root_path)
#             if csv_env_path:
#                 self._csv_env_lookup = _build_csv_env_lookup(csv_env_path)
#         elif csv_env_path is not None:
#             # Luôn build CSV lookup kể cả có Env_data (để bổ sung u500 raw)
#             self._csv_env_lookup = _build_csv_env_lookup(csv_env_path)

#         # Nếu có Env_data nhưng không có csv_env_path, vẫn thử auto-discover CSV
#         # để dùng d3d_u500_mean_raw (Env_data .npy đã có u500_mean normalized)
#         if not self._csv_env_lookup:
#             auto_csv = _auto_discover_csv(self.root_path)
#             if auto_csv:
#                 self._csv_env_lookup = _build_csv_env_lookup(auto_csv)

#         self.obs_len    = obs_len
#         self.pred_len   = pred_len
#         self.seq_len    = obs_len + pred_len
#         self.skip       = skip
#         self.modal_name = other_modal

#         self._synthetic_lookup: dict = {}
#         if self._is_train:
#             if synthetic_csv_path is None:
#                 synthetic_csv_path = _auto_discover_synthetic_csv(self.root_path)
#             if synthetic_csv_path:
#                 self._synthetic_lookup = self._build_synthetic_lookup(synthetic_csv_path)
#                 logger.info(f"Loaded synthetic: {len(self._synthetic_lookup)} storms")

#         if not os.path.exists(self.data1d_path):
#             logger.error(f"Missing Data1d: {self.data1d_path}")
#             self.num_seq       = 0
#             self.seq_start_end = []
#             self.tyID          = []
#             return

#         all_files = [
#             os.path.join(self.data1d_path, f)
#             for f in sorted(os.listdir(self.data1d_path))
#             if f.endswith(".txt") and (test_year is None or str(test_year) in f)
#         ]

#         self.obs_traj_raw    = []
#         self.pred_traj_raw   = []
#         self.obs_Me_raw      = []
#         self.pred_Me_raw     = []
#         self.obs_rel_raw     = []
#         self.pred_rel_raw    = []
#         self.obs_Me_rel_raw  = []
#         self.pred_Me_rel_raw = []
#         self.non_linear_ped  = []
#         self.tyID            = []
#         num_peds_in_seq      = []
#         self.env_cache: dict = {}

#         n_filtered_coord = 0
#         for path in all_files:
#             base   = os.path.splitext(os.path.basename(path))[0]
#             parts  = base.split("_")
#             f_year = parts[0] if parts else "unknown"
#             f_name = parts[1] if len(parts) > 1 else base

#             d    = self._read_file(path, delim)
#             data = d["main"]
#             add  = d["addition"]
#             if len(data) < self.seq_len:
#                 continue

#             frames     = np.unique(data[:, 0]).tolist()
#             frame_data = [data[data[:, 0] == f] for f in frames]
#             n_frames   = len(frames)
#             if n_frames < self.seq_len:
#                 continue
#             n_seq = (n_frames - self.seq_len) // skip + 1

#             for idx in range(0, n_seq * skip, skip):
#                 if idx + self.seq_len > len(frame_data):
#                     break

#                 seg  = np.concatenate(frame_data[idx: idx + self.seq_len])
#                 peds = np.unique(seg[:, 1])

#                 buf_obs_traj   = []
#                 buf_pred_traj  = []
#                 buf_obs_rel    = []
#                 buf_pred_rel   = []
#                 buf_obs_Me     = []
#                 buf_pred_Me    = []
#                 buf_obs_Me_rel = []
#                 buf_pred_Me_rel= []
#                 buf_nlp        = []
#                 cnt            = 0

#                 for pid in peds:
#                     ps = seg[seg[:, 1] == pid]
#                     if len(ps) != self.seq_len:
#                         continue
#                     ps_t = np.transpose(ps[:, 2:])

#                     # FIX-DATA-25: Filter outlier coordinates
#                     lon_norm_vals = ps_t[0, :]
#                     lat_norm_vals = ps_t[1, :]
#                     lon_deg = (lon_norm_vals * 50.0 + 1800.0) / 10.0
#                     lat_deg = (lat_norm_vals * 50.0) / 10.0
#                     if lon_deg.min() < _LON_VALID_MIN or lon_deg.max() > _LON_VALID_MAX:
#                         n_filtered_coord += 1
#                         continue
#                     if lat_deg.max() > _LAT_VALID_MAX:
#                         n_filtered_coord += 1
#                         continue

#                     # FIX-DATA-27: Clip lon_norm / lat_norm
#                     ps_t_clipped = ps_t.copy()
#                     ps_t_clipped[0, :] = np.clip(ps_t[0, :], _LON_NORM_MIN, _LON_NORM_MAX)
#                     ps_t_clipped[1, :] = np.clip(ps_t[1, :], _LAT_NORM_MIN, _LAT_NORM_MAX)

#                     rel  = np.zeros_like(ps_t_clipped)
#                     rel[:, 1:] = ps_t_clipped[:, 1:] - ps_t_clipped[:, :-1]

#                     buf_obs_traj.append(torch.from_numpy(ps_t_clipped[:2, :obs_len]).float())
#                     buf_pred_traj.append(torch.from_numpy(ps_t_clipped[:2, obs_len:]).float())
#                     buf_obs_rel.append(torch.from_numpy(rel[:2, :obs_len]).float())
#                     buf_pred_rel.append(torch.from_numpy(rel[:2, obs_len:]).float())
#                     buf_obs_Me.append(torch.from_numpy(ps_t_clipped[2:, :obs_len]).float())
#                     buf_pred_Me.append(torch.from_numpy(ps_t_clipped[2:, obs_len:]).float())
#                     buf_obs_Me_rel.append(torch.from_numpy(rel[2:, :obs_len]).float())
#                     buf_pred_Me_rel.append(torch.from_numpy(rel[2:, obs_len:]).float())
#                     buf_nlp.append(self._poly_fit(ps_t_clipped, pred_len, threshold))
#                     cnt += 1

#                 if cnt >= min_ped:
#                     self.obs_traj_raw.extend(buf_obs_traj)
#                     self.pred_traj_raw.extend(buf_pred_traj)
#                     self.obs_rel_raw.extend(buf_obs_rel)
#                     self.pred_rel_raw.extend(buf_pred_rel)
#                     self.obs_Me_raw.extend(buf_obs_Me)
#                     self.pred_Me_raw.extend(buf_pred_Me)
#                     self.obs_Me_rel_raw.extend(buf_obs_Me_rel)
#                     self.pred_Me_rel_raw.extend(buf_pred_Me_rel)
#                     self.non_linear_ped.extend(buf_nlp)
#                     num_peds_in_seq.append(cnt)
#                     self.tyID.append({
#                         "old":    [f_year, f_name, idx],
#                         "tydate": [add[i][0] for i in range(idx, idx + self.seq_len)],
#                     })

#         real_seqs = len(self.tyID)
#         logger.info(f"Loaded {real_seqs} real sequences "
#                     f"(filtered {n_filtered_coord} outlier coords)")

#         if self._is_train and self._synthetic_lookup:
#             self._load_synthetic_sequences(
#                 min_ped, threshold, obs_len, pred_len, delim)
#             logger.info(f"Total sequences after synthetic: {len(self.tyID)}")

#         self.num_seq = len(self.tyID)
#         cum = np.cumsum(num_peds_in_seq).tolist()
#         self.seq_start_end = list(zip([0] + cum[:-1], cum))

#     def _build_synthetic_lookup(self, csv_path: str) -> dict:
#         try:
#             import pandas as pd
#         except ImportError:
#             return {}
#         if not os.path.exists(csv_path):
#             return {}
#         try:
#             df = pd.read_csv(csv_path, dtype={"storm_name": str, "dt": str})
#             lookup = {}
#             for (yr, nm), grp in df.groupby(["year", "storm_name"]):
#                 lookup[(str(int(float(yr))), str(nm))] = grp.sort_values("step_i").reset_index(drop=True)
#             return lookup
#         except Exception as e:
#             logger.warning(f"Synthetic CSV load failed: {e}")
#             return {}

#     def _load_synthetic_sequences(self, min_ped, threshold, obs_len, pred_len, delim):
#         num_peds_in_seq_local = []
#         for (f_year, f_name), storm_df in self._synthetic_lookup.items():
#             n_steps = len(storm_df)
#             if n_steps < self.seq_len:
#                 continue
#             n_seq = (n_steps - self.seq_len) // self.skip + 1
#             dates_list = storm_df["dt"].tolist()
#             for idx in range(0, n_seq * self.skip, self.skip):
#                 if idx + self.seq_len > n_steps:
#                     break
#                 seg = storm_df.iloc[idx: idx + self.seq_len]
#                 lon_norm = seg["lon_norm"].values
#                 lat_norm = seg["lat_norm"].values
#                 pres_norm = seg["pres_norm"].values
#                 wnd_norm = seg["wnd_norm"].values
#                 lon_deg = (lon_norm * 50.0 + 1800.0) / 10.0
#                 lat_deg = (lat_norm * 50.0) / 10.0
#                 if lon_deg.min() < _LON_VALID_MIN or lon_deg.max() > _LON_VALID_MAX:
#                     continue
#                 if lat_deg.max() > _LAT_VALID_MAX:
#                     continue
#                 lon_norm = np.clip(lon_norm, _LON_NORM_MIN, _LON_NORM_MAX)
#                 lat_norm = np.clip(lat_norm, _LAT_NORM_MIN, _LAT_NORM_MAX)
#                 ps_t = np.stack([lon_norm, lat_norm, pres_norm, wnd_norm], axis=0)
#                 rel  = np.zeros_like(ps_t)
#                 rel[:, 1:] = ps_t[:, 1:] - ps_t[:, :-1]
#                 self.obs_traj_raw.append(torch.from_numpy(ps_t[:2, :obs_len]).float())
#                 self.pred_traj_raw.append(torch.from_numpy(ps_t[:2, obs_len:]).float())
#                 self.obs_rel_raw.append(torch.from_numpy(rel[:2, :obs_len]).float())
#                 self.pred_rel_raw.append(torch.from_numpy(rel[:2, obs_len:]).float())
#                 self.obs_Me_raw.append(torch.from_numpy(ps_t[2:, :obs_len]).float())
#                 self.pred_Me_raw.append(torch.from_numpy(ps_t[2:, obs_len:]).float())
#                 self.obs_Me_rel_raw.append(torch.from_numpy(rel[2:, :obs_len]).float())
#                 self.pred_Me_rel_raw.append(torch.from_numpy(rel[2:, obs_len:]).float())
#                 self.non_linear_ped.append(self._poly_fit(ps_t, pred_len, threshold))
#                 num_peds_in_seq_local.append(1)
#                 tydate = [str(dates_list[i]) for i in range(idx, idx + self.seq_len)]
#                 self.tyID.append({
#                     "old":    [f_year, f_name, idx],
#                     "tydate": tydate,
#                     "is_synthetic": True,
#                 })
#         existing_end = self.seq_start_end[-1][1] if self.seq_start_end else 0
#         new_cum = np.cumsum(num_peds_in_seq_local) + existing_end
#         new_pairs = list(zip([existing_end] + new_cum[:-1].tolist(), new_cum.tolist()))
#         self.seq_start_end.extend(new_pairs)

#     def _read_file(self, path: str, delim: str) -> dict:
#         data, add = [], []
#         with open(path, encoding="utf-8", errors="ignore") as f:
#             raw_lines = f.readlines()
#         for line in raw_lines:
#             line = line.strip()
#             if not line or line.startswith(("#", "//", "-", "=")):
#                 continue
#             parts = line.split()
#             if len(parts) < 7:
#                 continue
#             try:
#                 int(parts[0])
#             except ValueError:
#                 continue
#             try:
#                 frame_id  = float(parts[0])
#                 lon_norm  = float(parts[1])
#                 lat_norm  = float(parts[2])
#                 pres_norm = float(parts[3])
#                 wnd_norm  = float(parts[4])
#                 date      = parts[5]
#                 name      = parts[6]
#                 add.append([date, name])
#                 data.append([frame_id, 1.0, lon_norm, lat_norm, pres_norm, wnd_norm])
#             except (ValueError, IndexError):
#                 continue
#         return {
#             "main":     np.asarray(data, dtype=np.float32) if data else np.zeros((0, 6), dtype=np.float32),
#             "addition": add,
#         }

#     def _poly_fit(self, traj, tlen, threshold):
#         t  = np.linspace(0, tlen - 1, tlen)
#         rx = np.polyfit(t, traj[0, -tlen:], 2, full=True)[1]
#         ry = np.polyfit(t, traj[1, -tlen:], 2, full=True)[1]
#         return 1.0 if (len(rx) > 0 and rx[0] + ry[0] >= threshold) else 0.0

#     def _normalize_data3d(self, arr: np.ndarray) -> np.ndarray:
#         arr = arr.copy()
#         for c in range(DATA3D_CH):
#             ch = arr[:, :, c]
#             ch[ch > _DATA3D_SENTINEL_LARGE] = np.nan
#             if c < 4:  # FIX-v26: GPH channels — filter negative sentinels
#                 ch[ch < 0] = np.nan
#             if c in _DATA3D_SENTINEL_ZERO_CHANNELS:
#                 ch[ch == 0.0] = np.nan
#             if c == _DATA3D_SST_CHANNEL:
#                 ch[ch < _SST_VALID_MIN] = _SST_FILL_K
#             if np.any(np.isnan(ch)):
#                 valid_vals = ch[~np.isnan(ch)]
#                 fill_val = (float(np.median(valid_vals)) if len(valid_vals) > 0
#                             else float(DATA3D_MEAN[c]))
#                 ch[np.isnan(ch)] = fill_val
#             arr[:, :, c] = (ch - DATA3D_MEAN[c]) / (DATA3D_STD[c] + 1e-6)
#         return np.clip(arr, -5.0, 5.0)

#     def _load_data3d_file(self, path: str):
#         try:
#             if path.endswith(".npy"):
#                 arr = np.load(path).astype(np.float32)
#             elif path.endswith(".nc") and HAS_NC:
#                 with nc.Dataset(path) as ds:
#                     keys = list(ds.variables.keys())
#                     arr  = np.array(ds.variables[keys[-1]][:]).astype(np.float32)
#             else:
#                 return None
#             if arr.ndim == 2:
#                 arr = arr[:, :, np.newaxis]
#             if arr.ndim == 3:
#                 if arr.shape[0] == DATA3D_CH:
#                     arr = arr.transpose(1, 2, 0)
#                 H, W, C = arr.shape
#                 if H != DATA3D_H or W != DATA3D_W:
#                     if HAS_CV2:
#                         arr = cv2.resize(arr, (DATA3D_W, DATA3D_H))
#                     else:
#                         arr = arr[:DATA3D_H, :DATA3D_W, :]
#                         if arr.shape[0] < DATA3D_H:
#                             arr = np.pad(arr, ((0, DATA3D_H - arr.shape[0]), (0, 0), (0, 0)))
#                         if arr.shape[1] < DATA3D_W:
#                             arr = np.pad(arr, ((0, 0), (0, DATA3D_W - arr.shape[1]), (0, 0)))
#                 if arr.shape[2] < DATA3D_CH:
#                     arr = np.concatenate([
#                         arr,
#                         np.zeros((DATA3D_H, DATA3D_W, DATA3D_CH - arr.shape[2]), dtype=np.float32),
#                     ], axis=2)
#                 arr = arr[:, :, :DATA3D_CH]
#                 return self._normalize_data3d(arr)
#         except Exception as e:
#             logger.debug(f"Data3d load error {path}: {e}")
#         return None

#     def img_read(self, year, ty_name, timestamp) -> torch.Tensor:
#         folder = os.path.join(self.data3d_path, str(year), str(ty_name))
#         if not os.path.exists(folder):
#             return torch.zeros(DATA3D_H, DATA3D_W, DATA3D_CH)
#         prefix = f"WP{year}{ty_name}_{timestamp}"
#         for ext in (".npy", ".nc"):
#             p = os.path.join(folder, prefix + ext)
#             if os.path.exists(p):
#                 arr = self._load_data3d_file(p)
#                 if arr is not None:
#                     return torch.from_numpy(arr).float()
#         try:
#             for fname in sorted(os.listdir(folder)):
#                 if timestamp in fname and fname.endswith((".npy", ".nc")):
#                     arr = self._load_data3d_file(os.path.join(folder, fname))
#                     if arr is not None:
#                         return torch.from_numpy(arr).float()
#         except Exception:
#             pass
#         return torch.zeros(DATA3D_H, DATA3D_W, DATA3D_CH)

#     def _load_env_npy(self, year, ty_name, timestamp):
#         """
#         FIX-DATA-29: .npy mới (build_env_data_scs_v10) lưu gph500_mean
#         đã normalized bằng (gph-5900)/200 → range ~[-0.3, 0.3].
#         Set gph500_already_normed=True nếu file KHÔNG có "_n" suffix keys.
#         """
#         folder = os.path.join(self.env_path, str(year), str(ty_name))
#         if os.path.exists(folder):
#             candidates = []
#             for fname in [f"WP{year}{ty_name}_{timestamp}.npy",
#                           f"{timestamp}.npy"]:
#                 p = os.path.join(folder, fname)
#                 if os.path.exists(p):
#                     candidates.append(p)
#             if not candidates:
#                 try:
#                     candidates = [
#                         os.path.join(folder, f)
#                         for f in sorted(os.listdir(folder))
#                         if timestamp in f and f.endswith(".npy")
#                     ]
#                 except Exception:
#                     pass

#             for p in candidates:
#                 try:
#                     raw = np.load(p, allow_pickle=True).item()
#                     if not isinstance(raw, dict):
#                         continue

#                     remapped = dict(raw)
#                     has_old_gph = False
#                     has_old_n_suffix = any(k.endswith("_n") for k in remapped)

#                     for old_k, new_k in _NPY_KEY_REMAP.items():
#                         if old_k in remapped and new_k not in remapped:
#                             remapped[new_k] = remapped.pop(old_k)
#                             has_old_gph = True

#                     for old_k, new_k in _NPY_U500_KEY_REMAP.items():
#                         if old_k in remapped and new_k not in remapped:
#                             remapped[new_k] = remapped.pop(old_k)

#                     if has_old_gph:
#                         # .npy cũ với "_n" suffix: đã normalized bởi (gph-5900)/200
#                         # remapped["gph500_already_normed"] = True
#                         remapped["gph500_already_normed"] = True
#                     elif "gph500_mean" in remapped and not has_old_n_suffix:
#                         # FIX-DATA-29: .npy mới (build_env_data_scs_v10):
#                         # gph500_mean = (gph-5900)/200, u500_mean = clip(u/30,-1,1)
#                         # Đây đều là đã normalized
#                         # remapped["gph500_already_normed"] = True\
#                         remapped["gph500_already_normed"] = True   # FIX-v57: GPH cần normalize

                    
#                     result = env_data_processing(remapped)
#                     # if not hasattr(self, '_debug_printed'):
#                     #     self._debug_printed = True
#                     #     print(f"\n  [DEBUG _load_env_npy] file={p}")
#                     #     print(f"  raw keys     : {list(raw.keys())}")
#                     #     print(f"  raw has_data3d : {raw.get('has_data3d')}")
#                     #     print(f"  raw u500_mean  : {raw.get('u500_mean')}")
#                     #     print(f"  remapped has_data3d : {remapped.get('has_data3d')}")
#                     #     print(f"  remapped u500_mean  : {remapped.get('u500_mean')}")
#                     #     print(f"  result has_data3d   : {result.get('has_data3d')}")
#                     #     print(f"  result u500_mean    : {result.get('u500_mean')}")
#                     #     print()

#                     return result
#                 except Exception as e:
#                     logger.debug(f"env npy load error {p}: {e}")

#         # CSV fallback
#         if self._csv_env_lookup:
#             yr_str = str(year)
#             ts_str = str(timestamp)
#             name_strip  = str(ty_name).lstrip("0") or "0"
#             name_padded = str(ty_name).zfill(4)
#             name_padded2 = str(ty_name).zfill(2)
#             for name_try in (str(ty_name), name_strip, name_padded, name_padded2):
#                 raw_dict = self._csv_env_lookup.get((yr_str, name_try, ts_str))
#                 if raw_dict is not None:
#                     return env_data_processing(dict(raw_dict))

#         return None

#     def _get_env_features(self, year, ty_name, dates, obs_traj, obs_Me):
#         T         = len(dates)
#         all_feats = []
#         prev_speed = None

#         for t in range(T):
#             env_npy = self._load_env_npy(year, ty_name, dates[t])
#             feat    = build_env_features_one_step(
#                 lon_norm  = float(obs_traj[0, t]),
#                 lat_norm  = float(obs_traj[1, t]),
#                 wind_norm = float(obs_Me[1, t]),
#                 pres_norm = float(obs_Me[0, t]),
#                 timestamp = dates[t],
#                 env_npy   = env_npy,
#                 prev_speed_kmh = prev_speed,
#             )
#             # # DEBUG một lần
#             # if not hasattr(self, '_debug_feat_printed'):
#             #     self._debug_feat_printed = True
#             #     print(f"\n  [DEBUG _get_env_features] t={t} date={dates[t]}")
#             #     print(f"  env_npy has_data3d : {env_npy.get('has_data3d') if env_npy else None}")
#             #     print(f"  env_npy u500_mean  : {env_npy.get('u500_mean') if env_npy else None}")
#             #     print(f"  feat u500_mean     : {feat.get('u500_mean')}")
#             #     print(f"  feat gph500_mean   : {feat.get('gph500_mean')}")

#             all_feats.append(feat)
#             if isinstance(env_npy, dict):
#                 mv = float(env_npy.get("move_velocity", 0.0) or 0.0)
#                 prev_speed = mv if mv != -1 else 0.0

#         env_out = {}
#         for key in ENV_FEATURE_DIMS:
#             dim  = ENV_FEATURE_DIMS[key]
#             rows = []
#             for feat in all_feats:
#                 v = feat.get(key, [0.0] * dim)
#                 t = torch.tensor(v, dtype=torch.float)
#                 if t.numel() < dim:
#                     t = F.pad(t, (0, dim - t.numel()))
#                 rows.append(t[:dim])
#             env_out[key] = torch.stack(rows, dim=0)
#         return env_out

#     def _embed_time(self, date_list):
#         rows = []
#         for d in date_list:
#             try:
#                 rows.append([
#                     (float(d[:4]) - 1949) / 70.0 - 0.5,
#                     (float(d[4:6]) - 1)   / 11.0 - 0.5,
#                     (float(d[6:8]) - 1)   / 30.0 - 0.5,
#                     float(d[8:10])         / 18.0 - 0.5,
#                 ])
#             except Exception:
#                 rows.append([0.0, 0.0, 0.0, 0.0])
#         return torch.tensor(rows, dtype=torch.float).t().unsqueeze(0)

#     def __len__(self):
#         return self.num_seq

#     def __getitem__(self, index):
#         if self.num_seq == 0:
#             raise IndexError("Empty dataset")
#         s, e   = self.seq_start_end[index]
#         info   = self.tyID[index]
#         year   = str(info["old"][0])
#         tyname = str(info["old"][1])
#         dates  = info["tydate"]

#         imgs    = [self.img_read(year, tyname, ts) for ts in dates[:self.obs_len]]
#         img_obs = torch.stack(imgs, dim=0)
#         img_pred = torch.zeros(self.pred_len, DATA3D_H, DATA3D_W, DATA3D_CH)

#         obs_traj    = torch.stack([self.obs_traj_raw[i]    for i in range(s, e)])
#         pred_traj   = torch.stack([self.pred_traj_raw[i]   for i in range(s, e)])
#         obs_rel     = torch.stack([self.obs_rel_raw[i]     for i in range(s, e)])
#         pred_rel    = torch.stack([self.pred_rel_raw[i]    for i in range(s, e)])
#         obs_Me      = torch.stack([self.obs_Me_raw[i]      for i in range(s, e)])
#         pred_Me     = torch.stack([self.pred_Me_raw[i]     for i in range(s, e)])
#         obs_Me_rel  = torch.stack([self.obs_Me_rel_raw[i]  for i in range(s, e)])
#         pred_Me_rel = torch.stack([self.pred_Me_rel_raw[i] for i in range(s, e)])

#         n    = e - s
#         nlp  = [self.non_linear_ped[i] for i in range(s, e)]
#         mask = torch.ones(n, self.seq_len)

#         obs_traj_np = obs_traj[0].numpy()
#         obs_Me_np   = obs_Me[0].numpy()

#         cache_key = (year, tyname, tuple(dates[:self.obs_len]))
#         if cache_key not in self.env_cache:
#             self.env_cache[cache_key] = self._get_env_features(
#                 year, tyname, dates[:self.obs_len], obs_traj_np, obs_Me_np)
#         env_out = self.env_cache[cache_key]

#         return [
#             obs_traj, pred_traj, obs_rel, pred_rel, nlp, mask,
#             obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
#             self._embed_time(dates[:self.obs_len]),
#             self._embed_time(dates[self.obs_len:]),
#             img_obs, img_pred, env_out, info,
#         ]

"""
Model/data/trajectoriesWithMe_unet_training.py  ── v25 patch
=============================================================
FIXES vs v24:

  FIX-DATA-28  [P0-CRITICAL] _build_csv_env_lookup: normalize u/v500 đúng.

               build_env_data_scs_v10.py dùng U500_NORM=30.0:
                 "u500_mean_n": clip(u_mean / 30.0, -1.0, 1.0)

               → .npy lưu u500_mean trong [-1.0, 1.0].
               → CSV có d3d_u500_mean_raw (raw m²/s² ~5843).
               → Cần: clip(d3d_u500_mean_raw / 30.0, -1.0, 1.0)
                 để consistent với .npy format.
               → Lưu vào key "u500_mean" (không phải "u500_raw_mean")
                 để build_env_features_one_step đọc đúng.

  FIX-DATA-29  [P1] _load_env_npy: thêm flag gph500_already_normed=True
               khi đọc từ .npy mới (không có "_n" suffix).
               .npy mới (build_env_data_scs_v10) lưu:
                 "gph500_mean": (gph - 5900) / 200  → ~[-0.3, 0.3]
               Đây là đã normalized. Cần set flag để không z-score lại.

  FIX-DATA-30  [P0] check_env_features_in_batch(): thêm diagnostic
               function để verify u500/v500 != 0 trong batch đầu tiên.
               Gọi trong _check_gph500() của train script.

Kept from v24:
  FIX-DATA-24..27 (synthetic data, coord filter, clip norm)
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import netCDF4 as nc
    HAS_NC = True
except ImportError:
    HAS_NC = False

from Model.env_net_transformer_gphsplit import (
    bearing_to_scs_center_onehot, dist_to_scs_boundary_onehot,
    delta_velocity_onehot, intensity_class_onehot,
    build_env_features_one_step, feat_to_tensor, ENV_FEATURE_DIMS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA3D_H  = 81
DATA3D_W  = 81
DATA3D_CH = 13

DATA3D_MEAN = np.array([
    12444.037,  # CH0: GPH_200  — from compute_data3d_stats_v2
     5844.935,  # CH1: GPH_500
     1482.430,  # CH2: GPH_850
      752.466,  # CH3: GPH_925
       -1.052,  # CH4: U_200
       -0.188,  # CH5: U_500
       -0.407,  # CH6: U_850
       -0.888,  # CH7: U_925
       -0.055,  # CH8: V_200
        1.655,  # CH9: V_500
        1.343,  # CH10: V_850
        1.018,  # CH11: V_925
      301.133,  # CH12: SST
], dtype=np.float32)
 
DATA3D_STD = np.array([
    113.583,  # CH0: GPH_200
     57.135,  # CH1: GPH_500
     37.852,  # CH2: GPH_850
     37.274,  # CH3: GPH_925
     13.315,  # CH4: U_200
      8.042,  # CH5: U_500
      7.911,  # CH6: U_850
      7.494,  # CH7: U_925
      8.377,  # CH8: V_200
      5.999,  # CH9: V_500
      6.203,  # CH10: V_850
      6.555,  # CH11: V_925
      3.023,  # CH12: SST
], dtype=np.float32)

_DATA3D_SENTINEL_LARGE         = 20000.0
_DATA3D_SENTINEL_ZERO_CHANNELS = {0}
_DATA3D_GPH_VALID_MIN          = 25.0
_DATA3D_GPH_VALID_MAX          = 95.0
_DATA3D_SST_CHANNEL            = 12
_SST_VALID_MIN                 = 270.0
_SST_FILL_K                    = 298.0
_MOVE_VEL_NORM                 = 150.0

# FIX-DATA-29 [bug thật đã tìm ra, verify bằng scan_lonlat_range.py]:
# ngưỡng lọc/clip tọa độ cũ (_LON_VALID_MIN=100, _LON_NORM_MIN=-9.0 tức
# 135°E) SAI so với phạm vi địa lý THẬT của dataset này.
#
# Giá trị dưới đây lấy TRỰC TIẾP từ output của scan_lonlat_range.py chạy
# thật trên toàn bộ Data1d/{train,val,test} — KHÔNG phải ước lượng.
# Script đó quét mọi storm, chỉ tính min/max lon/lat trên các storm THẬT
# SỰ đi qua Biển Đông/Việt Nam (>=15% số điểm track nằm trong hộp lon
# 99-121E, lat 0-23N — loại đúng các trường hợp như 1989_GAY, storm chủ
# yếu ở Vịnh Bengal 74-92E chỉ chạm rìa Tây Biển Đông ở vài điểm cuối,
# tránh kéo ngưỡng lệch bởi 1 điểm outlier).
#
# HẬU QUẢ THỰC TẾ của ngưỡng cũ (đã xác nhận trên MOLAVE 2020 — đổ bộ
# Quảng Ngãi — và CONSON 2021 — đổ bộ Đà Nẵng): np.clip(lon_norm, -9.0,
# 2.0) ép cứng mọi điểm phía Tây 135°E về đúng 135°E, xóa mất toàn bộ
# đoạn quỹ đạo áp sát Việt Nam — không loại bỏ sample, mà âm thầm bóp
# méo tọa độ thật thành một giá trị cố định sai.
_LON_VALID_MIN  = 73.00
_LON_VALID_MAX  = 186.40
_LAT_VALID_MAX  = 64.00
_LON_NORM_MIN   = -21.4000
_LON_NORM_MAX   =   1.2800
_LAT_NORM_MIN   =   0.0000
_LAT_NORM_MAX   =  12.8000

# FIX-DATA-28: U/V500 normalization constant (same as build_env_data_scs_v10.py)
_UV500_NORM = 30.0

_NPY_KEY_REMAP = {
    "gph500_mean_n"   : "gph500_mean",
    "gph500_center_n" : "gph500_center",
}
_NPY_U500_KEY_REMAP = {
    "u500_mean_n"   : "u500_mean",
    "u500_center_n" : "u500_center",
    "v500_mean_n"   : "v500_mean",
    "v500_center_n" : "v500_center",
}


# ── FIX-DATA-30: Diagnostic function ─────────────────────────────────────────

def check_env_features_in_batch(bl, tag: str = "") -> None:
    """
    Verify env features (u500, v500, gph500) != 0 trong batch.
    Gọi từ train script tại epoch 0, batch 0.
    """
    env_data = bl[13] if len(bl) > 13 else None
    if env_data is None:
        print(f"  ⚠️  [{tag}] env_data is None!")
        return

    sep = "!" * 60
    ok  = "✅"
    warn = "⚠️ "

    for key in ["u500_mean", "v500_mean", "gph500_mean",
                "u500_center", "v500_center"]:
        if key not in env_data:
            print(f"  {warn} [{tag}] key '{key}' MISSING from env_data!")
            continue
        v = env_data[key]
        if torch.is_tensor(v):
            v_np = v.detach().cpu().float().numpy().flatten()
        else:
            v_np = np.array(v, dtype=float).flatten()

        mn   = float(v_np.mean())
        std  = float(v_np.std())
        zr   = float((v_np == 0.0).mean() * 100)
        vmin = float(v_np.min())
        vmax = float(v_np.max())

        if zr > 90.0:
            print(f"\n{sep}")
            print(f"  {warn} {key} = 0 cho {zr:.1f}% samples!")
            print(f"     mean={mn:.4f}, std={std:.4f}")
            print(f"     FIX-DATA-28 chưa được apply hoặc CSV không có d3d_u500_mean_raw")
            print(f"{sep}\n")
        else:
            print(f"  {ok}  [{tag}] {key:15}: mean={mn:+.4f}  std={std:.4f}  "
                  f"zero={zr:.1f}%  range=[{vmin:.3f}, {vmax:.3f}]")


# ── CSV fallback builder ──────────────────────────────────────────────────────

def _build_csv_env_lookup(csv_path: str) -> dict:
    """
    FIX-DATA-28: normalize d3d_u500_mean_raw → clip(raw/30.0, -1, 1).
                 Lưu vào key "u500_mean" để consistent với .npy format.
                 Không dùng env_u500_mean (boolean flag).
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas không có → CSV fallback không khả dụng")
        return {}

    if not os.path.exists(csv_path):
        logger.warning(f"CSV fallback không tìm thấy: {csv_path}")
        return {}

    logger.info(f"Loading CSV env fallback: {csv_path}")
    try:
        df = pd.read_csv(csv_path, dtype={"storm_name": str, "dt": str})
    except Exception as e:
        logger.warning(f"CSV load failed: {e}")
        return {}

    lookup: dict = {}
    for _, row in df.iterrows():
        yr         = str(int(float(row["year"])))
        name_raw   = str(row["storm_name"])
        name_strip = name_raw.lstrip("0") or "0"
        ts         = str(row["dt"])

        dir12   = [float(row.get(f"env_dir12_{i}",  0.0)) for i in range(8)]
        dir24   = [float(row.get(f"env_dir24_{i}",  0.0)) for i in range(8)]
        inten24 = [float(row.get(f"env_inten24_{i}", 0.0)) for i in range(4)]

        gph_mean   = float(row.get("env_gph500_mean",   -29.5))
        gph_center = float(row.get("env_gph500_center", -29.5))

        mv_norm = float(row.get("env_move_velocity", 0.0))
        mv_raw  = mv_norm * _MOVE_VEL_NORM

        # FIX-DATA-28: normalize u/v500 raw → [-1,1] consistent với .npy
        def _norm_uv(col_name):
            raw = float(row.get(col_name, 0.0))
            if raw <= 0.0 or raw > 50000.0:
                return 0.0
            return float(np.clip(raw / _UV500_NORM, -1.0, 1.0))

        u500_mean   = _norm_uv("d3d_u500_mean_raw")
        u500_center = _norm_uv("d3d_u500_center_raw") if "d3d_u500_center_raw" in df.columns else u500_mean
        v500_mean   = _norm_uv("d3d_v500_mean_raw")
        v500_center = _norm_uv("d3d_v500_center_raw") if "d3d_v500_center_raw" in df.columns else v500_mean

        d = {
            "gph500_mean"           : gph_mean,
            "gph500_center"         : gph_center,
            "gph500_already_normed" : True,  # CSV: raw dam → cần z-score
            # FIX-DATA-28: normalized [-1,1], key consistent với .npy
            "u500_mean"             : u500_mean,
            "u500_center"           : u500_center,
            "v500_mean"             : v500_mean,
            "v500_center"           : v500_center,
            "has_data3d"            : (u500_mean != 0.0),
            "move_velocity"         : mv_raw,
            "history_direction12"   : dir12,
            "history_direction24"   : dir24,
            "history_inte_change24" : inten24,
        }

        for name_try in (name_raw, name_strip, name_raw.zfill(4), name_raw.zfill(2)):
            lookup[(yr, name_try, ts)] = d

    n_storms = df['storm_name'].nunique()
    logger.info(f"CSV env lookup built: {len(lookup)} entries ({n_storms} storms)")

    # Quick sanity check u500
    sample_u = [v["u500_mean"] for v in list(lookup.values())[:100]]
    nonzero = sum(1 for x in sample_u if x != 0.0)
    logger.info(f"CSV u500_mean: {nonzero}/100 non-zero in first 100 entries "
                f"(mean={np.mean(sample_u):.4f})")

    return lookup


def _auto_discover_csv(root_path: str) -> str | None:
    candidates = [
        os.path.join(root_path, "all_storms_final.csv"),
        os.path.join(os.path.dirname(root_path), "all_storms_final.csv"),
        os.path.join(os.path.dirname(os.path.dirname(root_path)), "all_storms_final.csv"),
        "/kaggle/input/datasets/gmnguynhng/data-tc-finall/all_storms_final.csv",
        "/kaggle/working/all_storms_final.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            logger.info(f"Auto-discovered CSV: {p}")
            return p
    return None


def _auto_discover_synthetic_csv(root_path: str) -> str | None:
    candidates = [
        os.path.join(root_path, "all_storms_synthetic.csv"),
        os.path.join(os.path.dirname(root_path), "all_storms_synthetic.csv"),
        os.path.join(os.path.dirname(os.path.dirname(root_path)), "all_storms_synthetic.csv"),
        "/kaggle/working/all_storms_synthetic.csv",
        "/kaggle/input/datasets/gmnguynhng/data-tc-finall/all_storms_synthetic.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            logger.info(f"Auto-discovered synthetic CSV: {p}")
            return p
    return None


# [FIX-DATA-30, per scan_lonlat_range.py's corrected logic] Region
# filter — keep only storms whose track genuinely enters the South
# China Sea / Vietnam region, as requested: "không lẫn vùng khác mà
# những bão có đi vào biển đông, việt nam".
#
# Bounds: lon 99-121E, lat 0-23N (core South China Sea / Vietnam
# coastal region, standard nautical chart convention).
#
# IMPORTANT — uses PERCENTAGE OF TRACK POINTS inside the box, NOT a
# simple bounding-box overlap test. An earlier version of this filter
# used bbox overlap only, which was found to have a false-positive bug:
# storms like 1989_GAY (track mostly in the Bay of Bengal, 74-92E, only
# touching the western rim of the South China Sea at 99-105E for its
# LAST few points) would still pass a bbox-overlap test — its bounding
# box technically overlaps the SCS box even though almost none of its
# actual track lies inside it. This distorted the dataset's overall
# lon/lat range (pulling _LON_NORM_MIN down to -73E-equivalent from
# just this one mostly-irrelevant storm) and would silently include
# storms that have essentially nothing to do with the target region.
#
# min_pct default (15%) is low enough to keep storms that form far away
# and only enter the South China Sea in the SECOND HALF of their track
# (e.g. MOLAVE 2020, which forms east of Palau before crossing into the
# SCS and hitting Vietnam), while high enough to reject storms that
# merely clip a corner of the box in passing.
_SCS_LON_MIN, _SCS_LON_MAX = 99.0, 121.0
_SCS_LAT_MIN, _SCS_LAT_MAX = 0.0, 23.0
_SCS_MIN_PCT_DEFAULT = 15.0


def _storm_touches_scs_vietnam(filepath: str, min_pct: float = _SCS_MIN_PCT_DEFAULT) -> bool:
    """
    Reads a raw Data1d .txt file's lon/lat columns directly (same column
    convention as TrajectoryDataset._read_file: col 1 = lon_norm,
    col 2 = lat_norm), converts to real degrees, and checks whether AT
    LEAST min_pct% of the storm's track points fall inside the South
    China Sea / Vietnam box — not just whether its bounding box
    technically overlaps the region (see module-level docstring above
    for why that distinction matters).

    Returns True (keep the file) if the check cannot be completed for
    any reason (missing file, unparseable, empty) — a filter that
    silently discards data on error would be worse than one that's
    occasionally too permissive.
    """
    try:
        lons, lats = [], []
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "//", "-", "=")):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    int(parts[0])
                except ValueError:
                    continue
                try:
                    lon_norm = float(parts[1])
                    lat_norm = float(parts[2])
                except (ValueError, IndexError):
                    continue
                # Same formula as denorm_traj()/to_deg() elsewhere in
                # this codebase: LONG = (LONG_norm*50 + 1800)/10,
                # LAT = (LAT_norm*50)/10.
                lons.append((lon_norm * 50.0 + 1800.0) / 10.0)
                lats.append((lat_norm * 50.0) / 10.0)
        if not lons:
            return True  # couldn't read anything usable — don't drop silently

        n_in_box = sum(
            1 for lon, lat in zip(lons, lats)
            if _SCS_LON_MIN <= lon <= _SCS_LON_MAX and
               _SCS_LAT_MIN <= lat <= _SCS_LAT_MAX
        )
        pct_in_box = 100.0 * n_in_box / len(lons)
        return pct_in_box >= min_pct
    except Exception as e:
        logger.warning(f"_storm_touches_scs_vietnam failed on {filepath}: "
                       f"{e} — keeping file (fail-open, not fail-closed).")
        return True


def env_data_processing(env_dict: dict) -> dict:
    if not isinstance(env_dict, dict):
        return {}
    already_normed = bool(env_dict.get("gph500_already_normed", False))
    cleaned = {"gph500_already_normed": already_normed}
    for k, v in env_dict.items():
        if k in ("gph500_already_normed", "has_data3d",
                 "gph500_mean_already_normed", "gph500_center_already_normed"):
            continue
        if isinstance(v, (list, np.ndarray)):
            cleaned[k] = v
        elif isinstance(v, bool):
            cleaned[k] = v
        elif v == -1:
            cleaned[k] = 0.0
        else:
            cleaned[k] = v
    # Preserve has_data3d
    cleaned["has_data3d"] = env_dict.get("has_data3d", True)
    for sst_key in ("sst_mean", "sst_center", "sst"):
        if sst_key in cleaned:
            val = cleaned[sst_key]
            if val is None or val == 0 or (isinstance(val, float) and val < _SST_VALID_MIN):
                cleaned[sst_key] = _SST_FILL_K
    if not already_normed:
        for gph_key in ("gph500_mean", "gph500_center"):
            if gph_key in cleaned:
                val = cleaned[gph_key]
                if val is not None and isinstance(val, (int, float)):
                    if val < _DATA3D_GPH_VALID_MIN or val > _DATA3D_GPH_VALID_MAX:
                        cleaned[gph_key] = None
    return cleaned


# ── seq_collate ───────────────────────────────────────────────────────────────

def seq_collate(data):
    (obs_traj, pred_traj, obs_rel, pred_rel,
     nlp, mask, obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
     obs_date, pred_date, img_obs, img_pred, env_data_raw, tyID) = zip(*data)

    def traj_TBC(lst):
        cat = torch.cat(lst, dim=0)
        return cat.permute(2, 0, 1)

    obs_traj_out    = traj_TBC(obs_traj)
    pred_traj_out   = traj_TBC(pred_traj)
    obs_rel_out     = traj_TBC(obs_rel)
    pred_rel_out    = traj_TBC(pred_rel)
    obs_Me_out      = traj_TBC(obs_Me)
    pred_Me_out     = traj_TBC(pred_Me)
    obs_Me_rel_out  = traj_TBC(obs_Me_rel)
    pred_Me_rel_out = traj_TBC(pred_Me_rel)

    nlp_out = torch.tensor(
        [v for sl in nlp for v in (sl if hasattr(sl, "__iter__") else [sl])],
        dtype=torch.float,
    )
    mask_out = torch.cat(list(mask), dim=0).permute(1, 0)

    counts        = torch.tensor([t.shape[0] for t in obs_traj])
    cum           = torch.cumsum(counts, dim=0)
    starts        = torch.cat([torch.tensor([0]), cum[:-1]])
    seq_start_end = torch.stack([starts, cum], dim=1)

    img_obs_out  = torch.stack(list(img_obs),  dim=0).permute(0, 4, 1, 2, 3).float()
    img_pred_out = torch.stack(list(img_pred), dim=0).permute(0, 4, 1, 2, 3).float()

    env_out    = None
    valid_envs = [d for d in env_data_raw if isinstance(d, dict)]
    if valid_envs:
        env_out  = {}
        all_keys = set()
        for d in valid_envs:
            all_keys.update(d.keys())

        _skip_keys = {
            "gph500_already_normed", "has_data3d",
            "gph500_mean_already_normed", "gph500_center_already_normed",
            "history_direction12_valid", "history_direction24_valid",
            "history_inte_change24_valid",
        }
        all_keys -= _skip_keys

        # sorted(): set iteration order is not stable across processes
        # (hash randomization), which would give env_out a non-deterministic
        # key order. Sorting fixes the order without changing any values.
        for key in sorted(all_keys):
            vals = []
            for d in env_data_raw:
                if isinstance(d, dict) and key in d:
                    v = d[key]
                    v = torch.tensor(v, dtype=torch.float) if not torch.is_tensor(v) else v.float()
                    vals.append(v)
                else:
                    ref = next((d[key] for d in valid_envs if key in d), None)
                    if ref is not None:
                        rt = torch.tensor(ref, dtype=torch.float) if not torch.is_tensor(ref) else ref.float()
                        vals.append(torch.zeros_like(rt))
                    else:
                        vals.append(torch.zeros(1))
            try:
                env_out[key] = torch.stack(vals, dim=0)
            except Exception:
                try:
                    mx     = max(v.numel() for v in vals)
                    padded = [F.pad(v.flatten(), (0, mx - v.numel())) for v in vals]
                    env_out[key] = torch.stack(padded, dim=0)
                except Exception:
                    pass

    return (
        obs_traj_out, pred_traj_out, obs_rel_out, pred_rel_out,
        nlp_out, mask_out, seq_start_end,
        obs_Me_out, pred_Me_out, obs_Me_rel_out, pred_Me_rel_out,
        img_obs_out, img_pred_out, env_out, None, list(tyID),
    )


# ── TrajectoryDataset ─────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1,
                 threshold=0.002, min_ped=1, delim=" ", other_modal="gph",
                 test_year=None, type="train", split=None, is_test=False,
                 csv_env_path=None, synthetic_csv_path=None,
                 filter_region=False, min_pct_in_scs=_SCS_MIN_PCT_DEFAULT,
                 **kwargs):
        super().__init__()

        dtype = split if split is not None else type

        if isinstance(data_dir, dict):
            root  = data_dir["root"]
            dtype = data_dir.get("type", dtype)
        else:
            root = data_dir
        if is_test and dtype not in ("val", "test"):
            dtype = "test"

        root = os.path.abspath(root)
        bn   = os.path.basename(root)
        if bn in ("train", "test", "val"):
            self.root_path = os.path.dirname(os.path.dirname(root))
        elif bn == "Data1d":
            self.root_path = os.path.dirname(root)
        else:
            self.root_path = root

        self.data1d_path = os.path.join(self.root_path, "Data1d", dtype)
        self.data3d_path = os.path.join(self.root_path, "Data3d")
        for env_name in ("Env_data", "ENV_DATA", "env_data", "Env_Data", "Env_Data"):
            candidate = os.path.join(self.root_path, env_name)
            if os.path.exists(candidate):
                self.env_path = candidate
                break
        else:
            self.env_path = os.path.join(self.root_path, "Env_data")

        self._is_train = (dtype == "train")

        self._csv_env_lookup: dict  = {}
        self._env_path_missing      = not os.path.exists(self.env_path)

        if self._env_path_missing or csv_env_path is not None:
            if csv_env_path is None:
                csv_env_path = _auto_discover_csv(self.root_path)
            if csv_env_path:
                self._csv_env_lookup = _build_csv_env_lookup(csv_env_path)
        elif csv_env_path is not None:
            # Luôn build CSV lookup kể cả có Env_data (để bổ sung u500 raw)
            self._csv_env_lookup = _build_csv_env_lookup(csv_env_path)

        # Nếu có Env_data nhưng không có csv_env_path, vẫn thử auto-discover CSV
        # để dùng d3d_u500_mean_raw (Env_data .npy đã có u500_mean normalized)
        if not self._csv_env_lookup:
            auto_csv = _auto_discover_csv(self.root_path)
            if auto_csv:
                self._csv_env_lookup = _build_csv_env_lookup(auto_csv)

        self.obs_len    = obs_len
        self.pred_len   = pred_len
        self.seq_len    = obs_len + pred_len
        self.skip       = skip
        self.modal_name = other_modal

        self._synthetic_lookup: dict = {}
        if self._is_train:
            if synthetic_csv_path is None:
                synthetic_csv_path = _auto_discover_synthetic_csv(self.root_path)
            if synthetic_csv_path:
                self._synthetic_lookup = self._build_synthetic_lookup(synthetic_csv_path)
                logger.info(f"Loaded synthetic: {len(self._synthetic_lookup)} storms")

        if not os.path.exists(self.data1d_path):
            logger.error(f"Missing Data1d: {self.data1d_path}")
            self.num_seq       = 0
            self.seq_start_end = []
            self.tyID          = []
            return

        all_files = [
            os.path.join(self.data1d_path, f)
            for f in sorted(os.listdir(self.data1d_path))
            if f.endswith(".txt") and (test_year is None or str(test_year) in f)
        ]

        # [FIX-DATA-30] Optional region filter — keep only storms whose
        # track has >= min_pct_in_scs% of points inside the South China
        # Sea / Vietnam box. Off by default (filter_region=False) to
        # preserve existing behavior; caller must opt in explicitly.
        if filter_region:
            n_before = len(all_files)
            all_files = [f for f in all_files
                        if _storm_touches_scs_vietnam(f, min_pct=min_pct_in_scs)]
            n_after = len(all_files)
            logger.info(f"filter_region=True (min_pct={min_pct_in_scs}%): kept "
                       f"{n_after}/{n_before} storm files whose track has >= "
                       f"{min_pct_in_scs}% of points in the South China Sea / "
                       f"Vietnam region (lon {_SCS_LON_MIN}-{_SCS_LON_MAX}E, "
                       f"lat {_SCS_LAT_MIN}-{_SCS_LAT_MAX}N)")

        self.obs_traj_raw    = []
        self.pred_traj_raw   = []
        self.obs_Me_raw      = []
        self.pred_Me_raw     = []
        self.obs_rel_raw     = []
        self.pred_rel_raw    = []
        self.obs_Me_rel_raw  = []
        self.pred_Me_rel_raw = []
        self.non_linear_ped  = []
        self.tyID            = []
        num_peds_in_seq      = []
        self.env_cache: dict = {}

        n_filtered_coord = 0
        for path in all_files:
            base   = os.path.splitext(os.path.basename(path))[0]
            parts  = base.split("_")
            f_year = parts[0] if parts else "unknown"
            # FIX-DATA-31 [P0-CRITICAL]: lay TOAN BO phan con lai sau year
            # (noi lai bang "_"), KHONG chi lay parts[1]. Ten file co the
            # co dang "<year>_<name>_<so_thu_tu>.txt" (vd "2000_0019_2.txt"
            # cho con bao thu 2 trung ma so/ten "0019" trong nam 2000).
            # BUG CU: f_name = parts[1] cat mat phan "_2", khien nhieu con
            # bao khac nhau trong cung 1 nam (vd 2000_0019.txt den
            # 2000_0019_7.txt, 7 con bao rieng biet) DEU bi gan chung
            # f_name="0019" -> khi img_read()/_load_env_npy() tim thu muc
            # Data3d/Env theo (year, f_name), ca 7 con bao se doc NHAM
            # anh ve tinh/du lieu moi truong cua con bao goc "0019" thay
            # vi du lieu rieng cua chinh no trong "0019_2".."0019_7".
            f_name = "_".join(parts[1:]) if len(parts) > 1 else base

            d    = self._read_file(path, delim)
            data = d["main"]
            add  = d["addition"]
            if len(data) < self.seq_len:
                continue

            frames     = np.unique(data[:, 0]).tolist()
            frame_data = [data[data[:, 0] == f] for f in frames]
            n_frames   = len(frames)
            if n_frames < self.seq_len:
                continue
            n_seq = (n_frames - self.seq_len) // skip + 1

            for idx in range(0, n_seq * skip, skip):
                if idx + self.seq_len > len(frame_data):
                    break

                seg  = np.concatenate(frame_data[idx: idx + self.seq_len])
                peds = np.unique(seg[:, 1])

                buf_obs_traj   = []
                buf_pred_traj  = []
                buf_obs_rel    = []
                buf_pred_rel   = []
                buf_obs_Me     = []
                buf_pred_Me    = []
                buf_obs_Me_rel = []
                buf_pred_Me_rel= []
                buf_nlp        = []
                cnt            = 0

                for pid in peds:
                    ps = seg[seg[:, 1] == pid]
                    if len(ps) != self.seq_len:
                        continue
                    ps_t = np.transpose(ps[:, 2:])

                    # FIX-DATA-25: Filter outlier coordinates
                    lon_norm_vals = ps_t[0, :]
                    lat_norm_vals = ps_t[1, :]
                    lon_deg = (lon_norm_vals * 50.0 + 1800.0) / 10.0
                    lat_deg = (lat_norm_vals * 50.0) / 10.0
                    if lon_deg.min() < _LON_VALID_MIN or lon_deg.max() > _LON_VALID_MAX:
                        n_filtered_coord += 1
                        continue
                    if lat_deg.max() > _LAT_VALID_MAX:
                        n_filtered_coord += 1
                        continue

                    # FIX-DATA-27: Clip lon_norm / lat_norm
                    ps_t_clipped = ps_t.copy()
                    ps_t_clipped[0, :] = np.clip(ps_t[0, :], _LON_NORM_MIN, _LON_NORM_MAX)
                    ps_t_clipped[1, :] = np.clip(ps_t[1, :], _LAT_NORM_MIN, _LAT_NORM_MAX)

                    rel  = np.zeros_like(ps_t_clipped)
                    rel[:, 1:] = ps_t_clipped[:, 1:] - ps_t_clipped[:, :-1]

                    buf_obs_traj.append(torch.from_numpy(ps_t_clipped[:2, :obs_len]).float())
                    buf_pred_traj.append(torch.from_numpy(ps_t_clipped[:2, obs_len:]).float())
                    buf_obs_rel.append(torch.from_numpy(rel[:2, :obs_len]).float())
                    buf_pred_rel.append(torch.from_numpy(rel[:2, obs_len:]).float())
                    buf_obs_Me.append(torch.from_numpy(ps_t_clipped[2:, :obs_len]).float())
                    buf_pred_Me.append(torch.from_numpy(ps_t_clipped[2:, obs_len:]).float())
                    buf_obs_Me_rel.append(torch.from_numpy(rel[2:, :obs_len]).float())
                    buf_pred_Me_rel.append(torch.from_numpy(rel[2:, obs_len:]).float())
                    buf_nlp.append(self._poly_fit(ps_t_clipped, pred_len, threshold))
                    cnt += 1

                if cnt >= min_ped:
                    self.obs_traj_raw.extend(buf_obs_traj)
                    self.pred_traj_raw.extend(buf_pred_traj)
                    self.obs_rel_raw.extend(buf_obs_rel)
                    self.pred_rel_raw.extend(buf_pred_rel)
                    self.obs_Me_raw.extend(buf_obs_Me)
                    self.pred_Me_raw.extend(buf_pred_Me)
                    self.obs_Me_rel_raw.extend(buf_obs_Me_rel)
                    self.pred_Me_rel_raw.extend(buf_pred_Me_rel)
                    self.non_linear_ped.extend(buf_nlp)
                    num_peds_in_seq.append(cnt)
                    self.tyID.append({
                        "old":    [f_year, f_name, idx],
                        "tydate": [add[i][0] for i in range(idx, idx + self.seq_len)],
                    })

        real_seqs = len(self.tyID)
        logger.info(f"Loaded {real_seqs} real sequences "
                    f"(filtered {n_filtered_coord} outlier coords)")

        if self._is_train and self._synthetic_lookup:
            self._load_synthetic_sequences(
                min_ped, threshold, obs_len, pred_len, delim)
            logger.info(f"Total sequences after synthetic: {len(self.tyID)}")

        self.num_seq = len(self.tyID)
        cum = np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = list(zip([0] + cum[:-1], cum))

    def _build_synthetic_lookup(self, csv_path: str) -> dict:
        try:
            import pandas as pd
        except ImportError:
            return {}
        if not os.path.exists(csv_path):
            return {}
        try:
            df = pd.read_csv(csv_path, dtype={"storm_name": str, "dt": str})
            lookup = {}
            for (yr, nm), grp in df.groupby(["year", "storm_name"]):
                lookup[(str(int(float(yr))), str(nm))] = grp.sort_values("step_i").reset_index(drop=True)
            return lookup
        except Exception as e:
            logger.warning(f"Synthetic CSV load failed: {e}")
            return {}

    def _load_synthetic_sequences(self, min_ped, threshold, obs_len, pred_len, delim):
        num_peds_in_seq_local = []
        for (f_year, f_name), storm_df in self._synthetic_lookup.items():
            n_steps = len(storm_df)
            if n_steps < self.seq_len:
                continue
            n_seq = (n_steps - self.seq_len) // self.skip + 1
            dates_list = storm_df["dt"].tolist()
            for idx in range(0, n_seq * self.skip, self.skip):
                if idx + self.seq_len > n_steps:
                    break
                seg = storm_df.iloc[idx: idx + self.seq_len]
                lon_norm = seg["lon_norm"].values
                lat_norm = seg["lat_norm"].values
                pres_norm = seg["pres_norm"].values
                wnd_norm = seg["wnd_norm"].values
                lon_deg = (lon_norm * 50.0 + 1800.0) / 10.0
                lat_deg = (lat_norm * 50.0) / 10.0
                if lon_deg.min() < _LON_VALID_MIN or lon_deg.max() > _LON_VALID_MAX:
                    continue
                if lat_deg.max() > _LAT_VALID_MAX:
                    continue
                lon_norm = np.clip(lon_norm, _LON_NORM_MIN, _LON_NORM_MAX)
                lat_norm = np.clip(lat_norm, _LAT_NORM_MIN, _LAT_NORM_MAX)
                ps_t = np.stack([lon_norm, lat_norm, pres_norm, wnd_norm], axis=0)
                rel  = np.zeros_like(ps_t)
                rel[:, 1:] = ps_t[:, 1:] - ps_t[:, :-1]
                self.obs_traj_raw.append(torch.from_numpy(ps_t[:2, :obs_len]).float())
                self.pred_traj_raw.append(torch.from_numpy(ps_t[:2, obs_len:]).float())
                self.obs_rel_raw.append(torch.from_numpy(rel[:2, :obs_len]).float())
                self.pred_rel_raw.append(torch.from_numpy(rel[:2, obs_len:]).float())
                self.obs_Me_raw.append(torch.from_numpy(ps_t[2:, :obs_len]).float())
                self.pred_Me_raw.append(torch.from_numpy(ps_t[2:, obs_len:]).float())
                self.obs_Me_rel_raw.append(torch.from_numpy(rel[2:, :obs_len]).float())
                self.pred_Me_rel_raw.append(torch.from_numpy(rel[2:, obs_len:]).float())
                self.non_linear_ped.append(self._poly_fit(ps_t, pred_len, threshold))
                num_peds_in_seq_local.append(1)
                tydate = [str(dates_list[i]) for i in range(idx, idx + self.seq_len)]
                self.tyID.append({
                    "old":    [f_year, f_name, idx],
                    "tydate": tydate,
                    "is_synthetic": True,
                })
        existing_end = self.seq_start_end[-1][1] if self.seq_start_end else 0
        new_cum = np.cumsum(num_peds_in_seq_local) + existing_end
        new_pairs = list(zip([existing_end] + new_cum[:-1].tolist(), new_cum.tolist()))
        self.seq_start_end.extend(new_pairs)

    def _read_file(self, path: str, delim: str) -> dict:
        data, add = [], []
        with open(path, encoding="utf-8", errors="ignore") as f:
            raw_lines = f.readlines()
        for line in raw_lines:
            line = line.strip()
            if not line or line.startswith(("#", "//", "-", "=")):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                int(parts[0])
            except ValueError:
                continue
            try:
                frame_id  = float(parts[0])
                lon_norm  = float(parts[1])
                lat_norm  = float(parts[2])
                pres_norm = float(parts[3])
                wnd_norm  = float(parts[4])
                date      = parts[5]
                name      = parts[6]
                add.append([date, name])
                data.append([frame_id, 1.0, lon_norm, lat_norm, pres_norm, wnd_norm])
            except (ValueError, IndexError):
                continue
        return {
            "main":     np.asarray(data, dtype=np.float32) if data else np.zeros((0, 6), dtype=np.float32),
            "addition": add,
        }

    def _poly_fit(self, traj, tlen, threshold):
        t  = np.linspace(0, tlen - 1, tlen)
        rx = np.polyfit(t, traj[0, -tlen:], 2, full=True)[1]
        ry = np.polyfit(t, traj[1, -tlen:], 2, full=True)[1]
        return 1.0 if (len(rx) > 0 and rx[0] + ry[0] >= threshold) else 0.0

    def _normalize_data3d(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        for c in range(DATA3D_CH):
            ch = arr[:, :, c]
            ch[ch > _DATA3D_SENTINEL_LARGE] = np.nan
            if c < 4:  # FIX-v26: GPH channels — filter negative sentinels
                ch[ch < 0] = np.nan
            if c in _DATA3D_SENTINEL_ZERO_CHANNELS:
                ch[ch == 0.0] = np.nan
            if c == _DATA3D_SST_CHANNEL:
                ch[ch < _SST_VALID_MIN] = _SST_FILL_K
            if np.any(np.isnan(ch)):
                valid_vals = ch[~np.isnan(ch)]
                fill_val = (float(np.median(valid_vals)) if len(valid_vals) > 0
                            else float(DATA3D_MEAN[c]))
                ch[np.isnan(ch)] = fill_val
            arr[:, :, c] = (ch - DATA3D_MEAN[c]) / (DATA3D_STD[c] + 1e-6)
        return np.clip(arr, -5.0, 5.0)

    def _load_data3d_file(self, path: str):
        try:
            if path.endswith(".npy"):
                arr = np.load(path).astype(np.float32)
            elif path.endswith(".nc") and HAS_NC:
                with nc.Dataset(path) as ds:
                    keys = list(ds.variables.keys())
                    arr  = np.array(ds.variables[keys[-1]][:]).astype(np.float32)
            else:
                return None
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            if arr.ndim == 3:
                if arr.shape[0] == DATA3D_CH:
                    arr = arr.transpose(1, 2, 0)
                H, W, C = arr.shape
                if H != DATA3D_H or W != DATA3D_W:
                    if HAS_CV2:
                        arr = cv2.resize(arr, (DATA3D_W, DATA3D_H))
                    else:
                        arr = arr[:DATA3D_H, :DATA3D_W, :]
                        if arr.shape[0] < DATA3D_H:
                            arr = np.pad(arr, ((0, DATA3D_H - arr.shape[0]), (0, 0), (0, 0)))
                        if arr.shape[1] < DATA3D_W:
                            arr = np.pad(arr, ((0, 0), (0, DATA3D_W - arr.shape[1]), (0, 0)))
                if arr.shape[2] < DATA3D_CH:
                    arr = np.concatenate([
                        arr,
                        np.zeros((DATA3D_H, DATA3D_W, DATA3D_CH - arr.shape[2]), dtype=np.float32),
                    ], axis=2)
                arr = arr[:, :, :DATA3D_CH]
                return self._normalize_data3d(arr)
        except Exception as e:
            logger.debug(f"Data3d load error {path}: {e}")
        return None

    def img_read(self, year, ty_name, timestamp) -> torch.Tensor:
        folder = os.path.join(self.data3d_path, str(year), str(ty_name))
        if not os.path.exists(folder):
            return torch.zeros(DATA3D_H, DATA3D_W, DATA3D_CH)
        prefix = f"WP{year}{ty_name}_{timestamp}"
        for ext in (".npy", ".nc"):
            p = os.path.join(folder, prefix + ext)
            if os.path.exists(p):
                arr = self._load_data3d_file(p)
                if arr is not None:
                    return torch.from_numpy(arr).float()
        try:
            for fname in sorted(os.listdir(folder)):
                if timestamp in fname and fname.endswith((".npy", ".nc")):
                    arr = self._load_data3d_file(os.path.join(folder, fname))
                    if arr is not None:
                        return torch.from_numpy(arr).float()
        except Exception:
            pass
        return torch.zeros(DATA3D_H, DATA3D_W, DATA3D_CH)

    def _load_env_npy(self, year, ty_name, timestamp):
        """
        FIX-DATA-29: .npy mới (build_env_data_scs_v10) lưu gph500_mean
        đã normalized bằng (gph-5900)/200 → range ~[-0.3, 0.3].
        Set gph500_already_normed=True nếu file KHÔNG có "_n" suffix keys.
        """
        folder = os.path.join(self.env_path, str(year), str(ty_name))
        if os.path.exists(folder):
            candidates = []
            for fname in [f"WP{year}{ty_name}_{timestamp}.npy",
                          f"{timestamp}.npy"]:
                p = os.path.join(folder, fname)
                if os.path.exists(p):
                    candidates.append(p)
            if not candidates:
                try:
                    candidates = [
                        os.path.join(folder, f)
                        for f in sorted(os.listdir(folder))
                        if timestamp in f and f.endswith(".npy")
                    ]
                except Exception:
                    pass

            for p in candidates:
                try:
                    raw = np.load(p, allow_pickle=True).item()
                    if not isinstance(raw, dict):
                        continue

                    remapped = dict(raw)
                    has_old_gph = False
                    has_old_n_suffix = any(k.endswith("_n") for k in remapped)

                    for old_k, new_k in _NPY_KEY_REMAP.items():
                        if old_k in remapped and new_k not in remapped:
                            remapped[new_k] = remapped.pop(old_k)
                            has_old_gph = True

                    for old_k, new_k in _NPY_U500_KEY_REMAP.items():
                        if old_k in remapped and new_k not in remapped:
                            remapped[new_k] = remapped.pop(old_k)

                    if has_old_gph:
                        # .npy cũ với "_n" suffix: đã normalized bởi (gph-5900)/200
                        # remapped["gph500_already_normed"] = True
                        remapped["gph500_already_normed"] = True
                    elif "gph500_mean" in remapped and not has_old_n_suffix:
                        # FIX-DATA-29: .npy mới (build_env_data_scs_v10):
                        # gph500_mean = (gph-5900)/200, u500_mean = clip(u/30,-1,1)
                        # Đây đều là đã normalized
                        # remapped["gph500_already_normed"] = True\
                        remapped["gph500_already_normed"] = True   # FIX-v57: GPH cần normalize

                    
                    result = env_data_processing(remapped)
                    # if not hasattr(self, '_debug_printed'):
                    #     self._debug_printed = True
                    #     print(f"\n  [DEBUG _load_env_npy] file={p}")
                    #     print(f"  raw keys     : {list(raw.keys())}")
                    #     print(f"  raw has_data3d : {raw.get('has_data3d')}")
                    #     print(f"  raw u500_mean  : {raw.get('u500_mean')}")
                    #     print(f"  remapped has_data3d : {remapped.get('has_data3d')}")
                    #     print(f"  remapped u500_mean  : {remapped.get('u500_mean')}")
                    #     print(f"  result has_data3d   : {result.get('has_data3d')}")
                    #     print(f"  result u500_mean    : {result.get('u500_mean')}")
                    #     print()

                    return result
                except Exception as e:
                    logger.debug(f"env npy load error {p}: {e}")

        # CSV fallback
        if self._csv_env_lookup:
            yr_str = str(year)
            ts_str = str(timestamp)
            name_strip  = str(ty_name).lstrip("0") or "0"
            name_padded = str(ty_name).zfill(4)
            name_padded2 = str(ty_name).zfill(2)
            for name_try in (str(ty_name), name_strip, name_padded, name_padded2):
                raw_dict = self._csv_env_lookup.get((yr_str, name_try, ts_str))
                if raw_dict is not None:
                    return env_data_processing(dict(raw_dict))

        return None

    def _get_env_features(self, year, ty_name, dates, obs_traj, obs_Me):
        T         = len(dates)
        all_feats = []
        prev_speed = None

        for t in range(T):
            env_npy = self._load_env_npy(year, ty_name, dates[t])
            feat    = build_env_features_one_step(
                lon_norm  = float(obs_traj[0, t]),
                lat_norm  = float(obs_traj[1, t]),
                wind_norm = float(obs_Me[1, t]),
                pres_norm = float(obs_Me[0, t]),
                timestamp = dates[t],
                env_npy   = env_npy,
                prev_speed_kmh = prev_speed,
            )
            # # DEBUG một lần
            # if not hasattr(self, '_debug_feat_printed'):
            #     self._debug_feat_printed = True
            #     print(f"\n  [DEBUG _get_env_features] t={t} date={dates[t]}")
            #     print(f"  env_npy has_data3d : {env_npy.get('has_data3d') if env_npy else None}")
            #     print(f"  env_npy u500_mean  : {env_npy.get('u500_mean') if env_npy else None}")
            #     print(f"  feat u500_mean     : {feat.get('u500_mean')}")
            #     print(f"  feat gph500_mean   : {feat.get('gph500_mean')}")

            all_feats.append(feat)
            if isinstance(env_npy, dict):
                mv = float(env_npy.get("move_velocity", 0.0) or 0.0)
                prev_speed = mv if mv != -1 else 0.0

        env_out = {}
        for key in ENV_FEATURE_DIMS:
            dim  = ENV_FEATURE_DIMS[key]
            rows = []
            for feat in all_feats:
                v = feat.get(key, [0.0] * dim)
                t = torch.tensor(v, dtype=torch.float)
                if t.numel() < dim:
                    t = F.pad(t, (0, dim - t.numel()))
                rows.append(t[:dim])
            env_out[key] = torch.stack(rows, dim=0)
        return env_out

    def _embed_time(self, date_list):
        rows = []
        for d in date_list:
            try:
                rows.append([
                    (float(d[:4]) - 1949) / 70.0 - 0.5,
                    (float(d[4:6]) - 1)   / 11.0 - 0.5,
                    (float(d[6:8]) - 1)   / 30.0 - 0.5,
                    float(d[8:10])         / 18.0 - 0.5,
                ])
            except Exception:
                rows.append([0.0, 0.0, 0.0, 0.0])
        return torch.tensor(rows, dtype=torch.float).t().unsqueeze(0)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        if self.num_seq == 0:
            raise IndexError("Empty dataset")
        s, e   = self.seq_start_end[index]
        info   = self.tyID[index]
        year   = str(info["old"][0])
        tyname = str(info["old"][1])
        dates  = info["tydate"]

        imgs    = [self.img_read(year, tyname, ts) for ts in dates[:self.obs_len]]
        img_obs = torch.stack(imgs, dim=0)
        img_pred = torch.zeros(self.pred_len, DATA3D_H, DATA3D_W, DATA3D_CH)

        obs_traj    = torch.stack([self.obs_traj_raw[i]    for i in range(s, e)])
        pred_traj   = torch.stack([self.pred_traj_raw[i]   for i in range(s, e)])
        obs_rel     = torch.stack([self.obs_rel_raw[i]     for i in range(s, e)])
        pred_rel    = torch.stack([self.pred_rel_raw[i]    for i in range(s, e)])
        obs_Me      = torch.stack([self.obs_Me_raw[i]      for i in range(s, e)])
        pred_Me     = torch.stack([self.pred_Me_raw[i]     for i in range(s, e)])
        obs_Me_rel  = torch.stack([self.obs_Me_rel_raw[i]  for i in range(s, e)])
        pred_Me_rel = torch.stack([self.pred_Me_rel_raw[i] for i in range(s, e)])

        n    = e - s
        nlp  = [self.non_linear_ped[i] for i in range(s, e)]
        mask = torch.ones(n, self.seq_len)

        obs_traj_np = obs_traj[0].numpy()
        obs_Me_np   = obs_Me[0].numpy()

        cache_key = (year, tyname, tuple(dates[:self.obs_len]))
        if cache_key not in self.env_cache:
            self.env_cache[cache_key] = self._get_env_features(
                year, tyname, dates[:self.obs_len], obs_traj_np, obs_Me_np)
        env_out = self.env_cache[cache_key]

        return [
            obs_traj, pred_traj, obs_rel, pred_rel, nlp, mask,
            obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
            self._embed_time(dates[:self.obs_len]),
            self._embed_time(dates[self.obs_len:]),
            img_obs, img_pred, env_out, info,
        ]