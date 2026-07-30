# """
# evaluate_multi_model.py
# =========================
# Runs test-set evaluation for RNN / GRU / LSTM / ST-Trans / FM checkpoints,
# using ONE SHARED ATE/CTE formula for all five models.

# WHY A SEPARATE SCRIPT, NOT REUSING EACH MODEL'S OWN METRIC CODE
# -----------------------------------------------------------------
# paper_baseline_model.py (used by RNN/GRU/LSTM/ST-Trans) has its own
# _ate_cte_tensors() with a DIFFERENT convention than flow_matching_model.py's
# _ate_cte_full() (used by FM):
#   - flat-earth approx (111*cos(lat)) vs haversine — small effect (~0.2%)
#   - reference heading computed OUTGOING from each point (gt[t]->gt[t+1])
#     vs INCOMING to each point (gt[t-1]->gt[t]) — this is NOT a small
#     effect: verified numerically on a synthetic turning trajectory, the
#     two conventions can even disagree on the SIGN of CTE at a given step.
# Comparing FM's ATE/CTE (computed one way) against RNN/GRU/LSTM/ST-Trans's
# ATE/CTE (computed the other way) would not be a fair like-for-like
# comparison — any difference could be an artifact of the formula, not the
# model. This script computes ADE/ATE/CTE for ALL FIVE models using the
# SAME function (_ate_cte_full, _haversine_deg, _forward_azimuth, imported
# from flow_matching_model.py — the version with the off-by-one fix already
# verified in evaluate_full.py), so Table-10-style comparisons are sound.

# USAGE
# -----
# python evaluate_multi_model.py \
#     --dataset_root <root> \
#     --fm_checkpoint runs/fm_seed42/best_model.pth \
#     --st_trans_checkpoint runs/st_trans/best_model.pth \
#     --lstm_checkpoint runs/lstm/best_model.pth \
#     --gru_checkpoint runs/gru/best_model.pth \
#     --rnn_checkpoint runs/rnn/best_model.pth \
#     --output_dir eval_multi/

# Any checkpoint arg can be omitted to skip that model. Produces one
# combined JSON of per-window records (one row per (model, storm, window))
# that generate_comparison_table.py consumes directly for the Table-10-style
# statistical significance tables.
# """
# from __future__ import annotations
# import sys, os, argparse, json, random
# from typing import Dict, List, Optional

# import numpy as np
# import torch

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from Model.data.loader_training import data_loader
# from Model.flow_matching_model import (
#     TCFlowMatching, _norm_to_deg, _haversine_deg, _forward_azimuth, _unwrap,
# )
# from Model.paper_baseline_model import PaperBaseline, MODEL_TYPES
# from Model.st_trans_model import STTrans


# def _infer_seed(checkpoint_path: str, ck: dict) -> str:
#     """
#     Best-effort seed extraction: prefer an explicit 'seed' key saved in
#     the checkpoint (train_flowmatching.py / train_st_trans.py /
#     train_paper_baseline.py all save this per the project's multi-seed
#     convention). Falls back to parsing "_seed<N>" from the path/dirname
#     (the convention those same scripts use for output_dir), and finally
#     "unknown" if neither is found — records with seed="unknown" will
#     NOT be treated as genuinely pooled multi-seed data by
#     generate_paper_table.py (it warns explicitly in that case).
#     """
#     if isinstance(ck, dict) and "seed" in ck:
#         return str(ck["seed"])
#     import re
#     m = re.search(r"seed[_-]?(\d+)", checkpoint_path)
#     if m:
#         return m.group(1)
#     return "unknown"


# def move(batch, device):
#     return [x.to(device) if torch.is_tensor(x) else x for x in batch]


# def ate_cte_full(pred_deg: torch.Tensor, gt_deg: torch.Tensor):
#     """
#     SAME formula as evaluate_full.py's _ate_cte_full (off-by-one already
#     fixed there — see that file's own comments for the fix history).
#     Returns [T-1, B] each; index k holds the error at ORIGINAL step k+1.
#     """
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         z = pred_deg.new_zeros(1, pred_deg.shape[1])
#         return z, z
#     bear_ref = _forward_azimuth(gt_deg[:T - 1], gt_deg[1:T])
#     bear_err = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
#     dist_err = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
#     ang = bear_err - bear_ref
#     return dist_err * torch.cos(ang), dist_err * torch.sin(ang)


# def load_fm(checkpoint: str, device):
#     ck = torch.load(checkpoint, map_location="cpu")
#     model_cfg = ck.get("model_cfg") or {}
#     if not model_cfg:
#         print(f"  ⚠ FM checkpoint has no model_cfg — using constructor "
#               f"defaults (correct only if trained with default architecture).")
#     model = TCFlowMatching(**model_cfg).to(device)
#     state = ck.get("model", ck.get("model_state"))
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     if missing or unexpected:
#         print(f"  ⚠ FM load_state_dict: {len(missing)} missing, "
#               f"{len(unexpected)} unexpected keys")
#     model.eval()
#     seed = _infer_seed(checkpoint, ck)
#     return model, seed


# def load_paper_baseline(checkpoint: str, model_type: str, device,
#                          hidden_dim: int = 256, n_layers: int = 3,
#                          obs_len: int = 8, pred_len: int = 12,
#                          unet_in_ch: int = 13, dropout: float = 0.20):
#     """
#     [UPDATED] train_paper_baseline.py now saves a full model_cfg dict in
#     newer checkpoints (added alongside --seed support). If present, it is
#     used directly and the CLI-default args above are ignored — this is
#     the reliable path. If absent (checkpoint trained before that fix),
#     falls back to the CLI-default-matching values passed in here, with a
#     warning, same caveat as FM's missing-model_cfg case.
#     """
#     assert model_type in MODEL_TYPES, f"model_type must be one of {MODEL_TYPES}"
#     ck = torch.load(checkpoint, map_location="cpu")
#     saved_type = ck.get("model_type", model_type)
#     if saved_type != model_type:
#         print(f"  ⚠ Checkpoint's saved model_type='{saved_type}' differs "
#               f"from requested '{model_type}' — using requested value. "
#               f"Verify this checkpoint is really the {model_type.upper()} one.")

#     model_cfg = ck.get("model_cfg")
#     if model_cfg:
#         model = PaperBaseline(**model_cfg).to(device)
#     else:
#         print(f"  ⚠ {model_type.upper()} checkpoint has no model_cfg — "
#               f"using CLI-default-matching args (hidden_dim={hidden_dim}, "
#               f"n_layers={n_layers}, dropout={dropout}). Only correct if "
#               f"trained with train_paper_baseline.py's own defaults.")
#         model = PaperBaseline(model_type=model_type, pred_len=pred_len,
#                                obs_len=obs_len, hidden_dim=hidden_dim,
#                                n_layers=n_layers, unet_in_ch=unet_in_ch,
#                                dropout=dropout).to(device)
#     state = ck.get("model_state", ck.get("model"))
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     if missing or unexpected:
#         print(f"  ⚠ {model_type.upper()} load_state_dict: {len(missing)} "
#               f"missing, {len(unexpected)} unexpected keys")
#     model.eval()
#     seed = _infer_seed(checkpoint, ck)
#     return model, seed


# def load_st_trans(checkpoint: str, device,
#                    obs_len: int = 8, pred_len: int = 12, unet_in_ch: int = 13,
#                    d_model: int = 64, nhead: int = 4, num_enc_layers: int = 1,
#                    num_dec_layers: int = 3, dim_ff: int = 512, dropout: float = 0.1):
#     """
#     [UPDATED] Same model_cfg pattern as load_paper_baseline. Note STTrans
#     (non_ar) and STTransAR have DIFFERENT constructor signatures (AR has
#     no num_dec_layers) — the saved model_cfg already accounts for this
#     (see train_st_trans.py's checkpoint-save branch), so **kwargs here
#     naturally works for either. This loader only instantiates STTrans
#     (non-AR) — STTransAR support would need its own loader if you also
#     want to evaluate that variant.

#     [FIX-STTRANS-MODEL-CFG] Checkpoints saved by an OLDER version of
#     train_st_trans.py (before the "STTrans vs STTransAR have different
#     ctor signatures" fix was added there) can have a model_cfg dict that
#     includes extra keys STTrans.__init__() does not accept -- observed in
#     practice: 'model_type' (TypeError: STTrans.__init__() got an
#     unexpected keyword argument 'model_type'). Current train_st_trans.py
#     saves model_type as a SEPARATE top-level checkpoint key (ck['model_type']),
#     not inside model_cfg, so a clean checkpoint should not hit this -- but
#     since we cannot control what checkpoint file the user points this at
#     (e.g. one trained before that convention existed), filter model_cfg
#     down to exactly STTrans's known-valid non_ar constructor keys before
#     calling STTrans(**model_cfg), rather than trusting the checkpoint's
#     model_cfg blindly. Any dropped key is reported so a real config
#     mismatch (not just a harmless stale extra key) is still visible.
#     """
#     ck = torch.load(checkpoint, map_location="cpu")
#     model_cfg = ck.get("model_cfg")
#     if model_cfg:
#         _VALID_STTRANS_KEYS = {
#             "obs_len", "pred_len", "unet_in_ch", "d_model", "nhead",
#             "num_enc_layers", "num_dec_layers", "dim_ff", "dropout",
#         }
#         _dropped = {k: v for k, v in model_cfg.items() if k not in _VALID_STTRANS_KEYS}
#         if _dropped:
#             # [FIX-STTRANS-WRONG-CKPT] 'hidden_dim'/'n_layers' are
#             # PaperBaseline-specific (RNN/GRU/LSTM), never valid STTrans
#             # ctor args under ANY version of train_st_trans.py -- their
#             # presence means this file is almost certainly a PaperBaseline
#             # checkpoint pointed at by the wrong --st_trans_checkpoints
#             # path/filename, NOT a genuinely-older-but-valid STTrans
#             # checkpoint. Silently dropping and proceeding in that case
#             # would build a randomly-initialized STTrans and load a
#             # state_dict for a totally different architecture (near-total
#             # missing/unexpected key mismatch), producing a model that
#             # LOOKS like it loaded (no crash) but is untrained garbage --
#             # exactly the ADE~6000km symptom this warning exists to catch
#             # before it's mistaken for a legitimate eval result.
#             _paper_baseline_keys = _dropped.keys() & {"model_type", "hidden_dim", "n_layers"}
#             if _paper_baseline_keys:
#                 print(f"  🛑 ST-Trans checkpoint's model_cfg contains "
#                       f"{sorted(_paper_baseline_keys)} — these are PaperBaseline "
#                       f"(RNN/GRU/LSTM) architecture keys, NEVER valid for STTrans. "
#                       f"This file is very likely NOT an ST-Trans checkpoint at all "
#                       f"(wrong path in --st_trans_checkpoints?). Proceeding anyway "
#                       f"per current call, but expect load_state_dict to report a "
#                       f"large missing/unexpected key count below if so — if that "
#                       f"happens, the resulting ADE number is meaningless (near-random "
#                       f"weights), not a real evaluation of ST-Trans. Verify the "
#                       f"checkpoint path before trusting any metric from this run.")
#             else:
#                 print(f"  ⚠ ST-Trans checkpoint's model_cfg has keys STTrans.__init__() "
#                       f"doesn't accept — dropping before construction: {list(_dropped.keys())}. "
#                       f"(Likely saved by an older train_st_trans.py; verify this is really "
#                       f"an STTrans non_ar checkpoint, not e.g. STTransAR, if unsure.)")
#         model_cfg = {k: v for k, v in model_cfg.items() if k in _VALID_STTRANS_KEYS}
#         model = STTrans(**model_cfg).to(device)
#     else:
#         print(f"  ⚠ ST-Trans checkpoint has no model_cfg — using "
#               f"CLI-default-matching args (d_model={d_model}, nhead={nhead}, "
#               f"num_dec_layers={num_dec_layers}). Only correct if trained "
#               f"with train_st_trans.py's own defaults.")
#         model = STTrans(obs_len=obs_len, pred_len=pred_len, unet_in_ch=unet_in_ch,
#                          d_model=d_model, nhead=nhead, num_enc_layers=num_enc_layers,
#                          num_dec_layers=num_dec_layers, dim_ff=dim_ff,
#                          dropout=dropout).to(device)
#     state = ck.get("model_state", ck.get("model"))
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     if missing or unexpected:
#         print(f"  ⚠ ST-Trans load_state_dict: {len(missing)} missing, "
#               f"{len(unexpected)} unexpected keys")
#         # [FIX-STTRANS-WRONG-CKPT] A handful of missing/unexpected keys can
#         # legitimately happen (e.g. a minor architecture tweak between
#         # checkpoint versions). A LARGE fraction mismatching means the
#         # state_dict almost certainly belongs to a different architecture
#         # entirely -- most of the model is left at its random init, so any
#         # ADE/ATE/CTE computed from it is meaningless, not a real result.
#         try:
#             total_model_params = len(list(model.state_dict().keys()))
#             mismatch_frac = (len(missing) + len(unexpected)) / max(total_model_params, 1)
#         except Exception:
#             mismatch_frac = 0.0
#         if mismatch_frac > 0.25:
#             print(f"  🛑 {mismatch_frac*100:.0f}% of STTrans's parameter keys did not "
#                   f"match this checkpoint's state_dict -- this is FAR too many for a "
#                   f"minor version difference. The loaded weights are almost entirely "
#                   f"random-initialized, not the trained checkpoint. Any metric reported "
#                   f"below (ADE/ATE/CTE) is NOT a valid evaluation of ST-Trans -- stop and "
#                   f"verify the checkpoint path/architecture before using these numbers.")
#     model.eval()
#     seed = _infer_seed(checkpoint, ck)
#     return model, seed


# @torch.no_grad()
# def evaluate_one_model(model, loader, device, model_name: str,
#                         seed: str = "unknown",
#                         n_ensemble: int = 20,
#                         ddim_steps: Optional[int] = None) -> List[Dict]:
#     """
#     Returns a list of PER-LEAD-TIME records:
#       {"model": name, "seed": seed, "storm": storm_key, "window": idx,
#        "lead_time": t, "ade": .., "ate": .., "cte": .., "obs_speed": ..}
#     One record per (storm, window, lead_time) triple — matches the
#     paper's Table 10 pairing granularity (140 windows x 16 lead-times =
#     2240 matched pairs, i.e. paired PER FORECAST STEP, not averaged over
#     the whole trajectory first).

#     lead_time convention (1-indexed, 1..T; T=pred_len, e.g. 1=6h...12=72h
#     when T=12): this is the SAME convention as generate_paper_report.py's
#     HORIZON_LEAD_TIMES = {"6h":1,...,"72h":12}. ADE (d) has a value for
#     EVERY lead_time 1..T. ATE/CTE do not: there is no heading reference
#     at the very first predicted step, so ate/cte are only defined for
#     lead_time 2..T (None at lead_time=1/6h). [FIX] An earlier version of
#     this loop bounded lead_time by ate/cte's shorter range (T-1 instead
#     of T), which silently dropped the LAST lead_time (T, i.e. 72h when
#     T=12) from ADE too, and additionally mislabeled lead_time=1 as if it
#     were the first step (6h) when it was actually the second step (12h,
#     0-indexed step 1) — both bugs are fixed by this version: ADE now
#     covers the full 1..T range, and lead_time=1 genuinely is the first
#     predicted step.
#     """
#     records = []
#     is_fm = isinstance(model, TCFlowMatching) or hasattr(model, "sigma_inference")

#     for bi, batch in enumerate(loader):
#         bl = move(list(batch), device)
#         gt = bl[1]
#         obs = bl[0]
#         try:
#             tyid_list = bl[15]
#         except IndexError:
#             tyid_list = None

#         try:
#             if is_fm:
#                 # [FIX-ODE-STEPS-MISMATCH] Previously ignored any ddim_steps
#                 # override entirely and always used the checkpoint's own
#                 # self.n_inference_steps. Now CLI-configurable via --ddim_steps
#                 # to match evaluate_full.py's convention (None = defer to
#                 # checkpoint's trained value, same as before if not passed).
#                 pred, _, _ = model.sample(bl, num_ensemble=n_ensemble, ddim_steps=ddim_steps)
#             else:
#                 pred, _, _ = model.sample(bl, num_ensemble=1)
#         except Exception as e:
#             print(f"  [{model_name}] batch {bi}: sample error: {e}")
#             continue

#         T = min(pred.shape[0], gt.shape[0])
#         pd = _norm_to_deg(pred[:T])
#         gd = _norm_to_deg(gt[:T, :, :2])
#         d  = _haversine_deg(pd, gd)                  # [T, B] -- steps 0..T-1 (0=6h ... T-1=72h when T=12)
#         ate, cte = ate_cte_full(pd, gd)               # [T-1, B] -- ate[k] = error at step k+1 (0-indexed)
#         T_valid = ate.shape[0]                        # = T-1

#         obs_deg = _norm_to_deg(obs[:, :, :2])
#         if obs_deg.shape[0] >= 2:
#             step_km = _haversine_deg(obs_deg[:-1], obs_deg[1:])
#             obs_speed = step_km.mean(0) / 6.0
#         else:
#             obs_speed = torch.zeros(obs.shape[1], device=device)

#         B = obs.shape[1]
#         for b in range(B):
#             if tyid_list is not None and b < len(tyid_list) and \
#                isinstance(tyid_list[b], dict) and "old" in tyid_list[b]:
#                 info = tyid_list[b]
#                 storm_key = f"{info['old'][1]}_{info['old'][0]}"
#             else:
#                 storm_key = f"UNKNOWN_batch{bi}"
#             # [FIX] Bug thật đã tìm ra: trước đây vòng lặp chạy
#             # `for i in range(T_valid)` (T_valid = T-1), với
#             # lead_time = i+1 (i=0..T_valid-1 => lead_time=1..T_valid=1..T-1).
#             # Với T=12, lead_time chỉ chạy 1..11 -- KHÔNG BAO GIỜ đạt 12.
#             # generate_paper_report.py's HORIZON_LEAD_TIMES tra "72h"->12,
#             # nên luôn ra n=0 ở 72h (khớp đúng hiện tượng đã quan sát).
#             # Đồng thời "lead_time=1" trước đây thực chất ứng 0-indexed
#             # step 1 (=12h theo evaluate_full.py's HORIZONS convention),
#             # KHÔNG PHẢI 6h -- tên horizon "6h" ở nơi đọc dữ liệu cũng bị
#             # lệch 1 bước so với dữ liệu thật.
#             #
#             # Sửa: lead_time giờ là 1-indexed THẬT trên toàn bộ T bước
#             # (lead_time = step_0indexed + 1, chạy 1..T, tức 1=6h...T=72h
#             # khi T=12) -- khớp đúng HORIZON_LEAD_TIMES = {"6h":1,...,
#             # "72h":12} sau khi sửa ở generate_paper_report.py.
#             # ADE (d) có đủ giá trị cho MỌI lead_time 1..T.
#             # ATE/CTE (ate/cte) chỉ có giá trị cho lead_time 2..T (không
#             # định nghĩa được ở lead_time=1/6h, vì cần bước trước đó để
#             # biết hướng đi) -- ghi None thay vì bỏ hẳn record.
#             for step0 in range(T):           # step0 = 0-indexed step, 0..T-1
#                 lead_time = step0 + 1        # 1-indexed, 1..T (1=6h...T=72h)
#                 has_atecte = step0 >= 1      # ate/cte defined for step0=1..T-1
#                 ate_i = step0 - 1            # ate/cte array index when has_atecte
#                 records.append({
#                     "model":     model_name,
#                     "seed":      seed,
#                     "storm":     storm_key,
#                     "window":    b,
#                     "lead_time": lead_time,
#                     "ade":       float(d[step0, b]),
#                     "ate":       float(ate[ate_i, b].abs()) if has_atecte else None,
#                     "cte":       float(cte[ate_i, b].abs()) if has_atecte else None,
#                     "obs_speed": float(obs_speed[b]),
#                 })
#     return records


# # [FIX-DETERMINISM] Mirrors evaluate_full.py / visual_evaluate_mode.py's
# # set_seed(): this script never seeded RNGs before model.sample()'s
# # K-candidate torch.randn(...) draw, so repeated runs (and comparisons
# # against visualize's fixed-seed output) were not reproducible for FM.
# def set_seed(s: int = 42):
#     random.seed(s)
#     np.random.seed(s)
#     torch.manual_seed(s)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(s)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark     = False


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--seed", type=int, default=42,
#                    help="[FIX-DETERMINISM] RNG seed applied before any model.sample() call; "
#                         "matches visual_evaluate_mode.py's fixed seed(42) convention.")
#     p.add_argument("--dataset_root", required=True)
#     p.add_argument("--split", default="test", choices=["test", "val", "train"])
#     p.add_argument("--output_dir", default="eval_multi")
#     p.add_argument("--gpu", type=int, default=0)
#     p.add_argument("--n_ensemble", type=int, default=20)
#     p.add_argument("--ddim_steps", type=int, default=None,
#                    help="[FIX-ODE-STEPS-MISMATCH] Number of ODE integration steps for "
#                         "FM's model.sample(). Was previously not exposed here at all "
#                         "(evaluate_one_model always used the checkpoint's own trained "
#                         "n_inference_steps, silently ignoring any intended override). "
#                         "Default None matches evaluate_full.py's --ddim_steps convention: "
#                         "defer to the checkpoint's own value unless explicitly set.")
#     p.add_argument("--test_year", type=int, default=None)

#     p.add_argument("--fm_checkpoints",       nargs="+", default=None,
#                    help="One or more FM checkpoint paths, one per seed")
#     p.add_argument("--st_trans_checkpoints", nargs="+", default=None,
#                    help="One or more ST-Trans checkpoint paths, one per seed")
#     p.add_argument("--lstm_checkpoints",     nargs="+", default=None,
#                    help="One or more LSTM checkpoint paths, one per seed")
#     p.add_argument("--gru_checkpoints",      nargs="+", default=None,
#                    help="One or more GRU checkpoint paths, one per seed")
#     p.add_argument("--rnn_checkpoints",      nargs="+", default=None,
#                    help="One or more RNN checkpoint paths, one per seed")
#     # Backward-compat singular aliases (old single-checkpoint usage still works)
#     p.add_argument("--fm_checkpoint",       default=None, help="[legacy] single checkpoint, use --fm_checkpoints instead")
#     p.add_argument("--st_trans_checkpoint", default=None, help="[legacy] single checkpoint")
#     p.add_argument("--lstm_checkpoint",     default=None, help="[legacy] single checkpoint")
#     p.add_argument("--gru_checkpoint",      default=None, help="[legacy] single checkpoint")
#     p.add_argument("--rnn_checkpoint",      default=None, help="[legacy] single checkpoint")

#     p.add_argument("--paper_hidden_dim", type=int, default=256)
#     p.add_argument("--paper_n_layers",   type=int, default=3)
#     p.add_argument("--paper_dropout",    type=float, default=0.20)
#     p.add_argument("--st_d_model",        type=int, default=64)
#     p.add_argument("--st_nhead",          type=int, default=4)
#     p.add_argument("--st_num_enc_layers", type=int, default=1)
#     p.add_argument("--st_num_dec_layers", type=int, default=3)
#     p.add_argument("--st_dim_ff",         type=int, default=512)
#     p.add_argument("--st_dropout",        type=float, default=0.1)

#     args = p.parse_args()
#     set_seed(args.seed)   # [FIX-DETERMINISM] must run before any model.sample() call below
#     os.makedirs(args.output_dir, exist_ok=True)
#     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

#     print(f"  Loading test data...")
#     import argparse as _ap
#     _loader_args = _ap.Namespace(
#         dataset_root = args.dataset_root,
#         obs_len      = 8,
#         pred_len     = 12,
#         batch_size   = 64,
#         num_workers  = 2,
#         test_year    = args.test_year,
#         skip         = 1,
#         min_ped      = 1,
#         threshold    = 0.002,
#     )
#     _, loader = data_loader(_loader_args,
#                              {"root": args.dataset_root, "type": args.split},
#                              test=(args.split != "train"))
#     print(f"  Data: {len(loader)} batches")

#     def _collect(multi, single):
#         """Merge --xxx_checkpoints (list) and legacy --xxx_checkpoint (str) into one list."""
#         paths = list(multi) if multi else []
#         if single and single not in paths:
#             paths.append(single)
#         return paths

#     jobs = []  # (display_name, kind, checkpoint_path)
#     for display_name, kind, multi, single in [
#         ("FM",       "fm",       args.fm_checkpoints,       args.fm_checkpoint),
#         ("ST-Trans", "st_trans", args.st_trans_checkpoints, args.st_trans_checkpoint),
#         ("LSTM",     "lstm",     args.lstm_checkpoints,     args.lstm_checkpoint),
#         ("GRU",      "gru",      args.gru_checkpoints,      args.gru_checkpoint),
#         ("RNN",      "rnn",      args.rnn_checkpoints,      args.rnn_checkpoint),
#     ]:
#         for ckpt_path in _collect(multi, single):
#             jobs.append((display_name, kind, ckpt_path))

#     if not jobs:
#         print("  No checkpoints given — nothing to do.")
#         return

#     all_records = []
#     for display_name, kind, ckpt_path in jobs:
#         print(f"\n  {'='*70}\n  Loading {display_name}: {ckpt_path}\n  {'='*70}")
#         if kind == "fm":
#             model, seed = load_fm(ckpt_path, device)
#         elif kind == "st_trans":
#             model, seed = load_st_trans(ckpt_path, device,
#                                    d_model=args.st_d_model, nhead=args.st_nhead,
#                                    num_enc_layers=args.st_num_enc_layers,
#                                    num_dec_layers=args.st_num_dec_layers,
#                                    dim_ff=args.st_dim_ff, dropout=args.st_dropout)
#         else:
#             model, seed = load_paper_baseline(ckpt_path, kind, device,
#                                          hidden_dim=args.paper_hidden_dim,
#                                          n_layers=args.paper_n_layers,
#                                          dropout=args.paper_dropout)

#         n_params = sum(pm.numel() for pm in model.parameters())
#         print(f"  {display_name} (seed={seed}): {n_params:,} params")

#         recs = evaluate_one_model(model, loader, device, display_name,
#                                    seed=seed, n_ensemble=args.n_ensemble,
#                                    ddim_steps=args.ddim_steps)
#         all_records.extend(recs)

#         # [FIX] ate/cte là None ở lead_time=1 (6h) theo convention đã sửa
#         # (xem evaluate_one_model's docstring) — np.mean crash nếu None
#         # lẫn trong list. Lọc trước khi tính, giống mọi chỗ khác trong
#         # generate_paper_report.py đã áp dụng cùng bộ lọc này.
#         ade = np.mean([r["ade"] for r in recs if r["ade"] is not None])
#         ate_vals = [r["ate"] for r in recs if r["ate"] is not None]
#         cte_vals = [r["cte"] for r in recs if r["cte"] is not None]
#         ate = np.mean(ate_vals) if ate_vals else float("nan")
#         cte = np.mean(cte_vals) if cte_vals else float("nan")
#         print(f"  {display_name} seed={seed}: n={len(recs)}  ADE={ade:.2f}  "
#               f"ATE={ate:.2f}  CTE={cte:.2f}")

#         del model
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     out_path = os.path.join(args.output_dir, f"multi_model_{args.split}.json")
#     with open(out_path, "w") as f:
#         json.dump(all_records, f, indent=2)
#     print(f"\n  Saved {len(all_records)} records → {out_path}")
#     print(f"  Run generate_comparison_table.py --records {out_path} "
#           f"to produce the Table-10-style significance table.")


# if __name__ == "__main__":
#     main()
"""
evaluate_multi_model.py
=========================
Runs test-set evaluation for RNN / GRU / LSTM / ST-Trans / FM / MGTCF /
Phys-Diff checkpoints, using ONE SHARED ATE/CTE formula for all models.

WHY A SEPARATE SCRIPT, NOT REUSING EACH MODEL'S OWN METRIC CODE
-----------------------------------------------------------------
paper_baseline_model.py (used by RNN/GRU/LSTM/ST-Trans) has its own
_ate_cte_tensors() with a DIFFERENT convention than flow_matching_model.py's
_ate_cte_full() (used by FM):
  - flat-earth approx (111*cos(lat)) vs haversine — small effect (~0.2%)
  - reference heading computed OUTGOING from each point (gt[t]->gt[t+1])
    vs INCOMING to each point (gt[t-1]->gt[t]) — this is NOT a small
    effect: verified numerically on a synthetic turning trajectory, the
    two conventions can even disagree on the SIGN of CTE at a given step.
Comparing FM's ATE/CTE (computed one way) against RNN/GRU/LSTM/ST-Trans's
ATE/CTE (computed the other way) would not be a fair like-for-like
comparison — any difference could be an artifact of the formula, not the
model. This script computes ADE/ATE/CTE for ALL SEVEN models using the
SAME function (_ate_cte_full, _haversine_deg, _forward_azimuth, imported
from flow_matching_model.py — the version with the off-by-one fix already
verified in evaluate_full.py), so Table-10-style comparisons are sound.
MGTCF (Model/mgtcf_model.py) and Phys-Diff (Model/physdiff_model.py) use
this exact same shared formula too — no separate ATE/CTE code of their
own, so they slot into the comparison with the same guarantee.

USAGE
-----
python evaluate_multi_model.py \
    --dataset_root <root> \
    --fm_checkpoint runs/fm_seed42/best_model.pth \
    --st_trans_checkpoint runs/st_trans/best_model.pth \
    --lstm_checkpoint runs/lstm/best_model.pth \
    --gru_checkpoint runs/gru/best_model.pth \
    --rnn_checkpoint runs/rnn/best_model.pth \
    --mgtcf_checkpoint runs/mgtcf/best_model.pth \
    --physdiff_checkpoint runs/physdiff/best_model.pth \
    --output_dir eval_multi/

Any checkpoint arg can be omitted to skip that model. Produces one
combined JSON of per-window records (one row per (model, storm, window))
that generate_comparison_table.py consumes directly for the Table-10-style
statistical significance tables.
"""
from __future__ import annotations
import sys, os, argparse, json, random
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _haversine_deg, _forward_azimuth, _unwrap,
)
from Model.paper_baseline_model import PaperBaseline, MODEL_TYPES
from Model.st_trans_model import STTrans
# [MỚI] MGTCF (multi-generator GAN, Huang et al. 2023/2025) và Phys-Diff
# (latent diffusion + PIGA, Liu et al. 2026) — cả 2 đều là generative
# model như FM, nên dùng chung nhóm "ensemble" (num_ensemble=K) khi gọi
# .sample(), thay vì num_ensemble=1 như các baseline deterministic
# (LSTM/GRU/RNN/ST-Trans) — xem is_ensemble_model() bên dưới.
from Model.mgtcf_model import MGTCFModel
from Model.physdiff_model import PhysDiffModel


def _infer_seed(checkpoint_path: str, ck: dict) -> str:
    """
    Best-effort seed extraction: prefer an explicit 'seed' key saved in
    the checkpoint (train_flowmatching.py / train_st_trans.py /
    train_paper_baseline.py all save this per the project's multi-seed
    convention). Falls back to parsing "_seed<N>" from the path/dirname
    (the convention those same scripts use for output_dir), and finally
    "unknown" if neither is found — records with seed="unknown" will
    NOT be treated as genuinely pooled multi-seed data by
    generate_paper_table.py (it warns explicitly in that case).
    """
    if isinstance(ck, dict) and "seed" in ck:
        return str(ck["seed"])
    import re
    m = re.search(r"seed[_-]?(\d+)", checkpoint_path)
    if m:
        return m.group(1)
    return "unknown"


def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


def ate_cte_full(pred_deg: torch.Tensor, gt_deg: torch.Tensor):
    """
    SAME formula as evaluate_full.py's _ate_cte_full (off-by-one already
    fixed there — see that file's own comments for the fix history).
    Returns [T-1, B] each; index k holds the error at ORIGINAL step k+1.
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    bear_ref = _forward_azimuth(gt_deg[:T - 1], gt_deg[1:T])
    bear_err = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
    dist_err = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang = bear_err - bear_ref
    return dist_err * torch.cos(ang), dist_err * torch.sin(ang)


def load_fm(checkpoint: str, device):
    ck = torch.load(checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg") or {}
    if not model_cfg:
        print(f"  ⚠ FM checkpoint has no model_cfg — using constructor "
              f"defaults (correct only if trained with default architecture).")
    model = TCFlowMatching(**model_cfg).to(device)
    state = ck.get("model", ck.get("model_state"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ FM load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
    model.eval()
    seed = _infer_seed(checkpoint, ck)
    return model, seed


def load_paper_baseline(checkpoint: str, model_type: str, device,
                         hidden_dim: int = 256, n_layers: int = 3,
                         obs_len: int = 8, pred_len: int = 12,
                         unet_in_ch: int = 13, dropout: float = 0.20):
    """
    [UPDATED] train_paper_baseline.py now saves a full model_cfg dict in
    newer checkpoints (added alongside --seed support). If present, it is
    used directly and the CLI-default args above are ignored — this is
    the reliable path. If absent (checkpoint trained before that fix),
    falls back to the CLI-default-matching values passed in here, with a
    warning, same caveat as FM's missing-model_cfg case.
    """
    assert model_type in MODEL_TYPES, f"model_type must be one of {MODEL_TYPES}"
    ck = torch.load(checkpoint, map_location="cpu")
    saved_type = ck.get("model_type", model_type)
    if saved_type != model_type:
        print(f"  ⚠ Checkpoint's saved model_type='{saved_type}' differs "
              f"from requested '{model_type}' — using requested value. "
              f"Verify this checkpoint is really the {model_type.upper()} one.")

    model_cfg = ck.get("model_cfg")
    if model_cfg:
        model = PaperBaseline(**model_cfg).to(device)
    else:
        print(f"  ⚠ {model_type.upper()} checkpoint has no model_cfg — "
              f"using CLI-default-matching args (hidden_dim={hidden_dim}, "
              f"n_layers={n_layers}, dropout={dropout}). Only correct if "
              f"trained with train_paper_baseline.py's own defaults.")
        model = PaperBaseline(model_type=model_type, pred_len=pred_len,
                               obs_len=obs_len, hidden_dim=hidden_dim,
                               n_layers=n_layers, unet_in_ch=unet_in_ch,
                               dropout=dropout).to(device)
    state = ck.get("model_state", ck.get("model"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ {model_type.upper()} load_state_dict: {len(missing)} "
              f"missing, {len(unexpected)} unexpected keys")
    model.eval()
    seed = _infer_seed(checkpoint, ck)
    return model, seed


def load_st_trans(checkpoint: str, device,
                   obs_len: int = 8, pred_len: int = 12, unet_in_ch: int = 13,
                   d_model: int = 64, nhead: int = 4, num_enc_layers: int = 1,
                   num_dec_layers: int = 3, dim_ff: int = 512, dropout: float = 0.1):
    """
    [UPDATED] Same model_cfg pattern as load_paper_baseline. Note STTrans
    (non_ar) and STTransAR have DIFFERENT constructor signatures (AR has
    no num_dec_layers) — the saved model_cfg already accounts for this
    (see train_st_trans.py's checkpoint-save branch), so **kwargs here
    naturally works for either. This loader only instantiates STTrans
    (non-AR) — STTransAR support would need its own loader if you also
    want to evaluate that variant.

    [FIX-STTRANS-MODEL-CFG] Checkpoints saved by an OLDER version of
    train_st_trans.py (before the "STTrans vs STTransAR have different
    ctor signatures" fix was added there) can have a model_cfg dict that
    includes extra keys STTrans.__init__() does not accept -- observed in
    practice: 'model_type' (TypeError: STTrans.__init__() got an
    unexpected keyword argument 'model_type'). Current train_st_trans.py
    saves model_type as a SEPARATE top-level checkpoint key (ck['model_type']),
    not inside model_cfg, so a clean checkpoint should not hit this -- but
    since we cannot control what checkpoint file the user points this at
    (e.g. one trained before that convention existed), filter model_cfg
    down to exactly STTrans's known-valid non_ar constructor keys before
    calling STTrans(**model_cfg), rather than trusting the checkpoint's
    model_cfg blindly. Any dropped key is reported so a real config
    mismatch (not just a harmless stale extra key) is still visible.
    """
    ck = torch.load(checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg")
    if model_cfg:
        _VALID_STTRANS_KEYS = {
            "obs_len", "pred_len", "unet_in_ch", "d_model", "nhead",
            "num_enc_layers", "num_dec_layers", "dim_ff", "dropout",
        }
        _dropped = {k: v for k, v in model_cfg.items() if k not in _VALID_STTRANS_KEYS}
        if _dropped:
            # [FIX-STTRANS-WRONG-CKPT] 'hidden_dim'/'n_layers' are
            # PaperBaseline-specific (RNN/GRU/LSTM), never valid STTrans
            # ctor args under ANY version of train_st_trans.py -- their
            # presence means this file is almost certainly a PaperBaseline
            # checkpoint pointed at by the wrong --st_trans_checkpoints
            # path/filename, NOT a genuinely-older-but-valid STTrans
            # checkpoint. Silently dropping and proceeding in that case
            # would build a randomly-initialized STTrans and load a
            # state_dict for a totally different architecture (near-total
            # missing/unexpected key mismatch), producing a model that
            # LOOKS like it loaded (no crash) but is untrained garbage --
            # exactly the ADE~6000km symptom this warning exists to catch
            # before it's mistaken for a legitimate eval result.
            _paper_baseline_keys = _dropped.keys() & {"model_type", "hidden_dim", "n_layers"}
            if _paper_baseline_keys:
                print(f"  🛑 ST-Trans checkpoint's model_cfg contains "
                      f"{sorted(_paper_baseline_keys)} — these are PaperBaseline "
                      f"(RNN/GRU/LSTM) architecture keys, NEVER valid for STTrans. "
                      f"This file is very likely NOT an ST-Trans checkpoint at all "
                      f"(wrong path in --st_trans_checkpoints?). Proceeding anyway "
                      f"per current call, but expect load_state_dict to report a "
                      f"large missing/unexpected key count below if so — if that "
                      f"happens, the resulting ADE number is meaningless (near-random "
                      f"weights), not a real evaluation of ST-Trans. Verify the "
                      f"checkpoint path before trusting any metric from this run.")
            else:
                print(f"  ⚠ ST-Trans checkpoint's model_cfg has keys STTrans.__init__() "
                      f"doesn't accept — dropping before construction: {list(_dropped.keys())}. "
                      f"(Likely saved by an older train_st_trans.py; verify this is really "
                      f"an STTrans non_ar checkpoint, not e.g. STTransAR, if unsure.)")
        model_cfg = {k: v for k, v in model_cfg.items() if k in _VALID_STTRANS_KEYS}
        model = STTrans(**model_cfg).to(device)
    else:
        print(f"  ⚠ ST-Trans checkpoint has no model_cfg — using "
              f"CLI-default-matching args (d_model={d_model}, nhead={nhead}, "
              f"num_dec_layers={num_dec_layers}). Only correct if trained "
              f"with train_st_trans.py's own defaults.")
        model = STTrans(obs_len=obs_len, pred_len=pred_len, unet_in_ch=unet_in_ch,
                         d_model=d_model, nhead=nhead, num_enc_layers=num_enc_layers,
                         num_dec_layers=num_dec_layers, dim_ff=dim_ff,
                         dropout=dropout).to(device)
    state = ck.get("model_state", ck.get("model"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ ST-Trans load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
        # [FIX-STTRANS-WRONG-CKPT] A handful of missing/unexpected keys can
        # legitimately happen (e.g. a minor architecture tweak between
        # checkpoint versions). A LARGE fraction mismatching means the
        # state_dict almost certainly belongs to a different architecture
        # entirely -- most of the model is left at its random init, so any
        # ADE/ATE/CTE computed from it is meaningless, not a real result.
        try:
            total_model_params = len(list(model.state_dict().keys()))
            mismatch_frac = (len(missing) + len(unexpected)) / max(total_model_params, 1)
        except Exception:
            mismatch_frac = 0.0
        if mismatch_frac > 0.25:
            print(f"  🛑 {mismatch_frac*100:.0f}% of STTrans's parameter keys did not "
                  f"match this checkpoint's state_dict -- this is FAR too many for a "
                  f"minor version difference. The loaded weights are almost entirely "
                  f"random-initialized, not the trained checkpoint. Any metric reported "
                  f"below (ADE/ATE/CTE) is NOT a valid evaluation of ST-Trans -- stop and "
                  f"verify the checkpoint path/architecture before using these numbers.")
    model.eval()
    seed = _infer_seed(checkpoint, ck)
    return model, seed


def load_mgtcf(checkpoint: str, device,
               n_generators: int = 6, embedding_dim: int = 64,
               encoder_h_dim: int = 64, decoder_h_dim: int = 128,
               noise_dim: int = 8, disc_h_dim: int = 64, disc_mlp_dim: int = 256,
               obs_len: int = 8, pred_len: int = 12, unet_in_ch: int = 13,
               dropout: float = 0.0, best_k: int = 6):
    """
    [MỚI] Load MGTCFModel checkpoint (train_mgtcf.py). Cùng phong cách
    guard/cảnh báo với load_st_trans: model_cfg thiếu -> dùng CLI-default
    args với cảnh báo rõ; load_state_dict mismatch lớn -> cảnh báo mạnh
    (không chỉ im lặng cho ra ADE vô nghĩa).

    LƯU Ý QUAN TRỌNG (khác load_paper_baseline/load_st_trans): MGTCFModel
    KHÔNG nhận tham số "model_type" trong constructor (khác PaperBaseline).
    train_mgtcf.py lưu model_type="MGTCF" (PascalCase) ở ck["model_type"]
    CẤP NGOÀI model_cfg, không phải bên trong model_cfg như PaperBaseline
    — nên KHÔNG cần .pop("model_type") ở đây vì model_cfg vốn dĩ đã
    không chứa key này (đã verify trực tiếp trong train_mgtcf.py).
    """
    ck = torch.load(checkpoint, map_location="cpu")
    saved_type = ck.get("model_type", "MGTCF")
    if saved_type.lower() != "mgtcf":
        print(f"  ⚠ Checkpoint's saved model_type='{saved_type}' is not "
              f"'MGTCF' — verify this checkpoint path is really the MGTCF one.")

    model_cfg = ck.get("model_cfg")
    if model_cfg:
        model = MGTCFModel(**model_cfg).to(device)
    else:
        print(f"  ⚠ MGTCF checkpoint has no model_cfg — using CLI-default"
              f"-matching args (n_generators={n_generators}, "
              f"embedding_dim={embedding_dim}). Only correct if trained "
              f"with train_mgtcf.py's own defaults.")
        model = MGTCFModel(obs_len=obs_len, pred_len=pred_len, unet_in_ch=unet_in_ch,
                            n_generators=n_generators, embedding_dim=embedding_dim,
                            encoder_h_dim=encoder_h_dim, decoder_h_dim=decoder_h_dim,
                            noise_dim=noise_dim, disc_h_dim=disc_h_dim,
                            disc_mlp_dim=disc_mlp_dim, dropout=dropout,
                            best_k=best_k).to(device)

    state = ck.get("model_state", ck.get("model"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ MGTCF load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
        try:
            total_model_params = len(list(model.state_dict().keys()))
            mismatch_frac = (len(missing) + len(unexpected)) / max(total_model_params, 1)
        except Exception:
            mismatch_frac = 0.0
        if mismatch_frac > 0.25:
            print(f"  🛑 {mismatch_frac*100:.0f}% of MGTCF's parameter keys did not "
                  f"match this checkpoint's state_dict -- loaded weights are almost "
                  f"entirely random-initialized. Any ADE/ATE/CTE reported below is "
                  f"NOT a valid evaluation -- verify the checkpoint path/architecture "
                  f"(especially --mgtcf_* CLI args matching the ones train_mgtcf.py "
                  f"was actually run with, e.g. --n_generators) before trusting it.")
    model.eval()
    seed = _infer_seed(checkpoint, ck)
    return model, seed


def load_physdiff(checkpoint: str, device,
                  d_model: int = 64, nhead: int = 4, num_blocks: int = 3,
                  dim_ff: int = 256, dropout: float = 0.1,
                  obs_len: int = 8, pred_len: int = 12, unet_in_ch: int = 13,
                  T_diffusion: int = 1000, beta_start: float = 1e-4,
                  beta_end: float = 0.02, n_sample_steps: int = 50):
    """
    [MỚI] Load PhysDiffModel checkpoint (train_physdiff.py). Cùng
    convention lưu checkpoint như MGTCF: model_type ở ck["model_type"]
    cấp ngoài ("PhysDiff"), không nằm trong model_cfg.

    LƯU Ý: n_sample_steps trong model_cfg đã lưu có thể là giá trị dùng
    lúc TRAINING (thường nhỏ hơn, để validation nhanh) -- nếu muốn dùng
    số bước reverse-diffusion nhiều hơn cho đánh giá cuối cùng (chất
    lượng cao hơn, giống --test_n_sample_steps của train_physdiff.py),
    override model.n_sample_steps SAU khi load, xem cách gọi trong
    main() bên dưới.
    """
    ck = torch.load(checkpoint, map_location="cpu")
    saved_type = ck.get("model_type", "PhysDiff")
    if saved_type.lower() != "physdiff":
        print(f"  ⚠ Checkpoint's saved model_type='{saved_type}' is not "
              f"'PhysDiff' — verify this checkpoint path is really the Phys-Diff one.")

    model_cfg = ck.get("model_cfg")
    if model_cfg:
        model = PhysDiffModel(**model_cfg).to(device)
    else:
        print(f"  ⚠ Phys-Diff checkpoint has no model_cfg — using CLI-default"
              f"-matching args (d_model={d_model}, num_blocks={num_blocks}). "
              f"Only correct if trained with train_physdiff.py's own defaults.")
        model = PhysDiffModel(obs_len=obs_len, pred_len=pred_len, unet_in_ch=unet_in_ch,
                              d_model=d_model, nhead=nhead, num_blocks=num_blocks,
                              dim_ff=dim_ff, dropout=dropout, T_diffusion=T_diffusion,
                              beta_start=beta_start, beta_end=beta_end,
                              n_sample_steps=n_sample_steps).to(device)

    state = ck.get("model_state", ck.get("model"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠ Phys-Diff load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
        try:
            total_model_params = len(list(model.state_dict().keys()))
            mismatch_frac = (len(missing) + len(unexpected)) / max(total_model_params, 1)
        except Exception:
            mismatch_frac = 0.0
        if mismatch_frac > 0.25:
            print(f"  🛑 {mismatch_frac*100:.0f}% of Phys-Diff's parameter keys did not "
                  f"match this checkpoint's state_dict -- loaded weights are almost "
                  f"entirely random-initialized. Any ADE/ATE/CTE reported below is NOT "
                  f"a valid evaluation -- verify the checkpoint path/architecture "
                  f"before trusting it.")
    model.eval()
    seed = _infer_seed(checkpoint, ck)
    return model, seed


@torch.no_grad()
def evaluate_one_model(model, loader, device, model_name: str,
                        seed: str = "unknown",
                        n_ensemble: int = 20,
                        ddim_steps: Optional[int] = None) -> List[Dict]:
    """
    Returns a list of PER-LEAD-TIME records:
      {"model": name, "seed": seed, "storm": storm_key, "window": idx,
       "lead_time": t, "ade": .., "ate": .., "cte": .., "obs_speed": ..}
    One record per (storm, window, lead_time) triple — matches the
    paper's Table 10 pairing granularity (140 windows x 16 lead-times =
    2240 matched pairs, i.e. paired PER FORECAST STEP, not averaged over
    the whole trajectory first).

    lead_time convention (1-indexed, 1..T; T=pred_len, e.g. 1=6h...12=72h
    when T=12): this is the SAME convention as generate_paper_report.py's
    HORIZON_LEAD_TIMES = {"6h":1,...,"72h":12}. ADE (d) has a value for
    EVERY lead_time 1..T. ATE/CTE do not: there is no heading reference
    at the very first predicted step, so ate/cte are only defined for
    lead_time 2..T (None at lead_time=1/6h). [FIX] An earlier version of
    this loop bounded lead_time by ate/cte's shorter range (T-1 instead
    of T), which silently dropped the LAST lead_time (T, i.e. 72h when
    T=12) from ADE too, and additionally mislabeled lead_time=1 as if it
    were the first step (6h) when it was actually the second step (12h,
    0-indexed step 1) — both bugs are fixed by this version: ADE now
    covers the full 1..T range, and lead_time=1 genuinely is the first
    predicted step.
    """
    records = []
    # [MỚI] Mở rộng từ "is_fm" thành "is_ensemble_model": MGTCF (Roulette
    # sampling qua GC-Net) và Phys-Diff (reverse diffusion từ nhiều
    # Gaussian noise init khác nhau) đều là generative model có khả
    # năng sinh ensemble THẬT (khác LSTM/GRU/RNN/ST-Trans, deterministic,
    # num_ensemble>1 vô nghĩa với chúng) -- nên dùng num_ensemble=K=20
    # giống FM, KHÔNG xếp chung nhóm "else" (num_ensemble=1) như trước.
    is_ensemble_model = (
        isinstance(model, (TCFlowMatching, MGTCFModel, PhysDiffModel))
        or hasattr(model, "sigma_inference")
    )
    is_fm = isinstance(model, TCFlowMatching) or hasattr(model, "sigma_inference")

    for bi, batch in enumerate(loader):
        bl = move(list(batch), device)
        gt = bl[1]
        obs = bl[0]
        try:
            tyid_list = bl[15]
        except IndexError:
            tyid_list = None

        try:
            if is_fm:
                # [FIX-ODE-STEPS-MISMATCH] Previously ignored any ddim_steps
                # override entirely and always used the checkpoint's own
                # self.n_inference_steps. Now CLI-configurable via --ddim_steps
                # to match evaluate_full.py's convention (None = defer to
                # checkpoint's trained value, same as before if not passed).
                pred, _, _ = model.sample(bl, num_ensemble=n_ensemble, ddim_steps=ddim_steps)
            elif is_ensemble_model:
                # [MỚI] MGTCF/Phys-Diff: sample() không nhận ddim_steps (đó
                # là tham số riêng của FM's CFM ODE integration) -- chỉ
                # truyền num_ensemble=K, giống tinh thần "best-of-K" mà cả
                # 2 paper gốc báo cáo (MGTCF-Ens/Phys-Diff N=50 ensemble).
                pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
            else:
                pred, _, _ = model.sample(bl, num_ensemble=1)
        except Exception as e:
            print(f"  [{model_name}] batch {bi}: sample error: {e}")
            continue

        T = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T])
        gd = _norm_to_deg(gt[:T, :, :2])
        d  = _haversine_deg(pd, gd)                  # [T, B] -- steps 0..T-1 (0=6h ... T-1=72h when T=12)
        ate, cte = ate_cte_full(pd, gd)               # [T-1, B] -- ate[k] = error at step k+1 (0-indexed)
        T_valid = ate.shape[0]                        # = T-1

        obs_deg = _norm_to_deg(obs[:, :, :2])
        if obs_deg.shape[0] >= 2:
            step_km = _haversine_deg(obs_deg[:-1], obs_deg[1:])
            obs_speed = step_km.mean(0) / 6.0
        else:
            obs_speed = torch.zeros(obs.shape[1], device=device)

        B = obs.shape[1]
        for b in range(B):
            if tyid_list is not None and b < len(tyid_list) and \
               isinstance(tyid_list[b], dict) and "old" in tyid_list[b]:
                info = tyid_list[b]
                storm_key = f"{info['old'][1]}_{info['old'][0]}"
            else:
                storm_key = f"UNKNOWN_batch{bi}"
            # [FIX] Bug thật đã tìm ra: trước đây vòng lặp chạy
            # `for i in range(T_valid)` (T_valid = T-1), với
            # lead_time = i+1 (i=0..T_valid-1 => lead_time=1..T_valid=1..T-1).
            # Với T=12, lead_time chỉ chạy 1..11 -- KHÔNG BAO GIỜ đạt 12.
            # generate_paper_report.py's HORIZON_LEAD_TIMES tra "72h"->12,
            # nên luôn ra n=0 ở 72h (khớp đúng hiện tượng đã quan sát).
            # Đồng thời "lead_time=1" trước đây thực chất ứng 0-indexed
            # step 1 (=12h theo evaluate_full.py's HORIZONS convention),
            # KHÔNG PHẢI 6h -- tên horizon "6h" ở nơi đọc dữ liệu cũng bị
            # lệch 1 bước so với dữ liệu thật.
            #
            # Sửa: lead_time giờ là 1-indexed THẬT trên toàn bộ T bước
            # (lead_time = step_0indexed + 1, chạy 1..T, tức 1=6h...T=72h
            # khi T=12) -- khớp đúng HORIZON_LEAD_TIMES = {"6h":1,...,
            # "72h":12} sau khi sửa ở generate_paper_report.py.
            # ADE (d) có đủ giá trị cho MỌI lead_time 1..T.
            # ATE/CTE (ate/cte) chỉ có giá trị cho lead_time 2..T (không
            # định nghĩa được ở lead_time=1/6h, vì cần bước trước đó để
            # biết hướng đi) -- ghi None thay vì bỏ hẳn record.
            for step0 in range(T):           # step0 = 0-indexed step, 0..T-1
                lead_time = step0 + 1        # 1-indexed, 1..T (1=6h...T=72h)
                has_atecte = step0 >= 1      # ate/cte defined for step0=1..T-1
                ate_i = step0 - 1            # ate/cte array index when has_atecte
                records.append({
                    "model":     model_name,
                    "seed":      seed,
                    "storm":     storm_key,
                    "window":    b,
                    "lead_time": lead_time,
                    "ade":       float(d[step0, b]),
                    "ate":       float(ate[ate_i, b].abs()) if has_atecte else None,
                    "cte":       float(cte[ate_i, b].abs()) if has_atecte else None,
                    "obs_speed": float(obs_speed[b]),
                })
    return records


# [FIX-DETERMINISM] Mirrors evaluate_full.py / visual_evaluate_mode.py's
# set_seed(): this script never seeded RNGs before model.sample()'s
# K-candidate torch.randn(...) draw, so repeated runs (and comparisons
# against visualize's fixed-seed output) were not reproducible for FM.
def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42,
                   help="[FIX-DETERMINISM] RNG seed applied before any model.sample() call; "
                        "matches visual_evaluate_mode.py's fixed seed(42) convention.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--output_dir", default="eval_multi")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_ensemble", type=int, default=20)
    p.add_argument("--ddim_steps", type=int, default=None,
                   help="[FIX-ODE-STEPS-MISMATCH] Number of ODE integration steps for "
                        "FM's model.sample(). Was previously not exposed here at all "
                        "(evaluate_one_model always used the checkpoint's own trained "
                        "n_inference_steps, silently ignoring any intended override). "
                        "Default None matches evaluate_full.py's --ddim_steps convention: "
                        "defer to the checkpoint's own value unless explicitly set.")
    p.add_argument("--test_year", type=int, default=None)

    p.add_argument("--fm_checkpoints",       nargs="+", default=None,
                   help="One or more FM checkpoint paths, one per seed")
    p.add_argument("--st_trans_checkpoints", nargs="+", default=None,
                   help="One or more ST-Trans checkpoint paths, one per seed")
    p.add_argument("--lstm_checkpoints",     nargs="+", default=None,
                   help="One or more LSTM checkpoint paths, one per seed")
    p.add_argument("--gru_checkpoints",      nargs="+", default=None,
                   help="One or more GRU checkpoint paths, one per seed")
    p.add_argument("--rnn_checkpoints",      nargs="+", default=None,
                   help="One or more RNN checkpoint paths, one per seed")
    # [MỚI] MGTCF / Phys-Diff checkpoint args, cùng convention multi/legacy-single
    p.add_argument("--mgtcf_checkpoints",    nargs="+", default=None,
                   help="One or more MGTCF checkpoint paths, one per seed")
    p.add_argument("--physdiff_checkpoints", nargs="+", default=None,
                   help="One or more Phys-Diff checkpoint paths, one per seed")
    # Backward-compat singular aliases (old single-checkpoint usage still works)
    p.add_argument("--fm_checkpoint",       default=None, help="[legacy] single checkpoint, use --fm_checkpoints instead")
    p.add_argument("--st_trans_checkpoint", default=None, help="[legacy] single checkpoint")
    p.add_argument("--lstm_checkpoint",     default=None, help="[legacy] single checkpoint")
    p.add_argument("--gru_checkpoint",      default=None, help="[legacy] single checkpoint")
    p.add_argument("--rnn_checkpoint",      default=None, help="[legacy] single checkpoint")
    p.add_argument("--mgtcf_checkpoint",    default=None, help="[legacy] single checkpoint")
    p.add_argument("--physdiff_checkpoint", default=None, help="[legacy] single checkpoint")

    p.add_argument("--paper_hidden_dim", type=int, default=256)
    p.add_argument("--paper_n_layers",   type=int, default=3)
    p.add_argument("--paper_dropout",    type=float, default=0.20)
    p.add_argument("--st_d_model",        type=int, default=64)
    p.add_argument("--st_nhead",          type=int, default=4)
    p.add_argument("--st_num_enc_layers", type=int, default=1)
    p.add_argument("--st_num_dec_layers", type=int, default=3)
    p.add_argument("--st_dim_ff",         type=int, default=512)
    p.add_argument("--st_dropout",        type=float, default=0.1)

    # [MỚI] MGTCF architecture args (fallback nếu checkpoint thiếu model_cfg)
    p.add_argument("--mgtcf_n_generators",  type=int, default=6)
    p.add_argument("--mgtcf_embedding_dim", type=int, default=64)
    p.add_argument("--mgtcf_encoder_h_dim", type=int, default=64)
    p.add_argument("--mgtcf_decoder_h_dim", type=int, default=128)
    p.add_argument("--mgtcf_noise_dim",     type=int, default=8)
    p.add_argument("--mgtcf_best_k",        type=int, default=6)

    # [MỚI] Phys-Diff architecture args (fallback nếu checkpoint thiếu model_cfg)
    p.add_argument("--physdiff_d_model",    type=int, default=64)
    p.add_argument("--physdiff_num_blocks", type=int, default=3)
    p.add_argument("--physdiff_n_sample_steps", type=int, default=200,
                   help="Số bước reverse diffusion dùng lúc ĐÁNH GIÁ (khác "
                        "n_sample_steps lưu trong checkpoint, thường nhỏ hơn "
                        "vì dùng cho validation nhanh lúc train) -- override "
                        "sau khi load, giống --test_n_sample_steps của "
                        "train_physdiff.py.")

    args = p.parse_args()
    set_seed(args.seed)   # [FIX-DETERMINISM] must run before any model.sample() call below
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"  Loading test data...")
    import argparse as _ap
    _loader_args = _ap.Namespace(
        dataset_root = args.dataset_root,
        obs_len      = 8,
        pred_len     = 12,
        batch_size   = 64,
        num_workers  = 2,
        test_year    = args.test_year,
        skip         = 1,
        min_ped      = 1,
        threshold    = 0.002,
    )
    _, loader = data_loader(_loader_args,
                             {"root": args.dataset_root, "type": args.split},
                             test=(args.split != "train"))
    print(f"  Data: {len(loader)} batches")

    def _collect(multi, single):
        """Merge --xxx_checkpoints (list) and legacy --xxx_checkpoint (str) into one list."""
        paths = list(multi) if multi else []
        if single and single not in paths:
            paths.append(single)
        return paths

    jobs = []  # (display_name, kind, checkpoint_path)
    for display_name, kind, multi, single in [
        ("FM",       "fm",       args.fm_checkpoints,       args.fm_checkpoint),
        ("ST-Trans", "st_trans", args.st_trans_checkpoints, args.st_trans_checkpoint),
        ("LSTM",     "lstm",     args.lstm_checkpoints,     args.lstm_checkpoint),
        ("GRU",      "gru",      args.gru_checkpoints,      args.gru_checkpoint),
        ("RNN",      "rnn",      args.rnn_checkpoints,      args.rnn_checkpoint),
        ("MGTCF",    "mgtcf",    args.mgtcf_checkpoints,    args.mgtcf_checkpoint),
        ("Phys-Diff","physdiff", args.physdiff_checkpoints, args.physdiff_checkpoint),
    ]:
        for ckpt_path in _collect(multi, single):
            jobs.append((display_name, kind, ckpt_path))

    if not jobs:
        print("  No checkpoints given — nothing to do.")
        return

    all_records = []
    for display_name, kind, ckpt_path in jobs:
        print(f"\n  {'='*70}\n  Loading {display_name}: {ckpt_path}\n  {'='*70}")
        if kind == "fm":
            model, seed = load_fm(ckpt_path, device)
        elif kind == "st_trans":
            model, seed = load_st_trans(ckpt_path, device,
                                   d_model=args.st_d_model, nhead=args.st_nhead,
                                   num_enc_layers=args.st_num_enc_layers,
                                   num_dec_layers=args.st_num_dec_layers,
                                   dim_ff=args.st_dim_ff, dropout=args.st_dropout)
        elif kind == "mgtcf":
            model, seed = load_mgtcf(ckpt_path, device,
                                   n_generators=args.mgtcf_n_generators,
                                   embedding_dim=args.mgtcf_embedding_dim,
                                   encoder_h_dim=args.mgtcf_encoder_h_dim,
                                   decoder_h_dim=args.mgtcf_decoder_h_dim,
                                   noise_dim=args.mgtcf_noise_dim,
                                   best_k=args.mgtcf_best_k)
        elif kind == "physdiff":
            model, seed = load_physdiff(ckpt_path, device,
                                   d_model=args.physdiff_d_model,
                                   num_blocks=args.physdiff_num_blocks)
            # Override n_sample_steps cho đánh giá cuối cùng (chất lượng
            # cao hơn giá trị lưu trong checkpoint, thường nhỏ để val nhanh).
            model.n_sample_steps = args.physdiff_n_sample_steps
        else:
            model, seed = load_paper_baseline(ckpt_path, kind, device,
                                         hidden_dim=args.paper_hidden_dim,
                                         n_layers=args.paper_n_layers,
                                         dropout=args.paper_dropout)

        n_params = sum(pm.numel() for pm in model.parameters())
        print(f"  {display_name} (seed={seed}): {n_params:,} params")

        recs = evaluate_one_model(model, loader, device, display_name,
                                   seed=seed, n_ensemble=args.n_ensemble,
                                   ddim_steps=args.ddim_steps)
        all_records.extend(recs)

        # [FIX] ate/cte là None ở lead_time=1 (6h) theo convention đã sửa
        # (xem evaluate_one_model's docstring) — np.mean crash nếu None
        # lẫn trong list. Lọc trước khi tính, giống mọi chỗ khác trong
        # generate_paper_report.py đã áp dụng cùng bộ lọc này.
        ade = np.mean([r["ade"] for r in recs if r["ade"] is not None])
        ate_vals = [r["ate"] for r in recs if r["ate"] is not None]
        cte_vals = [r["cte"] for r in recs if r["cte"] is not None]
        ate = np.mean(ate_vals) if ate_vals else float("nan")
        cte = np.mean(cte_vals) if cte_vals else float("nan")
        print(f"  {display_name} seed={seed}: n={len(recs)}  ADE={ade:.2f}  "
              f"ATE={ate:.2f}  CTE={cte:.2f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = os.path.join(args.output_dir, f"multi_model_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n  Saved {len(all_records)} records → {out_path}")
    print(f"  Run generate_comparison_table.py --records {out_path} "
          f"to produce the Table-10-style significance table.")


if __name__ == "__main__":
    main()