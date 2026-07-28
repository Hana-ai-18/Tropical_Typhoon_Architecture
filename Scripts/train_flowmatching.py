# """

# # ─────────────────────────────────────────────────────────────────────────────
# # FILE PLACEMENT:
# #
# #   SOURCE:        train_fm_v24.py
# #   KAGGLE TARGET: /kaggle/working/train_fm.py    (root, cạnh Model/)
# #   LOCAL DEV:     train_fm.py
# #
# #   cp train_fm_v24.py /kaggle/working/train_fm.py
# # ─────────────────────────────────────────────────────────────────────────────

# scripts/train_fm.py  ──  TC-FlowMatching v2.1-XAI Training
# ═══════════════════════════════════════════════════════════════════════════════

# Cải tiến từ v2.1 (dựa trên phân tích log 150 epochs):

#   ROOT CAUSE: CTE gap val=45.7→test=71.6 (+25.9km, 57%) — thiếu direction training
#   [AUG-C]  Recurvature ±20° — PROVEN -7.1km CTE (v2.5)
#   [L_HDG]  L_heading bug fixed (no_grad removed), weight=0.07
#   [CALIB]  Speed calibration tại inference (safe, no training change)
#   [SWA]    Stochastic Weight Averaging (generalization)
#   [HVAL]   Hard-val checkpoint (hard storms = test distribution)
#   NO AUG-D: proven harmful (+4.5km ATE v2.6, +13.4km v2.7)

# Giữ nguyên (proven):
#   ✅ sigma_inference=0.04 FIXED
#   ✅ L_reg softmax-linspace weights
#   ✅ 2-group optimizer, encoder freeze 10ep
#   ✅ val loop NO augmentation
#   ✅ 1-shot inference
#   ✅ SWA sau plateau
#   ✅ Hard-val checkpoint (HVAL)
#   ✅ XAI logging mỗi 10 epoch
# """
# from __future__ import annotations
# import json, sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse, math, time
# from collections import defaultdict
# from typing import Dict, List, Optional

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.amp import autocast, GradScaler

# from Model.data.loader_training import data_loader
# from Model.flow_matching_model import (
#     TCFlowMatching, _norm_to_deg, _haversine_deg,
#     EMAModel, augment_batch, hard_score_from_obs,
#     compute_ensemble_uncertainty,
#     compute_heading_deviation, compute_cte_contribution,
#     compute_obs_attribution,
# )

# HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}
# ST_TRANS_VAL  = {"ADE": 172.68, "ATE": 142.21, "CTE": 42.04,
#                  "12h": 65.42, "24h": 104.67, "48h": 205.10, "72h": 321.39}
# ST_TRANS_TEST = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
#                  "12h": 77.5, "24h": 130.5, "48h": 269.9, "72h": 423.3}

# # ─────────────────────────────────────────────────────────────────────────────
# #  Utilities
# # ─────────────────────────────────────────────────────────────────────────────

# def _unwrap(m):
#     return m._orig_mod if hasattr(m, "_orig_mod") else m

# def move(batch, device):
#     out = list(batch)
#     for i, x in enumerate(out):
#         if torch.is_tensor(x):
#             out[i] = x.to(device)
#         elif isinstance(x, dict):
#             out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
#     return out

# def _ate_cte(pred_deg, gt_deg):
#     """Decompose error into along-track and cross-track components."""
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         z = pred_deg.new_zeros(1, pred_deg.shape[1])
#         return z, z
#     lo1 = torch.deg2rad(gt_deg[:T-1,:,0]); la1 = torch.deg2rad(gt_deg[:T-1,:,1])
#     lo2 = torch.deg2rad(gt_deg[1:T, :,0]); la2 = torch.deg2rad(gt_deg[1:T, :,1])
#     lo3 = torch.deg2rad(pred_deg[1:T,:,0]); la3 = torch.deg2rad(pred_deg[1:T,:,1])
#     ya  = torch.sin(lo2-lo1)*torch.cos(la2)
#     xa  = torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
#     be  = torch.atan2(ya, xa)
#     ye  = torch.sin(lo3-lo2)*torch.cos(la3)
#     xe  = torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
#     bee = torch.atan2(ye, xe)
#     tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
#     ang = bee - be
#     return tot*torch.cos(ang), tot*torch.sin(ang)


# # ─────────────────────────────────────────────────────────────────────────────
# #  2-group optimizer (encoder freeze pattern from v2.1)
# # ─────────────────────────────────────────────────────────────────────────────

# def build_optimizer(model, lr_velocity, lr_encoder, weight_decay,
#                      lr_logits_scale: float = 0.2):
#     raw = _unwrap(model)

#     # [CRITICAL FIX] v2.1-learn thêm 7 nhóm nn.Parameter trực tiếp trên
#     # TCFlowMatching (KHÔNG nằm trong raw.encoder hay raw.velocity):
#     #   speed_correction_logits, reg_step_logits, hard_score_weight_logits,
#     #   heading_step_logits, log_sigma_reg, log_sigma_heading, log_sigma_calib
#     #
#     # BUG ĐÃ PHÁT HIỆN: optimizer cũ (2 param_groups: encoder + velocity)
#     # KHÔNG cover các tham số này. loss.backward() vẫn tính đúng gradient
#     # cho chúng (đã verify), nhưng optimizer.step() chỉ update params nằm
#     # trong param_groups đã đăng ký -> 7 tham số "mồ côi": có gradient
#     # nhưng KHÔNG BAO GIỜ thay đổi giá trị. Mọi cơ chế "tự học" (speed
#     # calibration per-horizon, step weights, Kendall loss weighting...)
#     # sẽ hoàn toàn vô tác dụng nếu không sửa optimizer này.
#     #
#     # FIX: liệt kê params đã thuộc encoder/velocity, lấy phần CÒN LẠI
#     # (set difference) làm group thứ 3 — cách này tự động cover MỌI
#     # tham số mới thêm sau này vào TCFlowMatching mà không cần liệt kê
#     # tên cụ thể, tránh lặp lại lỗi tương tự nếu code model còn đổi nữa.
#     #
#     # [VAR-REDUCE v2, THỬ NGHIỆM — giả thuyết chưa chứng minh chặt, xem dưới]
#     # SPLIT thêm learnable_extra thành 2 group:
#     #   - "softmax_logits" (reg_step_logits, hard_score_weight_logits,
#     #     heading_step_logits): các softmax nhỏ (4-12 chiều), khởi tạo
#     #     GIỐNG HỆT NHAU mọi seed (zeros hoặc hằng số cố định — xem model.py,
#     #     đã verify qua code, không phải suy đoán).
#     #     QUAN SÁT: qua 4 lần train độc lập (seed 42/0/1/2), 72h heading
#     #     deviation (đo trên 1 batch XAI cố định, KHÔNG phải test set) chia
#     #     thành 2 cụm — {42,1}≈107° thấp hơn {0,2}≈118° — tương quan với
#     #     test CTE ở r=0.811 (n=4). ĐÂY LÀ TƯƠNG QUAN ĐÁNG CHÚ Ý NHƯNG CHƯA
#     #     ĐỦ MẠNH để khẳng định chắc chắn (cần |r|>0.95 với n=4 mới coi là
#     #     gần chắc chắn không phải trùng hợp ngẫu nhiên) — coi đây là GIẢ
#     #     THUYẾT đáng thử nghiệm, không phải nguyên nhân đã xác định.
#     #     reg_step_logits có ramp GIÁN TIẾP qua ramp_reg (ep10→30, xem
#     #     get_loss_breakdown) — cửa sổ gradient-mạnh-sớm là ep10-30, KHÔNG
#     #     phải ep0-2. Cơ chế rủi ro: softmax không có ràng buộc tốc độ thay
#     #     đổi; một chênh lệch gradient nhỏ trong cửa sổ ep10-30 (do random
#     #     init của TOÀN mạng, không phải của chính logits) có thể bị khuếch
#     #     đại qua softmax trước khi tín hiệu tích lũy nhiều epoch kịp điều
#     #     chỉnh.
#     #     FIX (thử nghiệm): KHÔNG ép giá trị đích về uniform (đã thử ở
#     #     hard_score_weight_logits — làm chậm cả model, ADE/ATE tệ hơn vì
#     #     early-horizon/dễ-học THẬT SỰ xứng đáng trọng số cao hơn, ép
#     #     uniform chống lại tín hiệu thật). Thay vào đó CHỈ giảm LR cho
#     #     riêng nhóm này (mặc định x0.2 so với velocity) — logits vẫn tự do
#     #     hội tụ về BẤT KỲ đích nào tín hiệu thật dẫn tới, chỉ di chuyển
#     #     CHẬM HƠN. speed_correction_logits KHÔNG đưa vào nhóm này: nó dùng
#     #     sigmoid (độc lập từng phần tử) chứ không phải softmax (ràng buộc
#     #     tổng=1 giữa các phần tử) — không có cùng cơ chế khuếch đại lẫn
#     #     nhau nên không có cùng loại rủi ro.
#     #   - "learnable_extra" (mọi thứ còn lại: speed_correction_logits,
#     #     log_sigma_*): giữ nguyên LR như cũ, không đổi hành vi.
#     encoder_ids  = {id(p) for p in raw.encoder.parameters()}
#     velocity_ids = {id(p) for p in raw.velocity.parameters()}
#     covered_ids  = encoder_ids | velocity_ids

#     softmax_logit_names = {"reg_step_logits", "hard_score_weight_logits",
#                             "heading_step_logits"}
#     softmax_logit_params, rest_extra_params = [], []
#     for name, p in raw.named_parameters():
#         if id(p) in covered_ids:
#             continue
#         short = name.rsplit(".", 1)[-1]
#         (softmax_logit_params if short in softmax_logit_names
#          else rest_extra_params).append(p)

#     groups = [
#         {"params": list(raw.encoder.parameters()),  "lr": lr_encoder,  "name": "encoder"},
#         {"params": list(raw.velocity.parameters()), "lr": lr_velocity, "name": "velocity"},
#     ]
#     if len(rest_extra_params) > 0:
#         # Cùng lr với velocity: các tham số này tham gia trực tiếp vào
#         # loss giống velocity (qua L_reg/L_heading/L_calib), không có lý
#         # do để đóng băng theo encoder-freeze schedule.
#         groups.append({"params": rest_extra_params, "lr": lr_velocity, "name": "learnable_extra"})
#         print(f"  [build_optimizer] learnable_extra group: {len(rest_extra_params)} tensors "
#               f"({sum(p.numel() for p in rest_extra_params)} params) — "
#               f"speed_correction/log_sigma*")
#     if len(softmax_logit_params) > 0:
#         groups.append({"params": softmax_logit_params,
#                         "lr": lr_velocity * lr_logits_scale,
#                         "name": "softmax_logits"})
#         print(f"  [build_optimizer] softmax_logits group: {len(softmax_logit_params)} tensors "
#               f"({sum(p.numel() for p in softmax_logit_params)} params) — "
#               f"reg_step/hard_score/heading_step @ lr×{lr_logits_scale} "
#               f"(slower convergence, less seed-init sensitivity)")

#     return optim.AdamW(groups, weight_decay=weight_decay)

# def get_lrs(opt):
#     lr_enc = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "encoder")
#     lr_vel = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "velocity")
#     return lr_enc, lr_vel

# class TwoGroupScheduler:
#     def __init__(self, opt, warmup_epochs, total_epochs,
#                  lr_vel, lr_vel_min, freeze_end_ep,
#                  lr_enc_peak, encoder_warmup_epochs=5):
#         self.opt         = opt
#         self.warmup      = warmup_epochs
#         self.total       = total_epochs
#         self.lr_vel      = lr_vel
#         self.lr_vel_min  = lr_vel_min
#         self.freeze_end  = freeze_end_ep
#         self.lr_enc_peak = lr_enc_peak
#         self.enc_warmup  = encoder_warmup_epochs
#         self.epoch       = 0
#         # [VAR-REDUCE v2 FIX] Without this, .step() below would overwrite
#         # EVERY non-"encoder" group's lr with lr_vel every epoch — silently
#         # erasing build_optimizer's lr_logits_scale after epoch 1 (the
#         # "softmax_logits" group's LR would only differ from velocity's for
#         # the very first .step() call, then get reset to lr_vel forever).
#         # Record each group's ORIGINAL ratio to lr_vel at construction time,
#         # so .step() can rescale proportionally instead of overwriting.
#         self._lr_ratio = {}
#         for pg in self.opt.param_groups:
#             name = pg.get("name")
#             if name not in (None, "encoder"):
#                 self._lr_ratio[name] = pg["lr"] / lr_vel if lr_vel > 0 else 1.0

#     def _cosine(self, ep_from, ep_to, lr_s, lr_e, ep):
#         t = max(0., min(1., (ep - ep_from) / max(ep_to - ep_from, 1)))
#         return lr_e + 0.5 * (lr_s - lr_e) * (1 + math.cos(math.pi * t))

#     def step(self):
#         ep = self.epoch
#         if ep < self.warmup:
#             lr_vel = self.lr_vel * (0.1 + 0.9 * ep / max(self.warmup - 1, 1))
#         else:
#             lr_vel = self._cosine(self.warmup, self.total, self.lr_vel, self.lr_vel_min, ep)
#         if ep < self.freeze_end:
#             lr_enc = 0.0
#         elif ep < self.freeze_end + self.enc_warmup:
#             lr_enc = self.lr_enc_peak * (ep - self.freeze_end) / self.enc_warmup
#         else:
#             lr_enc = self._cosine(self.freeze_end + self.enc_warmup, self.total,
#                                   self.lr_enc_peak, self.lr_vel_min, ep)
#         for pg in self.opt.param_groups:
#             name = pg.get("name")
#             if name == "encoder":
#                 pg["lr"] = lr_enc
#             elif name in self._lr_ratio:
#                 # Preserve this group's original ratio to lr_vel (e.g.
#                 # softmax_logits at lr_logits_scale=0.2) through the same
#                 # cosine schedule velocity follows, instead of resetting to
#                 # lr_vel outright.
#                 pg["lr"] = lr_vel * self._lr_ratio[name]
#             else:
#                 pg["lr"] = lr_vel
#         self.epoch += 1
#         return lr_vel, lr_enc


# # ─────────────────────────────────────────────────────────────────────────────
# #  SWA handler (same as v2.5 — proven)
# # ─────────────────────────────────────────────────────────────────────────────

# class SWAHandler:
#     """
#     Stochastic Weight Averaging after val ADE plateau.
#     Activation: val ADE improvement < threshold km in last `window` checks.
#     After activation: average model weights each epoch → flatter loss landscape
#     → better generalization.
#     """
#     def __init__(self, swa_lr: float = 2e-6):
#         self.swa_lr    = swa_lr
#         self.active    = False
#         self.start_ep  = None
#         self.n_updates = 0
#         self.avg_state = {}

#     def should_activate(self, ade_history: List[float],
#                          window: int = 3, threshold: float = 1.5) -> bool:
#         if len(ade_history) < window: return False
#         return (ade_history[-window] - ade_history[-1]) < threshold

#     def activate(self, model, opt, ep: int):
#         self.active = True; self.start_ep = ep
#         for pg in opt.param_groups: pg["lr"] = self.swa_lr
#         m = _unwrap(model)
#         self.avg_state = {k: v.detach().clone().float()
#                           for k, v in m.state_dict().items()
#                           if v.dtype.is_floating_point}
#         self.n_updates = 1
#         print(f"  *** SWA ACTIVATED @ ep{ep} (lr → {self.swa_lr:.1e}) ***")

#     def update(self, model):
#         if not self.active: return
#         m = _unwrap(model); sd = m.state_dict(); n = self.n_updates
#         for k in self.avg_state:
#             if k in sd:
#                 self.avg_state[k] = (n * self.avg_state[k] + sd[k].detach().float()) / (n + 1)
#         self.n_updates += 1

#     def apply_to_model(self, model):
#         if not self.active or not self.avg_state: return
#         m = _unwrap(model); sd = m.state_dict()
#         for k in self.avg_state:
#             if k in sd: sd[k].copy_(self.avg_state[k].to(sd[k].device))

#     def restore_from_backup(self, model, backup):
#         m = _unwrap(model); sd = m.state_dict()
#         for k, v in backup.items():
#             if k in sd: sd[k].copy_(v)

#     def save_avg_state(self, path: str, epoch: int, best_score: float,
#                        extra: Optional[dict] = None, model_cfg=None):
#         payload = {"epoch": epoch, "model": self.avg_state,
#                    "best_score": best_score, "is_swa": True,
#                    "swa_updates": self.n_updates,
#                    "model_cfg": model_cfg}
#         if extra: payload.update(extra)
#         torch.save(payload, path)


# # ─────────────────────────────────────────────────────────────────────────────
# #  Hard-val evaluator (HVAL)
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def evaluate_hard_val(model, val_loader, device, hard_threshold: float = 0.35,
#                        n_ensemble: int = 20, ema=None, epoch_for_loss: int = 9999):
#     """
#     Evaluate on hard storms only (hard_score > threshold).
#     Hard storms are closer to test distribution (more recurvature).
#     Best hard-val checkpoint often generalizes better than overall best.
#     """
#     bk = None
#     if ema is not None:
#         try: bk = ema.apply_to(model)
#         except Exception: pass

#     model.eval()
#     all_ade, all_ate, all_cte = [], [], []
#     n_hard = 0

#     for batch in val_loader:
#         bl = move(list(batch), device)
#         B  = bl[0].shape[1]
#         h_score   = hard_score_from_obs(bl[0][:, :, :2])
#         hard_mask = h_score > hard_threshold
#         if hard_mask.sum() == 0: continue
#         hard_idx  = hard_mask.nonzero(as_tuple=True)[0]

#         # Build sub-batch for hard storms
#         bl_h = list(bl)
#         for i, item in enumerate(bl_h):
#             if torch.is_tensor(item):
#                 if item.dim() >= 2 and item.shape[1] == B:
#                     bl_h[i] = item[:, hard_idx, ...]
#                 elif item.dim() >= 1 and item.shape[0] == B:
#                     bl_h[i] = item[hard_idx, ...]

#         try:
#             pred, _, _ = model.sample(bl_h, num_ensemble=n_ensemble)
#         except Exception as e:
#             print(f"  hard val error: {e}"); continue

#         gt = bl_h[1]
#         T  = min(pred.shape[0], gt.shape[0])
#         pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
#         dist = _haversine_deg(pd, gd)
#         ate, cte = _ate_cte(pd, gd)
#         all_ade.extend(dist.mean(0).tolist())
#         if ate.shape[0] > 0:
#             all_ate.extend(ate.abs().mean(0).tolist())
#             all_cte.extend(cte.abs().mean(0).tolist())
#         n_hard += len(hard_idx)

#     if bk is not None:
#         try: ema.restore(model, bk)
#         except Exception: pass

#     def _m(lst): return float(np.mean(lst)) if lst else float("nan")
#     return {
#         "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
#         "n_hard": n_hard,
#         "combined_score": 0.6*_m(all_ade) + 0.2*_m(all_ate) + 0.2*_m(all_cte),
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# #  Main evaluation
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def evaluate(model, loader, device, tag: str = "",
#              n_ensemble: int = 20, ema=None,
#              ref_targets=None, use_tta: bool = False,
#              n_tta: int = 5, epoch_for_loss: int = 9999,
#              run_xai: bool = False, xai_batch=None) -> Dict:
#     """
#     Full evaluation. NO augmentation (val_loss is reliable generalization signal).
#     XAI logging every 10 epochs shows training dynamics.
#     """
#     bk = None
#     if ema is not None:
#         try: bk = ema.apply_to(model)
#         except Exception as e: print(f"  ⚠ EMA: {e}")

#     model.eval()
#     all_ade, all_ate, all_cte = [], [], []
#     step_dist = defaultdict(list)
#     sum_loss = sum_cfm = sum_head = 0.0
#     sum_n = 0

#     for batch in loader:
#         bl = move(list(batch), device)
#         gt = bl[1]; B = bl[0].shape[1]

#         # Val loss (no augmentation)
#         try:
#             bd = model.get_loss_breakdown(bl, epoch=epoch_for_loss)
#             if torch.isfinite(bd["total"]):
#                 sum_loss += bd["total"].item() * B
#                 sum_cfm  += bd["l_cfm"] * B
#                 sum_head += bd["l_heading"] * B
#                 sum_n    += B
#         except Exception: pass

#         # Inference (standard or speed-scale TTA)
#         if use_tta:
#             obs = bl[0]; anchor = obs[-1:, :, :2].detach()
#             scales = [0.875, 0.9375, 1.0, 1.0625, 1.125][:n_tta]
#             preds_t, weights_t = [], []
#             for sc in scales:
#                 obs_s = obs.clone(); obs_s[..., :2] = anchor + (obs[..., :2] - anchor) * sc
#                 bl_s = list(bl); bl_s[0] = obs_s
#                 try:
#                     p, _, _ = model.sample(bl_s, num_ensemble=n_ensemble)
#                     preds_t.append(p)
#                     weights_t.append(2.0 if abs(sc - 1.0) < 1e-6 else 1.0)
#                 except Exception: continue
#             if not preds_t: continue
#             tw   = sum(weights_t)
#             pred = sum(w / tw * p for w, p in zip(weights_t, preds_t))
#         else:
#             try:
#                 # [MULTI-STEP] n_inference_steps default changed 1->10 (see
#                 # constructor comment) — this call doesn't override via
#                 # ddim_steps, so val/test loops now run ~10x more velocity
#                 # network forward passes per sample() call than before.
#                 # Expect val/test evaluation phases to take noticeably
#                 # longer; this is the real cost of the change, not a
#                 # separate slowdown bug.
#                 pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
#             except Exception as e:
#                 print(f"  sample error: {e}"); continue

#         T = min(pred.shape[0], gt.shape[0])
#         pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
#         dist = _haversine_deg(pd, gd)
#         ate, cte = _ate_cte(pd, gd)
#         all_ade.extend(dist.mean(0).tolist())
#         if ate.shape[0] > 0:
#             all_ate.extend(ate.abs().mean(0).tolist())
#             all_cte.extend(cte.abs().mean(0).tolist())
#         for h, s in HORIZON_STEPS.items():
#             if s < T: step_dist[h].extend(dist[s].tolist())

#     if bk is not None:
#         try: ema.restore(model, bk)
#         except Exception: pass

#     def _m(lst): return float(np.mean(lst)) if lst else float("nan")
#     val_loss = sum_loss / max(sum_n, 1)

#     result = {
#         "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
#         "n": len(all_ade),
#         "val_loss": val_loss,
#         "val_cfm_loss":  sum_cfm / max(sum_n, 1),
#         "val_head_loss": sum_head / max(sum_n, 1),
#         "val_mom_loss":  0.0,   # always 0
#     }
#     for h in HORIZON_STEPS: result[f"{h}h"] = _m(step_dist[h])
#     ade, ate_, cte_ = result["ADE"], result["ATE"], result["CTE"]
#     result["combined_score"] = (
#         0.6 * ade + 0.2 * ate_ + 0.2 * cte_
#         if all(np.isfinite(x) for x in [ade, ate_, cte_]) else ade)

#     ref = ref_targets or ST_TRANS_VAL
#     def _v(k): return result.get(k, float("nan"))
#     def _ok(k): return "✓" if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9) else "✗"

#     tta_str = " [TTA]" if use_tta else ""
#     print(f"\n  {'='*72}")
#     print(f"  [{tag}]{tta_str}  n={result['n']}")
#     print(f"  Val Loss : {val_loss:.6f}  cfm={result['val_cfm_loss']:.6f}  "
#           f"head4s={result['val_head_loss']:.6f}  [mom=DISABLED]")
#     print(f"  ADE={_v('ADE'):7.1f}km {_ok('ADE')}  "
#           f"ATE={_v('ATE'):7.1f}km {_ok('ATE')}  "
#           f"CTE={_v('CTE'):7.1f}km {_ok('CTE')}")
#     print(f"  Combined = {_v('combined_score'):.1f}")
#     print(f"  12h={_v('12h'):6.1f}  24h={_v('24h'):6.1f}  "
#           f"48h={_v('48h'):6.1f}  72h={_v('72h'):6.1f} km")
#     beat = [f"{k}={_v(k):.0f}<{ref.get(k,999):.0f}"
#             for k in ["ADE","ATE","CTE","12h","24h","48h","72h"]
#             if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9)]
#     print(f"  BEAT: {' | '.join(beat) if beat else 'none yet'}")

#     # XAI logging
#     if run_xai and xai_batch is not None:
#         try:
#             _, _, _, xai = _unwrap(model).sample(xai_batch, return_xai=True)

#             print(f"  {'─'*60}")
#             print(f"  XAI Summary (fixed val batch)")
#             print(f"  {'─'*60}")

#             # XAI-4: Uncertainty
#             print(f"  [XAI-4] Uncertainty:"
#                   f" 12h={xai['mean_12h_std']:.1f}km"
#                   f"  72h={xai['mean_72h_std']:.1f}km"
#                   f"  ratio={float(xai['uncertainty_ratio'].mean()):.2f}×"
#                   f"  high_uncert={xai['high_uncertainty'].sum().item()}")

#             # XAI-2: Hard score components
#             hc = xai.get("hard_components", {})
#             if hc:
#                 print(f"  [XAI-2] HardScore:"
#                       f" curv={float(hc['curvature'].mean()):.3f}"
#                       f"  spd_var={float(hc['speed_var'].mean()):.3f}"
#                       f"  dir_chg={float(hc['dir_change'].mean()):.3f}"
#                       f"  obs_spd_n={float(hc.get('obs_speed_norm', torch.zeros(1)).mean()):.3f}")

#             # XAI-3: Physics score components
#             pc = xai.get("physics_components", {})
#             if pc:
#                 print(f"  [XAI-3] Physics:"
#                       f" speed={float(pc['speed'].mean()):.3f}"
#                       f"  smooth={float(pc['smooth'].mean()):.3f}"
#                       f"  heading={float(pc['heading'].mean()):.3f}")

#             # XAI-5: Speed comparison (shows calibration effect)
#             sc = xai.get("speed_comparison", {})
#             if sc:
#                 ratio = sc.get("speed_ratio", 1.0)
#                 flag  = ("⚠ OVER" if ratio > 1.15 else
#                          "⚠ UNDER" if ratio < 0.85 else "✓ OK")
#                 n_over  = sc.get("over_predict",  torch.zeros(1)).sum().item()
#                 n_under = sc.get("under_predict", torch.zeros(1)).sum().item()
#                 print(f"  [XAI-5] Speed (post-calibration):"
#                       f" obs={sc['obs_speed_mean']:.1f}km/h"
#                       f"  pred={sc['pred_speed_mean']:.1f}km/h"
#                       f"  ratio={ratio:.2f} {flag}"
#                       f"  (over:{int(n_over)} under:{int(n_under)})")

#             # XAI-6: Heading deviation — key metric for heading loss efficacy
#             hd = xai.get("heading_deviation_deg")
#             if hd is not None and hd.shape[0] >= 1:
#                 hd_mean = hd.mean(1)
#                 print(f"  [XAI-6] Heading deviation:"
#                       f" 12h={hd_mean[0].item():.1f}°"
#                       f"  24h={hd_mean[min(2,len(hd_mean)-1)].item():.1f}°"
#                       f"  72h={hd_mean[min(10,len(hd_mean)-1)].item():.1f}°"
#                       f"  (target: 12h<20°, 72h<100°)")

#             # XAI-7: ATE/CTE decomposition
#             ac = xai.get("ate_cte_decomp", {})
#             if ac:
#                 print(f"  [XAI-7] Error:"
#                       f" ATE={ac['ate_abs_mean']:.1f}km"
#                       f"  CTE={ac['cte_abs_mean']:.1f}km"
#                       f"  ratio={ac['ate_abs_mean']/(ac['cte_abs_mean']+1e-3):.2f}")

#             # XAI-8: Per-horizon speed [v2.1-XAI]
#             sph = xai.get("speed_per_horizon", {})
#             if sph and "pred_kmh" in sph:
#                 ps = sph["pred_kmh"]; gs = sph["gt_kmh"]; rs = sph["ratio"]
#                 hz = [(0,"12h"),(2,"24h"),(6,"48h"),(10,"72h")]
#                 parts = []
#                 for idx, lbl in hz:
#                     if idx < len(rs):
#                         flag = "⚠" if rs[idx] > 1.3 or rs[idx] < 0.7 else "✓"
#                         parts.append(f"{lbl}:r={rs[idx]:.2f}{flag}")
#                 print(f"  [XAI-8] Speed/horizon: {' | '.join(parts)}")

#             # XAI-9: Storm category [v2.1-XAI]
#             sc9 = xai.get("storm_categories", {})
#             if sc9:
#                 print(f"  [XAI-9] Storms:"
#                       f" slow={sc9.get('n_slow',0)}"
#                       f"  med={sc9.get('n_medium',0)}"
#                       f"  fast={sc9.get('n_fast',0)}"
#                       f"  spd={sc9.get('speed_mean',0):.1f}±{sc9.get('speed_std',0):.1f}km/h")

#             # [LEARN] Learned parameter values — theo dõi model có thực sự
#             # học hay không qua các epoch (nếu các giá trị này đứng yên ở
#             # init suốt training, đó là dấu hiệu optimizer không cover
#             # chúng — xem build_optimizer's 'learnable_extra' param group)
#             lp = xai.get("learned_params", {})
#             if lp:
#                 sc_str  = ",".join(f"{v:.2f}" for v in lp.get("speed_correction", [])[:4])
#                 rw_str  = ",".join(f"{v:.3f}" for v in lp.get("reg_step_weights", [])[:4])
#                 # [FIX] full 12-horizon reg_step_weights + b_horizon, not just
#                 # the first 4 — the far-horizon values (index 7,11 = 48h,72h)
#                 # are exactly what --disable_horizon_nll's fix targets, and
#                 # were invisible in the old [:4]-truncated print.
#                 rw_full = lp.get("reg_step_weights", [])
#                 rw_far  = ",".join(f"{v:.3f}" for v in rw_full[7:12]) if len(rw_full) >= 12 else "?"
#                 bh_str  = ",".join(f"{v:.1f}" for v in lp.get("b_horizon", [])[:12])
#                 hw_str  = ",".join(f"{v:.3f}" for v in lp.get("hard_score_weights", []))
#                 hdw_str = ",".join(f"{v:.3f}" for v in lp.get("heading_step_weights", [])[:4])
#                 sig_inf = lp.get("sigma_inf", float("nan"))
#                 ls_r    = lp.get("log_sigma_reg",     float("nan"))
#                 ls_h    = lp.get("log_sigma_heading",  float("nan"))
#                 ls_c    = lp.get("log_sigma_calib",    float("nan"))
#                 el_r    = lp.get("eff_lambda_reg",     float("nan"))
#                 el_h    = lp.get("eff_lambda_heading", float("nan"))
#                 el_c    = lp.get("eff_lambda_calib",   float("nan"))
#                 print(f"  [LEARN] speed_corr(12h-24h)=[{sc_str}]"
#                       f"  reg_w(12h-24h)=[{rw_str}]")
#                 print(f"  [LEARN] reg_w(48h-72h,idx7-11)=[{rw_far}]"
#                       f"  b_horizon(6h..72h)=[{bh_str}]")
#                 print(f"  [LEARN] hard_w(curv,spdvar,dirchg,obsspd)=[{hw_str}]"
#                       f"  head_w(12h-24h)=[{hdw_str}]  sigma_inf={sig_inf:.4f}")
#                 print(f"  [LEARN] log_sigma: reg={ls_r:.3f}  heading={ls_h:.3f}  calib={ls_c:.3f}"
#                       f"  |  eff_lambda: reg={el_r:.3f}  heading={el_h:.3f}  calib={el_c:.3f}")

#             print(f"  {'─'*60}")
#             result["xai"] = xai
#         except Exception as e:
#             print(f"  XAI error: {e}")
#             import traceback; traceback.print_exc()

#     print(f"  {'='*72}\n")
#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# #  Checkpoint
# # ─────────────────────────────────────────────────────────────────────────────

# def _save(path, epoch, model, opt, sched, best_score,
#           ema=None, scaler=None, extra=None, model_cfg=None):
#     m   = _unwrap(model)
#     esd = None
#     if ema is not None:
#         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
#         except Exception: pass
#     payload = {
#         "epoch": epoch, "model": m.state_dict(),
#         "optimizer": opt.state_dict(), "scheduler": sched.epoch,
#         "scaler": scaler.state_dict() if scaler is not None else None,
#         "best_score": best_score, "best_ade": best_score, "ema": esd,
#         # [FIX] self-describing checkpoint — see call site comment in main()
#         "model_cfg": model_cfg,
#     }
#     if extra: payload.update(extra)
#     torch.save(payload, path)


# # ─────────────────────────────────────────────────────────────────────────────
# #  Args
# # ─────────────────────────────────────────────────────────────────────────────

# def get_args():
#     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     # Data
#     p.add_argument("--dataset_root",           default="TCND_vn")
#     p.add_argument("--obs_len",                default=8,      type=int)
#     p.add_argument("--pred_len",               default=12,     type=int)
#     p.add_argument("--num_workers",            default=2,      type=int)
#     p.add_argument("--other_modal",            default="gph")
#     p.add_argument("--delim",                  default=" ")
#     p.add_argument("--skip",                   default=1,      type=int)
#     p.add_argument("--min_ped",                default=1,      type=int)
#     p.add_argument("--threshold",              default=0.002,  type=float)
#     # Model
#     p.add_argument("--d_cond",                 default=256,    type=int)
#     p.add_argument("--d_model",                default=256,    type=int)
#     p.add_argument("--nhead",                  default=8,      type=int)
#     p.add_argument("--num_dec_layers",         default=4,      type=int)
#     p.add_argument("--dim_ff",                 default=512,    type=int)
#     p.add_argument("--dropout",                default=0.1,    type=float)
#     p.add_argument("--unet_in_ch",             default=13,     type=int)
#     p.add_argument("--sigma_min",              default=0.06,   type=float,
#                    help="[FIX, user-approved tradeoff] Min noise scale for "
#                         "CFM sampling. CHANGED default from 0.04 (~22km) to "
#                         "0.06 (~33km) — 0.04 was only ~10-20% of actual "
#                         "error magnitude, too small to seed meaningful "
#                         "ensemble divergence (suspected contributor to "
#                         "near-zero CRPS/Spread-Skill). NOT experimentally "
#                         "validated on a real checkpoint (unlike "
#                         "n_inference_steps) — this is a directional fix "
#                         "based on the km-scale analysis, not a proven-best "
#                         "value. Pass --sigma_min 0.04 to revert to the "
#                         "original.")
#     p.add_argument("--sigma_max",              default=0.15,   type=float,
#                    help="[FIX, user-approved tradeoff] Max noise scale. "
#                         "CHANGED default from 0.08 (~44km) to 0.15 (~83km), "
#                         "closer to actual CTE-scale error (67km), while "
#                         "staying below full ADE scale (~219km). Pass "
#                         "--sigma_max 0.08 to revert to the original.")
#     p.add_argument("--sigma_decay_end",        default=100,    type=int,
#                    help="[FIX, user-approved tradeoff] Epoch at which sigma "
#                         "reaches sigma_min and stays there. CHANGED default "
#                         "from 40 to 100 — with 40, >100 epochs of a typical "
#                         "140-165 epoch run trained at minimum noise. Pass "
#                         "--sigma_decay_end 40 to revert to the original.")
#     p.add_argument("--lambda_reg",             default=0.2,    type=float)
#     p.add_argument("--lambda_heading",         default=0.07,   type=float,
#                    help="[v2.1-XAI] multi-step heading, bug fixed, weight=0.07")
#     p.add_argument("--lambda_momentum",        default=0.0,    type=float,
#                    help="[v2.6] DISABLED — hurt test ATE by +7.9km")
#     p.add_argument("--lambda_hard_reg",        default=0.02,   type=float,
#                    help="[VAR-REDUCE] Fixed weight (not Kendall-learned) L2 "
#                         "penalty pulling hard_score_weight_logits' softmax "
#                         "toward uniform. Addresses seed-to-seed CTE variance "
#                         "traced to this 4-way weighting collapsing onto "
#                         "'curvature' at different rates per seed (confirmed "
#                         "across seed 42/1/2 runs at matched epochs). "
#                         "0.0 = fully disabled (same as --disable_hard_reg).")
#     p.add_argument("--log_sigma_reg_min_clamp", default=-3.0,   type=float,
#                    help="Numerical-stability clamp floor for log_sigma_reg "
#                         "(prevents exp() overflow), NOT a tuning knob. "
#                         "Default -3.0 -> cap~=202, far above the ~15-20 "
#                         "eff_lambda_reg converges to — reconsidered from an "
#                         "earlier proposal to tighten this as a way to force "
#                         "L_reg's weight down: that would fight a genuine "
#                         "loss-minimizing choice Kendall converges to (not a "
#                         "bug), same class of change that hurt ADE/ATE when "
#                         "tried on hard_score_weight_logits. See "
#                         "--disable_horizon_nll instead for the principled "
#                         "fix to far-horizon attention starvation.")
#     p.add_argument("--disable_horizon_nll",    action="store_true", default=False,
#                    help="[FIX] Disable log_b_horizon (Laplace-NLL per-"
#                         "horizon error normalization) — falls back to "
#                         "reg_step_logits' softmax attending to RAW "
#                         "haversine distance per horizon (the original "
#                         "formula). Default (flag absent) uses the NEW "
#                         "normalized formula: reg_step_logits' softmax was "
#                         "found to systematically starve far horizons "
#                         "(48h-72h get only ~7-13% combined attention) "
#                         "because raw error grows ~14x from 6h to 72h — not "
#                         "a bug, but a rational response to an unnormalized "
#                         "signal. The fix normalizes dist[t] by a LEARNED "
#                         "per-horizon scale b_horizon (via a stable Laplace "
#                         "NLL, same log(b) counterbalance structure as "
#                         "Kendall's Gaussian NLL elsewhere in this model — "
#                         "provably cannot degenerate to 0 or explode). "
#                         "Initializes as an exact no-op (b=1 -> nll_h=dist) "
#                         "so it only diverges from original behavior as "
#                         "training reveals genuine per-horizon difficulty. "
#                         "CAUTION: this is a NEW mechanism, not yet "
#                         "empirically validated on a real training run "
#                         "(unlike --n_inference_steps, which IS verified) — "
#                         "the math is sound but watch ADE/ATE/CTE together "
#                         "with CRPS/Spread-Skill to confirm it helps far-"
#                         "horizon diversity without hurting near-horizon "
#                         "accuracy before trusting the result.")
#     p.add_argument("--use_ot",                 default=True,   action="store_true")
#     p.add_argument("--no_ot",                  dest="use_ot",  action="store_false")
#     p.add_argument("--ot_epsilon",             default=0.05,   type=float)
#     p.add_argument("--n_ensemble",             default=20,     type=int)
#     p.add_argument("--sigma_inference",        default=0.04,   type=float)
#     p.add_argument("--n_inference_steps",      default=10,     type=int,
#                    help="[MULTI-STEP] Euler integration steps baked into "
#                         "the checkpoint (used at inference when eval "
#                         "doesn't override via --ddim_steps). CHANGED "
#                         "default from 1 (single-shot x0+v) to 10 — this is "
#                         "the ONE change in this codebase backed by direct "
#                         "experimental evidence (not just theory): evaluating "
#                         "an existing checkpoint with N=1->10 raised ensemble "
#                         "spread 4km->55km, improved CRPS -9.5% (216.8->"
#                         "196.3km), Spread/Skill ratio at 6h reached 1.07 "
#                         "(near-ideal 1.0), at a small ADE/ATE/CTE cost "
#                         "(+1.7%/+1.9%/+7.2%). N=3-4 tested WORSE than both "
#                         "N=1 and N=8-10 — standard Euler discretization "
#                         "error is non-monotonic here, don't assume any N "
#                         "value between 1 and 10 is safe without testing. "
#                         "Pass 1 to reproduce the exact original single-shot "
#                         "behavior. NOTE: this does NOT fix ensemble spread "
#                         "stalling at far horizons (48h-72h) — that traces "
#                         "to reg_step_logits' softmax naturally down-"
#                         "weighting hard-to-reduce far-horizon loss, a "
#                         "genuine optimization outcome verified NOT to be a "
#                         "bug (see flow_matching_model.py's _reg_loss). No "
#                         "fix for that is applied by default in this "
#                         "codebase — it would require overriding what the "
#                         "model has genuinely learned is loss-optimal, which "
#                         "needs deliberate experimentation, not a silent "
#                         "default change.")
#     # Training
#     p.add_argument("--num_epochs",             default=150,    type=int)
#     p.add_argument("--batch_size",             default=64,     type=int)
#     p.add_argument("--lr",                     default=2e-4,   type=float)
#     p.add_argument("--lr_logits_scale",        default=0.2,    type=float,
#                    help="[VAR-REDUCE v2] LR multiplier (vs --lr) for the "
#                         "small softmax logits (reg_step_logits, "
#                         "hard_score_weight_logits, heading_step_logits). "
#                         "Slower convergence for these reduces how much a "
#                         "seed's early-training noise can lock them into an "
#                         "extreme configuration before the true signal "
#                         "(accumulated over many epochs) has a chance to "
#                         "shape them — without pulling their VALUE toward "
#                         "uniform (that was tried on hard_score_weight_logits "
#                         "and hurt ADE/ATE by fighting the genuine easier-to"
#                         "-learn-early-horizon signal). 1.0 = old behavior "
#                         "(same LR as velocity, i.e. disabled).")
#     p.add_argument("--lr_min",                 default=1e-6,   type=float)
#     p.add_argument("--warmup_epochs",          default=5,      type=int)
#     p.add_argument("--weight_decay",           default=1e-4,   type=float)
#     p.add_argument("--grad_clip",              default=1.0,    type=float)
#     p.add_argument("--use_amp",                action="store_true", default=False)
#     p.add_argument("--use_ema",                default=True,   action="store_true")
#     p.add_argument("--no_ema",                 dest="use_ema", action="store_false")
#     # Encoder freeze
#     p.add_argument("--freeze_encoder_epochs",  default=10,     type=int)
#     p.add_argument("--encoder_warmup_epochs",  default=5,      type=int)
#     p.add_argument("--lr_enc_peak",            default=5e-5,   type=float)
#     # Eval
#     p.add_argument("--val_freq",               default=5,      type=int)
#     p.add_argument("--patience",               default=20,     type=int)
#     p.add_argument("--min_ep",                 default=20,     type=int)
#     p.add_argument("--hard_val_threshold",     default=0.35,   type=float)
#     p.add_argument("--hard_val_freq",          default=10,     type=int)
#     # SWA
#     p.add_argument("--swa_lr",                 default=2e-6,   type=float)
#     p.add_argument("--swa_window",             default=3,      type=int)
#     p.add_argument("--swa_threshold",          default=1.5,    type=float)
#     p.add_argument("--swa_min_ep",             default=50,     type=int)
#     # Test
#     p.add_argument("--tta_test",               default=True,   action="store_true")
#     p.add_argument("--n_tta",                  default=5,      type=int)
#     p.add_argument("--multiscale_test",        default=True,   action="store_true")
#     # IO
#     p.add_argument("--output_dir",             default="runs/fm_v26")
#     p.add_argument("--gpu_num",                default="0")
#     p.add_argument("--resume",                 default=None)
#     p.add_argument("--test_at_end",            action="store_true", default=True)
#     p.add_argument("--no_test",                dest="test_at_end", action="store_false")
#     # ── ESWA paper: reproducibility + ablation ────────────────────────────
#     p.add_argument("--seed",                   type=int, default=42,
#                    help="Random seed. Run 3-5 seeds for ESWA mean±std reporting.")
#     p.add_argument("--disable_l_heading",      action="store_true", default=False,
#                    help="Ablation: disable L_heading_ms")
#     p.add_argument("--disable_l_calib",        action="store_true", default=False,
#                    help="Ablation: disable L_calib (speed correction training)")
#     p.add_argument("--disable_l_reg",          action="store_true", default=False,
#                    help="Ablation: disable L_reg (CFM-only variant)")
#     p.add_argument("--disable_aug_c",          action="store_true", default=False,
#                    help="Ablation: disable AUG-C recurvature")
#     p.add_argument("--disable_learned_weights",action="store_true", default=False,
#                    help="Ablation: fixed lambda (no Kendall weighting)")
#     p.add_argument("--disable_hard_reg",       action="store_true", default=False,
#                    help="Ablation: disable hard_score_weight_logits uniform "
#                         "regularizer (same as --lambda_hard_reg 0.0)")
#     p.add_argument("--ablation_name",          type=str, default="",
#                    help="Suffix for output_dir to tag ablation variant")
#     return p.parse_args()


# # ─────────────────────────────────────────────────────────────────────────────
# #  Main
# # ─────────────────────────────────────────────────────────────────────────────

# def main(args):
#     # ── Seed (ESWA: run with 3-5 seeds, report mean±std) ──────────────────
#     import random
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.seed)
#     # Ablation: append name to output dir if provided
#     if args.ablation_name:
#         args.output_dir = f"{args.output_dir}_{args.ablation_name}"
#     if args.seed != 42:
#         args.output_dir = f"{args.output_dir}_seed{args.seed}"

#     if torch.cuda.is_available():
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output_dir, exist_ok=True)
#     # Wall-clock tracking
#     _wall_start = time.time()
#     best_ckpt      = os.path.join(args.output_dir, "best_model.pth")
#     hard_best_ckpt = os.path.join(args.output_dir, "hard_best_model.pth")
#     swa_ckpt       = os.path.join(args.output_dir, "swa_model.pth")
#     last_ckpt      = os.path.join(args.output_dir, "last_model.pth")

#     print("=" * 72)
#     print("  TC-FlowMatching v2.1-XAI")
#     print(f"  [AUG-C]  Recurvature ±20° (PROVEN -7.1km CTE in v2.5)")
#     print(f"  [L_HDG]  L_heading bug fixed, weight={args.lambda_heading}")
#     print(f"  [CALIB]  Speed calibration at inference (clip 0.85-1.15)")
#     print(f"  [NO-D]   AUG-D removed (proven +4.5km ATE worse)")
#     print(f"  KEEP:    sigma_inf={args.sigma_inference} FIXED, L_reg linear, OT")
#     print("=" * 72)

#     # ── Data ──────────────────────────────────────────────────────────────
#     print("\n  Loading data...")
#     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
#     vd, val_loader = data_loader(args, {"root": args.dataset_root, "type": "val"}, test=True)
#     print(f"  train: {len(trd)} ({len(trl)} batches/ep)")
#     print(f"  val:   {len(vd)} ({len(val_loader)} batches)")

#     # Fixed XAI batch (same every epoch for comparability)
#     try:
#         xai_batch = move(list(next(iter(val_loader))), device)
#     except Exception:
#         xai_batch = None

#     # ── Model ─────────────────────────────────────────────────────────────
#     model = TCFlowMatching(
#         pred_len=args.pred_len,        obs_len=args.obs_len,
#         unet_in_ch=args.unet_in_ch,   d_cond=args.d_cond,
#         d_model=args.d_model,          nhead=args.nhead,
#         num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
#         dropout=args.dropout,
#         sigma_min=args.sigma_min,      sigma_max=args.sigma_max,
#         sigma_decay_end=args.sigma_decay_end,
#         lambda_reg=args.lambda_reg,    lambda_heading=args.lambda_heading,
#         lambda_momentum=0.0,
#         lambda_hard_reg=(0.0 if args.disable_hard_reg else args.lambda_hard_reg),
#         log_sigma_reg_min_clamp=args.log_sigma_reg_min_clamp,
#         enable_horizon_nll=not args.disable_horizon_nll,
#         use_ot=args.use_ot,            ot_epsilon=args.ot_epsilon,
#         use_ema=args.use_ema,          n_ensemble=args.n_ensemble,
#         n_inference_steps=args.n_inference_steps,
#         sigma_inference=args.sigma_inference,
#     ).to(device)

#     # [FIX] Checkpoints must self-describe their architecture. Without this,
#     # evaluate_full.py's `TCFlowMatching(**ck.get("model_cfg", {}))` silently
#     # falls back to ALL constructor defaults whenever model_cfg is missing —
#     # harmless only by coincidence when CLI defaults happen to match
#     # constructor defaults (true today for d_model/nhead/num_dec_layers), but
#     # a real correctness risk the moment anyone trains with a non-default
#     # architecture (e.g. --d_model 128): eval would silently reconstruct the
#     # WRONG architecture, load_state_dict(strict=False) would mismatch
#     # shapes silently degrade to partial loading instead of a clear error.
#     # This dict mirrors the constructor call above exactly.
#     model_cfg = dict(
#         pred_len=args.pred_len,        obs_len=args.obs_len,
#         unet_in_ch=args.unet_in_ch,   d_cond=args.d_cond,
#         d_model=args.d_model,          nhead=args.nhead,
#         num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
#         dropout=args.dropout,
#         sigma_min=args.sigma_min,      sigma_max=args.sigma_max,
#         sigma_decay_end=args.sigma_decay_end,
#         lambda_reg=args.lambda_reg,    lambda_heading=args.lambda_heading,
#         lambda_momentum=0.0,
#         lambda_hard_reg=(0.0 if args.disable_hard_reg else args.lambda_hard_reg),
#         log_sigma_reg_min_clamp=args.log_sigma_reg_min_clamp,
#         enable_horizon_nll=not args.disable_horizon_nll,
#         use_ot=args.use_ot,            ot_epsilon=args.ot_epsilon,
#         use_ema=args.use_ema,          n_ensemble=args.n_ensemble,
#         n_inference_steps=args.n_inference_steps,
#         sigma_inference=args.sigma_inference,
#     )

#     model.init_ema()
#     ema = getattr(_unwrap(model), "_ema", None)
#     raw = _unwrap(model)
#     n_enc = sum(p.numel() for p in raw.encoder.parameters())
#     n_vel = sum(p.numel() for p in raw.velocity.parameters())
#     encoder_ids  = {id(p) for p in raw.encoder.parameters()}
#     velocity_ids = {id(p) for p in raw.velocity.parameters()}
#     n_extra = sum(p.numel() for p in raw.parameters()
#                   if id(p) not in encoder_ids and id(p) not in velocity_ids)
#     n_total = n_enc + n_vel + n_extra
#     mem_mb  = sum(p.numel()*p.element_size() for p in model.parameters()) / 1e6
#     print(f"\n  Encoder: {n_enc:,}  VelocityTrans: {n_vel:,}  "
#           f"LearnableExtra: {n_extra:,}  Total: {n_total:,}  Mem: {mem_mb:.1f}MB")
#     # Log computational footprint for ESWA paper Table E
#     footprint_info = {
#         "n_encoder": n_enc, "n_velocity": n_vel, "n_extra": n_extra,
#         "n_total": n_total, "mem_mb": mem_mb,
#         "seed": args.seed,
#         "ablation_name": args.ablation_name or "full",
#         "disable_l_heading": getattr(args, "disable_l_heading", False),
#         "disable_l_calib":   getattr(args, "disable_l_calib", False),
#         "disable_l_reg":     getattr(args, "disable_l_reg", False),
#         "disable_aug_c":     getattr(args, "disable_aug_c", False),
#         "disable_hard_reg":  getattr(args, "disable_hard_reg", False),
#         "lambda_hard_reg":   getattr(args, "lambda_hard_reg", 0.02),
#         # [FIX] 2 ablation flags dưới đây đã có mặt trong model_cfg (nên
#         # checkpoint vẫn load lại đúng kiến trúc), nhưng bị BỎ SÓT khỏi
#         # footprint_info — tức file auto_eval_summary.json / bất kỳ chỗ
#         # nào đọc footprint_info sẽ KHÔNG biết run này có --no_ot hay
#         # --disable_horizon_nll hay không, dễ gây nhầm lẫn khi tra cứu
#         # lại nhiều run cùng lúc (đặc biệt 3 ablation seed=0 chạy song
#         # song: no_ot / no_horizon_nll / no_hard_reg).
#         "disable_horizon_nll": getattr(args, "disable_horizon_nll", False),
#         "use_ot":              getattr(args, "use_ot", True),
#     }

#     # ── Ablation: patch get_loss_breakdown ─────────────────────────────────
#     # [FIX] --disable_horizon_nll và --no_ot đều là cấu hình model-level
#     # thật (nằm trong model_cfg, ảnh hưởng constructor TCFlowMatching),
#     # nhưng không thuộc nhóm ablation "patch get_loss_breakdown" bên dưới
#     # (chúng không cần patch gì — enable_horizon_nll/use_ot đã được
#     # constructor xử lý trực tiếp). Trước đây 2 flag này hoàn toàn
#     # KHÔNG được log ra console — chạy --no_ot hay --disable_horizon_nll
#     # sẽ không in dòng [ABLATION] nào, dễ tưởng nhầm flag không có tác
#     # dụng dù model_cfg (và do đó checkpoint) vẫn đúng. Thêm log riêng.
#     if args.disable_horizon_nll or not args.use_ot:
#         print(f"  [ABLATION] Model-level: "
#               f"{'no_horizon_nll(raw dist, no log_b_horizon) ' if args.disable_horizon_nll else ''}"
#               f"{'no_OT(random x0/x1 pairing) ' if not args.use_ot else ''}")

#     if any([args.disable_l_heading, args.disable_l_calib, args.disable_l_reg,
#             args.disable_aug_c, args.disable_learned_weights, args.disable_hard_reg]):
#         print(f"  [ABLATION] Disabled: "
#               f"{'L_heading ' if args.disable_l_heading else ''}"
#               f"{'L_calib ' if args.disable_l_calib else ''}"
#               f"{'L_reg ' if args.disable_l_reg else ''}"
#               f"{'AUG-C ' if args.disable_aug_c else ''}"
#               f"{'Kendall_weights ' if args.disable_learned_weights else ''}"
#               f"{'HardScoreReg ' if args.disable_hard_reg else ''}")
#         _orig_glb = raw.get_loss_breakdown.__func__
#         _abl_flags = {
#             "disable_l_heading": args.disable_l_heading,
#             "disable_l_calib":   args.disable_l_calib,
#             "disable_l_reg":     args.disable_l_reg,
#             "disable_learned_weights": args.disable_learned_weights,
#             "disable_hard_reg":  args.disable_hard_reg,
#         }
#         def _patched_glb(self, batch_list, epoch=0, **kwargs):
#             bd = _orig_glb(self, batch_list, epoch=epoch, **kwargs)
#             import math as _m
#             # [FIX] Use GRAPH-CONNECTED tensors (bd["_t_l_*"]), NOT the
#             # detached float values (bd["l_reg"] etc, from .item()). The
#             # previous version of this patch rebuilt `total` from those
#             # detached floats — wrapping a float in torch.tensor() carries
#             # NO gradient, so backward() through that `total` only reached
#             # the Kendall log_sigma* params, not the encoder/velocity
#             # network. Every ablation run using this patch trained a
#             # near-frozen network. Fixed by reading the graph-connected
#             # tensors the model now exposes (_t_l_cfm, _t_l_reg, etc).
#             l_cfm      = bd["_t_l_cfm"]
#             l_reg      = bd["_t_l_reg"]
#             l_heading  = bd["_t_l_heading"]
#             l_calib    = bd["_t_l_calib"]
#             # [VAR-REDUCE] l_hard_reg must be explicitly carried through this
#             # rebuild — it's NOT part of the original total = l_cfm+reg+
#             # heading+calib formula this patch reconstructs, so omitting it
#             # would silently disable the uniform-regularizer on EVERY
#             # ablation run (e.g. --disable_l_calib), not just when the user
#             # actually asked for --disable_hard_reg.
#             l_hard_reg = bd["_t_l_hard_reg"]
#             if _abl_flags["disable_l_reg"]:     l_reg      = l_reg * 0.0
#             if _abl_flags["disable_l_heading"]: l_heading  = l_heading * 0.0
#             if _abl_flags["disable_l_calib"]:   l_calib    = l_calib * 0.0
#             if _abl_flags["disable_hard_reg"]:  l_hard_reg = l_hard_reg * 0.0
#             HALF = 0.5 * _m.log(2.0 * _m.pi)
#             if _abl_flags["disable_learned_weights"]:
#                 total = (l_cfm
#                          + bd["lam_reg"]   * 0.20 * l_reg
#                          + bd["lam_dir"]   * 0.07 * l_heading
#                          + bd["lam_calib"] * 0.10 * l_calib)
#             else:
#                 prec_r = torch.exp(-2.0 * self.log_sigma_reg.clamp(min=-3.0))
#                 prec_h = torch.exp(-2.0 * self.log_sigma_heading.clamp(min=-3.0))
#                 prec_c = torch.exp(-2.0 * self.log_sigma_calib.clamp(min=-3.0))
#                 total = (l_cfm
#                          + bd["lam_reg"]   * (0.5*prec_r*l_reg     + self.log_sigma_reg.clamp(min=-3.0)     + HALF)
#                          + bd["lam_dir"]   * (0.5*prec_h*l_heading + self.log_sigma_heading.clamp(min=-3.0) + HALF)
#                          + bd["lam_calib"] * (0.5*prec_c*l_calib   + self.log_sigma_calib.clamp(min=-3.0)   + HALF))
#             # Re-add the fixed-weight (non-Kendall) hard_score regularizer —
#             # same formula as the unpatched total, always additive.
#             total = total + self.lambda_hard_reg * l_hard_reg
#             if not torch.isfinite(total):
#                 total = total.new_zeros(())
#             bd.update({"total": total,
#                        "l_reg":      float(l_reg.detach()),
#                        "l_heading":  float(l_heading.detach()),
#                        "l_calib":    float(l_calib.detach()),
#                        "l_hard_reg": float(l_hard_reg.detach())})
#             return bd
#         import types
#         raw.get_loss_breakdown = types.MethodType(_patched_glb, raw)

#     # ── Optimizer + Scheduler ─────────────────────────────────────────────
#     opt    = build_optimizer(model, lr_velocity=args.lr, lr_encoder=0.0,
#                              weight_decay=args.weight_decay,
#                              lr_logits_scale=args.lr_logits_scale)
#     scaler = GradScaler("cuda", enabled=args.use_amp)
#     sched  = TwoGroupScheduler(
#         opt=opt, warmup_epochs=args.warmup_epochs, total_epochs=args.num_epochs,
#         lr_vel=args.lr, lr_vel_min=args.lr_min,
#         freeze_end_ep=args.freeze_encoder_epochs, lr_enc_peak=args.lr_enc_peak,
#         encoder_warmup_epochs=args.encoder_warmup_epochs)
#     print(f"\n  LR vel: {args.lr:.0e} → {args.lr_min:.0e}  "
#           f"LR enc: 0 ({args.freeze_encoder_epochs}ep) → {args.lr_enc_peak:.0e}")

#     swa = SWAHandler(swa_lr=args.swa_lr)

#     # ── Resume ────────────────────────────────────────────────────────────
#     start_ep = 0; best_score = float("inf"); best_hard = float("inf")
#     patience_cnt = 0; val_ade_history = []

#     if args.resume and os.path.exists(args.resume):
#         ck = torch.load(args.resume, map_location=device)
#         _unwrap(model).load_state_dict(ck["model"], strict=False)
#         try: opt.load_state_dict(ck["optimizer"])
#         except Exception as e: print(f"  ⚠ Opt: {e}")
#         sched.epoch  = ck.get("scheduler", 0)
#         start_ep     = ck.get("epoch", 0) + 1
#         best_score   = ck.get("best_score", ck.get("best_ade", float("inf")))
#         patience_cnt = ck.get("patience_cnt", 0)
#         if scaler and ck.get("scaler"):
#             try: scaler.load_state_dict(ck["scaler"])
#             except Exception: pass
#         if ema and ck.get("ema"):
#             for k, v in ck["ema"].items():
#                 if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
#         print(f"  ↩ Resume ep{start_ep}  best={best_score:.1f}  patience={patience_cnt}")

#     try:
#         model = torch.compile(model, mode="reduce-overhead")
#         print("  torch.compile: ok")
#     except Exception: pass

#     # ── Training loop ─────────────────────────────────────────────────────
#     nstep = len(trl)
#     print(f"\n  TRAINING ({nstep} steps/ep × {args.num_epochs} ep)")
#     print(f"  Aug: shift±5km(25%) + speed×[0.85,1.15](20%) + recurv±20°(20%) + no-aug(35%)")
#     print(f"  Loss: L_CFM + L_reg(linear) + L_heading_ms(4steps,decay=0.5)")
#     print(f"  Inf:  1-shot sigma=0.04 + speed_calibrate(±15%) + top3 physics")
#     print()

#     for ep in range(start_ep, start_ep + args.num_epochs):
#         rel_ep = ep - start_ep
#         freeze = rel_ep < args.freeze_encoder_epochs
#         for p in _unwrap(model).encoder.parameters():
#             p.requires_grad_(not freeze)

#         if rel_ep == 0 and freeze:
#             print(f"  *** Ep{ep}: encoder frozen ***")
#         if rel_ep == args.freeze_encoder_epochs:
#             print(f"\n  *** Ep{ep}: encoder unfrozen ***")

#         model.train()
#         sum_loss = sum_cfm = sum_reg = sum_head = sum_ade1 = 0.0
#         t0_ep = time.perf_counter()

#         for i, batch in enumerate(trl):
#             bl = move(list(batch), device)
#             bl_aug = augment_batch(bl, disable_c=args.disable_aug_c)   # training augmentation
#             # [FIX] Bug thật đã tìm ra: trước đây augment_batch(bl) được
#             # gọi KHÔNG qua flag nào — --disable_aug_c chỉ được ghi vào
#             # footprint_info/console log ("[ABLATION] Disabled: AUG-C")
#             # nhưng KHÔNG có tác dụng thật, vì augment_batch() (định nghĩa
#             # trong flow_matching_model.py) không nhận tham số nào để tắt
#             # riêng nhánh C (recurvature). Đã thêm tham số disable_c vào
#             # augment_batch() (default=False, giữ nguyên hành vi cũ cho
#             # MỌI call site khác không truyền tham số này) và truyền đúng
#             # args.disable_aug_c vào đây. Mọi checkpoint no_aug_c/ablation
#             # đã train TRƯỚC fix này thực chất đã train VỚI AUG-C bật —
#             # cần re-run --disable_aug_c để có kết quả ablation thật.

#             opt.zero_grad()
#             with autocast(device_type="cuda", enabled=args.use_amp):
#                 bd = model.get_loss_breakdown(bl_aug, epoch=ep)

#             scaler.scale(bd["total"]).backward()
#             scaler.unscale_(opt)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

#             if freeze:
#                 for p in _unwrap(model).encoder.parameters():
#                     if p.grad is not None: p.grad.zero_()

#             scaler.step(opt); scaler.update()
#             model.ema_update()
#             swa.update(model)

#             sum_loss += bd["total"].item(); sum_cfm  += bd["l_cfm"]
#             sum_reg  += bd["l_reg"];        sum_head += bd["l_heading"]
#             sum_ade1 += bd["ade_1step"]

#             if i % 30 == 0:
#                 _, lr_vel = get_lrs(opt)
#                 enc_s = "frozen" if freeze else "active"
#                 swa_s = " [SWA]" if swa.active else ""
#                 print(f"  [{ep:>3}][{i:>3}/{nstep}]"
#                       f"  tot={bd['total'].item():.4f}"
#                       f"  cfm={bd['l_cfm']:.4f}"
#                       f"  reg={bd['l_reg']:.4f}"
#                       f"  h4s={bd['l_heading']:.4f}"
#                       f"  hreg={bd.get('l_hard_reg', 0.0):.4f}"
#                       f"  lam_d={bd['lam_dir']:.2f}"
#                       f"  ade1={bd['ade_1step']:.0f}km"
#                       f"  enc={enc_s}{swa_s}"
#                       f"  lr={lr_vel:.2e}")

#         train_loss = sum_loss / nstep
#         _, lr_vel_used = get_lrs(opt)
#         sched.step()

#         print(f"\n  ── Ep{ep:>3}"
#               f"  train={train_loss:.6f}"
#               f"  cfm={sum_cfm/nstep:.4f}"
#               f"  reg={sum_reg/nstep:.4f}"
#               f"  h4s={sum_head/nstep:.4f}"
#               f"  ade1={sum_ade1/nstep:.0f}km"
#               f"  lr={lr_vel_used:.2e}"
#               f"  t={time.perf_counter()-t0_ep:.0f}s")

#         _save(last_ckpt, ep, model, opt, sched, best_score, ema, scaler,
#               model_cfg=model_cfg)
#         if ep % 5 == 0:
#             ep_ckpt = os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth")
#             _save(ep_ckpt, ep, model, opt, sched, best_score, ema, scaler,
#                   model_cfg=model_cfg)
#             print(f"  💾 {ep_ckpt}")

#         # ── Val evaluation ───────────────────────────────────────────────
#         if rel_ep % args.val_freq == 0:
#             run_xai_this = (rel_ep % 10 == 0)
#             r = evaluate(model, val_loader, device, tag=f"VAL ep{ep}",
#                          n_ensemble=args.n_ensemble, ema=ema,
#                          ref_targets=ST_TRANS_VAL, epoch_for_loss=ep,
#                          run_xai=run_xai_this, xai_batch=xai_batch)

#             val_ade = r["ADE"]; score = r["combined_score"]
#             val_ade_history.append(val_ade)

#             # Trend
#             if len(val_ade_history) >= 4:
#                 trend = float(np.mean(val_ade_history[-2:])) - float(np.mean(val_ade_history[-4:-2]))
#                 trend_s = f"↑{trend:+.1f}km⚠" if trend > 5 else f"↓{trend:+.1f}km✓" if trend < -5 else f"→{trend:+.1f}flat"
#             else:
#                 trend_s = "—"
#             print(f"  train={train_loss:.6f}  val_ADE={val_ade:.1f}  combined={score:.1f}  trend={trend_s}")

#             # SWA activation check
#             if (not swa.active and ep >= args.swa_min_ep
#                     and swa.should_activate(val_ade_history, args.swa_window, args.swa_threshold)):
#                 swa.activate(model, opt, ep)

#             # Best checkpoint
#             if score < best_score:
#                 best_score = score; patience_cnt = 0
#                 _save(best_ckpt, ep, model, opt, sched, best_score, ema, scaler,
#                       extra={"val_ade": r["ADE"], "val_ate": r["ATE"],
#                              "val_cte": r["CTE"], "patience_cnt": 0},
#                       model_cfg=model_cfg)
#                 print(f"  ✅ Best! score={best_score:.2f}"
#                       f"  ADE={r['ADE']:.1f} ATE={r['ATE']:.1f} CTE={r['CTE']:.1f}")
#             else:
#                 if rel_ep >= args.min_ep: patience_cnt += args.val_freq
#                 print(f"  No improve {patience_cnt}/{args.patience} (best={best_score:.1f})")
#                 if rel_ep >= args.min_ep and patience_cnt >= args.patience:
#                     print(f"  ⛔ Early stop @ ep{ep}")
#                     break

#         # ── Hard-val evaluation ──────────────────────────────────────────
#         if rel_ep % args.hard_val_freq == 0 and rel_ep >= args.min_ep:
#             r_h = evaluate_hard_val(model, val_loader, device,
#                                      hard_threshold=args.hard_val_threshold,
#                                      n_ensemble=args.n_ensemble, ema=ema,
#                                      epoch_for_loss=ep)
#             print(f"  [HVAL] n={r_h['n_hard']}"
#                   f"  ADE={r_h['ADE']:.1f} ATE={r_h['ATE']:.1f} CTE={r_h['CTE']:.1f}"
#                   f"  combined={r_h['combined_score']:.1f}")
#             if r_h["combined_score"] < best_hard and r_h["n_hard"] >= 10:
#                 best_hard = r_h["combined_score"]
#                 _save(hard_best_ckpt, ep, model, opt, sched, best_hard, ema, scaler,
#                       extra={"hard_val_ade": r_h["ADE"], "selection_criterion": "hard_val"},
#                       model_cfg=model_cfg)
#                 print(f"  💎 Hard-best! score={best_hard:.2f} ADE={r_h['ADE']:.1f}")

#         # ── SWA eval ─────────────────────────────────────────────────────
#         if swa.active and rel_ep % args.val_freq == 0 and swa.n_updates >= 10:
#             backup = {k: v.detach().clone()
#                       for k, v in _unwrap(model).state_dict().items()
#                       if v.dtype.is_floating_point}
#             swa.apply_to_model(model)
#             r_swa = evaluate(model, val_loader, device, tag=f"SWA ep{ep}",
#                               n_ensemble=args.n_ensemble, ema=None,
#                               ref_targets=ST_TRANS_VAL, epoch_for_loss=ep)
#             swa.restore_from_backup(model, backup)

#             swa_score = r_swa["combined_score"]
#             print(f"  [SWA] score={swa_score:.2f} ({swa.n_updates} updates) vs best={best_score:.2f}")
#             if swa_score < best_score:
#                 best_score = swa_score; patience_cnt = 0
#                 swa.save_avg_state(swa_ckpt, ep, best_score,
#                                    extra={"val_ade": r_swa["ADE"],
#                                           "val_ate": r_swa["ATE"],
#                                           "val_cte": r_swa["CTE"]},
#                                    model_cfg=model_cfg)
#                 import shutil; shutil.copy(swa_ckpt, best_ckpt)
#                 print(f"  ✅ SWA best! score={best_score:.2f} ADE={r_swa['ADE']:.1f}")

#     # ── Wall-clock + footprint (ESWA Table E) ────────────────────────────
#     _wall_total = time.time() - _wall_start
#     print(f"\n  Training wall-clock: {_wall_total/3600:.2f}h ({_wall_total:.0f}s)")
#     try:
#         footprint_info.update({
#             "training_wall_clock_s": _wall_total,
#             "training_wall_clock_h": round(_wall_total/3600, 3),
#             "num_epochs": args.num_epochs,
#             "best_score": best_score,
#         })
#         import json as _json
#         fp_path = os.path.join(args.output_dir, "footprint.json")
#         with open(fp_path, "w") as _fp:
#             _json.dump(footprint_info, _fp, indent=2)
#         print(f"  Footprint saved → {fp_path}")
#     except Exception as _fe:
#         print(f"  Footprint save failed: {_fe}")

#     # ── Test evaluation ───────────────────────────────────────────────────
#     print(f"\n  Done! best_score={best_score:.2f}")
#     if not args.test_at_end: return

#     print("\n  Loading best checkpoint for TEST...")
#     if not os.path.exists(best_ckpt):
#         print("  No checkpoint found."); return

#     ck = torch.load(best_ckpt, map_location=device)
#     is_swa = ck.get("is_swa", False)
#     _unwrap(model).load_state_dict(ck["model"], strict=False)
#     if not is_swa and ema and ck.get("ema"):
#         for k, v in ck["ema"].items():
#             if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
#     print(f"  Loaded ep{ck.get('epoch','?')} (is_swa={is_swa})")

#     try:
#         _, test_loader = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
#         print(f"  Test: {len(test_loader)} batches")
#     except Exception:
#         print("  No test set → using val"); test_loader = val_loader

#     # Standard test
#     r_test = evaluate(model, test_loader, device, tag="TEST (best ckpt)",
#                       n_ensemble=args.n_ensemble, ema=None if is_swa else ema,
#                       ref_targets=ST_TRANS_TEST, run_xai=True, xai_batch=xai_batch)

#     # TTA test
#     if args.tta_test:
#         r_tta = evaluate(model, test_loader, device, tag="TEST+TTA",
#                          n_ensemble=args.n_ensemble, ema=None if is_swa else ema,
#                          ref_targets=ST_TRANS_TEST, use_tta=True, n_tta=args.n_tta)
#         print(f"\n  TTA: ADE {r_test['ADE']:.1f}→{r_tta['ADE']:.1f}  "
#               f"ATE {r_test['ATE']:.1f}→{r_tta['ATE']:.1f}  "
#               f"CTE {r_test['CTE']:.1f}→{r_tta['CTE']:.1f}")

#     # Multi-scale test
#     if args.multiscale_test:
#         raw_m = _unwrap(model)
#         if hasattr(raw_m, "sample_multiscale"):
#             print("\n  Multi-scale sigma test...")
#             ms_ades, ms_ates, ms_ctes = [], [], []
#             ms_steps = defaultdict(list)
#             raw_m.eval()
#             bk_ms = (ema.apply_to(model) if (ema and not is_swa) else None)
#             with torch.no_grad():
#                 for batch in test_loader:
#                     bl = move(list(batch), device)
#                     try:
#                         pred, _, _ = raw_m.sample_multiscale(bl)
#                     except Exception: continue
#                     gt = bl[1]; T = min(pred.shape[0], gt.shape[0])
#                     pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
#                     dist = _haversine_deg(pd, gd)
#                     ate, cte = _ate_cte(pd, gd)
#                     ms_ades.extend(dist.mean(0).tolist())
#                     if ate.shape[0] > 0:
#                         ms_ates.extend(ate.abs().mean(0).tolist())
#                         ms_ctes.extend(cte.abs().mean(0).tolist())
#                     for h, s in HORIZON_STEPS.items():
#                         if s < T: ms_steps[h].extend(dist[s].tolist())
#             if bk_ms: ema.restore(model, bk_ms)
#             def _mm(lst): return float(np.mean(lst)) if lst else float("nan")
#             print(f"  Multi-scale: ADE={_mm(ms_ades):.1f} ATE={_mm(ms_ates):.1f} CTE={_mm(ms_ctes):.1f}")

#     # Final comparison
#     val_ade_b = ck.get("val_ade", float("nan"))
#     val_ate_b = ck.get("val_ate", float("nan"))
#     val_cte_b = ck.get("val_cte", float("nan"))
#     v21 = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4}  # ST-Trans targets
#     print("\n" + "=" * 72)
#     print("  v2.1-XAI FINAL RESULTS vs ST-Trans target")
#     print("=" * 72)
#     print(f"  {'Metric':<10} {'Val':>10} {'Test':>10} {'Gap':>10} {'v2.1 Test':>12} Status")
#     print("  " + "─"*60)
#     for m_n, val_v, test_v, v21_v in [
#         ("ADE", val_ade_b, r_test["ADE"], v21["ADE"]),
#         ("ATE", val_ate_b, r_test["ATE"], v21["ATE"]),
#         ("CTE", val_cte_b, r_test["CTE"], v21["CTE"]),
#     ]:
#         gap  = test_v - val_v if (np.isfinite(test_v) and np.isfinite(val_v)) else float("nan")
#         impr = v21_v - test_v
#         flag = (f"↓{impr:+.1f}" if np.isfinite(impr) and impr > 0 else
#                 f"↑{-impr:+.1f}" if np.isfinite(impr) else "?")
#         print(f"  {m_n:<10} {val_v:>10.2f} {test_v:>10.2f} {gap:>+10.2f} {v21_v:>12.1f}  {flag}")
#     print("=" * 72)

#     # ── Auto evaluate_full + statistical_tests sau khi train xong ─────────
#     # Tự động chạy để không cần chạy tay khi train ngầm trên Kaggle.
#     # Kết quả lưu vào args.output_dir/eval/ và args.output_dir/stats/
#     _auto_eval(args, best_ckpt, device)


# def _auto_eval(args, best_ckpt: str, device):
#     """
#     Tự động chạy evaluate_full.py và statistical_tests.py sau khi train xong.
#     Tìm script trong cùng thư mục hoặc thư mục cha của train script.
#     Nếu không tìm thấy → bỏ qua, không làm crash training.
#     """
#     import subprocess, sys, json as _json

#     print("\n" + "="*72)
#     print("  AUTO EVALUATE + STATISTICAL TESTS")
#     print("="*72)

#     eval_dir  = os.path.join(args.output_dir, "eval")
#     stats_dir = os.path.join(args.output_dir, "stats")
#     os.makedirs(eval_dir,  exist_ok=True)
#     os.makedirs(stats_dir, exist_ok=True)

#     # Tìm evaluate_full.py: thử root project, thư mục cha scripts, cwd
#     script_dir  = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.abspath(os.path.join(script_dir, ".."))
#     candidates_eval = [
#         os.path.join(project_root, "evaluate_full.py"),
#         os.path.join(os.getcwd(),  "evaluate_full.py"),
#         os.path.join(script_dir,   "evaluate_full.py"),
#     ]
#     candidates_stat = [
#         os.path.join(project_root, "statistical_tests.py"),
#         os.path.join(os.getcwd(),  "statistical_tests.py"),
#         os.path.join(script_dir,   "statistical_tests.py"),
#     ]
#     eval_script = next((p for p in candidates_eval if os.path.exists(p)), None)
#     stat_script = next((p for p in candidates_stat if os.path.exists(p)), None)

#     # ── Step 1: evaluate_full.py ────────────────────────────────────────────
#     if eval_script is None:
#         print("  ⚠ evaluate_full.py không tìm thấy → bỏ qua auto-eval")
#         print(f"    Đặt file tại: {candidates_eval[0]}")
#     else:
#         print(f"  ▶ evaluate_full.py ({eval_script})")
#         ep = "best"
#         try:
#             import torch as _torch
#             ck_info = _torch.load(best_ckpt, map_location="cpu")
#             ep = ck_info.get("epoch", "best")
#         except Exception:
#             pass

#         cmd_eval = [
#             sys.executable, eval_script,
#             "--checkpoint",   best_ckpt,
#             "--dataset_root", args.dataset_root,
#             "--split",        "test",
#             "--output_dir",   eval_dir,
#             "--n_ensemble",   str(args.n_ensemble),
#             "--no_crps",      # tắt CRPS để chạy nhanh hơn (bật nếu cần)
#             "--gpu",          str(args.gpu_num),
#         ]
#         try:
#             result = subprocess.run(cmd_eval, capture_output=False, timeout=1800)
#             if result.returncode == 0:
#                 print(f"  ✅ evaluate_full done → {eval_dir}/")
#             else:
#                 print(f"  ❌ evaluate_full failed (code {result.returncode})")
#         except subprocess.TimeoutExpired:
#             print("  ⚠ evaluate_full timeout (30min) → bỏ qua")
#         except Exception as e:
#             print(f"  ⚠ evaluate_full error: {e}")

#     # ── Step 2: statistical_tests.py ────────────────────────────────────────
#     # Tìm JSON output từ evaluate_full
#     eval_json = None
#     if os.path.exists(eval_dir):
#         jsons = sorted([
#             os.path.join(eval_dir, f) for f in os.listdir(eval_dir)
#             if f.startswith("eval_test") and f.endswith(".json")
#         ])
#         if jsons:
#             eval_json = jsons[-1]  # file mới nhất

#     if stat_script is None:
#         print("  ⚠ statistical_tests.py không tìm thấy → bỏ qua")
#     elif eval_json is None:
#         print("  ⚠ Không tìm thấy eval JSON → bỏ qua statistical tests")
#         print(f"    Thử chạy lại evaluate_full.py trước")
#     else:
#         print(f"  ▶ statistical_tests.py ({stat_script})")
#         print(f"    FM result: {eval_json}")
#         cmd_stat = [
#             sys.executable, stat_script,
#             "--fm_results",       eval_json,
#             "--use_st_trans_ref",
#             "--fm_n_storms",      "420",
#             "--baseline_name",    "ST-Trans",
#             "--output_dir",       stats_dir,
#             "--n_bootstrap",      "10000",
#         ]
#         try:
#             result = subprocess.run(cmd_stat, capture_output=False, timeout=600)
#             if result.returncode == 0:
#                 print(f"  ✅ statistical_tests done → {stats_dir}/")
#             else:
#                 print(f"  ❌ statistical_tests failed (code {result.returncode})")
#         except subprocess.TimeoutExpired:
#             print("  ⚠ statistical_tests timeout (10min) → bỏ qua")
#         except Exception as e:
#             print(f"  ⚠ statistical_tests error: {e}")

#     # ── Summary JSON ─────────────────────────────────────────────────────────
#     summary_path = os.path.join(args.output_dir, "auto_eval_summary.json")
#     try:
#         summary = {
#             "checkpoint":    best_ckpt,
#             "eval_dir":      eval_dir,
#             "stats_dir":     stats_dir,
#             "eval_json":     eval_json,
#             "seed":          getattr(args, "seed", 42),
#             "ablation_name": getattr(args, "ablation_name", ""),
#         }
#         # Đọc kết quả ADE/ATE/CTE từ eval JSON nếu có
#         if eval_json and os.path.exists(eval_json):
#             with open(eval_json) as _f:
#                 ev = _json.load(_f)
#             summary["test_ADE"] = ev.get("ADE")
#             summary["test_ATE"] = ev.get("ATE")
#             summary["test_CTE"] = ev.get("CTE")
#             summary["test_RMSE"] = ev.get("RMSE")
#             summary["crps_mean"] = ev.get("crps", {}).get("mean")
#         with open(summary_path, "w") as _f:
#             _json.dump(summary, _f, indent=2)
#         print(f"\n  Summary → {summary_path}")
#         if summary.get("test_ADE"):
#             print(f"  ADE={summary['test_ADE']:.2f}  "
#                   f"ATE={summary['test_ATE']:.2f}  "
#                   f"CTE={summary['test_CTE']:.2f}")
#     except Exception as e:
#         print(f"  Summary save failed: {e}")

#     print("="*72)


# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(42); torch.manual_seed(42)
#     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
#     if args.dataset_root == "TCND_vn":
#         _auto = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
#         if os.path.isdir(_auto):
#             args.dataset_root = _auto
#     main(args)
"""

# ─────────────────────────────────────────────────────────────────────────────
# FILE PLACEMENT:
#
#   SOURCE:        train_fm_v24.py
#   KAGGLE TARGET: /kaggle/working/train_fm.py    (root, cạnh Model/)
#   LOCAL DEV:     train_fm.py
#
#   cp train_fm_v24.py /kaggle/working/train_fm.py
# ─────────────────────────────────────────────────────────────────────────────

scripts/train_fm.py  ──  TC-FlowMatching v2.1-XAI Training
═══════════════════════════════════════════════════════════════════════════════

Cải tiến từ v2.1 (dựa trên phân tích log 150 epochs):

  ROOT CAUSE: CTE gap val=45.7→test=71.6 (+25.9km, 57%) — thiếu direction training
  [AUG-C]  Recurvature ±20° — PROVEN -7.1km CTE (v2.5)
  [L_HDG]  L_heading bug fixed (no_grad removed), weight=0.07
  [CALIB]  Speed calibration tại inference (safe, no training change)
  [SWA]    Stochastic Weight Averaging (generalization)
  [HVAL]   Hard-val checkpoint (hard storms = test distribution)
  NO AUG-D: proven harmful (+4.5km ATE v2.6, +13.4km v2.7)

Giữ nguyên (proven):
  ✅ sigma_inference=0.04 FIXED
  ✅ L_reg softmax-linspace weights
  ✅ 2-group optimizer, encoder freeze 10ep
  ✅ val loop NO augmentation
  ✅ 1-shot inference
  ✅ SWA sau plateau
  ✅ Hard-val checkpoint (HVAL)
  ✅ XAI logging mỗi 10 epoch
"""
from __future__ import annotations
import json, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, math, time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler

from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _haversine_deg,
    EMAModel, augment_batch, hard_score_from_obs,
    compute_ensemble_uncertainty,
    compute_heading_deviation, compute_cte_contribution,
    compute_obs_attribution,
)

HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}
ST_TRANS_VAL  = {"ADE": 172.68, "ATE": 142.21, "CTE": 42.04,
                 "12h": 65.42, "24h": 104.67, "48h": 205.10, "72h": 321.39}
ST_TRANS_TEST = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
                 "12h": 77.5, "24h": 130.5, "48h": 269.9, "72h": 423.3}

# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m

def move(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
    return out

def _ate_cte(pred_deg, gt_deg):
    """Decompose error into along-track and cross-track components."""
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    lo1 = torch.deg2rad(gt_deg[:T-1,:,0]); la1 = torch.deg2rad(gt_deg[:T-1,:,1])
    lo2 = torch.deg2rad(gt_deg[1:T, :,0]); la2 = torch.deg2rad(gt_deg[1:T, :,1])
    lo3 = torch.deg2rad(pred_deg[1:T,:,0]); la3 = torch.deg2rad(pred_deg[1:T,:,1])
    ya  = torch.sin(lo2-lo1)*torch.cos(la2)
    xa  = torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
    be  = torch.atan2(ya, xa)
    ye  = torch.sin(lo3-lo2)*torch.cos(la3)
    xe  = torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
    bee = torch.atan2(ye, xe)
    tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang = bee - be
    return tot*torch.cos(ang), tot*torch.sin(ang)


# ─────────────────────────────────────────────────────────────────────────────
#  2-group optimizer (encoder freeze pattern from v2.1)
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model, lr_velocity, lr_encoder, weight_decay,
                     lr_logits_scale: float = 0.2):
    raw = _unwrap(model)

    # [CRITICAL FIX] v2.1-learn thêm 7 nhóm nn.Parameter trực tiếp trên
    # TCFlowMatching (KHÔNG nằm trong raw.encoder hay raw.velocity):
    #   speed_correction_logits, reg_step_logits, hard_score_weight_logits,
    #   heading_step_logits, log_sigma_reg, log_sigma_heading, log_sigma_calib
    #
    # BUG ĐÃ PHÁT HIỆN: optimizer cũ (2 param_groups: encoder + velocity)
    # KHÔNG cover các tham số này. loss.backward() vẫn tính đúng gradient
    # cho chúng (đã verify), nhưng optimizer.step() chỉ update params nằm
    # trong param_groups đã đăng ký -> 7 tham số "mồ côi": có gradient
    # nhưng KHÔNG BAO GIỜ thay đổi giá trị. Mọi cơ chế "tự học" (speed
    # calibration per-horizon, step weights, Kendall loss weighting...)
    # sẽ hoàn toàn vô tác dụng nếu không sửa optimizer này.
    #
    # FIX: liệt kê params đã thuộc encoder/velocity, lấy phần CÒN LẠI
    # (set difference) làm group thứ 3 — cách này tự động cover MỌI
    # tham số mới thêm sau này vào TCFlowMatching mà không cần liệt kê
    # tên cụ thể, tránh lặp lại lỗi tương tự nếu code model còn đổi nữa.
    #
    # [VAR-REDUCE v2, THỬ NGHIỆM — giả thuyết chưa chứng minh chặt, xem dưới]
    # SPLIT thêm learnable_extra thành 2 group:
    #   - "softmax_logits" (reg_step_logits, hard_score_weight_logits,
    #     heading_step_logits): các softmax nhỏ (4-12 chiều), khởi tạo
    #     GIỐNG HỆT NHAU mọi seed (zeros hoặc hằng số cố định — xem model.py,
    #     đã verify qua code, không phải suy đoán).
    #     QUAN SÁT: qua 4 lần train độc lập (seed 42/0/1/2), 72h heading
    #     deviation (đo trên 1 batch XAI cố định, KHÔNG phải test set) chia
    #     thành 2 cụm — {42,1}≈107° thấp hơn {0,2}≈118° — tương quan với
    #     test CTE ở r=0.811 (n=4). ĐÂY LÀ TƯƠNG QUAN ĐÁNG CHÚ Ý NHƯNG CHƯA
    #     ĐỦ MẠNH để khẳng định chắc chắn (cần |r|>0.95 với n=4 mới coi là
    #     gần chắc chắn không phải trùng hợp ngẫu nhiên) — coi đây là GIẢ
    #     THUYẾT đáng thử nghiệm, không phải nguyên nhân đã xác định.
    #     reg_step_logits có ramp GIÁN TIẾP qua ramp_reg (ep10→30, xem
    #     get_loss_breakdown) — cửa sổ gradient-mạnh-sớm là ep10-30, KHÔNG
    #     phải ep0-2. Cơ chế rủi ro: softmax không có ràng buộc tốc độ thay
    #     đổi; một chênh lệch gradient nhỏ trong cửa sổ ep10-30 (do random
    #     init của TOÀN mạng, không phải của chính logits) có thể bị khuếch
    #     đại qua softmax trước khi tín hiệu tích lũy nhiều epoch kịp điều
    #     chỉnh.
    #     FIX (thử nghiệm): KHÔNG ép giá trị đích về uniform (đã thử ở
    #     hard_score_weight_logits — làm chậm cả model, ADE/ATE tệ hơn vì
    #     early-horizon/dễ-học THẬT SỰ xứng đáng trọng số cao hơn, ép
    #     uniform chống lại tín hiệu thật). Thay vào đó CHỈ giảm LR cho
    #     riêng nhóm này (mặc định x0.2 so với velocity) — logits vẫn tự do
    #     hội tụ về BẤT KỲ đích nào tín hiệu thật dẫn tới, chỉ di chuyển
    #     CHẬM HƠN. speed_correction_logits KHÔNG đưa vào nhóm này: nó dùng
    #     sigmoid (độc lập từng phần tử) chứ không phải softmax (ràng buộc
    #     tổng=1 giữa các phần tử) — không có cùng cơ chế khuếch đại lẫn
    #     nhau nên không có cùng loại rủi ro.
    #   - "learnable_extra" (mọi thứ còn lại: speed_correction_logits,
    #     log_sigma_*): giữ nguyên LR như cũ, không đổi hành vi.
    encoder_ids  = {id(p) for p in raw.encoder.parameters()}
    velocity_ids = {id(p) for p in raw.velocity.parameters()}
    covered_ids  = encoder_ids | velocity_ids

    softmax_logit_names = {"reg_step_logits", "hard_score_weight_logits",
                            "heading_step_logits"}
    softmax_logit_params, rest_extra_params = [], []
    for name, p in raw.named_parameters():
        if id(p) in covered_ids:
            continue
        short = name.rsplit(".", 1)[-1]
        (softmax_logit_params if short in softmax_logit_names
         else rest_extra_params).append(p)

    groups = [
        {"params": list(raw.encoder.parameters()),  "lr": lr_encoder,  "name": "encoder"},
        {"params": list(raw.velocity.parameters()), "lr": lr_velocity, "name": "velocity"},
    ]
    if len(rest_extra_params) > 0:
        # Cùng lr với velocity: các tham số này tham gia trực tiếp vào
        # loss giống velocity (qua L_reg/L_heading/L_calib), không có lý
        # do để đóng băng theo encoder-freeze schedule.
        groups.append({"params": rest_extra_params, "lr": lr_velocity, "name": "learnable_extra"})
        print(f"  [build_optimizer] learnable_extra group: {len(rest_extra_params)} tensors "
              f"({sum(p.numel() for p in rest_extra_params)} params) — "
              f"speed_correction/log_sigma*")
    if len(softmax_logit_params) > 0:
        groups.append({"params": softmax_logit_params,
                        "lr": lr_velocity * lr_logits_scale,
                        "name": "softmax_logits"})
        print(f"  [build_optimizer] softmax_logits group: {len(softmax_logit_params)} tensors "
              f"({sum(p.numel() for p in softmax_logit_params)} params) — "
              f"reg_step/hard_score/heading_step @ lr×{lr_logits_scale} "
              f"(slower convergence, less seed-init sensitivity)")

    return optim.AdamW(groups, weight_decay=weight_decay)

def get_lrs(opt):
    lr_enc = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "encoder")
    lr_vel = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "velocity")
    return lr_enc, lr_vel

class TwoGroupScheduler:
    def __init__(self, opt, warmup_epochs, total_epochs,
                 lr_vel, lr_vel_min, freeze_end_ep,
                 lr_enc_peak, encoder_warmup_epochs=5):
        self.opt         = opt
        self.warmup      = warmup_epochs
        self.total       = total_epochs
        self.lr_vel      = lr_vel
        self.lr_vel_min  = lr_vel_min
        self.freeze_end  = freeze_end_ep
        self.lr_enc_peak = lr_enc_peak
        self.enc_warmup  = encoder_warmup_epochs
        self.epoch       = 0
        # [VAR-REDUCE v2 FIX] Without this, .step() below would overwrite
        # EVERY non-"encoder" group's lr with lr_vel every epoch — silently
        # erasing build_optimizer's lr_logits_scale after epoch 1 (the
        # "softmax_logits" group's LR would only differ from velocity's for
        # the very first .step() call, then get reset to lr_vel forever).
        # Record each group's ORIGINAL ratio to lr_vel at construction time,
        # so .step() can rescale proportionally instead of overwriting.
        self._lr_ratio = {}
        for pg in self.opt.param_groups:
            name = pg.get("name")
            if name not in (None, "encoder"):
                self._lr_ratio[name] = pg["lr"] / lr_vel if lr_vel > 0 else 1.0

    def _cosine(self, ep_from, ep_to, lr_s, lr_e, ep):
        t = max(0., min(1., (ep - ep_from) / max(ep_to - ep_from, 1)))
        return lr_e + 0.5 * (lr_s - lr_e) * (1 + math.cos(math.pi * t))

    def step(self):
        ep = self.epoch
        if ep < self.warmup:
            lr_vel = self.lr_vel * (0.1 + 0.9 * ep / max(self.warmup - 1, 1))
        else:
            lr_vel = self._cosine(self.warmup, self.total, self.lr_vel, self.lr_vel_min, ep)
        if ep < self.freeze_end:
            lr_enc = 0.0
        elif ep < self.freeze_end + self.enc_warmup:
            lr_enc = self.lr_enc_peak * (ep - self.freeze_end) / self.enc_warmup
        else:
            lr_enc = self._cosine(self.freeze_end + self.enc_warmup, self.total,
                                  self.lr_enc_peak, self.lr_vel_min, ep)
        for pg in self.opt.param_groups:
            name = pg.get("name")
            if name == "encoder":
                pg["lr"] = lr_enc
            elif name in self._lr_ratio:
                # Preserve this group's original ratio to lr_vel (e.g.
                # softmax_logits at lr_logits_scale=0.2) through the same
                # cosine schedule velocity follows, instead of resetting to
                # lr_vel outright.
                pg["lr"] = lr_vel * self._lr_ratio[name]
            else:
                pg["lr"] = lr_vel
        self.epoch += 1
        return lr_vel, lr_enc


# ─────────────────────────────────────────────────────────────────────────────
#  SWA handler (same as v2.5 — proven)
# ─────────────────────────────────────────────────────────────────────────────

class SWAHandler:
    """
    Stochastic Weight Averaging after val ADE plateau.
    Activation: val ADE improvement < threshold km in last `window` checks.
    After activation: average model weights each epoch → flatter loss landscape
    → better generalization.
    """
    def __init__(self, swa_lr: float = 2e-6):
        self.swa_lr    = swa_lr
        self.active    = False
        self.start_ep  = None
        self.n_updates = 0
        self.avg_state = {}

    def should_activate(self, ade_history: List[float],
                         window: int = 3, threshold: float = 1.5) -> bool:
        if len(ade_history) < window: return False
        return (ade_history[-window] - ade_history[-1]) < threshold

    def activate(self, model, opt, ep: int):
        self.active = True; self.start_ep = ep
        for pg in opt.param_groups: pg["lr"] = self.swa_lr
        m = _unwrap(model)
        self.avg_state = {k: v.detach().clone().float()
                          for k, v in m.state_dict().items()
                          if v.dtype.is_floating_point}
        self.n_updates = 1
        print(f"  *** SWA ACTIVATED @ ep{ep} (lr → {self.swa_lr:.1e}) ***")

    def update(self, model):
        if not self.active: return
        m = _unwrap(model); sd = m.state_dict(); n = self.n_updates
        for k in self.avg_state:
            if k in sd:
                self.avg_state[k] = (n * self.avg_state[k] + sd[k].detach().float()) / (n + 1)
        self.n_updates += 1

    def apply_to_model(self, model):
        if not self.active or not self.avg_state: return
        m = _unwrap(model); sd = m.state_dict()
        for k in self.avg_state:
            if k in sd: sd[k].copy_(self.avg_state[k].to(sd[k].device))

    def restore_from_backup(self, model, backup):
        m = _unwrap(model); sd = m.state_dict()
        for k, v in backup.items():
            if k in sd: sd[k].copy_(v)

    def save_avg_state(self, path: str, epoch: int, best_score: float,
                       extra: Optional[dict] = None, model_cfg=None):
        payload = {"epoch": epoch, "model": self.avg_state,
                   "best_score": best_score, "is_swa": True,
                   "swa_updates": self.n_updates,
                   "model_cfg": model_cfg}
        if extra: payload.update(extra)
        torch.save(payload, path)


# ─────────────────────────────────────────────────────────────────────────────
#  Hard-val evaluator (HVAL)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_hard_val(model, val_loader, device, hard_threshold: float = 0.35,
                       n_ensemble: int = 20, ema=None, epoch_for_loss: int = 9999):
    """
    Evaluate on hard storms only (hard_score > threshold).
    Hard storms are closer to test distribution (more recurvature).
    Best hard-val checkpoint often generalizes better than overall best.
    """
    bk = None
    if ema is not None:
        try: bk = ema.apply_to(model)
        except Exception: pass

    model.eval()
    all_ade, all_ate, all_cte = [], [], []
    n_hard = 0

    for batch in val_loader:
        bl = move(list(batch), device)
        B  = bl[0].shape[1]
        h_score   = hard_score_from_obs(bl[0][:, :, :2])
        hard_mask = h_score > hard_threshold
        if hard_mask.sum() == 0: continue
        hard_idx  = hard_mask.nonzero(as_tuple=True)[0]

        # Build sub-batch for hard storms
        bl_h = list(bl)
        for i, item in enumerate(bl_h):
            if torch.is_tensor(item):
                if item.dim() >= 2 and item.shape[1] == B:
                    bl_h[i] = item[:, hard_idx, ...]
                elif item.dim() >= 1 and item.shape[0] == B:
                    bl_h[i] = item[hard_idx, ...]

        try:
            pred, _, _ = model.sample(bl_h, num_ensemble=n_ensemble)
        except Exception as e:
            print(f"  hard val error: {e}"); continue

        gt = bl_h[1]
        T  = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
        dist = _haversine_deg(pd, gd)
        ate, cte = _ate_cte(pd, gd)
        all_ade.extend(dist.mean(0).tolist())
        if ate.shape[0] > 0:
            all_ate.extend(ate.abs().mean(0).tolist())
            all_cte.extend(cte.abs().mean(0).tolist())
        n_hard += len(hard_idx)

    if bk is not None:
        try: ema.restore(model, bk)
        except Exception: pass

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")
    return {
        "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
        "n_hard": n_hard,
        "combined_score": 0.6*_m(all_ade) + 0.2*_m(all_ate) + 0.2*_m(all_cte),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, tag: str = "",
             n_ensemble: int = 20, ema=None,
             ref_targets=None, use_tta: bool = False,
             n_tta: int = 5, epoch_for_loss: int = 9999,
             run_xai: bool = False, xai_batch=None) -> Dict:
    """
    Full evaluation. NO augmentation (val_loss is reliable generalization signal).
    XAI logging every 10 epochs shows training dynamics.
    """
    bk = None
    if ema is not None:
        try: bk = ema.apply_to(model)
        except Exception as e: print(f"  ⚠ EMA: {e}")

    model.eval()
    all_ade, all_ate, all_cte = [], [], []
    step_dist = defaultdict(list)
    sum_loss = sum_cfm = sum_head = 0.0
    sum_n = 0

    for batch in loader:
        bl = move(list(batch), device)
        gt = bl[1]; B = bl[0].shape[1]

        # Val loss (no augmentation)
        try:
            bd = model.get_loss_breakdown(bl, epoch=epoch_for_loss)
            if torch.isfinite(bd["total"]):
                sum_loss += bd["total"].item() * B
                sum_cfm  += bd["l_cfm"] * B
                sum_head += bd["l_heading"] * B
                sum_n    += B
        except Exception: pass

        # Inference (standard or speed-scale TTA)
        if use_tta:
            obs = bl[0]; anchor = obs[-1:, :, :2].detach()
            scales = [0.875, 0.9375, 1.0, 1.0625, 1.125][:n_tta]
            preds_t, weights_t = [], []
            for sc in scales:
                obs_s = obs.clone(); obs_s[..., :2] = anchor + (obs[..., :2] - anchor) * sc
                bl_s = list(bl); bl_s[0] = obs_s
                try:
                    p, _, _ = model.sample(bl_s, num_ensemble=n_ensemble)
                    preds_t.append(p)
                    weights_t.append(2.0 if abs(sc - 1.0) < 1e-6 else 1.0)
                except Exception: continue
            if not preds_t: continue
            tw   = sum(weights_t)
            pred = sum(w / tw * p for w, p in zip(weights_t, preds_t))
        else:
            try:
                # [MULTI-STEP] n_inference_steps default changed 1->10 (see
                # constructor comment) — this call doesn't override via
                # ddim_steps, so val/test loops now run ~10x more velocity
                # network forward passes per sample() call than before.
                # Expect val/test evaluation phases to take noticeably
                # longer; this is the real cost of the change, not a
                # separate slowdown bug.
                pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
            except Exception as e:
                print(f"  sample error: {e}"); continue

        T = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
        dist = _haversine_deg(pd, gd)
        ate, cte = _ate_cte(pd, gd)
        all_ade.extend(dist.mean(0).tolist())
        if ate.shape[0] > 0:
            all_ate.extend(ate.abs().mean(0).tolist())
            all_cte.extend(cte.abs().mean(0).tolist())
        for h, s in HORIZON_STEPS.items():
            if s < T: step_dist[h].extend(dist[s].tolist())

    if bk is not None:
        try: ema.restore(model, bk)
        except Exception: pass

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")
    val_loss = sum_loss / max(sum_n, 1)

    result = {
        "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
        "n": len(all_ade),
        "val_loss": val_loss,
        "val_cfm_loss":  sum_cfm / max(sum_n, 1),
        "val_head_loss": sum_head / max(sum_n, 1),
        "val_mom_loss":  0.0,   # always 0
    }
    for h in HORIZON_STEPS: result[f"{h}h"] = _m(step_dist[h])
    ade, ate_, cte_ = result["ADE"], result["ATE"], result["CTE"]
    result["combined_score"] = (
        0.6 * ade + 0.2 * ate_ + 0.2 * cte_
        if all(np.isfinite(x) for x in [ade, ate_, cte_]) else ade)

    ref = ref_targets or ST_TRANS_VAL
    def _v(k): return result.get(k, float("nan"))
    def _ok(k): return "✓" if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9) else "✗"

    tta_str = " [TTA]" if use_tta else ""
    print(f"\n  {'='*72}")
    print(f"  [{tag}]{tta_str}  n={result['n']}")
    print(f"  Val Loss : {val_loss:.6f}  cfm={result['val_cfm_loss']:.6f}  "
          f"head4s={result['val_head_loss']:.6f}  [mom=DISABLED]")
    print(f"  ADE={_v('ADE'):7.1f}km {_ok('ADE')}  "
          f"ATE={_v('ATE'):7.1f}km {_ok('ATE')}  "
          f"CTE={_v('CTE'):7.1f}km {_ok('CTE')}")
    print(f"  Combined = {_v('combined_score'):.1f}")
    print(f"  12h={_v('12h'):6.1f}  24h={_v('24h'):6.1f}  "
          f"48h={_v('48h'):6.1f}  72h={_v('72h'):6.1f} km")
    beat = [f"{k}={_v(k):.0f}<{ref.get(k,999):.0f}"
            for k in ["ADE","ATE","CTE","12h","24h","48h","72h"]
            if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9)]
    print(f"  BEAT: {' | '.join(beat) if beat else 'none yet'}")

    # XAI logging
    if run_xai and xai_batch is not None:
        try:
            _, _, _, xai = _unwrap(model).sample(xai_batch, return_xai=True)

            print(f"  {'─'*60}")
            print(f"  XAI Summary (fixed val batch)")
            print(f"  {'─'*60}")

            # XAI-4: Uncertainty
            print(f"  [XAI-4] Uncertainty:"
                  f" 12h={xai['mean_12h_std']:.1f}km"
                  f"  72h={xai['mean_72h_std']:.1f}km"
                  f"  ratio={float(xai['uncertainty_ratio'].mean()):.2f}×"
                  f"  high_uncert={xai['high_uncertainty'].sum().item()}")

            # XAI-2: Hard score components
            hc = xai.get("hard_components", {})
            if hc:
                print(f"  [XAI-2] HardScore:"
                      f" curv={float(hc['curvature'].mean()):.3f}"
                      f"  spd_var={float(hc['speed_var'].mean()):.3f}"
                      f"  dir_chg={float(hc['dir_change'].mean()):.3f}"
                      f"  obs_spd_n={float(hc.get('obs_speed_norm', torch.zeros(1)).mean()):.3f}")

            # XAI-3: Physics score components
            pc = xai.get("physics_components", {})
            if pc:
                print(f"  [XAI-3] Physics:"
                      f" speed={float(pc['speed'].mean()):.3f}"
                      f"  smooth={float(pc['smooth'].mean()):.3f}"
                      f"  heading={float(pc['heading'].mean()):.3f}")

            # XAI-5: Speed comparison (shows calibration effect)
            sc = xai.get("speed_comparison", {})
            if sc:
                ratio = sc.get("speed_ratio", 1.0)
                flag  = ("⚠ OVER" if ratio > 1.15 else
                         "⚠ UNDER" if ratio < 0.85 else "✓ OK")
                n_over  = sc.get("over_predict",  torch.zeros(1)).sum().item()
                n_under = sc.get("under_predict", torch.zeros(1)).sum().item()
                print(f"  [XAI-5] Speed (post-calibration):"
                      f" obs={sc['obs_speed_mean']:.1f}km/h"
                      f"  pred={sc['pred_speed_mean']:.1f}km/h"
                      f"  ratio={ratio:.2f} {flag}"
                      f"  (over:{int(n_over)} under:{int(n_under)})")

            # XAI-6: Heading deviation — key metric for heading loss efficacy
            hd = xai.get("heading_deviation_deg")
            if hd is not None and hd.shape[0] >= 1:
                hd_mean = hd.mean(1)
                print(f"  [XAI-6] Heading deviation:"
                      f" 12h={hd_mean[0].item():.1f}°"
                      f"  24h={hd_mean[min(2,len(hd_mean)-1)].item():.1f}°"
                      f"  72h={hd_mean[min(10,len(hd_mean)-1)].item():.1f}°"
                      f"  (target: 12h<20°, 72h<100°)")

            # XAI-7: ATE/CTE decomposition
            ac = xai.get("ate_cte_decomp", {})
            if ac:
                print(f"  [XAI-7] Error:"
                      f" ATE={ac['ate_abs_mean']:.1f}km"
                      f"  CTE={ac['cte_abs_mean']:.1f}km"
                      f"  ratio={ac['ate_abs_mean']/(ac['cte_abs_mean']+1e-3):.2f}")

            # XAI-8: Per-horizon speed [v2.1-XAI]
            sph = xai.get("speed_per_horizon", {})
            if sph and "pred_kmh" in sph:
                ps = sph["pred_kmh"]; gs = sph["gt_kmh"]; rs = sph["ratio"]
                hz = [(0,"12h"),(2,"24h"),(6,"48h"),(10,"72h")]
                parts = []
                for idx, lbl in hz:
                    if idx < len(rs):
                        flag = "⚠" if rs[idx] > 1.3 or rs[idx] < 0.7 else "✓"
                        parts.append(f"{lbl}:r={rs[idx]:.2f}{flag}")
                print(f"  [XAI-8] Speed/horizon: {' | '.join(parts)}")

            # XAI-9: Storm category [v2.1-XAI]
            sc9 = xai.get("storm_categories", {})
            if sc9:
                print(f"  [XAI-9] Storms:"
                      f" slow={sc9.get('n_slow',0)}"
                      f"  med={sc9.get('n_medium',0)}"
                      f"  fast={sc9.get('n_fast',0)}"
                      f"  spd={sc9.get('speed_mean',0):.1f}±{sc9.get('speed_std',0):.1f}km/h")

            # [LEARN] Learned parameter values — theo dõi model có thực sự
            # học hay không qua các epoch (nếu các giá trị này đứng yên ở
            # init suốt training, đó là dấu hiệu optimizer không cover
            # chúng — xem build_optimizer's 'learnable_extra' param group)
            lp = xai.get("learned_params", {})
            if lp:
                sc_str  = ",".join(f"{v:.2f}" for v in lp.get("speed_correction", [])[:4])
                rw_str  = ",".join(f"{v:.3f}" for v in lp.get("reg_step_weights", [])[:4])
                # [FIX] full 12-horizon reg_step_weights + b_horizon, not just
                # the first 4 — the far-horizon values (index 7,11 = 48h,72h)
                # are exactly what --disable_horizon_nll's fix targets, and
                # were invisible in the old [:4]-truncated print.
                rw_full = lp.get("reg_step_weights", [])
                rw_far  = ",".join(f"{v:.3f}" for v in rw_full[7:12]) if len(rw_full) >= 12 else "?"
                bh_str  = ",".join(f"{v:.1f}" for v in lp.get("b_horizon", [])[:12])
                hw_str  = ",".join(f"{v:.3f}" for v in lp.get("hard_score_weights", []))
                hdw_str = ",".join(f"{v:.3f}" for v in lp.get("heading_step_weights", [])[:4])
                sig_inf = lp.get("sigma_inf", float("nan"))
                ls_r    = lp.get("log_sigma_reg",     float("nan"))
                ls_h    = lp.get("log_sigma_heading",  float("nan"))
                ls_c    = lp.get("log_sigma_calib",    float("nan"))
                el_r    = lp.get("eff_lambda_reg",     float("nan"))
                el_h    = lp.get("eff_lambda_heading", float("nan"))
                el_c    = lp.get("eff_lambda_calib",   float("nan"))
                print(f"  [LEARN] speed_corr(12h-24h)=[{sc_str}]"
                      f"  reg_w(12h-24h)=[{rw_str}]")
                print(f"  [LEARN] reg_w(48h-72h,idx7-11)=[{rw_far}]"
                      f"  b_horizon(6h..72h)=[{bh_str}]")
                print(f"  [LEARN] hard_w(curv,spdvar,dirchg,obsspd)=[{hw_str}]"
                      f"  head_w(12h-24h)=[{hdw_str}]  sigma_inf={sig_inf:.4f}")
                print(f"  [LEARN] log_sigma: reg={ls_r:.3f}  heading={ls_h:.3f}  calib={ls_c:.3f}"
                      f"  |  eff_lambda: reg={el_r:.3f}  heading={el_h:.3f}  calib={el_c:.3f}")

            print(f"  {'─'*60}")
            result["xai"] = xai
        except Exception as e:
            print(f"  XAI error: {e}")
            import traceback; traceback.print_exc()

    print(f"  {'='*72}\n")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _save(path, epoch, model, opt, sched, best_score,
          ema=None, scaler=None, extra=None, model_cfg=None):
    m   = _unwrap(model)
    esd = None
    if ema is not None:
        try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
        except Exception: pass
    payload = {
        "epoch": epoch, "model": m.state_dict(),
        "optimizer": opt.state_dict(), "scheduler": sched.epoch,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_score": best_score, "best_ade": best_score, "ema": esd,
        # [FIX] self-describing checkpoint — see call site comment in main()
        "model_cfg": model_cfg,
    }
    if extra: payload.update(extra)
    torch.save(payload, path)


# ─────────────────────────────────────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    p.add_argument("--dataset_root",           default="TCND_vn")
    p.add_argument("--obs_len",                default=8,      type=int)
    p.add_argument("--pred_len",               default=12,     type=int)
    p.add_argument("--num_workers",            default=2,      type=int)
    p.add_argument("--other_modal",            default="gph")
    p.add_argument("--delim",                  default=" ")
    p.add_argument("--skip",                   default=1,      type=int)
    p.add_argument("--min_ped",                default=1,      type=int)
    p.add_argument("--threshold",              default=0.002,  type=float)
    # [FIX-DATA-30] Region filter — keep only storms whose track has
    # >= min_pct_in_scs% of points in the South China Sea / Vietnam
    # region. Off by default to preserve prior behavior; opt in
    # explicitly with --filter_region for a Vietnam/SCS-focused model.
    p.add_argument("--filter_region",          action="store_true", default=False,
                   help="Keep only storms whose track substantially enters "
                        "the South China Sea / Vietnam region (see "
                        "--min_pct_in_scs). Off by default.")
    p.add_argument("--min_pct_in_scs",         default=15.0,   type=float,
                   help="Minimum %% of a storm's track points that must fall "
                        "inside the South China Sea / Vietnam box (lon "
                        "99-121E, lat 0-23N) for it to be kept when "
                        "--filter_region is set.")
    # Model
    p.add_argument("--d_cond",                 default=256,    type=int)
    p.add_argument("--d_model",                default=256,    type=int)
    p.add_argument("--nhead",                  default=8,      type=int)
    p.add_argument("--num_dec_layers",         default=4,      type=int)
    p.add_argument("--dim_ff",                 default=512,    type=int)
    p.add_argument("--dropout",                default=0.1,    type=float)
    p.add_argument("--unet_in_ch",             default=13,     type=int)
    p.add_argument("--sigma_min",              default=0.06,   type=float,
                   help="[FIX, user-approved tradeoff] Min noise scale for "
                        "CFM sampling. CHANGED default from 0.04 (~22km) to "
                        "0.06 (~33km) — 0.04 was only ~10-20% of actual "
                        "error magnitude, too small to seed meaningful "
                        "ensemble divergence (suspected contributor to "
                        "near-zero CRPS/Spread-Skill). NOT experimentally "
                        "validated on a real checkpoint (unlike "
                        "n_inference_steps) — this is a directional fix "
                        "based on the km-scale analysis, not a proven-best "
                        "value. Pass --sigma_min 0.04 to revert to the "
                        "original.")
    p.add_argument("--sigma_max",              default=0.15,   type=float,
                   help="[FIX, user-approved tradeoff] Max noise scale. "
                        "CHANGED default from 0.08 (~44km) to 0.15 (~83km), "
                        "closer to actual CTE-scale error (67km), while "
                        "staying below full ADE scale (~219km). Pass "
                        "--sigma_max 0.08 to revert to the original.")
    p.add_argument("--sigma_decay_end",        default=100,    type=int,
                   help="[FIX, user-approved tradeoff] Epoch at which sigma "
                        "reaches sigma_min and stays there. CHANGED default "
                        "from 40 to 100 — with 40, >100 epochs of a typical "
                        "140-165 epoch run trained at minimum noise. Pass "
                        "--sigma_decay_end 40 to revert to the original.")
    p.add_argument("--lambda_reg",             default=0.2,    type=float)
    p.add_argument("--lambda_heading",         default=0.07,   type=float,
                   help="[v2.1-XAI] multi-step heading, bug fixed, weight=0.07")
    p.add_argument("--lambda_momentum",        default=0.0,    type=float,
                   help="[v2.6] DISABLED — hurt test ATE by +7.9km")
    p.add_argument("--lambda_hard_reg",        default=0.02,   type=float,
                   help="[VAR-REDUCE] Fixed weight (not Kendall-learned) L2 "
                        "penalty pulling hard_score_weight_logits' softmax "
                        "toward uniform. Addresses seed-to-seed CTE variance "
                        "traced to this 4-way weighting collapsing onto "
                        "'curvature' at different rates per seed (confirmed "
                        "across seed 42/1/2 runs at matched epochs). "
                        "0.0 = fully disabled (same as --disable_hard_reg).")
    p.add_argument("--log_sigma_reg_min_clamp", default=-3.0,   type=float,
                   help="Numerical-stability clamp floor for log_sigma_reg "
                        "(prevents exp() overflow), NOT a tuning knob. "
                        "Default -3.0 -> cap~=202, far above the ~15-20 "
                        "eff_lambda_reg converges to — reconsidered from an "
                        "earlier proposal to tighten this as a way to force "
                        "L_reg's weight down: that would fight a genuine "
                        "loss-minimizing choice Kendall converges to (not a "
                        "bug), same class of change that hurt ADE/ATE when "
                        "tried on hard_score_weight_logits. See "
                        "--disable_horizon_nll instead for the principled "
                        "fix to far-horizon attention starvation.")
    p.add_argument("--disable_horizon_nll",    action="store_true", default=False,
                   help="[FIX] Disable log_b_horizon (Laplace-NLL per-"
                        "horizon error normalization) — falls back to "
                        "reg_step_logits' softmax attending to RAW "
                        "haversine distance per horizon (the original "
                        "formula). Default (flag absent) uses the NEW "
                        "normalized formula: reg_step_logits' softmax was "
                        "found to systematically starve far horizons "
                        "(48h-72h get only ~7-13% combined attention) "
                        "because raw error grows ~14x from 6h to 72h — not "
                        "a bug, but a rational response to an unnormalized "
                        "signal. The fix normalizes dist[t] by a LEARNED "
                        "per-horizon scale b_horizon (via a stable Laplace "
                        "NLL, same log(b) counterbalance structure as "
                        "Kendall's Gaussian NLL elsewhere in this model — "
                        "provably cannot degenerate to 0 or explode). "
                        "Initializes as an exact no-op (b=1 -> nll_h=dist) "
                        "so it only diverges from original behavior as "
                        "training reveals genuine per-horizon difficulty. "
                        "CAUTION: this is a NEW mechanism, not yet "
                        "empirically validated on a real training run "
                        "(unlike --n_inference_steps, which IS verified) — "
                        "the math is sound but watch ADE/ATE/CTE together "
                        "with CRPS/Spread-Skill to confirm it helps far-"
                        "horizon diversity without hurting near-horizon "
                        "accuracy before trusting the result.")
    p.add_argument("--use_ot",                 default=True,   action="store_true")
    p.add_argument("--no_ot",                  dest="use_ot",  action="store_false")
    p.add_argument("--ot_epsilon",             default=0.05,   type=float)
    p.add_argument("--n_ensemble",             default=20,     type=int)
    p.add_argument("--sigma_inference",        default=0.04,   type=float)
    p.add_argument("--n_inference_steps",      default=10,     type=int,
                   help="[MULTI-STEP] Euler integration steps baked into "
                        "the checkpoint (used at inference when eval "
                        "doesn't override via --ddim_steps). CHANGED "
                        "default from 1 (single-shot x0+v) to 10 — this is "
                        "the ONE change in this codebase backed by direct "
                        "experimental evidence (not just theory): evaluating "
                        "an existing checkpoint with N=1->10 raised ensemble "
                        "spread 4km->55km, improved CRPS -9.5% (216.8->"
                        "196.3km), Spread/Skill ratio at 6h reached 1.07 "
                        "(near-ideal 1.0), at a small ADE/ATE/CTE cost "
                        "(+1.7%/+1.9%/+7.2%). N=3-4 tested WORSE than both "
                        "N=1 and N=8-10 — standard Euler discretization "
                        "error is non-monotonic here, don't assume any N "
                        "value between 1 and 10 is safe without testing. "
                        "Pass 1 to reproduce the exact original single-shot "
                        "behavior. NOTE: this does NOT fix ensemble spread "
                        "stalling at far horizons (48h-72h) — that traces "
                        "to reg_step_logits' softmax naturally down-"
                        "weighting hard-to-reduce far-horizon loss, a "
                        "genuine optimization outcome verified NOT to be a "
                        "bug (see flow_matching_model.py's _reg_loss). No "
                        "fix for that is applied by default in this "
                        "codebase — it would require overriding what the "
                        "model has genuinely learned is loss-optimal, which "
                        "needs deliberate experimentation, not a silent "
                        "default change.")
    # Training
    p.add_argument("--num_epochs",             default=150,    type=int)
    p.add_argument("--batch_size",             default=64,     type=int)
    p.add_argument("--lr",                     default=2e-4,   type=float)
    p.add_argument("--lr_logits_scale",        default=0.2,    type=float,
                   help="[VAR-REDUCE v2] LR multiplier (vs --lr) for the "
                        "small softmax logits (reg_step_logits, "
                        "hard_score_weight_logits, heading_step_logits). "
                        "Slower convergence for these reduces how much a "
                        "seed's early-training noise can lock them into an "
                        "extreme configuration before the true signal "
                        "(accumulated over many epochs) has a chance to "
                        "shape them — without pulling their VALUE toward "
                        "uniform (that was tried on hard_score_weight_logits "
                        "and hurt ADE/ATE by fighting the genuine easier-to"
                        "-learn-early-horizon signal). 1.0 = old behavior "
                        "(same LR as velocity, i.e. disabled).")
    p.add_argument("--lr_min",                 default=1e-6,   type=float)
    p.add_argument("--warmup_epochs",          default=5,      type=int)
    p.add_argument("--weight_decay",           default=1e-4,   type=float)
    p.add_argument("--grad_clip",              default=1.0,    type=float)
    p.add_argument("--use_amp",                action="store_true", default=False)
    p.add_argument("--use_ema",                default=True,   action="store_true")
    p.add_argument("--no_ema",                 dest="use_ema", action="store_false")
    # Encoder freeze
    p.add_argument("--freeze_encoder_epochs",  default=10,     type=int)
    p.add_argument("--encoder_warmup_epochs",  default=5,      type=int)
    p.add_argument("--lr_enc_peak",            default=5e-5,   type=float)
    # Eval
    p.add_argument("--val_freq",               default=5,      type=int)
    p.add_argument("--patience",               default=20,     type=int)
    p.add_argument("--min_ep",                 default=20,     type=int)
    p.add_argument("--hard_val_threshold",     default=0.35,   type=float)
    p.add_argument("--hard_val_freq",          default=10,     type=int)
    # SWA
    p.add_argument("--swa_lr",                 default=2e-6,   type=float)
    p.add_argument("--swa_window",             default=3,      type=int)
    p.add_argument("--swa_threshold",          default=1.5,    type=float)
    p.add_argument("--swa_min_ep",             default=50,     type=int)
    # Test
    p.add_argument("--tta_test",               default=True,   action="store_true")
    p.add_argument("--n_tta",                  default=5,      type=int)
    p.add_argument("--multiscale_test",        default=True,   action="store_true")
    # IO
    p.add_argument("--output_dir",             default="runs/fm_v26")
    p.add_argument("--gpu_num",                default="0")
    p.add_argument("--resume",                 default=None)
    p.add_argument("--test_at_end",            action="store_true", default=True)
    p.add_argument("--no_test",                dest="test_at_end", action="store_false")
    # ── ESWA paper: reproducibility + ablation ────────────────────────────
    p.add_argument("--seed",                   type=int, default=42,
                   help="Random seed. Run 3-5 seeds for ESWA mean±std reporting.")
    p.add_argument("--disable_l_heading",      action="store_true", default=False,
                   help="Ablation: disable L_heading_ms")
    p.add_argument("--disable_l_calib",        action="store_true", default=False,
                   help="Ablation: disable L_calib (speed correction training)")
    p.add_argument("--disable_l_reg",          action="store_true", default=False,
                   help="Ablation: disable L_reg (CFM-only variant)")
    p.add_argument("--disable_aug_c",          action="store_true", default=False,
                   help="Ablation: disable AUG-C recurvature")
    p.add_argument("--disable_learned_weights",action="store_true", default=False,
                   help="Ablation: fixed lambda (no Kendall weighting)")
    p.add_argument("--disable_hard_reg",       action="store_true", default=False,
                   help="Ablation: disable hard_score_weight_logits uniform "
                        "regularizer (same as --lambda_hard_reg 0.0)")
    p.add_argument("--ablation_name",          type=str, default="",
                   help="Suffix for output_dir to tag ablation variant")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── Seed (ESWA: run with 3-5 seeds, report mean±std) ──────────────────
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Ablation: append name to output dir if provided
    if args.ablation_name:
        args.output_dir = f"{args.output_dir}_{args.ablation_name}"
    if args.seed != 42:
        args.output_dir = f"{args.output_dir}_seed{args.seed}"

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # [FIX] Print device/GPU diagnostics immediately — without this, a
    # silent CPU fallback (or a slow/shared GPU) is only noticed hours
    # into training when epoch timing looks off, instead of in the
    # first few seconds. Same fix already applied to
    # train_paper_baseline.py and train_st_trans.py — this brings
    # train_flowmatching.py in line with those.
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"(CUDA {torch.version.cuda}, {torch.cuda.device_count()} visible)")
    else:
        print(f"  ⚠ No GPU detected — training on CPU will be MUCH slower "
              f"(often 10-50x). If you expected a GPU here, check Kaggle's "
              f"Accelerator setting for this session.")
    os.makedirs(args.output_dir, exist_ok=True)
    # Wall-clock tracking
    _wall_start = time.time()
    best_ckpt      = os.path.join(args.output_dir, "best_model.pth")
    hard_best_ckpt = os.path.join(args.output_dir, "hard_best_model.pth")
    swa_ckpt       = os.path.join(args.output_dir, "swa_model.pth")
    last_ckpt      = os.path.join(args.output_dir, "last_model.pth")

    print("=" * 72)
    print("  TC-FlowMatching v2.1-XAI")
    print(f"  [AUG-C]  Recurvature ±20° (PROVEN -7.1km CTE in v2.5)")
    print(f"  [L_HDG]  L_heading bug fixed, weight={args.lambda_heading}")
    print(f"  [CALIB]  Speed calibration at inference (clip 0.85-1.15)")
    print(f"  [NO-D]   AUG-D removed (proven +4.5km ATE worse)")
    print(f"  KEEP:    sigma_inf={args.sigma_inference} FIXED, L_reg linear, OT")
    print("=" * 72)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n  Loading data...")
    trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
    vd, val_loader = data_loader(args, {"root": args.dataset_root, "type": "val"}, test=True)
    print(f"  train: {len(trd)} ({len(trl)} batches/ep)")
    print(f"  val:   {len(vd)} ({len(val_loader)} batches)")

    # Fixed XAI batch (same every epoch for comparability)
    try:
        xai_batch = move(list(next(iter(val_loader))), device)
    except Exception:
        xai_batch = None

    # ── Model ─────────────────────────────────────────────────────────────
    model = TCFlowMatching(
        pred_len=args.pred_len,        obs_len=args.obs_len,
        unet_in_ch=args.unet_in_ch,   d_cond=args.d_cond,
        d_model=args.d_model,          nhead=args.nhead,
        num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
        dropout=args.dropout,
        sigma_min=args.sigma_min,      sigma_max=args.sigma_max,
        sigma_decay_end=args.sigma_decay_end,
        lambda_reg=args.lambda_reg,    lambda_heading=args.lambda_heading,
        lambda_momentum=0.0,
        lambda_hard_reg=(0.0 if args.disable_hard_reg else args.lambda_hard_reg),
        log_sigma_reg_min_clamp=args.log_sigma_reg_min_clamp,
        enable_horizon_nll=not args.disable_horizon_nll,
        use_ot=args.use_ot,            ot_epsilon=args.ot_epsilon,
        use_ema=args.use_ema,          n_ensemble=args.n_ensemble,
        n_inference_steps=args.n_inference_steps,
        sigma_inference=args.sigma_inference,
    ).to(device)

    # [FIX] Checkpoints must self-describe their architecture. Without this,
    # evaluate_full.py's `TCFlowMatching(**ck.get("model_cfg", {}))` silently
    # falls back to ALL constructor defaults whenever model_cfg is missing —
    # harmless only by coincidence when CLI defaults happen to match
    # constructor defaults (true today for d_model/nhead/num_dec_layers), but
    # a real correctness risk the moment anyone trains with a non-default
    # architecture (e.g. --d_model 128): eval would silently reconstruct the
    # WRONG architecture, load_state_dict(strict=False) would mismatch
    # shapes silently degrade to partial loading instead of a clear error.
    # This dict mirrors the constructor call above exactly.
    model_cfg = dict(
        pred_len=args.pred_len,        obs_len=args.obs_len,
        unet_in_ch=args.unet_in_ch,   d_cond=args.d_cond,
        d_model=args.d_model,          nhead=args.nhead,
        num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
        dropout=args.dropout,
        sigma_min=args.sigma_min,      sigma_max=args.sigma_max,
        sigma_decay_end=args.sigma_decay_end,
        lambda_reg=args.lambda_reg,    lambda_heading=args.lambda_heading,
        lambda_momentum=0.0,
        lambda_hard_reg=(0.0 if args.disable_hard_reg else args.lambda_hard_reg),
        log_sigma_reg_min_clamp=args.log_sigma_reg_min_clamp,
        enable_horizon_nll=not args.disable_horizon_nll,
        use_ot=args.use_ot,            ot_epsilon=args.ot_epsilon,
        use_ema=args.use_ema,          n_ensemble=args.n_ensemble,
        n_inference_steps=args.n_inference_steps,
        sigma_inference=args.sigma_inference,
    )

    model.init_ema()
    ema = getattr(_unwrap(model), "_ema", None)
    raw = _unwrap(model)
    n_enc = sum(p.numel() for p in raw.encoder.parameters())
    n_vel = sum(p.numel() for p in raw.velocity.parameters())
    encoder_ids  = {id(p) for p in raw.encoder.parameters()}
    velocity_ids = {id(p) for p in raw.velocity.parameters()}
    n_extra = sum(p.numel() for p in raw.parameters()
                  if id(p) not in encoder_ids and id(p) not in velocity_ids)
    n_total = n_enc + n_vel + n_extra
    mem_mb  = sum(p.numel()*p.element_size() for p in model.parameters()) / 1e6
    print(f"\n  Encoder: {n_enc:,}  VelocityTrans: {n_vel:,}  "
          f"LearnableExtra: {n_extra:,}  Total: {n_total:,}  Mem: {mem_mb:.1f}MB")
    # Log computational footprint for ESWA paper Table E
    footprint_info = {
        "n_encoder": n_enc, "n_velocity": n_vel, "n_extra": n_extra,
        "n_total": n_total, "mem_mb": mem_mb,
        "seed": args.seed,
        "ablation_name": args.ablation_name or "full",
        "disable_l_heading": getattr(args, "disable_l_heading", False),
        "disable_l_calib":   getattr(args, "disable_l_calib", False),
        "disable_l_reg":     getattr(args, "disable_l_reg", False),
        "disable_aug_c":     getattr(args, "disable_aug_c", False),
        "disable_hard_reg":  getattr(args, "disable_hard_reg", False),
        "lambda_hard_reg":   getattr(args, "lambda_hard_reg", 0.02),
        # [FIX] 2 ablation flags dưới đây đã có mặt trong model_cfg (nên
        # checkpoint vẫn load lại đúng kiến trúc), nhưng bị BỎ SÓT khỏi
        # footprint_info — tức file auto_eval_summary.json / bất kỳ chỗ
        # nào đọc footprint_info sẽ KHÔNG biết run này có --no_ot hay
        # --disable_horizon_nll hay không, dễ gây nhầm lẫn khi tra cứu
        # lại nhiều run cùng lúc (đặc biệt 3 ablation seed=0 chạy song
        # song: no_ot / no_horizon_nll / no_hard_reg).
        "disable_horizon_nll": getattr(args, "disable_horizon_nll", False),
        "use_ot":              getattr(args, "use_ot", True),
    }

    # ── Ablation: patch get_loss_breakdown ─────────────────────────────────
    # [FIX] --disable_horizon_nll và --no_ot đều là cấu hình model-level
    # thật (nằm trong model_cfg, ảnh hưởng constructor TCFlowMatching),
    # nhưng không thuộc nhóm ablation "patch get_loss_breakdown" bên dưới
    # (chúng không cần patch gì — enable_horizon_nll/use_ot đã được
    # constructor xử lý trực tiếp). Trước đây 2 flag này hoàn toàn
    # KHÔNG được log ra console — chạy --no_ot hay --disable_horizon_nll
    # sẽ không in dòng [ABLATION] nào, dễ tưởng nhầm flag không có tác
    # dụng dù model_cfg (và do đó checkpoint) vẫn đúng. Thêm log riêng.
    if args.disable_horizon_nll or not args.use_ot:
        print(f"  [ABLATION] Model-level: "
              f"{'no_horizon_nll(raw dist, no log_b_horizon) ' if args.disable_horizon_nll else ''}"
              f"{'no_OT(random x0/x1 pairing) ' if not args.use_ot else ''}")

    if any([args.disable_l_heading, args.disable_l_calib, args.disable_l_reg,
            args.disable_aug_c, args.disable_learned_weights, args.disable_hard_reg]):
        print(f"  [ABLATION] Disabled: "
              f"{'L_heading ' if args.disable_l_heading else ''}"
              f"{'L_calib ' if args.disable_l_calib else ''}"
              f"{'L_reg ' if args.disable_l_reg else ''}"
              f"{'AUG-C ' if args.disable_aug_c else ''}"
              f"{'Kendall_weights ' if args.disable_learned_weights else ''}"
              f"{'HardScoreReg ' if args.disable_hard_reg else ''}")
        _orig_glb = raw.get_loss_breakdown.__func__
        _abl_flags = {
            "disable_l_heading": args.disable_l_heading,
            "disable_l_calib":   args.disable_l_calib,
            "disable_l_reg":     args.disable_l_reg,
            "disable_learned_weights": args.disable_learned_weights,
            "disable_hard_reg":  args.disable_hard_reg,
        }
        def _patched_glb(self, batch_list, epoch=0, **kwargs):
            bd = _orig_glb(self, batch_list, epoch=epoch, **kwargs)
            import math as _m
            # [FIX] Use GRAPH-CONNECTED tensors (bd["_t_l_*"]), NOT the
            # detached float values (bd["l_reg"] etc, from .item()). The
            # previous version of this patch rebuilt `total` from those
            # detached floats — wrapping a float in torch.tensor() carries
            # NO gradient, so backward() through that `total` only reached
            # the Kendall log_sigma* params, not the encoder/velocity
            # network. Every ablation run using this patch trained a
            # near-frozen network. Fixed by reading the graph-connected
            # tensors the model now exposes (_t_l_cfm, _t_l_reg, etc).
            l_cfm      = bd["_t_l_cfm"]
            l_reg      = bd["_t_l_reg"]
            l_heading  = bd["_t_l_heading"]
            l_calib    = bd["_t_l_calib"]
            # [VAR-REDUCE] l_hard_reg must be explicitly carried through this
            # rebuild — it's NOT part of the original total = l_cfm+reg+
            # heading+calib formula this patch reconstructs, so omitting it
            # would silently disable the uniform-regularizer on EVERY
            # ablation run (e.g. --disable_l_calib), not just when the user
            # actually asked for --disable_hard_reg.
            l_hard_reg = bd["_t_l_hard_reg"]
            if _abl_flags["disable_l_reg"]:     l_reg      = l_reg * 0.0
            if _abl_flags["disable_l_heading"]: l_heading  = l_heading * 0.0
            if _abl_flags["disable_l_calib"]:   l_calib    = l_calib * 0.0
            if _abl_flags["disable_hard_reg"]:  l_hard_reg = l_hard_reg * 0.0
            HALF = 0.5 * _m.log(2.0 * _m.pi)
            if _abl_flags["disable_learned_weights"]:
                total = (l_cfm
                         + bd["lam_reg"]   * 0.20 * l_reg
                         + bd["lam_dir"]   * 0.07 * l_heading
                         + bd["lam_calib"] * 0.10 * l_calib)
            else:
                prec_r = torch.exp(-2.0 * self.log_sigma_reg.clamp(min=-3.0))
                prec_h = torch.exp(-2.0 * self.log_sigma_heading.clamp(min=-3.0))
                prec_c = torch.exp(-2.0 * self.log_sigma_calib.clamp(min=-3.0))
                total = (l_cfm
                         + bd["lam_reg"]   * (0.5*prec_r*l_reg     + self.log_sigma_reg.clamp(min=-3.0)     + HALF)
                         + bd["lam_dir"]   * (0.5*prec_h*l_heading + self.log_sigma_heading.clamp(min=-3.0) + HALF)
                         + bd["lam_calib"] * (0.5*prec_c*l_calib   + self.log_sigma_calib.clamp(min=-3.0)   + HALF))
            # Re-add the fixed-weight (non-Kendall) hard_score regularizer —
            # same formula as the unpatched total, always additive.
            total = total + self.lambda_hard_reg * l_hard_reg
            if not torch.isfinite(total):
                total = total.new_zeros(())
            bd.update({"total": total,
                       "l_reg":      float(l_reg.detach()),
                       "l_heading":  float(l_heading.detach()),
                       "l_calib":    float(l_calib.detach()),
                       "l_hard_reg": float(l_hard_reg.detach())})
            return bd
        import types
        raw.get_loss_breakdown = types.MethodType(_patched_glb, raw)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    opt    = build_optimizer(model, lr_velocity=args.lr, lr_encoder=0.0,
                             weight_decay=args.weight_decay,
                             lr_logits_scale=args.lr_logits_scale)
    scaler = GradScaler("cuda", enabled=args.use_amp)
    sched  = TwoGroupScheduler(
        opt=opt, warmup_epochs=args.warmup_epochs, total_epochs=args.num_epochs,
        lr_vel=args.lr, lr_vel_min=args.lr_min,
        freeze_end_ep=args.freeze_encoder_epochs, lr_enc_peak=args.lr_enc_peak,
        encoder_warmup_epochs=args.encoder_warmup_epochs)
    print(f"\n  LR vel: {args.lr:.0e} → {args.lr_min:.0e}  "
          f"LR enc: 0 ({args.freeze_encoder_epochs}ep) → {args.lr_enc_peak:.0e}")

    swa = SWAHandler(swa_lr=args.swa_lr)

    # ── Resume ────────────────────────────────────────────────────────────
    start_ep = 0; best_score = float("inf"); best_hard = float("inf")
    patience_cnt = 0; val_ade_history = []

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        _unwrap(model).load_state_dict(ck["model"], strict=False)
        try: opt.load_state_dict(ck["optimizer"])
        except Exception as e: print(f"  ⚠ Opt: {e}")
        sched.epoch  = ck.get("scheduler", 0)
        start_ep     = ck.get("epoch", 0) + 1
        best_score   = ck.get("best_score", ck.get("best_ade", float("inf")))
        patience_cnt = ck.get("patience_cnt", 0)
        if scaler and ck.get("scaler"):
            try: scaler.load_state_dict(ck["scaler"])
            except Exception: pass
        if ema and ck.get("ema"):
            for k, v in ck["ema"].items():
                if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
        print(f"  ↩ Resume ep{start_ep}  best={best_score:.1f}  patience={patience_cnt}")

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: ok")
    except Exception: pass

    # ── Training loop ─────────────────────────────────────────────────────
    nstep = len(trl)
    print(f"\n  TRAINING ({nstep} steps/ep × {args.num_epochs} ep)")
    print(f"  Aug: shift±5km(25%) + speed×[0.85,1.15](20%) + recurv±20°(20%) + no-aug(35%)")
    print(f"  Loss: L_CFM + L_reg(linear) + L_heading_ms(4steps,decay=0.5)")
    print(f"  Inf:  1-shot sigma=0.04 + speed_calibrate(±15%) + top3 physics")
    print()

    for ep in range(start_ep, start_ep + args.num_epochs):
        rel_ep = ep - start_ep
        freeze = rel_ep < args.freeze_encoder_epochs
        for p in _unwrap(model).encoder.parameters():
            p.requires_grad_(not freeze)

        if rel_ep == 0 and freeze:
            print(f"  *** Ep{ep}: encoder frozen ***")
        if rel_ep == args.freeze_encoder_epochs:
            print(f"\n  *** Ep{ep}: encoder unfrozen ***")

        model.train()
        sum_loss = sum_cfm = sum_reg = sum_head = sum_ade1 = 0.0
        t0_ep = time.perf_counter()

        for i, batch in enumerate(trl):
            bl = move(list(batch), device)
            bl_aug = augment_batch(bl, disable_c=args.disable_aug_c)   # training augmentation
            # [FIX] Bug thật đã tìm ra: trước đây augment_batch(bl) được
            # gọi KHÔNG qua flag nào — --disable_aug_c chỉ được ghi vào
            # footprint_info/console log ("[ABLATION] Disabled: AUG-C")
            # nhưng KHÔNG có tác dụng thật, vì augment_batch() (định nghĩa
            # trong flow_matching_model.py) không nhận tham số nào để tắt
            # riêng nhánh C (recurvature). Đã thêm tham số disable_c vào
            # augment_batch() (default=False, giữ nguyên hành vi cũ cho
            # MỌI call site khác không truyền tham số này) và truyền đúng
            # args.disable_aug_c vào đây. Mọi checkpoint no_aug_c/ablation
            # đã train TRƯỚC fix này thực chất đã train VỚI AUG-C bật —
            # cần re-run --disable_aug_c để có kết quả ablation thật.

            opt.zero_grad()
            with autocast(device_type="cuda", enabled=args.use_amp):
                bd = model.get_loss_breakdown(bl_aug, epoch=ep)

            scaler.scale(bd["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if freeze:
                for p in _unwrap(model).encoder.parameters():
                    if p.grad is not None: p.grad.zero_()

            scaler.step(opt); scaler.update()
            model.ema_update()
            swa.update(model)

            sum_loss += bd["total"].item(); sum_cfm  += bd["l_cfm"]
            sum_reg  += bd["l_reg"];        sum_head += bd["l_heading"]
            sum_ade1 += bd["ade_1step"]

            if i % 30 == 0:
                _, lr_vel = get_lrs(opt)
                enc_s = "frozen" if freeze else "active"
                swa_s = " [SWA]" if swa.active else ""
                print(f"  [{ep:>3}][{i:>3}/{nstep}]"
                      f"  tot={bd['total'].item():.4f}"
                      f"  cfm={bd['l_cfm']:.4f}"
                      f"  reg={bd['l_reg']:.4f}"
                      f"  h4s={bd['l_heading']:.4f}"
                      f"  hreg={bd.get('l_hard_reg', 0.0):.4f}"
                      f"  lam_d={bd['lam_dir']:.2f}"
                      f"  ade1={bd['ade_1step']:.0f}km"
                      f"  enc={enc_s}{swa_s}"
                      f"  lr={lr_vel:.2e}")

        train_loss = sum_loss / nstep
        _, lr_vel_used = get_lrs(opt)
        sched.step()

        print(f"\n  ── Ep{ep:>3}"
              f"  train={train_loss:.6f}"
              f"  cfm={sum_cfm/nstep:.4f}"
              f"  reg={sum_reg/nstep:.4f}"
              f"  h4s={sum_head/nstep:.4f}"
              f"  ade1={sum_ade1/nstep:.0f}km"
              f"  lr={lr_vel_used:.2e}"
              f"  t={time.perf_counter()-t0_ep:.0f}s")

        _save(last_ckpt, ep, model, opt, sched, best_score, ema, scaler,
              model_cfg=model_cfg)
        if ep % 5 == 0:
            ep_ckpt = os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth")
            _save(ep_ckpt, ep, model, opt, sched, best_score, ema, scaler,
                  model_cfg=model_cfg)
            print(f"  💾 {ep_ckpt}")

        # ── Val evaluation ───────────────────────────────────────────────
        if rel_ep % args.val_freq == 0:
            run_xai_this = (rel_ep % 10 == 0)
            r = evaluate(model, val_loader, device, tag=f"VAL ep{ep}",
                         n_ensemble=args.n_ensemble, ema=ema,
                         ref_targets=ST_TRANS_VAL, epoch_for_loss=ep,
                         run_xai=run_xai_this, xai_batch=xai_batch)

            val_ade = r["ADE"]; score = r["combined_score"]
            val_ade_history.append(val_ade)

            # Trend
            if len(val_ade_history) >= 4:
                trend = float(np.mean(val_ade_history[-2:])) - float(np.mean(val_ade_history[-4:-2]))
                trend_s = f"↑{trend:+.1f}km⚠" if trend > 5 else f"↓{trend:+.1f}km✓" if trend < -5 else f"→{trend:+.1f}flat"
            else:
                trend_s = "—"
            print(f"  train={train_loss:.6f}  val_ADE={val_ade:.1f}  combined={score:.1f}  trend={trend_s}")

            # SWA activation check
            if (not swa.active and ep >= args.swa_min_ep
                    and swa.should_activate(val_ade_history, args.swa_window, args.swa_threshold)):
                swa.activate(model, opt, ep)

            # Best checkpoint
            if score < best_score:
                best_score = score; patience_cnt = 0
                _save(best_ckpt, ep, model, opt, sched, best_score, ema, scaler,
                      extra={"val_ade": r["ADE"], "val_ate": r["ATE"],
                             "val_cte": r["CTE"], "patience_cnt": 0},
                      model_cfg=model_cfg)
                print(f"  ✅ Best! score={best_score:.2f}"
                      f"  ADE={r['ADE']:.1f} ATE={r['ATE']:.1f} CTE={r['CTE']:.1f}")
            else:
                if rel_ep >= args.min_ep: patience_cnt += args.val_freq
                print(f"  No improve {patience_cnt}/{args.patience} (best={best_score:.1f})")
                if rel_ep >= args.min_ep and patience_cnt >= args.patience:
                    print(f"  ⛔ Early stop @ ep{ep}")
                    break

        # ── Hard-val evaluation ──────────────────────────────────────────
        if rel_ep % args.hard_val_freq == 0 and rel_ep >= args.min_ep:
            r_h = evaluate_hard_val(model, val_loader, device,
                                     hard_threshold=args.hard_val_threshold,
                                     n_ensemble=args.n_ensemble, ema=ema,
                                     epoch_for_loss=ep)
            print(f"  [HVAL] n={r_h['n_hard']}"
                  f"  ADE={r_h['ADE']:.1f} ATE={r_h['ATE']:.1f} CTE={r_h['CTE']:.1f}"
                  f"  combined={r_h['combined_score']:.1f}")
            if r_h["combined_score"] < best_hard and r_h["n_hard"] >= 10:
                best_hard = r_h["combined_score"]
                _save(hard_best_ckpt, ep, model, opt, sched, best_hard, ema, scaler,
                      extra={"hard_val_ade": r_h["ADE"], "selection_criterion": "hard_val"},
                      model_cfg=model_cfg)
                print(f"  💎 Hard-best! score={best_hard:.2f} ADE={r_h['ADE']:.1f}")

        # ── SWA eval ─────────────────────────────────────────────────────
        if swa.active and rel_ep % args.val_freq == 0 and swa.n_updates >= 10:
            backup = {k: v.detach().clone()
                      for k, v in _unwrap(model).state_dict().items()
                      if v.dtype.is_floating_point}
            swa.apply_to_model(model)
            r_swa = evaluate(model, val_loader, device, tag=f"SWA ep{ep}",
                              n_ensemble=args.n_ensemble, ema=None,
                              ref_targets=ST_TRANS_VAL, epoch_for_loss=ep)
            swa.restore_from_backup(model, backup)

            swa_score = r_swa["combined_score"]
            print(f"  [SWA] score={swa_score:.2f} ({swa.n_updates} updates) vs best={best_score:.2f}")
            if swa_score < best_score:
                best_score = swa_score; patience_cnt = 0
                swa.save_avg_state(swa_ckpt, ep, best_score,
                                   extra={"val_ade": r_swa["ADE"],
                                          "val_ate": r_swa["ATE"],
                                          "val_cte": r_swa["CTE"]},
                                   model_cfg=model_cfg)
                import shutil; shutil.copy(swa_ckpt, best_ckpt)
                print(f"  ✅ SWA best! score={best_score:.2f} ADE={r_swa['ADE']:.1f}")

    # ── Wall-clock + footprint (ESWA Table E) ────────────────────────────
    _wall_total = time.time() - _wall_start
    print(f"\n  Training wall-clock: {_wall_total/3600:.2f}h ({_wall_total:.0f}s)")
    try:
        footprint_info.update({
            "training_wall_clock_s": _wall_total,
            "training_wall_clock_h": round(_wall_total/3600, 3),
            "num_epochs": args.num_epochs,
            "best_score": best_score,
        })
        import json as _json
        fp_path = os.path.join(args.output_dir, "footprint.json")
        with open(fp_path, "w") as _fp:
            _json.dump(footprint_info, _fp, indent=2)
        print(f"  Footprint saved → {fp_path}")
    except Exception as _fe:
        print(f"  Footprint save failed: {_fe}")

    # ── Test evaluation ───────────────────────────────────────────────────
    print(f"\n  Done! best_score={best_score:.2f}")
    if not args.test_at_end: return

    print("\n  Loading best checkpoint for TEST...")
    if not os.path.exists(best_ckpt):
        print("  No checkpoint found."); return

    ck = torch.load(best_ckpt, map_location=device)
    is_swa = ck.get("is_swa", False)
    _unwrap(model).load_state_dict(ck["model"], strict=False)
    if not is_swa and ema and ck.get("ema"):
        for k, v in ck["ema"].items():
            if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
    print(f"  Loaded ep{ck.get('epoch','?')} (is_swa={is_swa})")

    try:
        _, test_loader = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
        print(f"  Test: {len(test_loader)} batches")
    except Exception:
        print("  No test set → using val"); test_loader = val_loader

    # Standard test
    r_test = evaluate(model, test_loader, device, tag="TEST (best ckpt)",
                      n_ensemble=args.n_ensemble, ema=None if is_swa else ema,
                      ref_targets=ST_TRANS_TEST, run_xai=True, xai_batch=xai_batch)

    # TTA test
    if args.tta_test:
        r_tta = evaluate(model, test_loader, device, tag="TEST+TTA",
                         n_ensemble=args.n_ensemble, ema=None if is_swa else ema,
                         ref_targets=ST_TRANS_TEST, use_tta=True, n_tta=args.n_tta)
        print(f"\n  TTA: ADE {r_test['ADE']:.1f}→{r_tta['ADE']:.1f}  "
              f"ATE {r_test['ATE']:.1f}→{r_tta['ATE']:.1f}  "
              f"CTE {r_test['CTE']:.1f}→{r_tta['CTE']:.1f}")

    # Multi-scale test
    if args.multiscale_test:
        raw_m = _unwrap(model)
        if hasattr(raw_m, "sample_multiscale"):
            print("\n  Multi-scale sigma test...")
            ms_ades, ms_ates, ms_ctes = [], [], []
            ms_steps = defaultdict(list)
            raw_m.eval()
            bk_ms = (ema.apply_to(model) if (ema and not is_swa) else None)
            with torch.no_grad():
                for batch in test_loader:
                    bl = move(list(batch), device)
                    try:
                        pred, _, _ = raw_m.sample_multiscale(bl)
                    except Exception: continue
                    gt = bl[1]; T = min(pred.shape[0], gt.shape[0])
                    pd = _norm_to_deg(pred[:T]); gd = _norm_to_deg(gt[:T])
                    dist = _haversine_deg(pd, gd)
                    ate, cte = _ate_cte(pd, gd)
                    ms_ades.extend(dist.mean(0).tolist())
                    if ate.shape[0] > 0:
                        ms_ates.extend(ate.abs().mean(0).tolist())
                        ms_ctes.extend(cte.abs().mean(0).tolist())
                    for h, s in HORIZON_STEPS.items():
                        if s < T: ms_steps[h].extend(dist[s].tolist())
            if bk_ms: ema.restore(model, bk_ms)
            def _mm(lst): return float(np.mean(lst)) if lst else float("nan")
            print(f"  Multi-scale: ADE={_mm(ms_ades):.1f} ATE={_mm(ms_ates):.1f} CTE={_mm(ms_ctes):.1f}")

    # Final comparison
    val_ade_b = ck.get("val_ade", float("nan"))
    val_ate_b = ck.get("val_ate", float("nan"))
    val_cte_b = ck.get("val_cte", float("nan"))
    v21 = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4}  # ST-Trans targets
    print("\n" + "=" * 72)
    print("  v2.1-XAI FINAL RESULTS vs ST-Trans target")
    print("=" * 72)
    print(f"  {'Metric':<10} {'Val':>10} {'Test':>10} {'Gap':>10} {'v2.1 Test':>12} Status")
    print("  " + "─"*60)
    for m_n, val_v, test_v, v21_v in [
        ("ADE", val_ade_b, r_test["ADE"], v21["ADE"]),
        ("ATE", val_ate_b, r_test["ATE"], v21["ATE"]),
        ("CTE", val_cte_b, r_test["CTE"], v21["CTE"]),
    ]:
        gap  = test_v - val_v if (np.isfinite(test_v) and np.isfinite(val_v)) else float("nan")
        impr = v21_v - test_v
        flag = (f"↓{impr:+.1f}" if np.isfinite(impr) and impr > 0 else
                f"↑{-impr:+.1f}" if np.isfinite(impr) else "?")
        print(f"  {m_n:<10} {val_v:>10.2f} {test_v:>10.2f} {gap:>+10.2f} {v21_v:>12.1f}  {flag}")
    print("=" * 72)

    # ── Auto evaluate_full + statistical_tests sau khi train xong ─────────
    # Tự động chạy để không cần chạy tay khi train ngầm trên Kaggle.
    # Kết quả lưu vào args.output_dir/eval/ và args.output_dir/stats/
    _auto_eval(args, best_ckpt, device)


def _auto_eval(args, best_ckpt: str, device):
    """
    Tự động chạy evaluate_full.py và statistical_tests.py sau khi train xong.
    Tìm script trong cùng thư mục hoặc thư mục cha của train script.
    Nếu không tìm thấy → bỏ qua, không làm crash training.
    """
    import subprocess, sys, json as _json

    print("\n" + "="*72)
    print("  AUTO EVALUATE + STATISTICAL TESTS")
    print("="*72)

    eval_dir  = os.path.join(args.output_dir, "eval")
    stats_dir = os.path.join(args.output_dir, "stats")
    os.makedirs(eval_dir,  exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    # Tìm evaluate_full.py: thử root project, thư mục cha scripts, cwd
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    candidates_eval = [
        os.path.join(project_root, "evaluate_full.py"),
        os.path.join(os.getcwd(),  "evaluate_full.py"),
        os.path.join(script_dir,   "evaluate_full.py"),
    ]
    candidates_stat = [
        os.path.join(project_root, "statistical_tests.py"),
        os.path.join(os.getcwd(),  "statistical_tests.py"),
        os.path.join(script_dir,   "statistical_tests.py"),
    ]
    eval_script = next((p for p in candidates_eval if os.path.exists(p)), None)
    stat_script = next((p for p in candidates_stat if os.path.exists(p)), None)

    # ── Step 1: evaluate_full.py ────────────────────────────────────────────
    if eval_script is None:
        print("  ⚠ evaluate_full.py không tìm thấy → bỏ qua auto-eval")
        print(f"    Đặt file tại: {candidates_eval[0]}")
    else:
        print(f"  ▶ evaluate_full.py ({eval_script})")
        ep = "best"
        try:
            import torch as _torch
            ck_info = _torch.load(best_ckpt, map_location="cpu")
            ep = ck_info.get("epoch", "best")
        except Exception:
            pass

        cmd_eval = [
            sys.executable, eval_script,
            "--checkpoint",   best_ckpt,
            "--dataset_root", args.dataset_root,
            "--split",        "test",
            "--output_dir",   eval_dir,
            "--n_ensemble",   str(args.n_ensemble),
            "--no_crps",      # tắt CRPS để chạy nhanh hơn (bật nếu cần)
            "--gpu",          str(args.gpu_num),
        ]
        try:
            result = subprocess.run(cmd_eval, capture_output=False, timeout=1800)
            if result.returncode == 0:
                print(f"  ✅ evaluate_full done → {eval_dir}/")
            else:
                print(f"  ❌ evaluate_full failed (code {result.returncode})")
        except subprocess.TimeoutExpired:
            print("  ⚠ evaluate_full timeout (30min) → bỏ qua")
        except Exception as e:
            print(f"  ⚠ evaluate_full error: {e}")

    # ── Step 2: statistical_tests.py ────────────────────────────────────────
    # Tìm JSON output từ evaluate_full
    eval_json = None
    if os.path.exists(eval_dir):
        jsons = sorted([
            os.path.join(eval_dir, f) for f in os.listdir(eval_dir)
            if f.startswith("eval_test") and f.endswith(".json")
        ])
        if jsons:
            eval_json = jsons[-1]  # file mới nhất

    if stat_script is None:
        print("  ⚠ statistical_tests.py không tìm thấy → bỏ qua")
    elif eval_json is None:
        print("  ⚠ Không tìm thấy eval JSON → bỏ qua statistical tests")
        print(f"    Thử chạy lại evaluate_full.py trước")
    else:
        print(f"  ▶ statistical_tests.py ({stat_script})")
        print(f"    FM result: {eval_json}")
        cmd_stat = [
            sys.executable, stat_script,
            "--fm_results",       eval_json,
            "--use_st_trans_ref",
            "--fm_n_storms",      "420",
            "--baseline_name",    "ST-Trans",
            "--output_dir",       stats_dir,
            "--n_bootstrap",      "10000",
        ]
        try:
            result = subprocess.run(cmd_stat, capture_output=False, timeout=600)
            if result.returncode == 0:
                print(f"  ✅ statistical_tests done → {stats_dir}/")
            else:
                print(f"  ❌ statistical_tests failed (code {result.returncode})")
        except subprocess.TimeoutExpired:
            print("  ⚠ statistical_tests timeout (10min) → bỏ qua")
        except Exception as e:
            print(f"  ⚠ statistical_tests error: {e}")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "auto_eval_summary.json")
    try:
        summary = {
            "checkpoint":    best_ckpt,
            "eval_dir":      eval_dir,
            "stats_dir":     stats_dir,
            "eval_json":     eval_json,
            "seed":          getattr(args, "seed", 42),
            "ablation_name": getattr(args, "ablation_name", ""),
        }
        # Đọc kết quả ADE/ATE/CTE từ eval JSON nếu có
        if eval_json and os.path.exists(eval_json):
            with open(eval_json) as _f:
                ev = _json.load(_f)
            summary["test_ADE"] = ev.get("ADE")
            summary["test_ATE"] = ev.get("ATE")
            summary["test_CTE"] = ev.get("CTE")
            summary["test_RMSE"] = ev.get("RMSE")
            summary["crps_mean"] = ev.get("crps", {}).get("mean")
        with open(summary_path, "w") as _f:
            _json.dump(summary, _f, indent=2)
        print(f"\n  Summary → {summary_path}")
        if summary.get("test_ADE"):
            print(f"  ADE={summary['test_ADE']:.2f}  "
                  f"ATE={summary['test_ATE']:.2f}  "
                  f"CTE={summary['test_CTE']:.2f}")
    except Exception as e:
        print(f"  Summary save failed: {e}")

    print("="*72)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    if args.dataset_root == "TCND_vn":
        _auto = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
        if os.path.isdir(_auto):
            args.dataset_root = _auto
    main(args)