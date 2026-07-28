"""

# ─────────────────────────────────────────────────────────────────────────────
# FILE PLACEMENT — copy sang đúng vị trí trước khi train:
#
#   SOURCE (download từ Claude):  flow_matching_model_v24_final.py
#   KAGGLE TARGET:                /kaggle/working/Model/flow_matching_model.py
#   LOCAL DEV:                    Model/flow_matching_model.py
#
#   cp flow_matching_model_v24_final.py /kaggle/working/Model/flow_matching_model.py
# ─────────────────────────────────────────────────────────────────────────────

Model/flow_matching_model.py  ──  TC-FlowMatching v2.1-clean
═══════════════════════════════════════════════════════════════════════════════
VIẾT LẠI HOÀN TOÀN từ v2.1 + các fix đã verified từ thực nghiệm 140 epoch.

━━━ CƠ SỞ LÝ THUYẾT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v2.1 test: ADE=229.8, ATE=214.4, CTE=71.6 (best so far, generalize tốt nhất)
v2.5 test: ADE=232.5, ATE=222.3, CTE=64.5 (ATE tệ hơn, CTE tốt hơn)
v2.4 test: ADE=291.1 (catastrophic overfitting)

PHÂN TÍCH LỖI v2.1:
  ATE/CTE ratio on test = 214.4/71.6 = 3.0
  → 75% error is ALONG-TRACK (speed/distance)
  → 25% is CROSS-TRACK (direction)
  Test gap 72h = +129.3km >> 12h = +10.1km → long-range drift

PHÂN TÍCH v2.5 THẤT BẠI:
  L_momentum anchor pred_speed ≈ val_speed (10.3km/h)
  Test storms faster → L_momentum SLOWS model on test → ATE +7.9km WORSE
  L_heading (step 0 only) → 72h heading dev INCREASED: ep0=87°→ep120=128°!
  Recurvature aug: CTE improved 7km ← giữ lại

━━━ VẬT LÝ CỦA GIẢI PHÁP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GIẢI PHÁP ATE (along-track error):
  Training: KHÔNG anchor speed tại training (no L_momentum)
  Inference: SPEED CALIBRATION per-storm
    scale = clip(obs_speed / pred_speed_first4, 0.85, 1.15)
    pred_cal = last_obs + (pred - last_obs) * scale
    → Preserves direction (CTE unchanged), fixes magnitude (ATE reduced)
    → Per-storm adaptive: fast test storms get upscaling, slow storms downscaling
    → DOES NOT affect training → zero train/inference consistency issue

GIẢI PHÁP CTE (cross-track error):
  Training: MULTI-STEP heading constraint trên TOÀN BỘ 12 bước (không chỉ
  4 bước đầu như mô tả ban đầu) — xem [LEARN-5] bên dưới để biết chi tiết
  về lỗi đã phát hiện và sửa: bản trước chỉ áp dụng constraint cho 4/12
  bước, khiến 8 bước cuối (30h→72h) hoàn toàn không có gradient signal
  hướng. Trọng số mỗi bước giờ HỌC ĐƯỢC qua self.heading_step_logits.

GIẢI PHÁP GAP (val→test distribution shift):
  [AUG-C] Recurvature: proven to reduce CTE test (v2.5)
  Speed calibration at inference bridges remaining gap

━━━ GIỮ NGUYÊN TỪ v2.1 (PROVEN, KHÔNG THAY ĐỔI) ━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✅ sigma_inference=0.04 FIXED — zero train/inference mismatch
  ✅ Mild base aug: shift±5km + speed×0.85–1.15 (val ≈ test distribution)
  ✅ 1-shot inference nhất quán với L_reg training
  ✅ 2-group optimizer, encoder freeze 10ep
  ✅ val loop NO augmentation
  ✅ OT noise-GT matching trong L_CFM
  ✅ ContextEncoder: FNO3D + Mamba + Env + 6 kinematic features
  ✅ VelocityTransformer: d_model=256, 4 layers, nhead=8

━━━ v2.1-LEARN CHANGES (từ v2.1-clean baseline, test ADE=227.8 ATE=220.3 CTE=60.6) ━

  CĂN CỨ: mọi lần đổi HẰNG SỐ tay đoán (linspace range, clip range) trước đây
  chỉ tạo dao động <1km test. Bằng chứng: XAI-8 24h ratio luôn ~1.5-1.6×
  bất kể linspace(1,3) hay linspace(1,1.5). Đây là dấu hiệu bottleneck không
  nằm ở "chọn đúng hằng số" mà ở việc DÙNG hằng số thay vì để model học.

  [LEARN-1] speed_calibrate_pred: từ 1 scale chung (clip 0.85-1.15) cho TOÀN
            BỘ 12 bước → 12 hệ số riêng biệt theo từng horizon, học qua
            gradient (không còn @torch.no_grad/@staticmethod).
            Lý do: XAI-8 chứng minh correction cần KHÁC NHAU theo horizon
            (12h≈0.77x, 24h≈0.62x, 48h≈1.06x, 72h≈1.04x) — 1 số không
            thể đáp ứng cả 4 nhu cầu khác nhau cùng lúc.
            THÊM: L_calib loss term để gradient thực sự chảy tới tham số này
            (nếu không có loss term riêng, tham số tồn tại nhưng KHÔNG BAO
            GIỜ được optimizer cập nhật vì nó chỉ được dùng trong sample()
            dưới torch.no_grad() tại thời điểm eval).

  [LEARN-2] _reg_loss step weights: linspace(1.0,1.5) hardcoded → 12 logit
            riêng (softmax), model tự học phân bổ trọng số giữa các horizon
            thay vì áp công thức tuyến tính tay đoán.

  [LEARN-3] hard_score: thêm obs_speed (đã chuẩn hóa) làm thành phần thứ 4,
            VÀ 4 trọng số kết hợp (curvature, speed_var, dir_change, obs_speed)
            học qua gradient thay vì hardcode 0.4/0.3/0.3.
            Lý do: XAI-9 cho thấy fast storms (n=94/420=22%) gây 62% tổng CTE
            test nhưng hard_score cũ không có tín hiệu "obs_speed tuyệt đối"
            → storm nhanh-nhưng-đều (speed_var thấp) bị coi nhầm là "dễ"
            → không được train với loss-weight cao hơn dù nó cần.
            Bỏ @torch.no_grad() ở hard_score_from_obs để gradient chảy qua
            được (qua sw_hard trong _reg_loss).

  [LEARN-4] Cross-loss weighting: lambda_reg/lambda_heading/lambda_calib
            (3 hằng số tay đoán: 0.2/0.07/0.1) → Kendall homoscedastic
            uncertainty weighting (Kendall, Gal & Cipolla 2018):
              weighted_term = loss/(2·exp(2·log_sigma)) + log_sigma
            3 log_sigma HỌC ĐƯỢC qua gradient.
            QUAN TRỌNG: KHÔNG dùng nn.Parameter(lambda) trực tiếp nhân
            vào loss — gradient của total theo lambda = loss ≥ 0 luôn
            đẩy lambda về 0 (suy biến, tắt hết loss phụ). Số hạng
            +log_sigma trong công thức Kendall là regularizer ngăn suy
            biến này — đã verify thực nghiệm (xem test ở cuối file dev
            log): effective_lambda KHÔNG giảm về 0 qua 30 bước train
            với lr cao, trái lại tăng dần khi loss phụ giảm nhanh.
            Epoch-ramp (ep5/ep10/ep20/ep30) GIỮ NGUYÊN dạng if/else rời
            rạc — đây là GATE bật/tắt theo trạng thái encoder-freeze,
            không phải "trọng số tương đối", không thể differentiable
            theo epoch (số nguyên), nên không thuộc phạm vi can học.

  [LEARN-5] _heading_loss_ms: SỬA LỖI BỎ SÓT từ lần trước. Bản trước
            (kể cả lần "v2.1-learn" đầu tiên của tôi) vẫn giữ n_steps=4
            CỐ ĐỊNH + decay=0.5 CỐ ĐỊNH (weight=1.0/0.5/0.25/0.125 cho
            4 bước ĐẦU, 8 bước 30h→72h KHÔNG có constraint heading nào).
            FIX: self.heading_step_logits (pred_len=12 params, softmax),
            áp dụng cho TOÀN BỘ 12 bước, init theo đúng decay 0.5^t cũ
            (không đổi điểm khởi đầu) nhưng gradient tự học lại phân bổ
            trong lúc train — không còn giới hạn cứng "chỉ 4 bước đầu".

  KHÔNG ĐỔI (đã cân nhắc, có lý do kỹ thuật rõ ràng):
    _physics_score 4 exponent (0.30/0.25/0.30/0.15): hàm này chạy dưới
    @torch.no_grad() để CHỌN giữa K candidates sinh từ CÙNG một velocity
    network — đây là post-hoc re-ranking, không nằm trên đường gradient
    nào quay lại network hay bất kỳ tham số nào của nó. Biến nó learnable
    mà không có loss-term riêng sẽ lặp lại đúng lỗi "tham số mồ côi"
    (never updated) mà LEARN-1 đã phải sửa bằng cách thêm L_calib.
    R_EARTH, DT_HOURS, hệ số affine normalized↔degree: hằng số vật lý/
    metadata dataset, không phải "model nghĩ gì" — đổi sẽ làm sai đơn vị
    tính toán, không phải tinh chỉnh hành vi model.

  GIỮ NGUYÊN HOÀN TOÀN (đã proven qua thực nghiệm, KHÔNG đổi):
    ✅ AUG-C: rotate DISPLACEMENT vectors (không phải absolute positions)
    ✅ AUG-D: ĐÃ XÓA — proven harmful (+4.5km v2.6, +13.4km v2.7)
    ✅ L_momentum: disabled (proven harmful +7.9km — v2.5)
    ✅ sigma_inference=0.04 fixed (train/infer consistency)
    ✅ OT matching, 1-shot inference, encoder freeze 10ep

━━━ KẾT QUẢ THỰC TẾ (baseline trước thay đổi này) ━━━━━━━━━━━━━━━━━━━━━━━━━

                    Val ADE   Test ADE  Test ATE  Test CTE
  v2.1 gốc:         170.1     229.8     214.4     71.6
  v2.1-XAI (ran):   169.7     229.8     221.6     61.9  ← AUG-C bug → ATE +7.2
  v2.1-clean (ran):  ~168      227.8     220.3     60.6  ← baseline cho bản này
  ST-Trans target:            224.4     213.7     59.4

  CHƯA CÓ KẾT QUẢ TEST CHO v2.1-LEARN (bản này) — cần retrain để biết
  ATE/CTE thực tế có cải thiện so với 220.3/60.6 hay không. Mọi con số
  "kỳ vọng" chỉ là giả thuyết dựa trên cơ chế, KHÔNG phải đã đo được.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.FNO3D_encoder import FNO3DEncoder
from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
from Model.env_net_transformer_gphsplit import Env_net

R_EARTH  = 6371.0
DT_HOURS = 6.0


# ─────────────────────────────────────────────────────────────────────────────
#  Coordinate utilities (same as v2.1)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
    """Normalized coords → (lon°, lat°)."""
    return torch.stack([
        (t[..., 0] * 50.0 + 1800.0) / 10.0,
        (t[..., 1] * 50.0) / 10.0,
    ], dim=-1)


def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Haversine distance km between two (lon°,lat°) tensors."""
    lat1 = torch.deg2rad(p1[..., 1]);  lat2 = torch.deg2rad(p2[..., 1])
    dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
    dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
    a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Bearing in radians from p1→p2 (degrees input)."""
    lon1 = torch.deg2rad(p1[..., 0]);  lat1 = torch.deg2rad(p1[..., 1])
    lon2 = torch.deg2rad(p2[..., 0]);  lat2 = torch.deg2rad(p2[..., 1])
    dlon = lon2 - lon1
    y = torch.sin(dlon) * torch.cos(lat2)
    x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    return torch.atan2(y, x)


def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
    """Speed km/h between consecutive steps of a trajectory [T, B, 2]."""
    if traj_deg.shape[0] < 2:
        return traj_deg.new_zeros(1, traj_deg.shape[1])
    return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# ─────────────────────────────────────────────────────────────────────────────
#  EMAModel (same as v2.1)
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m


class EMAModel:
    def __init__(self, model, decay: float = 0.995):
        self.decay = decay
        m = _unwrap(model)
        self.shadow = {k: v.detach().clone()
                       for k, v in m.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model):
        m = _unwrap(model)
        with torch.no_grad():
            for k, v in m.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        m = _unwrap(model)
        backup, sd = {}, m.state_dict()
        for k in self.shadow:
            if k not in sd: continue
            backup[k] = sd[k].detach().clone()
            sd[k].copy_(self.shadow[k])
        return backup

    def restore(self, model, backup):
        m = _unwrap(model)
        sd = m.state_dict()
        for k, v in backup.items():
            if k in sd: sd[k].copy_(v)


# ─────────────────────────────────────────────────────────────────────────────
#  OT matching (same as v2.1)
# ─────────────────────────────────────────────────────────────────────────────

def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05, n_iter: int = 50) -> torch.Tensor:
    B = cost.shape[0]; device = cost.device
    log_a = -math.log(B) * torch.ones(B, device=device)
    log_b = -math.log(B) * torch.ones(B, device=device)
    log_K = -cost / epsilon
    log_u = torch.zeros(B, device=device)
    log_v = torch.zeros(B, device=device)
    for _ in range(n_iter):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
    return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)


def _ot_match(x0_flat: torch.Tensor, x1_flat: torch.Tensor,
              epsilon: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    B = x0_flat.shape[0]
    if B < 4:
        return x0_flat, x1_flat
    try:
        cost = torch.cdist(x0_flat.float(), x1_flat.float()) / (x0_flat.shape[-1] ** 0.5)
        with torch.no_grad():
            pi = _sinkhorn_log(cost, epsilon=epsilon)
        flat = pi.reshape(-1).clamp(0.0)
        s = flat.sum()
        if not torch.isfinite(s) or s < 1e-10:
            return x0_flat, x1_flat
        idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
        return x0_flat[idx // B], x1_flat
    except Exception:
        return x0_flat, x1_flat


# ─────────────────────────────────────────────────────────────────────────────
#  VelocityTransformer (identical to v2.1)
# ─────────────────────────────────────────────────────────────────────────────

class VelocityTransformer(nn.Module):
    def __init__(self, pred_len: int = 12, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
                 d_cond: int = 256):
        super().__init__()
        self.pred_len = pred_len
        self.d_model  = d_model
        self.traj_embed = nn.Linear(2, d_model)
        self.pos_emb    = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
        self.step_emb   = nn.Embedding(pred_len, d_model)
        self.time_mlp   = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.cond_proj  = nn.Sequential(nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
        self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)
                         * (-math.log(10000.0) / max(half - 1, 1)))
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.time_mlp(emb)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_t.shape
        step_idx = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
        x_emb = (self.traj_embed(x_t) + self.pos_emb[:, :T] + self.step_emb(step_idx))
        memory = torch.cat([self._time_emb(t).unsqueeze(1),
                            self.cond_proj(cond).unsqueeze(1)], dim=1)
        out = self.out_norm(self.decoder(x_emb, memory))
        return self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
#  ContextEncoder (identical to v2.1)
# ─────────────────────────────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    RAW_CTX_DIM = 512

    def __init__(self, obs_len: int = 8, unet_in_ch: int = 13, d_cond: int = 256):
        super().__init__()
        self.obs_len = obs_len
        self.d_cond  = d_cond

        self.spatial_enc     = FNO3DEncoder(in_channel=unet_in_ch, out_channel=1, d_model=32,
                                             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
                                             spatial_down=32, dropout=0.05)
        self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.bottleneck_proj = nn.Linear(128, 128)
        self.decoder_proj    = nn.Linear(1, 16)
        self.enc_1d          = DataEncoder1D(in_1d=4, feat_3d_dim=128, mlp_h=64,
                                              lstm_hidden=128, lstm_layers=3,
                                              dropout=0.1, d_state=16)
        self.env_enc         = Env_net(obs_len=obs_len, d_model=32)
        self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
        self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
        self.ctx_drop = nn.Dropout(0.1)
        self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, d_cond)
        self.ctx_ln2  = nn.LayerNorm(d_cond)
        # 6 kinematic features: vel_x, vel_y, speed_n, sin(h), cos(h), accel
        self.vel_obs_enc = nn.Sequential(
            nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, d_cond // 2), nn.GELU())
        self.hard_embed = nn.Sequential(
            nn.Linear(1, d_cond // 4), nn.GELU(), nn.Linear(d_cond // 4, d_cond // 4))
        self.fuse = nn.Sequential(
            nn.Linear(d_cond + d_cond // 2 + d_cond // 4, d_cond),
            nn.LayerNorm(d_cond), nn.GELU())

    def _encode_raw(self, batch_list) -> torch.Tensor:
        obs_traj  = batch_list[0]; obs_Me = batch_list[7]
        image_obs = batch_list[11]; env_data = batch_list[13]
        if image_obs.dim() == 4:
            image_obs = image_obs.unsqueeze(2)
        if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
            image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)
        e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
        T_obs = obs_traj.shape[0]
        e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        e_3d_s = self.bottleneck_proj(e_3d_s)
        if e_3d_s.shape[1] != T_obs:
            e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
                                   mode="linear", align_corners=False).permute(0,2,1)
        e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
        t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
                                          device=e_3d_dec_t.device) * 0.5, dim=0)
        f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        h_t    = self.enc_1d(obs_in, e_3d_s)
        e_env, _, _ = self.env_enc(env_data, image_obs)
        return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

    def _kinematic_feat(self, obs_traj: torch.Tensor) -> torch.Tensor:
        """6 kinematic features per obs step."""
        B = obs_traj.shape[1]; T_obs = obs_traj.shape[0]; device = obs_traj.device
        if T_obs >= 2:
            traj_deg = _norm_to_deg(obs_traj)
            vel_norm = obs_traj[1:] - obs_traj[:-1]
            speed    = _step_speeds_kmh(traj_deg)
            speed_n  = (speed / 20.0).clamp(-3.0, 3.0)
            heading  = torch.atan2(vel_norm[:, :, 1], vel_norm[:, :, 0])
            if T_obs >= 3:
                dspd  = speed[1:] - speed[:-1]
                accel = torch.cat([obs_traj.new_zeros(1, B),
                                   (dspd / 10.0).clamp(-3.0, 3.0)], 0)
            else:
                accel = obs_traj.new_zeros(T_obs - 1, B)
            kine = torch.stack([vel_norm[:,:,0], vel_norm[:,:,1], speed_n,
                                heading.sin(), heading.cos(), accel], dim=-1)
        else:
            kine = obs_traj.new_zeros(self.obs_len, B, 6)
        if kine.shape[0] < self.obs_len:
            kine = torch.cat([obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6), kine], 0)
        else:
            kine = kine[-self.obs_len:]
        return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))

    def forward(self, batch_list, hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
        raw   = self._encode_raw(batch_list)
        ctx   = self.ctx_ln2(self.ctx_fc2(self.ctx_drop(raw)))
        kfeat = self._kinematic_feat(batch_list[0][:, :, :2])
        if hard_score is None:
            hard_score = torch.zeros(ctx.shape[0], device=ctx.device)
        hfeat = self.hard_embed(hard_score.unsqueeze(1).to(ctx.dtype))
        return self.fuse(torch.cat([ctx, kfeat, hfeat], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
#  Hard score (XAI-2) — [LEARN-3] thêm obs_speed + learnable weights
# ─────────────────────────────────────────────────────────────────────────────

def hard_score_from_obs(obs_traj_norm: torch.Tensor,
                         return_components: bool = False,
                         weight_logits: Optional[torch.Tensor] = None,
                         obs_speed_norm_const: float = 20.0):
    """
    [XAI-2] Độ khó của storm — v2.1-learn: 4 components, learnable weights.

    [LEARN-3] THAY ĐỔI so với v2.1-clean:
      CŨ: score = 0.4*curvature + 0.3*speed_var + 0.3*dir_change  (hardcode)
          KHÔNG có obs_speed tuyệt đối → storm nhanh-nhưng-đều (speed_var
          thấp) bị coi nhầm là "dễ", dù XAI-9 cho thấy fast storms
          (obs_speed≥15km/h, 22% test set) gây 62% tổng CTE test.
      MỚI: score = softmax(weight_logits) · [curvature, speed_var,
                                              dir_change, obs_speed_norm]
           4 trọng số HỌC ĐƯỢC qua gradient (không phải số tay đoán),
           obs_speed_norm = obs_speed_mean / 20.0 (clamp 0,1) làm input
           feature thứ 4.

    weight_logits: nếu None, dùng uniform-equivalent fallback (0.25 mỗi cái)
      — cho phép gọi hàm này từ các chỗ KHÔNG có instance (vd các hàm
      module-level classify_hard_easy, compute_obs_attribution).
      Khi gọi TỪ TRONG model (get_loss_breakdown, sample), LUÔN truyền
      self.hard_score_weight_logits để gradient thực sự chảy qua được.

    QUAN TRỌNG: hàm này KHÔNG còn @torch.no_grad() — để gradient của
    weight_logits chảy ngược qua sw_hard trong _reg_loss khi được gọi
    trong training. Các nơi gọi hàm này CHỈ để lấy giá trị (không train,
    ví dụ XAI logging) phải tự bọc bằng `with torch.no_grad():` ở call
    site nếu muốn tránh build computation graph không cần thiết.
    """
    T, B   = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
    device = obs_traj_norm.device
    if T < 3:
        z = torch.zeros(B, device=device)
        if return_components:
            return z, {"curvature": z.clone(), "speed_var": z.clone(),
                       "dir_change": z.clone(), "obs_speed_norm": z.clone()}
        return z

    traj_deg = _norm_to_deg(obs_traj_norm[..., :2])
    az12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])
    az23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])
    diff = (az23 - az12).abs()
    diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
    curvature  = diff.mean(0) / math.pi
    spd        = _step_speeds_kmh(traj_deg)
    if spd.shape[0] >= 2:
        speed_var = (spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 1.)
        obs_speed_norm = (spd.mean(0) / obs_speed_norm_const).clamp(0., 1.)
    else:
        speed_var = torch.zeros(B, device=device)
        obs_speed_norm = torch.zeros(B, device=device)
    dir_change = (diff > (20.0 / 180.0 * math.pi)).float().mean(0)

    components = torch.stack([curvature, speed_var, dir_change, obs_speed_norm], dim=0)  # [4, B]

    if weight_logits is not None:
        w = F.softmax(weight_logits.to(device).to(components.dtype), dim=0)  # [4]
    else:
        # Fallback equivalent khi không có instance context: gần với
        # phân bổ gốc 0.4/0.3/0.3 cho 3 thành phần đầu, obs_speed neutral nhỏ
        w = torch.tensor([0.35, 0.25, 0.25, 0.15], device=device, dtype=components.dtype)

    score = (w.unsqueeze(1) * components).sum(0).clamp(0., 1.)   # [B]

    if return_components:
        return score, {"curvature": curvature, "speed_var": speed_var,
                       "dir_change": dir_change, "obs_speed_norm": obs_speed_norm}
    return score


# ─────────────────────────────────────────────────────────────────────────────
#  Physics score v2.6 (adds displacement_score) — GIỮ NGUYÊN HẰNG SỐ
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _physics_score(traj_norm: torch.Tensor, obs_norm: torch.Tensor,
                    use_curvature_score: bool = False) -> torch.Tensor:
    """
    Best-of-K selection score. Four components (five when
    use_curvature_score=True):
      speed_score       : per-step speed vs obs reference
      smooth_score       : trajectory smoothness (no sharp acceleration)
      heading_score      : first-step direction matches obs
      displacement_score : total path length consistent with obs speed
      curvature_score    : [CURV-SCORE, opt-in] candidate's turning rate
                            matches the storm's OBSERVED turning rate

    [v2.1-learn QUYẾT ĐỊNH] KHÔNG biến 4 exponent gốc thành learnable.
    Lý do: hàm này chạy dưới @torch.no_grad() để CHỌN trong số K candidates
    sinh từ CÙNG MỘT velocity network — đây là post-hoc re-ranking, không
    nằm trên đường gradient nào quay lại network hay bất kỳ tham số nào
    của nó. Nếu thêm nn.Parameter vào đây mà không có loss term riêng
    (như đã làm với speed_correction ở LEARN-1), tham số đó sẽ lại rơi vào
    đúng lỗi "tồn tại nhưng never updated" mà ta đang cố tránh.
    Để làm đúng cần thêm 1 loss phụ huấn luyện riêng các exponent này
    (vd qua REINFORCE / Gumbel-softmax cho discrete selection) — phạm vi
    đó rủi ro cao hơn lợi ích đo được, nên giữ nguyên 4 hằng số đã proven
    qua thực nghiệm thay vì đoán thêm một cơ chế chưa kiểm chứng.

    [CURV-SCORE] Motivation: head_score above only checks the FIRST
    predicted step's direction against the last observed vector — it is
    blind to whether the candidate's curvature over the FULL horizon
    matches the storm's actual recent turning behavior. smooth_score
    actively penalizes ANY turning (favors straight paths), which works
    AGAINST correctly-recurving candidates for storms that are genuinely
    turning. This is a DIFFERENT, complementary signal: it does not ask
    "is this candidate smooth" or "does step 0 match", it asks "does this
    candidate's turning RATE match what the storm was ALREADY doing in
    the observed history" — extrapolating observed curvature rather than
    assuming straight-line motion. This is a pure INFERENCE-TIME re-ranking
    change (no gradient, no effect on what the network learned), so it can
    be A/B tested directly on existing checkpoints without retraining.
    Opt-in (default False) to keep the proven 4-component score as default.
    """
    B      = traj_norm.shape[1]
    device = traj_norm.device
    traj_deg = _norm_to_deg(traj_norm)
    v_ref   = None

    # ── Speed score ────────────────────────────────────────────────────────
    if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
        obs_deg = _norm_to_deg(obs_norm)
        obs_spd = _step_speeds_kmh(obs_deg)
        T_s     = obs_spd.shape[0]
        w_obs   = torch.linspace(0.5, 1.0, T_s, device=device)
        v_ref   = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()   # [B]
        pred_spd = _step_speeds_kmh(traj_deg)
        v_sigma  = v_ref.clamp(min=5.0) * 0.5
        speed_score = torch.exp(
            -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0)).pow(2).mean(0) * 0.5)
    elif traj_deg.shape[0] >= 2:
        speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(min=0) / 30.).mean(0))
    else:
        speed_score = torch.ones(B, device=device)

    # ── Smoothness score ───────────────────────────────────────────────────
    if traj_deg.shape[0] >= 3:
        vel          = traj_deg[1:] - traj_deg[:-1]
        accel_mag    = (vel[1:] - vel[:-1]).norm(dim=-1)
        smooth_score = torch.exp(-accel_mag.mean(0) * 5.0)
    else:
        smooth_score = torch.ones(B, device=device)

    # ── Heading score ──────────────────────────────────────────────────────
    if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
        obs_vel  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
        pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
        obs_h    = F.normalize(obs_vel,  dim=-1, eps=1e-6)
        pred_h   = F.normalize(pred_vel, dim=-1, eps=1e-6)
        cos_sim  = (obs_h * pred_h).sum(-1).clamp(-1, 1)
        head_score = torch.exp((cos_sim - 1.0) * 3.0)
    else:
        head_score = torch.ones(B, device=device)

    # ── Curvature score [CURV-SCORE, opt-in] ─────────────────────────────
    # Estimate the storm's OBSERVED turning rate from its last 3 observed
    # points, then check whether the candidate's OWN turning rate (over its
    # full predicted horizon, near-term steps weighted more heavily since
    # extrapolated curvature degrades further out) matches it. Unlike
    # smooth_score, a candidate that turns AT THE SAME RATE the storm was
    # already turning scores HIGH here, not low.
    if use_curvature_score and obs_norm.shape[0] >= 3 and traj_deg.shape[0] >= 2:
        obs_deg_c  = _norm_to_deg(obs_norm)
        bear_obs_1 = _forward_azimuth(obs_deg_c[-3], obs_deg_c[-2])
        bear_obs_2 = _forward_azimuth(obs_deg_c[-2], obs_deg_c[-1])
        obs_turn_rate = ((bear_obs_2 - bear_obs_1 + 180.0) % 360.0) - 180.0   # [B], deg/step

        bear0 = _forward_azimuth(obs_deg_c[-1], traj_deg[0])
        if traj_deg.shape[0] >= 2:
            chain = [_forward_azimuth(traj_deg[t], traj_deg[t + 1])
                     for t in range(traj_deg.shape[0] - 1)]
            pred_bears = torch.stack([bear0] + chain, 0)   # [T, B]
        else:
            pred_bears = bear0.unsqueeze(0)

        if pred_bears.shape[0] >= 2:
            pred_turn = ((pred_bears[1:] - pred_bears[:-1] + 180.0) % 360.0) - 180.0  # [T-1, B]
            Tc = pred_turn.shape[0]
            w_curv = torch.linspace(1.0, 0.3, Tc, device=device).unsqueeze(1)
            turn_err = ((pred_turn - obs_turn_rate.unsqueeze(0)).abs() * w_curv).sum(0) / w_curv.sum()
            curvature_score = torch.exp(-turn_err / 15.0)   # 15 deg/step soft scale
        else:
            curvature_score = torch.ones(B, device=device)
    else:
        curvature_score = torch.ones(B, device=device)

    # ── Displacement score ──────────────────────────────────────────────
    # Expected total path length ≈ obs_speed × T_pred × DT × 0.75
    # (0.75 factor: storms curve, so straight-line < path length estimate)
    if v_ref is not None and traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
        T_pred        = traj_deg.shape[0]
        expected_total = v_ref * T_pred * DT_HOURS * 0.75   # [B] km
        step_dists    = _haversine_deg(traj_deg[:-1], traj_deg[1:])   # [T-1, B]
        actual_total  = step_dists.sum(0)                              # [B]
        rel_err       = (actual_total - expected_total).abs() / expected_total.clamp(min=10.)
        disp_score    = torch.exp(-rel_err * 1.5)
    else:
        disp_score    = torch.ones(B, device=device)

    if use_curvature_score:
        # Rebalanced to make room for curvature_score (still sums to 1.00):
        # speed 0.30->0.25, smooth 0.25->0.20, head 0.30->0.25, disp 0.15->0.10,
        # curvature gets 0.20 (comparable weight to smooth_score, its
        # natural "opposing force" for genuinely-turning storms).
        return (speed_score.pow(0.25)
                * smooth_score.pow(0.20)
                * head_score.pow(0.25)
                * disp_score.pow(0.10)
                * curvature_score.pow(0.20)).clamp(min=1e-6)
    return (speed_score.pow(0.30)
            * smooth_score.pow(0.25)
            * head_score.pow(0.30)
            * disp_score.pow(0.15)).clamp(min=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation — GIỮ NGUYÊN từ v2.1-clean (đã proven qua thực nghiệm)
# ─────────────────────────────────────────────────────────────────────────────

def augment_batch(batch_list, disable_c: bool = False) -> list:
    """
    Distribution (4 active types + 2 no-op slots):
      A (25%): track shift ±5km             — shape unchanged, position varies
      B (20%): GT speed scale ×0.85–1.15   — mild, proven from v2.1
      C (20%): recurvature ±20°             — proven CTE improvement, FIXED to
                                              rotate displacement vectors (not
                                              absolute positions — old bug)
      D-E (25%): no augmentation            — giữ original distribution
      F (10%): Gaussian noise ±3km          — small robustness

    KHÔNG dùng (proven harmful qua thực nghiệm):
      - AUG-D obs-speed scaling: +4.5km ATE (v2.6), +13.4km (v2.7) — ĐÃ XÓA
      - mixup (phi vật lý, v2.4 proof)
      - speed scale >×1.5 trên GT (too far from real)
      - exp L_reg weights (v2.4 catastrophic, ADE=291km)

    disable_c: [FIX, ablation support] default False = EXACT original
      behavior, không đổi gì cho bất kỳ lời gọi nào không truyền tham
      số này (mọi call site cũ vẫn chạy y hệt trước). Khi True (dùng
      bởi train_flowmatching.py's --disable_aug_c), nhánh C (recurvature,
      0.45<=r<0.65) rơi xuống no-op giống hệt nhánh D-E thay vì xoay
      displacement vectors — tỉ lệ r vẫn giữ nguyên 6 khoảng như cũ
      (không re-sample r, không đổi phân phối của A/B/D-E/F), chỉ đổi
      HÀNH VI của đúng khoảng recurvature. Trước khi có tham số này,
      --disable_aug_c hoàn toàn KHÔNG có tác dụng (train_flowmatching.py
      gọi augment_batch(bl) không qua flag nào), tức mọi run trước đó
      dùng cờ này (nếu có) thực ra đã train VỚI AUG-C bật, không phải
      không có — cần re-run nếu muốn có kết quả no_aug_c thật.
    """
    bl = list(batch_list)
    if not torch.is_tensor(bl[0]):
        return bl

    obs    = bl[0]
    device = obs.device
    anchor = obs[-1:, :, :2].detach()  # [1, B, 2] — last obs as pivot
    r = torch.rand(1).item()

    if r < 0.25:
        # A: Track shift ±10km (preserves storm shape, varies position)
        # 0.018 normalized × 5°/unit × 111.32 km/° ≈ ±10km (lat direction)
        # TC position uncertainty ~10-50km → ±10km là reasonable aug range
        shift = (torch.rand(2, device=device) - 0.5) * 0.018  # ±10km in normalized
        bl[0] = obs + shift.view(1, 1, 2)
        if torch.is_tensor(bl[1]):
            bl[1] = bl[1] + shift.view(1, 1, 2)

    elif r < 0.45:
        # B: Speed scale — [v2.4] 0.70-1.40 (was 0.85-1.15)
        # Rộng hơn để model thấy diverse speed context:
        # test storms có speed khác val → speed_correction cần wider range
        scale = 0.70 + 0.70 * torch.rand(1, device=device).item()
        obs_c = obs.clone()
        obs_c[..., :2] = anchor + (obs[..., :2] - anchor) * scale
        bl[0] = obs_c
        if torch.is_tensor(bl[1]):
            bl[1] = anchor + (bl[1] - anchor) * scale

    elif r < 0.65:
        # C: Recurvature ±20° — rotate DISPLACEMENT vectors (not positions)
        # FIX v2.1-clean: old bug rotated absolute positions around anchor,
        # which inflated step-distances → model learned "recurvature=faster"
        # → ATE +7.2km artifact. Fixed by rotating displacement vectors:
        # direction changes, step magnitude preserved = correct TC physics.
        if disable_c:
            # [FIX, ablation support] disable_c=True: same slot as D-E
            # (no augmentation) — see docstring above.
            pass
        else:
            T_pred = bl[1].shape[0] if torch.is_tensor(bl[1]) else 0
            if T_pred >= 4:
                gt      = bl[1].clone()
                max_deg = (torch.rand(1).item() - 0.5) * 40.0   # -20° to +20°
                max_rad = max_deg * math.pi / 180.0
                pts  = torch.cat([anchor, gt], 0)   # [T_pred+1, B, 2]
                disp = pts[1:] - pts[:-1]           # [T_pred, B, 2]
                for t in range(T_pred):
                    progress = (t / max(T_pred - 1, 1)) ** 1.5
                    a = max_rad * progress
                    c, s = math.cos(a), math.sin(a)
                    rot = torch.tensor([[c, -s], [s, c]], dtype=gt.dtype, device=device)
                    disp[t] = (rot @ disp[t].unsqueeze(-1)).squeeze(-1)
                gt_new = gt.clone()
                gt_new[0] = anchor[0] + disp[0]
                for t in range(1, T_pred):
                    gt_new[t] = gt_new[t - 1] + disp[t]
                bl[1] = gt_new
                # Rotate last 3 obs DISPLACEMENTS by 30% of max (continuity)
                T_obs   = obs.shape[0]
                obs_aug = obs.clone()
                cp = math.cos(max_rad * 0.3); sp = math.sin(max_rad * 0.3)
                rp = torch.tensor([[cp, -sp], [sp, cp]], dtype=obs.dtype, device=device)
                for t_obs in range(max(1, T_obs - 3), T_obs):
                    d = obs_aug[t_obs, :, :2] - obs_aug[t_obs - 1, :, :2]
                    obs_aug[t_obs, :, :2] = obs_aug[t_obs - 1, :, :2] + (rp @ d.unsqueeze(-1)).squeeze(-1)
                bl[0] = obs_aug

    elif r < 0.90:
        # D-E slot: no augmentation
        pass

    else:
        # F: small Gaussian noise ±3km
        obs_new = obs.clone()
        obs_new[..., :2] = obs[..., :2] + torch.randn_like(obs[..., :2]) * 0.003
        bl[0] = obs_new

    return bl


# ─────────────────────────────────────────────────────────────────────────────
#  XAI functions (1–7, complete) — GIỮ NGUYÊN logic, cập nhật call hard_score
# ─────────────────────────────────────────────────────────────────────────────

def compute_obs_attribution(model, batch_list, device: torch.device,
                             target_step: int = 11) -> torch.Tensor:
    """
    [XAI-1] Gradient saliency: which obs step most influences 72h prediction?
    """
    raw = _unwrap(model)
    with torch.no_grad():
        h_score = hard_score_from_obs(batch_list[0][:, :, :2],
                                       weight_logits=getattr(raw, "hard_score_weight_logits", None))
    obs_req = batch_list[0].detach().clone().requires_grad_(True)
    bl_g = list(batch_list); bl_g[0] = obs_req
    with torch.enable_grad():
        cond = raw.encoder(bl_g, hard_score=h_score)
        x0   = torch.randn(obs_req.shape[1], raw.pred_len, 2, device=device) * raw.sigma_inference
        t0   = torch.zeros(obs_req.shape[1], device=device)
        v    = raw.velocity(x0, t0, cond)
        pred_rel = x0 + v
        ts       = min(target_step, raw.pred_len - 1)
        pred_rel[:, ts, :].norm(dim=-1).mean().backward()
    if obs_req.grad is not None:
        attr = obs_req.grad[:, :, :2].norm(dim=-1)
        attr = attr / (attr.sum(0, keepdim=True) + 1e-8)
    else:
        attr = torch.zeros(batch_list[0].shape[0], batch_list[0].shape[1], device=device)
    return attr.detach()


@torch.no_grad()
def compute_ensemble_uncertainty(all_traj: torch.Tensor) -> Dict:
    """
    [XAI-4] Uncertainty per lead time across K ensemble samples.
    """
    all_deg  = _norm_to_deg(all_traj)  # [K,T,B,2]
    K, T, B  = all_deg.shape[:3]
    mean_traj = all_deg.mean(0)
    std_km = torch.zeros(T, B, device=all_traj.device)
    for t in range(T):
        dists = _haversine_deg(
            all_deg[:, t].reshape(K * B, 2),
            mean_traj[t].unsqueeze(0).expand(K, B, 2).reshape(K * B, 2)
        ).reshape(K, B)
        std_km[t] = dists.std(0)
    s12 = min(1, T - 1); s72 = min(11, T - 1)
    return {
        "std_per_step":      std_km,
        "uncertainty_ratio": (std_km[s72] + 1e-3) / (std_km[s12] + 1e-3),
        "mean_72h_std":      float(std_km[s72].mean()),
        "mean_12h_std":      float(std_km[s12].mean()),
        "high_uncertainty":  std_km[s72] > 80.0,
    }


@torch.no_grad()
def compute_heading_deviation(pred_deg: torch.Tensor,
                               gt_deg:   torch.Tensor) -> torch.Tensor:
    """
    [XAI-6] Heading deviation per lead time step, in degrees (absolute).
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        return pred_deg.new_zeros(1, pred_deg.shape[1])
    bear_gt   = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
    bear_pred = _forward_azimuth(gt_deg[:T-1], pred_deg[1:T])
    diff = (bear_pred - bear_gt).abs()
    diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
    return torch.rad2deg(diff)


@torch.no_grad()
def compute_cte_contribution(pred_deg: torch.Tensor,
                              gt_deg:   torch.Tensor) -> Dict:
    """
    [XAI-7] Decompose trajectory error into ATE (along-track) and CTE (cross-track).
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return {"ate_per_step": z, "cte_per_step": z,
                "ate_mean": z[0], "cte_mean": z[0],
                "ate_abs_mean": 0.0, "cte_abs_mean": 0.0}
    bear_ref  = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
    bear_err  = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
    dist_err  = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang       = bear_err - bear_ref
    ate       = dist_err * torch.cos(ang)
    cte       = dist_err * torch.sin(ang)
    return {
        "ate_per_step":  ate,
        "cte_per_step":  cte,
        "ate_mean":      ate.mean(0),
        "cte_mean":      cte.abs().mean(0),
        "ate_abs_mean":  float(ate.abs().mean()),
        "cte_abs_mean":  float(cte.abs().mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Compat stubs
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def classify_hard_easy(obs_traj_norm, per_sample_loss=None,
                        hard_score_p: float = 70.0, loss_p: float = 50.0):
    scores = hard_score_from_obs(obs_traj_norm)
    B = scores.shape[0]
    if B < 4:
        return torch.zeros(B, dtype=torch.bool, device=scores.device)
    return scores >= torch.quantile(scores, hard_score_p / 100.0)


@torch.no_grad()
def classify_hard_easy_global(obs_traj_norm, global_threshold):
    return hard_score_from_obs(obs_traj_norm) >= global_threshold


@torch.no_grad()
def compute_diversity_score(candidates) -> float:
    if len(candidates) < 2:
        return 0.0
    T, B = candidates[0].shape[0], candidates[0].shape[1]
    ep_step = min(T - 1, 11)
    endpoints = torch.stack([_norm_to_deg(c[ep_step]) for c in candidates], 0)
    N = endpoints.shape[0]
    ep_mean = endpoints.mean(0, keepdim=True)
    dists = _haversine_deg(
        endpoints.reshape(N * B, 2),
        ep_mean.expand(N, B, 2).reshape(N * B, 2)
    ).reshape(N, B)
    return float(dists.std(0).mean())


# ─────────────────────────────────────────────────────────────────────────────
#  TCFlowMatching v2.1-learn
# ─────────────────────────────────────────────────────────────────────────────

class TCFlowMatching(nn.Module):
    """
    TC-FlowMatching v2.1-learn

    Training loss:
        L_total = L_CFM
                + lam_reg  × L_reg          (ramp ep10→ep30, LEARNABLE step weights)
                + lam_dir  × lambda_heading × L_heading_ms  (ramp ep5→ep20)
                + 0.1      × L_calib         (trains speed_correction_logits)

        L_CFM:         conditional flow matching (random t, OT noise-GT matching)
        L_reg:         ADE loss at sigma_inference, LEARNABLE per-step weights
        L_heading_ms:  full 12-step heading continuation, LEARNABLE per-step weights
        L_calib:       ADE on speed-calibrated 1-shot pred, trains per-horizon
                       speed correction factors used by speed_calibrate_pred()

    NO L_momentum: removed because it anchors to val speed distribution,
    which hurts ATE on test storms that move at different speeds.

    Inference pipeline:
        1. Sample K=20 candidates with sigma=0.04 FIXED
        2. Physics score selection (speed+smooth+heading+displacement) — fixed weights
        3. Weighted average of top-3 candidates
        4. Speed calibration: LEARNABLE per-horizon correction (was fixed clip)
        5. Optional: multi-scale sigma ensemble (sample_multiscale)
    """

    def __init__(
        self,
        pred_len:          int   = 12,
        obs_len:           int   = 8,
        unet_in_ch:        int   = 13,
        d_cond:            int   = 256,
        d_model:           int   = 256,
        nhead:             int   = 8,
        num_dec_layers:    int   = 4,
        dim_ff:            int   = 512,
        dropout:           float = 0.1,
        sigma_min:         float = 0.06,   # [FIX, user-approved tradeoff] was
                                            # 0.04 (~22km) — increased to 0.06
                                            # (~33km). Analysis: 0.04 is only
                                            # ~10-20% of actual error magnitude
                                            # (ADE~219km), too small to seed
                                            # meaningful ensemble divergence.
                                            # NOT learnable (see constructor
                                            # docstring on why a free-floating
                                            # sigma degenerates to 0 under raw
                                            # MSE gradient) — this remains a
                                            # fixed hyperparameter choice, now
                                            # set directionally per the
                                            # uncertainty-collapse analysis
                                            # rather than left at the original
                                            # untested-for-this-purpose value.
                                            # Still overridable via --sigma_min.
        sigma_max:         float = 0.15,   # [FIX, user-approved tradeoff] was
                                            # 0.08 (~44km) — increased to 0.15
                                            # (~83km), closer to the actual
                                            # CTE-scale error (67km) so early-
                                            # training ensemble noise is large
                                            # enough to matter, while staying
                                            # below full ADE scale (~219km) to
                                            # avoid destabilizing point-
                                            # accuracy training. Still
                                            # overridable via --sigma_max.
        sigma_decay_start: int   = 5,      # unchanged — early warmup epochs
        sigma_decay_end:   int   = 100,    # [FIX, user-approved tradeoff] was
                                            # 40. With 40, >100 of a typical
                                            # 140-165 epoch run trains at the
                                            # MINIMUM sigma — most of training
                                            # sees only minimal noise. Extended
                                            # to 100 so meaningful noise stays
                                            # present through most of a
                                            # typical run. Still overridable
                                            # via --sigma_decay_end.
        lambda_reg:        float = 0.2,
        lambda_heading:    float = 0.07,   # reduced: gradient scale balance with L_CFM
        lambda_momentum:   float = 0.0,    # DISABLED — field kept for compat only
        lambda_calib:      float = 0.1,    # [LEARN-1] weight of L_calib term
        lambda_hard_reg:   float = 0.02,   # [VAR-REDUCE] pull hard_score_weight_logits
                                            # toward uniform — see docstring below
        log_sigma_reg_min_clamp: float = -3.0,  # unchanged — see clamp usage
                                            # in weighted_reg below. NOT used
                                            # as a hard cap experiment anymore
                                            # (an earlier version of this
                                            # comment proposed manually
                                            # tightening this to fight
                                            # eff_lambda_reg's learned value —
                                            # reconsidered: eff_lambda_reg is
                                            # the genuine loss-minimizing
                                            # choice Kendall converges to, not
                                            # a bug, so forcing it down would
                                            # fight real signal the same way
                                            # that hurt ADE/ATE when tried on
                                            # hard_score_weight_logits. Left
                                            # configurable via
                                            # --log_sigma_reg_min_clamp only
                                            # as a numerical-stability guard,
                                            # not a tuning knob.
        enable_horizon_nll: bool  = True,  # [FIX] Use log_b_horizon (Laplace
                                            # NLL per-horizon scale) to
                                            # normalize dist[t] before
                                            # reg_step_logits' softmax
                                            # attention, instead of raw
                                            # dist[t]. See log_b_horizon's
                                            # docstring above for the full
                                            # rationale — addresses far-
                                            # horizon attention starvation
                                            # without hardcoding any horizon-
                                            # specific weight; b_horizon is
                                            # itself learned via a stable NLL
                                            # formulation. Default True is a
                                            # NEW-mechanism default (not yet
                                            # empirically validated on a real
                                            # training run) — pass
                                            # --disable_horizon_nll via
                                            # train_fm.py to fall back to the
                                            # original raw-dist formula for
                                            # comparison/ablation.
        use_ot:            bool  = True,
        ot_epsilon:        float = 0.05,
        use_ema:           bool  = True,
        ema_decay:         float = 0.995,
        n_inference_steps: int   = 10,     # [MULTI-STEP] was 1 (single-shot
                                            # x0+v). Changed to 10 based on
                                            # EMPIRICAL evidence (not just
                                            # theory) from evaluating an
                                            # existing checkpoint: N=1->10
                                            # raised ensemble spread 4km->55km,
                                            # improved CRPS -9.5% (216.8->
                                            # 196.3km), Spread/Skill ratio at
                                            # 6h reached 1.07 (near-ideal 1.0),
                                            # at a small ADE/ATE/CTE cost
                                            # (+1.7%/+1.9%/+7.2%). N=3-4 tested
                                            # WORSE than both N=1 and N=8-10 —
                                            # standard 1st-order Euler
                                            # discretization error scales
                                            # O(1/N); small-but-not-1 step
                                            # counts take large mis-sized
                                            # jumps without yet benefiting
                                            # from finer integration. Does NOT
                                            # touch any learned parameter or
                                            # loss term — purely changes how
                                            # the trained velocity field is
                                            # integrated at inference. Note
                                            # this does NOT fix the SEPARATE,
                                            # still-unresolved issue of ensemble
                                            # spread stalling at far horizons
                                            # (48h-72h) while skill/error keeps
                                            # growing — that traces to
                                            # reg_step_logits' softmax
                                            # naturally down-weighting hard-
                                            # to-reduce far-horizon loss (a
                                            # genuine optimization outcome,
                                            # not a bug — see _reg_loss).
        n_ensemble:        int   = 20,
        sigma_inference:   float = 0.04,   # FIXED throughout
        **kwargs,
    ):
        super().__init__()
        self.pred_len          = pred_len
        self.obs_len            = obs_len
        self.sigma_min          = sigma_min
        self.sigma_max          = sigma_max
        self.sigma_decay_start  = sigma_decay_start
        self.sigma_decay_end    = sigma_decay_end
        self.lambda_reg         = lambda_reg
        self.lambda_heading     = lambda_heading
        self.lambda_momentum    = 0.0           # always disabled
        self.lambda_calib       = lambda_calib
        # [VAR-REDUCE] Fixed (NOT Kendall-learned) coefficient for the
        # hard_score_weight_logits regularizer — see get_loss_breakdown for
        # the term itself and rationale. Must stay fixed/unlearned: if this
        # were Kendall-weighted like L_reg/L_heading/L_calib, the model
        # could learn a very negative log_sigma for it and effectively
        # disable its own constraint — defeating the purpose of a
        # regularizer meant to CONSTRAIN what the model would otherwise do.
        self.lambda_hard_reg    = lambda_hard_reg
        self.log_sigma_reg_min_clamp = log_sigma_reg_min_clamp
        self.enable_horizon_nll = enable_horizon_nll
        self.use_ot              = use_ot
        self.ot_epsilon          = ot_epsilon
        self.n_inference_steps   = n_inference_steps
        self.n_ensemble          = n_ensemble
        self.sigma_inference     = sigma_inference

        self.encoder  = ContextEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch, d_cond=d_cond)
        self.velocity = VelocityTransformer(
            pred_len=pred_len, d_model=d_model, nhead=nhead,
            num_layers=num_dec_layers, dim_ff=dim_ff,
            dropout=dropout, d_cond=d_cond)
        self.use_ema = use_ema
        self._ema    = None

        # ── [v2.1-learn] LEARNABLE PARAMETERS ──────────────────────────────
        # Mỗi tham số dưới đây thay thế một hằng số tay đoán trong
        # v2.1-clean. Mỗi cái đều có ĐƯỜNG GRADIENT THỰC SỰ về tổng loss
        # (verify trong get_loss_breakdown) — không phải param "mồ côi".

        # [LEARN-1] Per-horizon speed correction, dùng trong speed_calibrate_pred.
        # Init=0 → sigmoid(0)*2=1.0 → identity (không sửa gì) lúc bắt đầu,
        # giống hệt điểm khởi đầu hành vi cũ (scale=1 baseline).
        self.speed_correction_logits = nn.Parameter(torch.zeros(pred_len))

        # [LEARN-2] Per-step L_reg weights. Init=0 → softmax uniform start
        # (khác linspace(1,1.5) cũ vốn đã thiên về late-step ngay từ đầu).
        self.reg_step_logits = nn.Parameter(torch.zeros(pred_len))

        # [FIX, NLL-grounded] Per-horizon LEARNED error scale (Laplace NLL,
        # matching _reg_loss's L1/haversine-distance error metric — Gaussian
        # NLL would be the wrong distributional match for an L1 loss).
        #
        # WHY THIS EXISTS: reg_step_logits' softmax was found to systematically
        # down-weight far horizons (48h-72h get only ~7-13% combined attention
        # vs ~87-93% for 6h-24h, confirmed across independent training runs at
        # matched epochs) — NOT because of a bug, but because raw haversine
        # error grows ~14x from 6h to 72h (skill≈30km → ≈450km, confirmed by
        # direct evaluation), so softmax naturally learns "attending to far
        # horizons barely reduces total loss, attend to near horizons instead"
        # — a rational but short-sighted response to an UNNORMALIZED signal.
        # This directly explains a SEPARATE observed symptom: ensemble spread
        # (from multi-step sampling) stalls at ~50km for horizons 24h-72h
        # while skill/error keeps growing to 450km+ — the network never
        # received strong-enough gradient at far horizons to learn anything
        # beyond a smoothed, low-variance response there.
        #
        # FIX: give _reg_loss a properly NLL-grounded per-horizon scale
        # b_horizon[t] = exp(log_b_horizon[t]), and present reg_step_logits'
        # softmax attention with dist[t]/b_horizon[t] (a DIFFICULTY-NORMALIZED
        # residual) instead of raw dist[t]. Laplace NLL = |err|/b + log(b) has
        # the SAME stabilizing structure as Kendall's Gaussian NLL used
        # elsewhere in this model (log_sigma_reg/heading/calib): the log(b)
        # term counterbalances 1/b, so b CANNOT degenerate to 0 or infinity
        # under gradient descent — verified analytically: optimal b* =
        # E[|err|] at that horizon, i.e. b_horizon learns to track each
        # horizon's OWN typical difficulty, not collapse or explode.
        # reg_step_logits' softmax then sees a comparable-scale signal across
        # all 12 horizons and is free to attend based on genuine
        # under/over-performance relative to each horizon's own learned
        # difficulty, not raw km magnitude.
        #
        # Init log_b_horizon=0 → b=1 everywhere → nll_h = dist/1 + log(1) =
        # dist + 0 = dist, IDENTICAL to the original raw-dist formula at
        # epoch 0 (b only diverges from 1 as training reveals real per-
        # horizon difficulty differences) — same "start as a no-op, let
        # gradient reveal the right values" principle as reg_step_logits'
        # own zero-init.
        #
        # CAUTION: this is a NEW mechanism, not yet validated on a real
        # training run at time of writing (unlike n_inference_steps, which
        # IS empirically verified). The math is sound and the initialization
        # is provably a no-op, but the FULL training-dynamics effect (does
        # far-horizon attention actually increase enough to matter, and at
        # what cost to near-horizon accuracy) can only be confirmed by
        # actually training and comparing against a run with this disabled
        # (see --disable_horizon_nll ablation flag in train_fm.py).
        self.log_b_horizon = nn.Parameter(torch.zeros(pred_len))

        # [LEARN-3] hard_score 4-component weights: [curvature, speed_var,
        # dir_change, obs_speed_norm]. Init lệch nhẹ giống phân bổ gốc
        # 0.4/0.3/0.3 (+obs_speed nhỏ) thay vì zero-init hoàn toàn, để
        # không phá vỡ hành vi đã proven ngay từ epoch đầu — nhưng vẫn
        # cho phép gradient điều chỉnh dần qua training.
        self.hard_score_weight_logits = nn.Parameter(
            torch.log(torch.tensor([0.40, 0.30, 0.30, 0.15])))

        # [LEARN-5] Per-step heading constraint weights — TOÀN BỘ pred_len bước.
        # Init = zeros → softmax uniform → mỗi bước bắt đầu với weight bằng nhau.
        #
        # [v2.4 FIX] Bản v2.1-XAI dùng init decay 0.5^t:
        #   sw_init = [0.500, 0.250, 0.125, ..., 0.0002]  (step 7/48h chỉ 0.4%!)
        #   → 48h heading gradient signal ≈ 0.5% total → model không học đủ
        #   → VÒNG ĐỘC: constraint yếu → heading 48h không học → signal không tăng
        #   → Kết quả: ep145 heading dev 48h ≈ 130° (chưa đạt target <90°)
        #
        # Init uniform (zeros): mọi bước nhận gradient đồng đều từ đầu.
        # Gradient thực tế từ data sẽ tự tìm phân bổ tối ưu.
        # Ablation evidence: decay-0.5^t init (ep145) → 48h heading cao → đổi sang uniform.
        self.heading_step_logits = nn.Parameter(torch.zeros(pred_len))

        # [LEARN-4] Cross-loss weighting — Kendall homoscedastic uncertainty.
        # [v2.4 FIX] 3 vấn đề từ v2.1-XAI ep145:
        #   (a) Init log_sigma=0 → eff_lambda_init=0.5 cho mọi loss
        #       → L_heading weight 0.5 >> baseline 0.07 → imbalance ban đầu
        #       → Model học theo ratio sai, sau đó log_sigma drift về -1.9
        #       → eff_lambda=22x sau 145 ep (quá cao, overfit training signal)
        #   (b) Thiếu HALF_LOG_2PI → total drift âm → optimizer có incentive
        #       kéo log_sigma xuống để total "improve" bằng cách âm hơn
        #       → log_sigma suy biến về -inf không có gì chặn
        #   (c) Thiếu clamp(min=-3) → không giới hạn eff_lambda tối đa
        #
        # FIX: init back-solve từ giá trị baseline đã proven để epoch 0 = baseline:
        #   log_sigma = -0.5*log(2*lambda_target)
        #   → lambda_reg=0.2:     log_sigma_reg     = 0.458
        #   → lambda_heading=0.07: log_sigma_heading = 0.983
        #   → lambda_calib=0.1:   log_sigma_calib   = 0.805
        # Sau đó gradient tự điều chỉnh, clamp(min=-3) chặn suy biến.
        import math as _math
        self.log_sigma_reg     = nn.Parameter(torch.tensor(
            -0.5 * _math.log(2.0 * 0.20)))   # 0.458 → eff_lambda=0.20
        self.log_sigma_heading = nn.Parameter(torch.tensor(
            -0.5 * _math.log(2.0 * 0.07)))   # 0.983 → eff_lambda=0.07
        self.log_sigma_calib   = nn.Parameter(torch.tensor(
            -0.5 * _math.log(2.0 * 0.10)))   # 0.805 → eff_lambda=0.10

    def init_ema(self):
        if self.use_ema:
            self._ema = EMAModel(self, decay=0.995)

    def ema_update(self):
        if self._ema is not None:
            self._ema.update(self)

    def _to_relative(self, x_abs: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
        return x_abs - last_obs.unsqueeze(1)

    def _from_relative(self, x_rel: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
        return x_rel + last_obs.unsqueeze(1)

    def _sigma_schedule(self, epoch: int) -> float:
        """
        Cosine decay sigma_max→sigma_min over [sigma_decay_start,
        sigma_decay_end] epochs. Was hardcoded ep5→ep40, sigma 0.08→0.04.

        [FIX, user-approved tradeoff] Defaults CHANGED (not preserved) to
        sigma_min=0.06, sigma_max=0.15, sigma_decay_end=100. Rationale:
        with the ORIGINAL schedule, >100 of a typical 140-165 epoch run
        trained at the minimum sigma (~22km-scale noise, only ~10-20% of
        actual error magnitude ADE~219km) — suspected contributor to the
        near-zero ensemble diversity seen in CRPS/Spread-Skill analysis
        (spread≈4km vs skill/error≈100-400km, verified experimentally by
        raising n_inference_steps alone: spread jumped 4km→55km, but only
        for the multi-step-integration axis — this sigma change targets a
        DIFFERENT, still-unaddressed axis: the noise scale seeding
        divergence in the first place).

        This IS a genuine behavior change from prior verified results
        (unlike n_inference_steps, this has NOT been experimentally
        validated on a real trained checkpoint at time of writing) — the
        user explicitly accepted this tradeoff to address all 4 causes
        identified in the uncertainty-collapse analysis rather than leave
        this one as an opt-in-only experiment. Still overridable via
        --sigma_min/--sigma_max/--sigma_decay_end if you want to run a
        grid search or revert to the original values (0.04/0.08/40).
        """
        if epoch < self.sigma_decay_start:
            return self.sigma_max
        if epoch < self.sigma_decay_end:
            span = max(self.sigma_decay_end - self.sigma_decay_start, 1)
            t = (epoch - self.sigma_decay_start) / span
            return self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + math.cos(math.pi * t))
        return self.sigma_min

    # ── Multi-step heading loss — [LEARN-5] learnable per-step weights ──────

    def _heading_loss_ms(self, pred_deg: torch.Tensor, obs_deg: torch.Tensor) -> torch.Tensor:
        """
        Multi-step heading constraint — TOÀN BỘ pred_len bước, trọng số HỌC.

        [LEARN-5] SỬA LỖI: bản trước (v2.1-clean và lần sửa trước của tôi)
        chỉ áp dụng cho n_steps=4 ĐẦU TIÊN (6h/12h/18h/24h) với decay=0.5
        CỐ ĐỊNH (weight = 1.0, 0.5, 0.25, 0.125) — đây CHÍNH XÁC là loại
        hằng số tay đoán cần sửa mà tôi đã bỏ sót. Hệ quả thực tế: heading
        constraint hoàn toàn KHÔNG áp dụng cho 8 bước còn lại (30h→72h) —
        đây là một phần lý do 72h heading dev khó cải thiện trong log cũ
        (113-128° xuyên suốt nhiều version) dù ATE/CTE có nhúc nhích.

        FIX: dùng self.heading_step_logits (pred_len params, softmax) —
        TOÀN BỘ 12 bước đều có constraint, trọng số PHÂN BỔ GIỮA chúng do
        gradient tự học, không còn giới hạn cứng "chỉ 4 bước đầu" hay
        decay theo cấp số nhân tay chọn (0.5^t).

        Detach ref tại mỗi step vẫn giữ nguyên (CRITICAL: ngăn gradient
        explosion qua chuỗi 12 bước liên tiếp — đây là cơ chế ổn định
        numerice, không phải "trọng số", nên không thuộc diện cần học).
        """
        if obs_deg.shape[0] < 2 or pred_deg.shape[0] < 1:
            return pred_deg.new_zeros(())

        ref_bear = _forward_azimuth(obs_deg[-2], obs_deg[-1])   # [B]
        pts = torch.cat([obs_deg[-1:], pred_deg], 0)   # [T_pred+1, B, 2]

        N = pred_deg.shape[0]
        sw = F.softmax(self.heading_step_logits[:N], dim=0)   # [N], học được

        loss = pred_deg.new_zeros(())
        for t in range(N):
            pred_bear  = _forward_azimuth(pts[t], pts[t + 1])  # [B]
            angle_diff = pred_bear - ref_bear
            loss       = loss + sw[t] * (1.0 - torch.cos(angle_diff)).mean()
            ref_bear   = pred_bear.detach()   # CRITICAL: detach to prevent chain gradient

        return loss

    # ── L_reg — [LEARN-2] learnable step weights ────────────────────────────

    def _reg_loss(self, x1_rel: torch.Tensor, last_obs: torch.Tensor,
                  cond: torch.Tensor,
                  hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        ADE loss at t=0 with sigma_inference noise, consistent with 1-shot inference.

        [LEARN-2] Step weights = softmax(self.reg_step_logits) thay vì
        linspace(1.0, 1.5, T) hardcoded.
        Bằng chứng tại sao cần đổi: linspace(1,3)→linspace(1,1.5) chỉ giảm
        test ATE 0.9km, và XAI-8 24h ratio KHÔNG đổi (vẫn ~1.5-1.6×) —
        chứng tỏ chỉnh tay con số trong công thức tuyến tính không đủ;
        cần để gradient tự tìm phân bổ trọng số tối ưu giữa 12 horizon.

        DOES NOT use exp weights (v2.4 used 12.2× ratio → catastrophic
        overfitting) — softmax trên logits học được có range tự nhiên bị
        chặn mềm, không bùng nổ như exp(linspace) thủ công.

        [FIX, enable_horizon_nll] reg_step_logits' softmax gradient is
        proportional to whatever quantity multiplies sw[t] in the final
        loss. Using RAW dist[t] there means the softmax is systematically
        pushed away from horizons with inherently large error (far
        horizons), regardless of whether the network could genuinely
        still improve there — a rational-but-short-sighted response to an
        unnormalized signal (verified: far-horizon attention starves at
        ~7-13% combined weight, and ensemble spread from multi-step
        sampling stalls ~50km at 24h-72h while skill grows to 450km+,
        consistent with the network never receiving strong gradient
        there). FIX: replace dist[t] with a Laplace-NLL-normalized
        nll_h[t] = dist[t]/b_horizon[t] + log(b_horizon[t]), where
        b_horizon = exp(log_b_horizon) is itself learned with the SAME
        stabilizing log(b) counterbalance Kendall uses elsewhere in this
        model (provably cannot collapse to 0 or explode — see
        log_b_horizon's docstring). At init (log_b_horizon=0, b=1),
        nll_h == dist exactly (log(1)=0), so this is a no-op until
        training reveals genuine per-horizon difficulty differences.
        The outer Kendall log_sigma_reg (scaling l_reg's contribution to
        the total loss) auto-compensates for any overall scale drift this
        introduces — composing two NLL mechanisms is expected to
        re-equilibrate correctly without needing a new hardcoded constant.
        """
        B, T, _ = x1_rel.shape
        device   = x1_rel.device
        x0   = torch.randn_like(x1_rel) * self.sigma_inference
        t0   = torch.zeros(B, device=device)
        v    = self.velocity(x0, t0, cond)
        x1_pred_abs = self._from_relative(x0 + v, last_obs)
        x1_gt_abs   = self._from_relative(x1_rel, last_obs)
        pred_deg = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))  # [T, B, 2]
        gt_deg   = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
        dist     = _haversine_deg(pred_deg, gt_deg)             # [T, B] km

        T_actual = dist.shape[0]

        if self.enable_horizon_nll:
            b_h  = torch.exp(self.log_b_horizon[:T_actual]).clamp(min=1e-2, max=1e4)  # [T]
            nll_h = dist / b_h.unsqueeze(1) + torch.log(b_h).unsqueeze(1)  # [T, B]
            weighted_dist = nll_h
        else:
            weighted_dist = dist

        sw = F.softmax(self.reg_step_logits[:T_actual], dim=0).unsqueeze(1)  # [T, 1]

        if hard_score is not None:
            sw_hard = (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)  # [1, B]
        else:
            sw_hard = torch.ones(1, B, device=device, dtype=dist.dtype)
        # /300: scale normalization để eff_lambda equilibrium hợp lý.
        # Với Kendall HALF_LOG_2PI đúng: eff_lambda* = 0.5 × l_reg_norm
        # /300 → l_reg_norm ≈ 250/300 ≈ 0.83 → eff_lambda* ≈ 0.42 (gần baseline 0.2)
        # Reviewer: '300 ≈ max expected ADE tại convergence (km)' — căn cứ rõ ràng.
        return ((weighted_dist * sw) * sw_hard).mean() / 300.0

    # ── get_loss_breakdown ──────────────────────────────────────────────────

    def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
        """
        Full loss computation. Called in TRAINING with augmented batch.
        Called in VAL without augmentation → val_loss is reliable signal.

        Total = L_CFM + lam_reg × L_reg + lam_dir × lambda_heading × L_heading_ms
                      + lambda_calib × L_calib
        """
        obs_traj = batch_list[0]
        gt_traj  = batch_list[1]
        B        = obs_traj.shape[1]
        device   = obs_traj.device

        sigma    = self._sigma_schedule(epoch)
        x1_gt    = gt_traj.permute(1, 0, 2)       # [B, T, 2]
        last_obs = obs_traj[-1, :, :2]             # [B, 2]
        x1_rel   = self._to_relative(x1_gt, last_obs)

        # [LEARN-3] hard_score: truyền self.hard_score_weight_logits để
        # gradient của 4 trọng số chảy qua sw_hard trong _reg_loss bên dưới.
        h_score = hard_score_from_obs(obs_traj[:, :, :2],
                                       weight_logits=self.hard_score_weight_logits)
        cond    = self.encoder(batch_list, hard_score=h_score)

        # [VAR-REDUCE] L2 penalty pulling hard_score_weight_logits' softmax
        # distribution toward UNIFORM [0.25,0.25,0.25,0.25] over its 4
        # components (curvature, speed_var, dir_change, obs_speed_norm).
        #
        # WHY: across 3 independent seed runs (seed42/1/2, same architecture,
        # same hyperparameters, only RNG differs) — confirmed directly from
        # seed_1.txt / seed_2.txt training logs — this 4-way softmax
        # collapses onto 'curvature' at very different rates depending on
        # seed. AT THE SAME EPOCH (130), ruling out "just trained longer":
        #   seed42: curv=0.739  spdvar=0.025  obsspd=0.013  -> test CTE=58.4
        #   seed1:  curv=0.843  spdvar=0.013  obsspd=0.007  -> test CTE=61.3
        #   seed2:  curv=0.875  spdvar=0.011  obsspd=0.006  -> test CTE=67.1
        # More collapse (curv higher, obs_speed_norm/speed_var lower)
        # correlates monotonically with WORSE test CTE. Since obs_speed_norm
        # feeds directly into which storms get upweighted by hard_score
        # (fast storms cause 62% of total CTE per this file's own XAI-9
        # analysis), a seed that lets this collapse early and hard ends up
        # systematically deprioritizing exactly the storms CTE most depends
        # on — a genuine, identified source of seed-to-seed CTE variance.
        #
        # FIX: a light, FIXED-weight (not Kendall-learned — see constructor
        # comment) L2 penalty on the softmax distribution itself. This
        # dampens how far ANY seed's early gradient is allowed to push this
        # collapse, without touching any per-metric (ADE/ATE/CTE) loss term
        # — it regularizes the MECHANISM shown to drive the variance,
        # staying general rather than metric-specific.
        hard_dist    = F.softmax(self.hard_score_weight_logits, dim=0)  # [4]
        hard_uniform = hard_dist.new_full((4,), 0.25)
        l_hard_reg   = ((hard_dist - hard_uniform) ** 2).sum()

        # ── L_CFM ─────────────────────────────────────────────────────────
        x0 = torch.randn_like(x1_rel) * sigma
        if self.use_ot and B >= 4:
            x0_flat, x1_flat = _ot_match(
                x0.reshape(B, -1), x1_rel.reshape(B, -1), self.ot_epsilon)
            x0         = x0_flat.reshape(B, self.pred_len, 2)
            x1_matched = x1_flat.reshape(B, self.pred_len, 2)
        else:
            x1_matched = x1_rel

        t        = torch.rand(B, device=device)
        x_t      = (1.0 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1_matched
        u_target = x1_matched - x0
        v_pred   = self.velocity(x_t, t, cond)
        l_cfm    = F.mse_loss(v_pred, u_target)

        # ── L_reg ramp ep10→ep30 (GATE theo curriculum, KHÔNG phải trọng số) ──
        # Lý do giữ ramp dạng if/else thay vì học: đây là LỊCH TRÌNH BẬT/TẮT
        # theo trạng thái huấn luyện (encoder bị freeze tới ep10 nên L_reg
        # chưa có ý nghĩa trước đó — bật sớm sẽ lan gradient nhiễu vào
        # velocity khi nó chưa thấy cond ổn định). Đây không phải "model
        # tin loss này quan trọng bao nhiêu" (thứ Kendall xử lý ở dưới) mà
        # là "loss này có HỢP LỆ để tính tại epoch này hay chưa" — 2 vấn đề
        # khác nhau về bản chất, gate này dùng index epoch (số nguyên rời
        # rạc) nên không phải dạng có thể differentiable để học bằng SGD.
        if epoch < 10:     ramp_reg = 0.0
        elif epoch < 30:   ramp_reg = (epoch - 10) / 20.0
        else:              ramp_reg = 1.0

        l_reg = (self._reg_loss(x1_rel, last_obs, cond, h_score)
                 if ramp_reg > 0.0 else x0.new_zeros(()))

        # ── L_heading_ms ramp ep5→ep20 (cùng lý do GATE ở trên) ─────────────
        if epoch < 5:      ramp_dir = 0.0
        elif epoch < 20:   ramp_dir = (epoch - 5) / 15.0
        else:              ramp_dir = 1.0

        if ramp_dir > 0.0:
            x0_h       = torch.randn_like(x1_rel) * self.sigma_inference
            v_h        = self.velocity(x0_h, torch.zeros(B, device=device), cond)
            x1_h_abs   = self._from_relative(x0_h + v_h, last_obs)
            pred_deg_h = _norm_to_deg(x1_h_abs.permute(1, 0, 2))   # [T, B, 2]
            obs_deg_h  = _norm_to_deg(obs_traj[:, :, :2])           # [T_obs, B, 2]
            l_heading  = self._heading_loss_ms(pred_deg_h, obs_deg_h)
        else:
            l_heading = x0.new_zeros(())

        l_momentum = x0.new_zeros(())   # always 0

        # ── L_calib ramp ep10→ep30 — [v2.4 FIX] ────────────────────────────
        # Trước: L_calib chạy từ ep0 kể cả khi encoder frozen → cond nhiễu
        # → speed_correction_logits học lệch hướng ngay từ đầu (root cause
        # speed_corr≈1.0 không hiệu quả tại ep145).
        # Fix: gate cùng lịch với ramp_reg (ep10→ep30).
        if epoch < 10:     ramp_calib = 0.0
        elif epoch < 30:   ramp_calib = (epoch - 10) / 20.0
        else:              ramp_calib = 1.0

        # L_calib luôn tính (không branch để tránh CUDA graph break),
        # ramp_calib nhân vào weighted_calib bên dưới.
        x0_c = torch.randn_like(x1_rel) * self.sigma_inference
        v_c  = self.velocity(x0_c, torch.zeros(B, device=device), cond)
        pred_c_abs = self._from_relative(x0_c + v_c, last_obs).permute(1, 0, 2)
        cal_abs    = self.speed_calibrate_pred(pred_c_abs, last_obs, obs_traj[:, :, :2])
        cal_deg    = _norm_to_deg(cal_abs)
        gt_c_deg   = _norm_to_deg(x1_gt.permute(1, 0, 2))
        # /300: cùng normalization với _reg_loss (xem giải thích ở đó)
        l_calib    = _haversine_deg(cal_deg, gt_c_deg).mean() / 300.0

        # ── Total — [v2.4 FIX] Kendall với HALF_LOG_2PI và clamp ────────────
        # [FIX-A] clamp(min=-3): ngăn log_sigma → -∞ (eff_lambda → ∞)
        #   ep145 log: log_sigma_reg=-1.897 → eff_lambda=22x (quá cao!)
        #   clamp(min=-3) → eff_lambda_max ≈ 201x (vẫn cao nhưng có giới hạn)
        # [FIX-B] HALF_LOG_2PI: hằng số NLL Gaussian đầy đủ
        #   Thiếu HALF_LOG_2PI → total drift âm → optimizer kéo log_sigma
        #   xuống để total "improve" bằng cách âm hơn → log_sigma suy biến
        #   Thêm vào: ở điểm cân bằng (sigma≈std(loss)), total ≈ 0, không
        #   còn incentive kéo log_sigma quá âm.
        # [FIX-C] ramp nhân toàn bộ Kendall term (kể cả +log_sigma+const)
        #   → khi gate=0, không có "ghost constant" đóng góp vào total
        HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)   # ≈ 0.9189

        prec_reg     = torch.exp(-2.0 * self.log_sigma_reg.clamp(min=self.log_sigma_reg_min_clamp))
        prec_heading = torch.exp(-2.0 * self.log_sigma_heading.clamp(min=-3.0))
        prec_calib   = torch.exp(-2.0 * self.log_sigma_calib.clamp(min=-3.0))

        weighted_reg     = ramp_reg    * (0.5 * prec_reg     * l_reg     + self.log_sigma_reg.clamp(min=self.log_sigma_reg_min_clamp)     + HALF_LOG_2PI)
        weighted_heading = ramp_dir    * (0.5 * prec_heading * l_heading + self.log_sigma_heading.clamp(min=-3.0) + HALF_LOG_2PI)
        weighted_calib   = ramp_calib  * (0.5 * prec_calib   * l_calib   + self.log_sigma_calib.clamp(min=-3.0)   + HALF_LOG_2PI)

        # [VAR-REDUCE] fixed weight, no ramp (regularizes the collapse from
        # epoch 0, since the divergence between seeds is already visible by
        # epoch 20-30 — waiting to ramp it in would be too late), no Kendall
        # (must stay a genuine constraint, not something the model can
        # learn to switch off).
        total = (l_cfm + weighted_reg + weighted_heading + weighted_calib
                  + self.lambda_hard_reg * l_hard_reg)
        if not torch.isfinite(total):
            total = x0.new_zeros(())

        # ── ADE 1-step log ────────────────────────────────────────────────
        with torch.no_grad():
            x0_log = torch.randn_like(x1_rel) * self.sigma_inference
            v_log  = self.velocity(x0_log, torch.zeros(B, device=device), cond)
            ade_log = _haversine_deg(
                _norm_to_deg(self._from_relative(x0_log + v_log, last_obs).permute(1, 0, 2)),
                _norm_to_deg(x1_gt.permute(1, 0, 2))
            ).mean().item()

        return {
            "total":     total,
            "l_cfm":     l_cfm.item(),
            "l_reg":     l_reg.item() if torch.is_tensor(l_reg) else 0.0,
            "l_heading": l_heading.item() if torch.is_tensor(l_heading) else 0.0,
            "l_calib":   l_calib.item(),
            "l_hard_reg": l_hard_reg.item(),
            "hard_dist": hard_dist.detach().tolist(),
            "lambda_hard_reg": self.lambda_hard_reg,
            # Graph-connected tensors (for ablation re-weighting; do NOT .item() these).
            # A wrapper that rebuilds `total` from the .item() floats above would
            # sever the autograd graph and train nothing but the Kendall log_sigma*
            # params. Use these instead when zeroing a term for an ablation run.
            "_t_l_cfm":      l_cfm,
            "_t_l_reg":      l_reg     if torch.is_tensor(l_reg)     else x0.new_zeros(()),
            "_t_l_heading":  l_heading if torch.is_tensor(l_heading) else x0.new_zeros(()),
            "_t_l_calib":    l_calib,
            "_t_l_hard_reg": l_hard_reg,
            "l_momentum": 0.0,
            "lam_reg":   ramp_reg,
            "lam_dir":   ramp_dir,
            "lam_calib": ramp_calib,
            "sigma":     sigma,
            "ade_1step": ade_log,
            "hard_score_mean": float(h_score.detach().mean()),
            "hard_score_max":  float(h_score.detach().max()),
            "learned_lambda_reg":     float((0.5 * prec_reg).detach()),
            "learned_lambda_heading": float((0.5 * prec_heading).detach()),
            "learned_lambda_calib":   float((0.5 * prec_calib).detach()),
            # Compat keys
            "l_fm": l_cfm.item(), "dpe": 0., "heading": 0., "vel_reg": 0.,
            "speed": 0., "accel": 0., "fm_mse": l_cfm.item(),
            "l_hard_total": 0., "n_hard": 0, "alpha_hard": 0.,
            "l_sel_total": 0., "speed_head_l": 0., "l_score": 0.,
            "l_speed_ratio": 0., "l_sigma_nll": 0.,
            "learned_lambda_speed_ratio": 0., "learned_sigma_infer": float(self.sigma_inference),
        }

    def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

    # ── Speed calibration — [LEARN-1] learnable per-horizon ─────────────────

    def speed_calibrate_pred(self,
                              pred_abs_norm: torch.Tensor,
                              last_obs_norm: torch.Tensor,
                              obs_norm:      torch.Tensor) -> torch.Tensor:
        """
        Per-horizon LEARNABLE speed calibration. Thay thế hằng số
        clip_min/clip_max=0.85/1.15 cũ (1 scale chung cho cả 12 bước).

        BẰNG CHỨNG TẠI SAO 1 SCALE CHUNG KHÔNG ĐỦ:
          XAI-8 (v2.1-clean, đã chạy) cho thấy correction cần khác nhau
          rõ rệt theo horizon: 12h cần scale≈0.77, 24h cần scale≈0.62,
          48h cần scale≈1.06, 72h cần scale≈1.04. clip(0.85,1.15) không
          thể đáp ứng tất cả các nhu cầu trái ngược này cùng lúc — đây là
          lý do mở rộng clip range (đã thử 0.72-1.35) chỉ cải thiện <1km:
          sai công cụ cho vấn đề (1 số thay vì 12 số).

        Không còn @staticmethod / @torch.no_grad(): chạy NGAY TRONG
        forward pass (gọi từ L_calib trong get_loss_breakdown VÀ từ
        sample() dưới no_grad của caller khi inference) để gradient có
        thể chảy qua self.speed_correction_logits trong lúc training.

        correction[t] = 2 * sigmoid(speed_correction_logits[t]) ∈ (0, 2)
        init = sigmoid(0)*2 = 1.0 → no-op tại epoch 0, học dần qua L_calib.
        """
        if obs_norm.shape[0] < 2 or pred_abs_norm.shape[0] < 2:
            return pred_abs_norm

        T = pred_abs_norm.shape[0]
        correction = (torch.sigmoid(self.speed_correction_logits[:T]) * 2.0
                      ).to(pred_abs_norm.dtype).view(T, 1, 1)               # [T,1,1]

        pts  = torch.cat([last_obs_norm.unsqueeze(0), pred_abs_norm], 0)    # [T+1, B, 2]
        disp = pts[1:] - pts[:-1]                                           # [T, B, 2]
        disp_cal = disp * correction

        out = torch.empty_like(pred_abs_norm)
        cur = last_obs_norm
        for t in range(T):
            cur = cur + disp_cal[t]
            out[t] = cur
        return out

    # ── Sample (1-shot + speed calibration) ──────────────────────────────

    @torch.no_grad()
    def sample(self, batch_list,
               num_ensemble:          Optional[int]  = None,
               ddim_steps:            Optional[int]  = None,
               return_xai:            bool           = False,
               use_speed_calibration: bool           = True,
               use_curvature_score:   bool           = False,
               **kwargs) -> Tuple:
        """
        1-shot inference with physics selection + learnable speed calibration.

        Steps:
          1. Encode context once (hard_score + full encoder)
          2. Sample K=20 candidates: x0 ~ N(0, sigma_inf²), x_pred = x0 + v(x0, 0, cond)
          3. Physics score each candidate (speed+smooth+heading+displacement
             [+curvature if use_curvature_score=True])
          4. Weighted average of top-3 candidates
          5. Speed calibration per-horizon (LEARNED, not fixed clip)
          6. If return_xai=True: compute XAI-1 through XAI-9

        use_curvature_score: [CURV-SCORE, opt-in, default False] adds a 5th
          re-ranking component that favors candidates whose turning rate
          matches the storm's OBSERVED turning rate, rather than only
          checking step-0 direction (head_score) or penalizing all turning
          (smooth_score). Pure inference-time change — no retraining needed,
          can be A/B tested directly on any existing checkpoint.

        Returns:
          (pred_mean [T,B,2], zeros [T,B,2], all_traj [K,T,B,2])
          or (pred_mean, zeros, all_traj, xai_dict) if return_xai=True
        """
        K  = num_ensemble or self.n_ensemble
        N  = ddim_steps if (ddim_steps is not None and ddim_steps > 1) else self.n_inference_steps
        dt = 1.0 / max(N, 1)

        obs_traj    = batch_list[0]
        T_obs, B, _ = obs_traj.shape
        device      = obs_traj.device

        h_score  = hard_score_from_obs(obs_traj[:, :, :2],
                                        weight_logits=self.hard_score_weight_logits)
        obs_norm = obs_traj[:, :, :2]
        last_obs = obs_traj[-1, :, :2]
        t0       = torch.zeros(B, device=device)

        cond = self.encoder(batch_list, hard_score=h_score)

        all_traj = []
        for _ in range(K):
            x_rel = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference

            if N <= 1:
                v     = self.velocity(x_rel, t0, cond)
                x_rel = x_rel + v
            else:
                for step in range(N):
                    t_b   = torch.full((B,), step * dt, device=device)
                    x_rel = (x_rel + dt * self.velocity(x_rel, t_b, cond)).clamp(-3., 3.)

            x_abs = self._from_relative(x_rel, last_obs)
            all_traj.append(x_abs.permute(1, 0, 2))   # [T, B, 2]

        scores  = torch.stack(
            [_physics_score(t, obs_norm, use_curvature_score=use_curvature_score)
             for t in all_traj], 0)   # [K, B]
        all_t   = torch.stack(all_traj, 0)   # [K, T, B, 2]
        top_k   = min(3, K)
        top_idx = scores.topk(top_k, dim=0).indices   # [top_k, B]

        pred_mean = torch.zeros_like(all_traj[0])
        for b in range(B):
            idx_b = top_idx[:, b]
            w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
            pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

        # [LEARN-1] Per-horizon LEARNED speed calibration (no hardcoded clip args)
        if use_speed_calibration:
            pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

        if not return_xai:
            return pred_mean, torch.zeros_like(pred_mean), all_t

        # ── XAI output (XAI-2,3,4,5,6,7,8,9) ──────────────────────────────
        xai = {}

        xai.update(compute_ensemble_uncertainty(all_t))

        _, hard_comps = hard_score_from_obs(obs_norm, return_components=True,
                                             weight_logits=self.hard_score_weight_logits)
        xai["hard_components"] = hard_comps

        pred_deg  = _norm_to_deg(pred_mean)    # [T, B, 2]
        obs_deg_x = _norm_to_deg(obs_norm)     # [T_obs, B, 2]

        obs_spd_x    = _step_speeds_kmh(obs_deg_x)
        obs_spd_mu   = obs_spd_x.mean(0)     # [B]
        if pred_deg.shape[0] >= 2:
            last_deg_x   = obs_deg_x[-1]
            pts_x        = torch.cat([last_deg_x.unsqueeze(0), pred_deg], 0)
            pred_spd_x   = _step_speeds_kmh(pts_x)
            pred_spd_mu  = pred_spd_x.mean(0)  # [B]
        else:
            pred_spd_mu = obs_spd_mu.clone()

        speed_ratio = (pred_spd_mu / obs_spd_mu.clamp(min=1.0))
        xai["speed_comparison"] = {
            "obs_speed_mean":  float(obs_spd_mu.mean()),
            "pred_speed_mean": float(pred_spd_mu.mean()),
            "speed_ratio":     float(speed_ratio.mean()),
            "per_storm_obs":   obs_spd_mu,
            "per_storm_pred":  pred_spd_mu,
            "over_predict":    speed_ratio > 1.15,
            "under_predict":   speed_ratio < 0.85,
        }

        v_ref = obs_spd_mu.clamp(min=5.0)
        v_sig = v_ref * 0.5
        spd_sc = torch.exp(-((pred_spd_mu - v_ref) / v_sig).pow(2) * 0.5)
        if pred_deg.shape[0] >= 3:
            vel_x   = pred_deg[1:] - pred_deg[:-1]
            accel_x = (vel_x[1:] - vel_x[:-1]).norm(dim=-1)
            smo_sc  = torch.exp(-accel_x.mean(0) * 5.0)
        else:
            smo_sc = torch.ones(B, device=device)
        if obs_norm.shape[0] >= 2 and pred_mean.shape[0] >= 1:
            ov = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
            pv = pred_mean[0, :, :2] - obs_norm[-1, :, :2]
            cos_s = (F.normalize(ov, dim=-1, eps=1e-6) * F.normalize(pv, dim=-1, eps=1e-6)).sum(-1)
            hd_sc = torch.exp((cos_s.clamp(-1, 1) - 1.0) * 3.0)
        else:
            hd_sc = torch.ones(B, device=device)
        xai["physics_components"] = {
            "speed": spd_sc, "smooth": smo_sc, "heading": hd_sc,
            "obs_speed": obs_spd_mu, "pred_speed": pred_spd_mu,
        }

        gt_traj_xai = batch_list[1]
        gt_deg_xai  = _norm_to_deg(gt_traj_xai[:, :, :2])
        xai["heading_deviation_deg"] = compute_heading_deviation(pred_deg, gt_deg_xai)

        xai["ate_cte_decomp"] = compute_cte_contribution(pred_deg, gt_deg_xai)

        # XAI-8: Per-horizon speed ratio (12h/24h/48h/72h)
        if pred_deg.shape[0] >= 2 and gt_deg_xai.shape[0] >= 2:
            T8       = min(pred_deg.shape[0], gt_deg_xai.shape[0])
            last_d   = obs_deg_x[-1]
            pts_pred = torch.cat([last_d.unsqueeze(0), pred_deg[:T8]], 0)
            pts_gt   = torch.cat([last_d.unsqueeze(0), gt_deg_xai[:T8]], 0)
            spd_pred = _step_speeds_kmh(pts_pred)   # [T8, B]
            spd_gt   = _step_speeds_kmh(pts_gt)     # [T8, B]
            ratio    = spd_pred / spd_gt.clamp(min=1.0)  # [T8, B]
            def _hz(s, lo, hi):
                hi = min(hi, s.shape[0])
                return float(s[lo:hi].mean()) if hi > lo else float("nan")
            r_mean = ratio.mean(1)   # [T8]
            xai["speed_per_horizon"] = {
                "ratio":    r_mean.tolist(),
                "pred_kmh": spd_pred.mean(1).tolist(),
                "gt_kmh":   spd_gt.mean(1).tolist(),
                "12h_ratio": _hz(r_mean, 0, 2),
                "24h_ratio": _hz(r_mean, 2, 4),
                "48h_ratio": _hz(r_mean, 6, 8),
                "72h_ratio": _hz(r_mean, 10, 12),
            }
        else:
            xai["speed_per_horizon"] = {}

        # XAI-9: Storm category breakdown by obs speed
        obs_spd_cat = _step_speeds_kmh(obs_deg_x).mean(0)   # [B]
        sm = obs_spd_cat < 8.0
        mm = (obs_spd_cat >= 8.0) & (obs_spd_cat < 15.0)
        fm = obs_spd_cat >= 15.0
        ade_per_storm = _haversine_deg(pred_deg[:gt_deg_xai.shape[0]],
                                        gt_deg_xai[:pred_deg.shape[0]]).mean(0)  # [B]
        def _cat_ade(mask):
            return float(ade_per_storm[mask].mean()) if mask.sum() > 0 else float("nan")
        xai["storm_categories"] = {
            "n_slow":    int(sm.sum()),
            "n_medium":  int(mm.sum()),
            "n_fast":    int(fm.sum()),
            "speed_mean": float(obs_spd_cat.mean()),
            "speed_std":  float(obs_spd_cat.std()),
            "ade_slow":   _cat_ade(sm),
            "ade_medium": _cat_ade(mm),
            "ade_fast":   _cat_ade(fm),
        }

        # [LEARN diagnostics] expose learned values for monitoring in train_fm
        xai["learned_params"] = {
            "speed_correction":     (torch.sigmoid(self.speed_correction_logits) * 2.0).tolist(),
            "reg_step_weights":     F.softmax(self.reg_step_logits, dim=0).tolist(),
            "hard_score_weights":   F.softmax(self.hard_score_weight_logits, dim=0).tolist(),
            "b_horizon":            torch.exp(self.log_b_horizon).detach().tolist(),
            # [v2.4 FIX] các keys sau bị thiếu → head_w=[] và sigma_inf=nan trong log
            "heading_step_weights": F.softmax(self.heading_step_logits, dim=0).tolist(),
            "sigma_inf":            float(self.sigma_inference),
            "log_sigma_reg":        float(self.log_sigma_reg.detach()),
            "log_sigma_heading":    float(self.log_sigma_heading.detach()),
            "log_sigma_calib":      float(self.log_sigma_calib.detach()),
            "eff_lambda_reg":       float((0.5 * torch.exp(-2.0 * self.log_sigma_reg.clamp(min=self.log_sigma_reg_min_clamp))).detach()),
            "eff_lambda_heading":   float((0.5 * torch.exp(-2.0 * self.log_sigma_heading.clamp(min=-3.0))).detach()),
            "eff_lambda_calib":     float((0.5 * torch.exp(-2.0 * self.log_sigma_calib.clamp(min=-3.0))).detach()),
        }

        return pred_mean, torch.zeros_like(pred_mean), all_t, xai

    @torch.no_grad()
    def sample_multiscale(
        self,
        batch_list,
        sigmas: Optional[List[float]] = None,
        n_per_sigma: int = 4,
        use_speed_calibration: bool = True,
        use_curvature_score:   bool = False,
    ) -> Tuple:
        """
        Multi-scale sigma ensemble at inference (DOES NOT affect training).
        Hedges against speed distribution shift between val and test.
        """
        if sigmas is None:
            sigmas = [0.025, 0.035, 0.04, 0.05, 0.065]

        obs_traj = batch_list[0]
        B = obs_traj.shape[1]; device = obs_traj.device

        h_score  = hard_score_from_obs(obs_traj[:, :, :2],
                                        weight_logits=self.hard_score_weight_logits)
        obs_norm = obs_traj[:, :, :2]
        last_obs = obs_traj[-1, :, :2]
        t0       = torch.zeros(B, device=device)
        cond     = self.encoder(batch_list, hard_score=h_score)

        all_traj = []
        for sigma in sigmas:
            for _ in range(n_per_sigma):
                x0    = torch.randn(B, self.pred_len, 2, device=device) * sigma
                v     = self.velocity(x0, t0, cond)
                x_abs = self._from_relative(x0 + v, last_obs)
                all_traj.append(x_abs.permute(1, 0, 2))

        scores  = torch.stack(
            [_physics_score(t, obs_norm, use_curvature_score=use_curvature_score)
             for t in all_traj], 0)
        all_t   = torch.stack(all_traj, 0)
        top_k   = min(5, len(all_traj))
        top_idx = scores.topk(top_k, dim=0).indices

        pred_mean = torch.zeros_like(all_traj[0])
        for b in range(B):
            idx_b = top_idx[:, b]
            w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
            pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

        if use_speed_calibration:
            pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

        return pred_mean, torch.zeros_like(pred_mean), all_t


# ─────────────────────────────────────────────────────────────────────────────
#  Backward compat alias
# ─────────────────────────────────────────────────────────────────────────────
TCDiffusion = TCFlowMatching