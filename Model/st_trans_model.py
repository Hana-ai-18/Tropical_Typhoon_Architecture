"""
Model/st_trans_model.py  ── ST-Trans Baseline
===============================================
THUẬT TOÁN: Faiaz et al. (2026)
"Physics-guided non-autoregressive transformer for lightweight
cyclone track prediction in the Bay of Bengal"
Expert Systems With Applications 317 (2026) 131972

THAY ĐỔI SO VỚI PHIÊN BẢN TRƯỚC:
  ✅ Dùng cùng PaperEncoder (FNO3D + Mamba + Env_net) với paper baseline
     thay vì chỉ encode obs_traj qua CNN đơn giản.
  ✅ Thêm ATE / CTE metrics (Along-Track / Cross-Track Error)
  ✅ Interface nhất quán: forward(batch_list) thay vì forward(obs_traj)

KIẾN TRÚC MỚI:
  PaperEncoder(batch_list)   → raw_ctx [B, 512]          (context phong phú)
  obs_traj features          → obs_memory [B, S, d_model] (temporal structure)
  full_memory = cat([raw_ctx_token, obs_memory], dim=1)   [B, S+1, d_model]
  Transformer decoder (learned horizon queries) → [B, H, d_model]
  Regression head  → [H, B, 2]

LOSS (§3.5.1 - Physics-guided composite):
  L = L_DPE + 0.05*L_MSE + λ_speed*L_speed + λ_accel*L_accel
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import shared encoder và metric helpers ──────────────────────────────────
from Model.paper_baseline_model import (
    PaperEncoder,
    _norm_to_deg,
    _ate_cte_tensors,
    haversine_km,
    compute_ade_per_horizon,
    compute_ate_cte_per_horizon,
    compute_full_metrics,
    HORIZON_STEPS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Sinusoidal Positional Encoding
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 300):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ══════════════════════════════════════════════════════════════════════════════
#  Lightweight obs-traj encoder  (thay CNNStateEncoder cũ)
#  8D kinematic features → [B, S, d_model] sequence memory
# ══════════════════════════════════════════════════════════════════════════════

class ObsKinematicEncoder(nn.Module):
    """
    Encode obs_traj [T_obs, B, 2] → sequence memory [B, T_obs, d_model].

    Tính 8 kinematic features (position, velocity, acceleration, speed, step-idx)
    rồi project qua MLP → transformer encoder để capture temporal dependencies.
    """

    FEAT_DIM = 8

    def __init__(self, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 1, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        # 8 → d_model projection
        self.proj = nn.Sequential(
            nn.Linear(self.FEAT_DIM, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.pe = SinusoidalPE(d_model, max_len=64)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            activation="relu", batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    @staticmethod
    def _extract_features(obs_traj: torch.Tensor) -> torch.Tensor:
        """obs_traj [T, B, 2] → features [B, T, 8]."""
        T, B, _ = obs_traj.shape
        device  = obs_traj.device

        lon = obs_traj[:, :, 0]
        lat = obs_traj[:, :, 1]

        if T >= 2:
            d_lon = torch.cat([obs_traj[1:, :, 0] - obs_traj[:-1, :, 0],
                               torch.zeros(1, B, device=device)], dim=0)
            d_lat = torch.cat([obs_traj[1:, :, 1] - obs_traj[:-1, :, 1],
                               torch.zeros(1, B, device=device)], dim=0)
        else:
            d_lon = torch.zeros(T, B, device=device)
            d_lat = torch.zeros(T, B, device=device)

        if T >= 3:
            dd_lon = torch.cat([d_lon[1:] - d_lon[:-1],
                                torch.zeros(1, B, device=device)], dim=0)
            dd_lat = torch.cat([d_lat[1:] - d_lat[:-1],
                                torch.zeros(1, B, device=device)], dim=0)
        else:
            dd_lon = torch.zeros(T, B, device=device)
            dd_lat = torch.zeros(T, B, device=device)

        step_idx = torch.linspace(0, 1, T, device=device).unsqueeze(1).expand(T, B)
        speed    = (d_lon.pow(2) + d_lat.pow(2)).sqrt()

        feat = torch.stack([lon, lat, d_lon, d_lat,
                            dd_lon, dd_lat, step_idx, speed], dim=-1)
        return feat.permute(1, 0, 2)   # [B, T, 8]

    def forward(self, obs_traj: torch.Tensor) -> torch.Tensor:
        """→ [B, T_obs, d_model]"""
        feat = self._extract_features(obs_traj)   # [B, T, 8]
        h    = self.proj(feat)                     # [B, T, d_model]
        h    = self.pe(h)
        return self.enc(h)                         # [B, T, d_model]


# ══════════════════════════════════════════════════════════════════════════════
#  ST-Trans Main Model
# ══════════════════════════════════════════════════════════════════════════════

class STTrans(nn.Module):
    """
    Physics-guided Non-Autoregressive Transformer for TC track prediction.

    Kiến trúc (sau khi sửa để dùng cùng encoder với PaperBaseline):
      1. PaperEncoder(batch_list)  → raw_ctx [B, 512]
      2. Project raw_ctx → ctx_token [B, 1, d_model]  (global context token)
      3. ObsKinematicEncoder(obs_traj) → obs_memory [B, S, d_model]
      4. full_memory = cat([ctx_token, obs_memory], dim=1)  [B, S+1, d_model]
      5. Learned horizon queries [B, H, d_model] + dec_pe
      6. Transformer decoder (cross-attention) → [B, H, d_model]
      7. Regression head → [H, B, 2]

    Loss: Physics-guided composite (§3.5.1)
      L = L_DPE + 0.05*L_MSE + 0.1*L_speed + 0.01*L_accel
    """

    def __init__(
        self,
        obs_len:        int   = 8,
        pred_len:       int   = 12,
        unet_in_ch:     int   = 13,
        d_model:        int   = 64,
        nhead:          int   = 4,
        num_enc_layers: int   = 1,
        num_dec_layers: int   = 3,
        dim_ff:         int   = 512,
        dropout:        float = 0.1,
        # Physics loss weights
        lambda_speed:   float = 0.1,
        lambda_accel:   float = 0.01,
        w_mse:          float = 0.05,
        v_max_kmh:      float = 80.0,
        dt_h:           float = 6.0,
    ):
        super().__init__()
        self.obs_len      = obs_len
        self.pred_len     = pred_len
        self.d_model      = d_model
        self.lambda_speed = lambda_speed
        self.lambda_accel = lambda_accel
        self.w_mse        = w_mse
        # v_max trong normalised units
        self.v_max_norm   = v_max_kmh * dt_h / (111.0 * 50.0)

        # ── Shared encoder (cùng với PaperBaseline) ───────────────────────
        self.encoder = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)

        # Project raw_ctx [B, 512] → context token [B, 1, d_model]
        self.ctx_proj = nn.Sequential(
            nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Obs trajectory kinematic encoder → temporal memory ────────────
        self.obs_enc = ObsKinematicEncoder(
            d_model=d_model, nhead=nhead,
            num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout,
        )

        # ── Learned horizon queries (paper §3.3.4) ────────────────────────
        self.horizon_queries = nn.Parameter(
            torch.randn(1, pred_len, d_model) * 0.02
        )
        self.dec_pe = SinusoidalPE(d_model, max_len=pred_len + 10)

        # ── Transformer decoder ───────────────────────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            activation="relu", batch_first=True,
        )
        self.transformer_dec = nn.TransformerDecoder(
            dec_layer, num_layers=num_dec_layers)

        # ── Regression head (paper §3.3.4) ───────────────────────────────
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, batch_list) -> torch.Tensor:
        """
        batch_list: cùng format với PaperBaseline (full batch với ảnh, env, ...)
        → pred [pred_len, B, 2] normalised
        """
        obs_traj = batch_list[0]     # [T_obs, B, 2]
        B        = obs_traj.shape[1]

        # 1. Rich context từ full encoder
        raw_ctx    = self.encoder(batch_list)          # [B, 512]
        ctx_token  = self.ctx_proj(raw_ctx).unsqueeze(1)  # [B, 1, d_model]

        # 2. Temporal obs memory
        obs_memory = self.obs_enc(obs_traj)            # [B, S, d_model]

        # 3. Kết hợp context token + temporal memory
        full_memory = torch.cat([ctx_token, obs_memory], dim=1)  # [B, S+1, d_model]

        # 4. Horizon queries
        Q = self.horizon_queries.expand(B, -1, -1)    # [B, H, d_model]
        Q = self.dec_pe(Q)

        # 5. Non-autoregressive decoder
        D   = self.transformer_dec(Q, full_memory)     # [B, H, d_model]
        out = self.reg_head(D)                         # [B, H, 2]

        return out.permute(1, 0, 2)                    # [pred_len, B, 2]

    # ── Physics-guided loss (§3.5.1) ─────────────────────────────────────

    def physics_loss(
        self,
        pred_norm: torch.Tensor,
        gt_norm:   torch.Tensor,
    ) -> Dict:
        T    = min(pred_norm.shape[0], gt_norm.shape[0])
        pred = pred_norm[:T]
        gt   = gt_norm[:T]

        pred_deg = _norm_to_deg(pred)
        gt_deg   = _norm_to_deg(gt)

        # L_DPE: mean great-circle distance (eq. 16)
        l_dpe = haversine_km(pred_deg, gt_deg).mean()

        # L_MSE: coordinate MSE (eq. 17)
        l_mse = F.mse_loss(pred, gt)

        # L_speed: penalise speeds > v_max (eq. 18-20)
        if T >= 2:
            step_dist = (pred[1:] - pred[:-1]).norm(dim=-1)      # [T-1, B]
            l_speed   = F.relu(step_dist - self.v_max_norm).pow(2).mean()
        else:
            l_speed = pred_norm.new_zeros(())

        # L_accel: penalise acceleration changes (eq. 21-22)
        if T >= 3:
            vel     = pred[1:] - pred[:-1]                        # [T-1, B, 2]
            l_accel = (vel[1:].norm(dim=-1) - vel[:-1].norm(dim=-1)).pow(2).mean()
        else:
            l_accel = pred_norm.new_zeros(())

        total = (l_dpe
                 + self.w_mse        * l_mse
                 + self.lambda_speed * l_speed
                 + self.lambda_accel * l_accel)

        return dict(
            total=total,
            dpe=l_dpe.item(),
            mse=l_mse.item(),
            speed=l_speed.item(),
            accel=l_accel.item(),
        )

    # ── Training / inference interface (nhất quán với PaperBaseline) ─────

    def get_loss(self, batch_list) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list)["total"]

    def get_loss_breakdown(self, batch_list) -> Dict:
        traj_gt = batch_list[1]
        pred    = self.forward(batch_list)
        bd      = self.physics_loss(pred, traj_gt)

        with torch.no_grad():
            ade_m = compute_ade_per_horizon(pred.detach(), traj_gt)
            atc_m = compute_ate_cte_per_horizon(pred.detach(), traj_gt)

        bd.update(ade_m)
        bd.update(atc_m)
        return bd

    @torch.no_grad()
    def sample(
        self,
        batch_list,
        num_ensemble: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred     = self.forward(batch_list)
        T, B, _  = pred.shape
        me_mean  = torch.zeros(T, B, 2, device=pred.device)
        return pred, me_mean, pred.unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
#  ST-Trans-AR  ── Autoregressive variant (paper §3.4)
# ══════════════════════════════════════════════════════════════════════════════

class STTransAR(nn.Module):
    """
    Autoregressive ST-Trans.
    Cùng encoder backbone với STTrans và PaperBaseline, decoder AR-GRU.
    Dùng để ablation so sánh với non-AR.
    """

    def __init__(
        self,
        obs_len:        int   = 8,
        pred_len:       int   = 12,
        unet_in_ch:     int   = 13,
        d_model:        int   = 64,
        nhead:          int   = 4,
        num_enc_layers: int   = 1,
        dim_ff:         int   = 512,
        dropout:        float = 0.1,
        lambda_speed:   float = 0.1,
        lambda_accel:   float = 0.01,
        w_mse:          float = 0.05,
        v_max_kmh:      float = 80.0,
        dt_h:           float = 6.0,
    ):
        super().__init__()
        self.obs_len      = obs_len
        self.pred_len     = pred_len
        self.d_model      = d_model
        self.lambda_speed = lambda_speed
        self.lambda_accel = lambda_accel
        self.w_mse        = w_mse
        self.v_max_norm   = v_max_kmh * dt_h / (111.0 * 50.0)

        # ── Shared encoder ────────────────────────────────────────────────
        self.encoder  = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
        self.ctx_proj = nn.Sequential(
            nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model),
            nn.LayerNorm(d_model),
        )

        self.obs_enc = ObsKinematicEncoder(
            d_model=d_model, nhead=nhead,
            num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout,
        )

        # ── AR-GRU decoder ────────────────────────────────────────────────
        # Input: cur_pos(2) + pooled_memory(d_model)
        self.ar_gru   = nn.GRUCell(2 + d_model, d_model)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        batch_list,
        gt_traj:         Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
    ) -> torch.Tensor:
        obs_traj = batch_list[0]
        B        = obs_traj.shape[1]

        raw_ctx    = self.encoder(batch_list)               # [B, 512]
        ctx_token  = self.ctx_proj(raw_ctx).unsqueeze(1)    # [B, 1, d_model]
        obs_memory = self.obs_enc(obs_traj)                 # [B, S, d_model]
        full_mem   = torch.cat([ctx_token, obs_memory], dim=1)  # [B, S+1, d_model]

        # Pooled context for AR decoder
        ctx = full_mem.mean(dim=1)     # [B, d_model]
        cur_pos = obs_traj[-1].clone()
        hx      = ctx
        preds   = []

        for i in range(self.pred_len):
            inp = torch.cat([cur_pos, ctx], dim=-1)
            hx  = self.ar_gru(inp, hx)
            out = self.reg_head(hx)
            preds.append(out)
            if teacher_forcing and gt_traj is not None and i < gt_traj.shape[0]:
                cur_pos = gt_traj[i]
            else:
                cur_pos = out.detach()

        return torch.stack(preds, dim=0)   # [pred_len, B, 2]

    def _physics_loss(self, pred_norm, gt_norm):
        T    = min(pred_norm.shape[0], gt_norm.shape[0])
        pred = pred_norm[:T]; gt = gt_norm[:T]
        pred_deg = _norm_to_deg(pred); gt_deg = _norm_to_deg(gt)
        l_dpe    = haversine_km(pred_deg, gt_deg).mean()
        l_mse    = F.mse_loss(pred, gt)
        if T >= 2:
            step_dist = (pred[1:] - pred[:-1]).norm(dim=-1)
            l_speed   = F.relu(step_dist - self.v_max_norm).pow(2).mean()
        else:
            l_speed   = pred_norm.new_zeros(())
        if T >= 3:
            vel     = pred[1:] - pred[:-1]
            l_accel = (vel[1:].norm(dim=-1) - vel[:-1].norm(dim=-1)).pow(2).mean()
        else:
            l_accel = pred_norm.new_zeros(())
        total = (l_dpe + self.w_mse * l_mse
                 + self.lambda_speed * l_speed + self.lambda_accel * l_accel)
        return dict(total=total, dpe=l_dpe.item(), mse=l_mse.item(),
                    speed=l_speed.item(), accel=l_accel.item())

    def get_loss(self, batch_list) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list)["total"]

    def get_loss_breakdown(self, batch_list) -> Dict:
        traj_gt = batch_list[1]
        pred    = self.forward(batch_list, traj_gt, teacher_forcing=True)
        bd      = self._physics_loss(pred, traj_gt)
        with torch.no_grad():
            bd.update(compute_ade_per_horizon(pred.detach(), traj_gt))
            bd.update(compute_ate_cte_per_horizon(pred.detach(), traj_gt))
        return bd

    @torch.no_grad()
    def sample(self, batch_list, **kwargs):
        pred    = self.forward(batch_list, teacher_forcing=False)
        T, B, _ = pred.shape
        me_mean = torch.zeros(T, B, 2, device=pred.device)
        return pred, me_mean, pred.unsqueeze(0)