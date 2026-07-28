"""
Model/paper_baseline_model.py  ── PAPER BASELINE
==================================================
THUẬT TOÁN GỐC: Rahman et al. (2025)
"Tropical cyclone track prediction harnessing deep learning algorithms"
Results in Engineering 26, 105009

CHIẾN LƯỢC:
  - Giữ nguyên encoder pipeline của bạn (FNO3D + Mamba + Env_net → raw_ctx [B, 512])
  - Thay toàn bộ prediction head bằng paper algorithm:
      * LSTM_Model  : LSTMCell stepwise (Algorithm 3)
      * GRU_Model   : GRUCell  stepwise với dropout=0.2 (Algorithm 4)
      * RNN_Model   : RNNCell  stepwise (Algorithm 2)
  - Loss: MSE (paper §2.7: "Mean Squared Error loss function")
  - Optimizer: Adam lr=0.001 (paper Table 2)
  - Epochs: 7000 (paper Table 2) → dùng early stopping trong train script
  - LR Scheduler: ReduceLROnPlateau (paper §2.8)

METRICS:
  - ADE / FDE / 12h / 24h / 48h / 72h  (haversine km)
  - ATE (Along-Track Error)  ── thành phần lỗi dọc theo hướng di chuyển bão
  - CTE (Cross-Track Error)  ── thành phần lỗi vuông góc với hướng di chuyển
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


# ══════════════════════════════════════════════════════════════════════════════
#  Normalisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
    """Denormalise từ [0,1]-ish space về degrees."""
    out = arr.clone()
    out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0   # lon
    out[..., 1] = (arr[..., 1] * 50.0) / 10.0              # lat
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Haversine distance
# ══════════════════════════════════════════════════════════════════════════════

def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """
    Haversine distance [km] giữa p1, p2 dạng degrees.
    p1, p2: [..., 2] (lon, lat)
    """
    lon1 = torch.deg2rad(p1[..., 0]);  lat1 = torch.deg2rad(p1[..., 1])
    lon2 = torch.deg2rad(p2[..., 0]);  lat2 = torch.deg2rad(p2[..., 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
    return 2.0 * 6371.0 * torch.asin(a.clamp(1e-12, 1.0).sqrt())


# Horizon steps (6h per step): 12h=step1, 24h=step3, 48h=step7, 72h=step11
HORIZON_STEPS: Dict[int, int] = {12: 1, 24: 3, 48: 7, 72: 11}


# ══════════════════════════════════════════════════════════════════════════════
#  ATE / CTE helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ate_cte_tensors(
    pred_norm: torch.Tensor,   # [T, B, 2] normalised
    gt_norm:   torch.Tensor,   # [T, B, 2] normalised
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trả về ATE [T, B] và CTE [T, B] theo km (signed).

    ATE (Along-Track Error):  thành phần lỗi dọc theo hướng di chuyển GT của bão.
        ATE > 0 → dự báo đi xa hơn GT (over-shoot)
        ATE < 0 → dự báo đi ít hơn GT (under-shoot)

    CTE (Cross-Track Error):  thành phần lỗi vuông góc với hướng di chuyển.
        CTE > 0 → lệch sang trái hướng di chuyển
        CTE < 0 → lệch sang phải
    """
    T = min(pred_norm.shape[0], gt_norm.shape[0])
    pred_deg = _norm_to_deg(pred_norm[:T])   # [T, B, 2]
    gt_deg   = _norm_to_deg(gt_norm[:T])     # [T, B, 2]

    # ── Sai số vị trí → km ────────────────────────────────────────────────
    err     = pred_deg - gt_deg              # [T, B, 2] degrees
    lat_rad = torch.deg2rad(gt_deg[..., 1])  # [T, B]
    err_km  = torch.stack([
        err[..., 0] * 111.0 * torch.cos(lat_rad),   # Δlon → km
        err[..., 1] * 111.0,                          # Δlat → km
    ], dim=-1)                                # [T, B, 2]

    # ── Hướng di chuyển GT (forward difference, last step backward) ──────
    if T >= 2:
        dir_raw = torch.cat([
            gt_deg[1:] - gt_deg[:-1],
            gt_deg[-1:] - gt_deg[-2:-1],
        ], dim=0)                             # [T, B, 2]
    else:
        # Không đủ bước: giả định hướng đông
        dir_raw = torch.zeros_like(gt_deg)
        dir_raw[..., 0] = 1.0

    # Chuyển hướng sang km để đồng nhất đơn vị
    dir_km = torch.stack([
        dir_raw[..., 0] * 111.0 * torch.cos(lat_rad),
        dir_raw[..., 1] * 111.0,
    ], dim=-1)                                # [T, B, 2]
    dir_norm = dir_km.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dir_unit = dir_km / dir_norm              # [T, B, 2] unit vector

    # ── ATE = dot(err_km, dir_unit) ───────────────────────────────────────
    ate = (err_km * dir_unit).sum(dim=-1)     # [T, B]

    # ── CTE = 2D cross-product (signed perpendicular component) ──────────
    cte = (err_km[..., 0] * dir_unit[..., 1]
           - err_km[..., 1] * dir_unit[..., 0])  # [T, B]

    return ate, cte


def compute_ade_per_horizon(
    pred_norm: torch.Tensor,   # [T, B, 2] normalised
    gt_norm:   torch.Tensor,   # [T, B, 2] normalised
) -> Dict[str, float]:
    """
    ADE (km) tại các mốc 12h, 24h, 48h, 72h + overall ADE/FDE.
    """
    T = min(pred_norm.shape[0], gt_norm.shape[0])
    pred_deg = _norm_to_deg(pred_norm[:T])
    gt_deg   = _norm_to_deg(gt_norm[:T])

    dist = haversine_km(pred_deg, gt_deg)    # [T, B]

    result: Dict[str, float] = {}
    result["ADE"] = float(dist.mean().item())
    result["FDE"] = float(dist[-1].mean().item())

    for h_label, step_idx in HORIZON_STEPS.items():
        if step_idx < T:
            result[f"{h_label}h"] = float(dist[step_idx].mean().item())
        else:
            result[f"{h_label}h"] = float("nan")

    return result


def compute_ate_cte_per_horizon(
    pred_norm: torch.Tensor,   # [T, B, 2] normalised
    gt_norm:   torch.Tensor,   # [T, B, 2] normalised
) -> Dict[str, float]:
    """
    ATE / CTE (km) tại các mốc 12h, 24h, 48h, 72h + overall mean.

    Trả về cả signed mean và absolute mean:
      ATE_{H}h     : signed mean (bias dọc track)
      ATE_abs_{H}h : |ATE| mean  (magnitude)
      CTE_{H}h     : signed mean (bias ngang track)
      CTE_abs_{H}h : |CTE| mean  (magnitude)
    """
    T = min(pred_norm.shape[0], gt_norm.shape[0])
    ate, cte = _ate_cte_tensors(pred_norm[:T], gt_norm[:T])   # each [T, B]

    result: Dict[str, float] = {
        "ATE":     float(ate.mean().item()),
        "ATE_abs": float(ate.abs().mean().item()),
        "CTE":     float(cte.mean().item()),
        "CTE_abs": float(cte.abs().mean().item()),
    }

    for h_label, step_idx in HORIZON_STEPS.items():
        if step_idx < T:
            result[f"ATE_{h_label}h"]     = float(ate[step_idx].mean().item())
            result[f"CTE_{h_label}h"]     = float(cte[step_idx].mean().item())
            result[f"ATE_abs_{h_label}h"] = float(ate[step_idx].abs().mean().item())
            result[f"CTE_abs_{h_label}h"] = float(cte[step_idx].abs().mean().item())
        else:
            for key in [f"ATE_{h_label}h", f"CTE_{h_label}h",
                        f"ATE_abs_{h_label}h", f"CTE_abs_{h_label}h"]:
                result[key] = float("nan")

    return result


def compute_full_metrics(
    pred_norm: torch.Tensor,
    gt_norm:   torch.Tensor,
) -> Dict[str, float]:
    """Tổng hợp ADE + ATE/CTE metrics vào một dict."""
    m = compute_ade_per_horizon(pred_norm, gt_norm)
    m.update(compute_ate_cte_per_horizon(pred_norm, gt_norm))
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  Paper RNN_Model  ── Algorithm 2
# ══════════════════════════════════════════════════════════════════════════════

class PaperRNNHead(nn.Module):
    """RNN stepwise prediction (Algorithm 2 của paper)."""

    def __init__(
        self,
        input_dim:  int = 512 + 2,
        hidden_dim: int = 128,
        n_layers:   int = 3,
        pred_len:   int = 12,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.pred_len   = pred_len

        self.cells = nn.ModuleList()
        self.cells.append(nn.RNNCell(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.cells.append(nn.RNNCell(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        raw_ctx:  torch.Tensor,
        obs_traj: torch.Tensor,
        feed_forward: bool = True,
    ) -> torch.Tensor:
        B      = raw_ctx.shape[0]
        device = raw_ctx.device

        h_list = [torch.zeros(B, self.hidden_dim, device=device)
                  for _ in range(self.n_layers)]
        cur_pos = obs_traj[-1].clone()
        preds   = []

        for _ in range(self.pred_len):
            inp = torch.cat([raw_ctx, cur_pos], dim=-1)
            h_list[0] = self.cells[0](inp, h_list[0].detach())
            h = h_list[0]
            for j in range(1, self.n_layers):
                h_list[j] = self.cells[j](h, h_list[j].detach())
                h = h_list[j]
            output = self.fc(h)
            preds.append(output)
            if feed_forward:
                cur_pos = output.detach()

        return torch.stack(preds, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
#  Paper LSTM_Model  ── Algorithm 3
# ══════════════════════════════════════════════════════════════════════════════

class PaperLSTMHead(nn.Module):
    """LSTMCell stepwise prediction (Algorithm 3 của paper)."""

    def __init__(
        self,
        input_dim:  int = 512 + 2,
        hidden_dim: int = 256,
        n_layers:   int = 3,
        pred_len:   int = 12,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.pred_len   = pred_len

        self.cells = nn.ModuleList()
        self.cells.append(nn.LSTMCell(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.cells.append(nn.LSTMCell(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        raw_ctx:  torch.Tensor,
        obs_traj: torch.Tensor,
        feed_forward: bool = True,
    ) -> torch.Tensor:
        B      = raw_ctx.shape[0]
        device = raw_ctx.device

        h_list = [torch.zeros(B, self.hidden_dim, device=device)
                  for _ in range(self.n_layers)]
        c_list = [torch.zeros(B, self.hidden_dim, device=device)
                  for _ in range(self.n_layers)]
        cur_pos = obs_traj[-1].clone()
        preds   = []

        for _ in range(self.pred_len):
            inp = torch.cat([raw_ctx, cur_pos], dim=-1)
            h_list[0], c_list[0] = self.cells[0](
                inp, (h_list[0].detach(), c_list[0].detach()))
            h = h_list[0]
            for j in range(1, self.n_layers):
                h_list[j], c_list[j] = self.cells[j](
                    h, (h_list[j].detach(), c_list[j].detach()))
                h = h_list[j]
            output = self.fc(h)
            preds.append(output)
            if feed_forward:
                cur_pos = output.detach()

        return torch.stack(preds, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
#  Paper GRU_Model  ── Algorithm 4
# ══════════════════════════════════════════════════════════════════════════════

class PaperGRUHead(nn.Module):
    """GRUCell stepwise prediction với dropout (Algorithm 4 của paper)."""

    def __init__(
        self,
        input_dim:  int   = 512 + 2,
        hidden_dim: int   = 256,
        n_layers:   int   = 3,
        pred_len:   int   = 12,
        dropout:    float = 0.20,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.pred_len   = pred_len

        self.cells = nn.ModuleList()
        self.cells.append(nn.GRUCell(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.cells.append(nn.GRUCell(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        raw_ctx:  torch.Tensor,
        obs_traj: torch.Tensor,
        feed_forward: bool = True,
    ) -> torch.Tensor:
        B      = raw_ctx.shape[0]
        device = raw_ctx.device

        h_list = [torch.zeros(B, self.hidden_dim, device=device)
                  for _ in range(self.n_layers)]
        cur_pos = obs_traj[-1].clone()
        preds   = []

        for _ in range(self.pred_len):
            inp = torch.cat([raw_ctx, cur_pos], dim=-1)
            h_list[0] = self.cells[0](inp, h_list[0].detach())
            h = self.dropout(h_list[0])
            for j in range(1, self.n_layers):
                h_list[j] = self.cells[j](h, h_list[j].detach())
                h = self.dropout(h_list[j])
            output = self.fc(h)
            preds.append(output)
            if feed_forward:
                cur_pos = output.detach()

        return torch.stack(preds, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
#  Encoder Pipeline  ── GIỮ NGUYÊN (dùng chung với STTrans)
# ══════════════════════════════════════════════════════════════════════════════

class PaperEncoder(nn.Module):
    """
    Encoder pipeline dùng chung cho PaperBaseline và STTrans.
    FNO3D + Mamba + Env_net → raw_ctx [B, RAW_CTX_DIM=512]
    """
    RAW_CTX_DIM = 512

    def __init__(
        self,
        obs_len:    int = 8,
        unet_in_ch: int = 13,
        ctx_dim:    int = 256,
    ):
        super().__init__()
        self.obs_len = obs_len

        self.spatial_enc = FNO3DEncoder(
            in_channel   = unet_in_ch,
            out_channel  = 1,
            d_model      = 32,
            n_layers     = 4,
            modes_t      = 4,
            modes_h      = 4,
            modes_w      = 4,
            spatial_down = 32,
            dropout      = 0.05,
        )

        self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.bottleneck_proj = nn.Linear(128, 128)
        self.decoder_proj    = nn.Linear(1, 16)

        self.enc_1d = DataEncoder1D(
            in_1d       = 4,
            feat_3d_dim = 128,
            mlp_h       = 64,
            lstm_hidden = 128,
            lstm_layers = 3,
            dropout     = 0.1,
            d_state     = 16,
        )

        self.env_enc = Env_net(obs_len=obs_len, d_model=32)

        self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
        self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
        self.ctx_drop = nn.Dropout(0.15)
        self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

    def forward(self, batch_list) -> torch.Tensor:
        """→ raw_ctx [B, RAW_CTX_DIM=512]"""
        obs_traj  = batch_list[0]
        obs_Me    = batch_list[7]
        image_obs = batch_list[11]
        env_data  = batch_list[13]

        if image_obs.dim() == 4:
            image_obs = image_obs.unsqueeze(2)

        expected_ch = self.spatial_enc.in_channel
        if image_obs.shape[1] == 1 and expected_ch != 1:
            image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

        e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
        T_obs = obs_traj.shape[0]

        e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)
        e_3d_s = e_3d_s.permute(0, 2, 1)
        e_3d_s = self.bottleneck_proj(e_3d_s)

        T_bot = e_3d_s.shape[1]
        if T_bot != T_obs:
            e_3d_s = F.interpolate(
                e_3d_s.permute(0, 2, 1), size=T_obs,
                mode="linear", align_corners=False,
            ).permute(0, 2, 1)

        e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
        t_weights  = torch.softmax(
            torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
                         device=e_3d_dec_t.device) * 0.5, dim=0,
        )
        f_spatial_scalar = (e_3d_dec_t * t_weights.unsqueeze(0)).sum(1, keepdim=True)
        f_spatial = self.decoder_proj(f_spatial_scalar)

        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        h_t    = self.enc_1d(obs_in, e_3d_s)

        e_env, _, _ = self.env_enc(env_data, image_obs)

        raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
        raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))
        return raw   # [B, 512]


# ══════════════════════════════════════════════════════════════════════════════
#  Paper Baseline Model  ── Wrapper chính
# ══════════════════════════════════════════════════════════════════════════════

MODEL_TYPES = ("lstm", "gru", "rnn")


class PaperBaseline(nn.Module):
    """
    Paper baseline model theo Rahman et al. (2025).
    Encoder: FNO3D + Mamba + Env_net (shared với STTrans)
    Prediction head: LSTMCell / GRUCell / RNNCell
    Loss: MSE (paper §2.7)
    """

    def __init__(
        self,
        model_type:  str   = "lstm",
        pred_len:    int   = 12,
        obs_len:     int   = 8,
        hidden_dim:  int   = 256,
        n_layers:    int   = 3,
        unet_in_ch:  int   = 13,
        dropout:     float = 0.20,
    ):
        super().__init__()
        assert model_type in MODEL_TYPES, f"model_type phải là {MODEL_TYPES}"
        self.model_type = model_type
        self.pred_len   = pred_len
        self.obs_len    = obs_len

        self.encoder  = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
        input_dim     = PaperEncoder.RAW_CTX_DIM + 2

        if model_type == "lstm":
            self.head = PaperLSTMHead(input_dim=input_dim, hidden_dim=hidden_dim,
                                      n_layers=n_layers, pred_len=pred_len)
        elif model_type == "gru":
            self.head = PaperGRUHead(input_dim=input_dim, hidden_dim=hidden_dim,
                                     n_layers=n_layers, pred_len=pred_len, dropout=dropout)
        else:
            self.head = PaperRNNHead(input_dim=input_dim, hidden_dim=hidden_dim,
                                     n_layers=n_layers, pred_len=pred_len)

    # ── Loss ─────────────────────────────────────────────────────────────────

    def mse_loss(self, pred_norm: torch.Tensor, gt_norm: torch.Tensor) -> torch.Tensor:
        """MSE loss (paper §2.7)."""
        T = min(pred_norm.shape[0], gt_norm.shape[0])
        return F.mse_loss(pred_norm[:T], gt_norm[:T])

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch_list) -> torch.Tensor:
        """→ pred [pred_len, B, 2] normalised."""
        raw_ctx  = self.encoder(batch_list)
        obs_traj = batch_list[0]
        return self.head(raw_ctx, obs_traj, feed_forward=True)

    # ── Training interface ────────────────────────────────────────────────────

    def get_loss(self, batch_list) -> torch.Tensor:
        traj_gt = batch_list[1]
        pred    = self.forward(batch_list)
        return self.mse_loss(pred, traj_gt)

    def get_loss_breakdown(self, batch_list, **kwargs) -> Dict:
        traj_gt = batch_list[1]
        pred    = self.forward(batch_list)
        loss    = self.mse_loss(pred, traj_gt)

        with torch.no_grad():
            ade_m = compute_ade_per_horizon(pred.detach(), traj_gt)
            atc_m = compute_ate_cte_per_horizon(pred.detach(), traj_gt)

        return dict(total=loss, mse=loss.item(), **ade_m, **atc_m)

    # ── Inference interface ───────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        batch_list,
        num_ensemble: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred_mean : [T, B, 2] normalised
            me_mean   : [T, B, 2] zeros
            all_trajs : [1, T, B, 2]
        """
        pred = self.forward(batch_list)
        T, B, _ = pred.shape
        me_mean   = torch.zeros(T, B, 2, device=pred.device)
        all_trajs = pred.unsqueeze(0)
        return pred, me_mean, all_trajs