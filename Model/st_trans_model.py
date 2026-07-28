from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 300):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class ObsKinematicEncoder(nn.Module):

    FEAT_DIM = 8

    def __init__(self, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 1, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()

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
        return feat.permute(1, 0, 2)

    def forward(self, obs_traj: torch.Tensor) -> torch.Tensor:
        feat = self._extract_features(obs_traj)
        h    = self.proj(feat)
        h    = self.pe(h)
        return self.enc(h)


class STTrans(nn.Module):

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


        self.encoder = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)


        self.ctx_proj = nn.Sequential(
            nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model),
            nn.LayerNorm(d_model),
        )


        self.obs_enc = ObsKinematicEncoder(
            d_model=d_model, nhead=nhead,
            num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout,
        )


        self.horizon_queries = nn.Parameter(
            torch.randn(1, pred_len, d_model) * 0.02
        )
        self.dec_pe = SinusoidalPE(d_model, max_len=pred_len + 10)


        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            activation="relu", batch_first=True,
        )
        self.transformer_dec = nn.TransformerDecoder(
            dec_layer, num_layers=num_dec_layers)


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


    def forward(self, batch_list) -> torch.Tensor:
        obs_traj = batch_list[0]
        B        = obs_traj.shape[1]


        raw_ctx    = self.encoder(batch_list)
        ctx_token  = self.ctx_proj(raw_ctx).unsqueeze(1)


        obs_memory = self.obs_enc(obs_traj)


        full_memory = torch.cat([ctx_token, obs_memory], dim=1)


        Q = self.horizon_queries.expand(B, -1, -1)
        Q = self.dec_pe(Q)


        D   = self.transformer_dec(Q, full_memory)
        out = self.reg_head(D)

        return out.permute(1, 0, 2)


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


        l_dpe = haversine_km(pred_deg, gt_deg).mean()


        l_mse = F.mse_loss(pred, gt)


        if T >= 2:
            step_dist = (pred[1:] - pred[:-1]).norm(dim=-1)
            l_speed   = F.relu(step_dist - self.v_max_norm).pow(2).mean()
        else:
            l_speed = pred_norm.new_zeros(())


        if T >= 3:
            vel     = pred[1:] - pred[:-1]
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


class STTransAR(nn.Module):

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


        self.encoder  = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
        self.ctx_proj = nn.Sequential(
            nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model),
            nn.LayerNorm(d_model),
        )

        self.obs_enc = ObsKinematicEncoder(
            d_model=d_model, nhead=nhead,
            num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout,
        )


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

        raw_ctx    = self.encoder(batch_list)
        ctx_token  = self.ctx_proj(raw_ctx).unsqueeze(1)
        obs_memory = self.obs_enc(obs_traj)
        full_mem   = torch.cat([ctx_token, obs_memory], dim=1)


        ctx = full_mem.mean(dim=1)
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

        return torch.stack(preds, dim=0)

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