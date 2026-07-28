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


def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        (t[..., 0] * 50.0 + 1800.0) / 10.0,
        (t[..., 1] * 50.0) / 10.0,
    ], dim=-1)


def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    lat1 = torch.deg2rad(p1[..., 1]);  lat2 = torch.deg2rad(p2[..., 1])
    dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
    dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
    a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    lon1 = torch.deg2rad(p1[..., 0]);  lat1 = torch.deg2rad(p1[..., 1])
    lon2 = torch.deg2rad(p2[..., 0]);  lat2 = torch.deg2rad(p2[..., 1])
    dlon = lon2 - lon1
    y = torch.sin(dlon) * torch.cos(lat2)
    x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    return torch.atan2(y, x)


def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
    if traj_deg.shape[0] < 2:
        return traj_deg.new_zeros(1, traj_deg.shape[1])
    return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


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


        self._attn_cache: list = []
        self._xai_hooks_registered = False

    def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)
                         * (-math.log(10000.0) / max(half - 1, 1)))
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.time_mlp(emb)


    def _register_xai_hooks(self):
        if self._xai_hooks_registered:
            return


        def _make_pre_hook():
            def _pre_hook(module, args, kwargs):
                kwargs = dict(kwargs)
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                return args, kwargs
            return _pre_hook

        def _make_hook(layer_idx: int, kind: str):
            def _hook(module, inputs, output):
                if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                    self._attn_cache.append({
                        "layer": layer_idx, "kind": kind,
                        "weights": output[1].detach()
                    })
            return _hook

        for i, layer in enumerate(self.decoder.layers):
            layer.self_attn.register_forward_pre_hook(_make_pre_hook(), with_kwargs=True)
            layer.self_attn.register_forward_hook(_make_hook(i, "self_attn"))


            layer.multihead_attn.register_forward_pre_hook(_make_pre_hook(), with_kwargs=True)
            layer.multihead_attn.register_forward_hook(_make_hook(i, "cross_attn"))

        self._xai_hooks_registered = True

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor, return_attention: bool = False):
        if return_attention:
            self._register_xai_hooks()
            self._attn_cache = []

        B, T, _ = x_t.shape
        step_idx = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
        x_emb = (self.traj_embed(x_t) + self.pos_emb[:, :T] + self.step_emb(step_idx))
        memory = torch.cat([self._time_emb(t).unsqueeze(1),
                            self.cond_proj(cond).unsqueeze(1)], dim=1)
        out = self.out_norm(self.decoder(x_emb, memory))
        velocity = self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)

        if return_attention:
            return velocity, list(self._attn_cache)
        return velocity


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


def hard_score_from_obs(obs_traj_norm: torch.Tensor,
                         return_components: bool = False,
                         weight_logits: Optional[torch.Tensor] = None,
                         obs_speed_norm_const: float = 20.0):
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

    components = torch.stack([curvature, speed_var, dir_change, obs_speed_norm], dim=0)

    if weight_logits is not None:
        w = F.softmax(weight_logits.to(device).to(components.dtype), dim=0)
    else:


        w = torch.tensor([0.35, 0.25, 0.25, 0.15], device=device, dtype=components.dtype)

    score = (w.unsqueeze(1) * components).sum(0).clamp(0., 1.)

    if return_components:
        return score, {"curvature": curvature, "speed_var": speed_var,
                       "dir_change": dir_change, "obs_speed_norm": obs_speed_norm}
    return score


@torch.no_grad()
def _physics_score(traj_norm: torch.Tensor, obs_norm: torch.Tensor,
                    use_curvature_score: bool = False) -> torch.Tensor:
    B      = traj_norm.shape[1]
    device = traj_norm.device
    traj_deg = _norm_to_deg(traj_norm)
    v_ref   = None


    if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
        obs_deg = _norm_to_deg(obs_norm)
        obs_spd = _step_speeds_kmh(obs_deg)
        T_s     = obs_spd.shape[0]
        w_obs   = torch.linspace(0.5, 1.0, T_s, device=device)
        v_ref   = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()
        pred_spd = _step_speeds_kmh(traj_deg)
        v_sigma  = v_ref.clamp(min=5.0) * 0.5
        speed_score = torch.exp(
            -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0)).pow(2).mean(0) * 0.5)
    elif traj_deg.shape[0] >= 2:
        speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(min=0) / 30.).mean(0))
    else:
        speed_score = torch.ones(B, device=device)


    if traj_deg.shape[0] >= 3:
        vel          = traj_deg[1:] - traj_deg[:-1]
        accel_mag    = (vel[1:] - vel[:-1]).norm(dim=-1)
        smooth_score = torch.exp(-accel_mag.mean(0) * 5.0)
    else:
        smooth_score = torch.ones(B, device=device)


    if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
        obs_vel  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
        pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
        obs_h    = F.normalize(obs_vel,  dim=-1, eps=1e-6)
        pred_h   = F.normalize(pred_vel, dim=-1, eps=1e-6)
        cos_sim  = (obs_h * pred_h).sum(-1).clamp(-1, 1)
        head_score = torch.exp((cos_sim - 1.0) * 3.0)
    else:
        head_score = torch.ones(B, device=device)


    if use_curvature_score and obs_norm.shape[0] >= 3 and traj_deg.shape[0] >= 2:
        obs_deg_c  = _norm_to_deg(obs_norm)
        bear_obs_1 = _forward_azimuth(obs_deg_c[-3], obs_deg_c[-2])
        bear_obs_2 = _forward_azimuth(obs_deg_c[-2], obs_deg_c[-1])
        obs_turn_rate = ((bear_obs_2 - bear_obs_1 + 180.0) % 360.0) - 180.0

        bear0 = _forward_azimuth(obs_deg_c[-1], traj_deg[0])
        if traj_deg.shape[0] >= 2:
            chain = [_forward_azimuth(traj_deg[t], traj_deg[t + 1])
                     for t in range(traj_deg.shape[0] - 1)]
            pred_bears = torch.stack([bear0] + chain, 0)
        else:
            pred_bears = bear0.unsqueeze(0)

        if pred_bears.shape[0] >= 2:
            pred_turn = ((pred_bears[1:] - pred_bears[:-1] + 180.0) % 360.0) - 180.0
            Tc = pred_turn.shape[0]
            w_curv = torch.linspace(1.0, 0.3, Tc, device=device).unsqueeze(1)
            turn_err = ((pred_turn - obs_turn_rate.unsqueeze(0)).abs() * w_curv).sum(0) / w_curv.sum()
            curvature_score = torch.exp(-turn_err / 15.0)
        else:
            curvature_score = torch.ones(B, device=device)
    else:
        curvature_score = torch.ones(B, device=device)


    if v_ref is not None and traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
        T_pred        = traj_deg.shape[0]
        expected_total = v_ref * T_pred * DT_HOURS * 0.75
        step_dists    = _haversine_deg(traj_deg[:-1], traj_deg[1:])
        actual_total  = step_dists.sum(0)
        rel_err       = (actual_total - expected_total).abs() / expected_total.clamp(min=10.)
        disp_score    = torch.exp(-rel_err * 1.5)
    else:
        disp_score    = torch.ones(B, device=device)

    if use_curvature_score:


        return (speed_score.pow(0.25)
                * smooth_score.pow(0.20)
                * head_score.pow(0.25)
                * disp_score.pow(0.10)
                * curvature_score.pow(0.20)).clamp(min=1e-6)
    return (speed_score.pow(0.30)
            * smooth_score.pow(0.25)
            * head_score.pow(0.30)
            * disp_score.pow(0.15)).clamp(min=1e-6)


def augment_batch(batch_list, disable_c: bool = False) -> list:
    bl = list(batch_list)
    if not torch.is_tensor(bl[0]):
        return bl

    obs    = bl[0]
    device = obs.device
    anchor = obs[-1:, :, :2].detach()
    r = torch.rand(1).item()

    if r < 0.25:


        shift = (torch.rand(2, device=device) - 0.5) * 0.018
        bl[0] = obs + shift.view(1, 1, 2)
        if torch.is_tensor(bl[1]):
            bl[1] = bl[1] + shift.view(1, 1, 2)

    elif r < 0.45:


        scale = 0.70 + 0.70 * torch.rand(1, device=device).item()
        obs_c = obs.clone()
        obs_c[..., :2] = anchor + (obs[..., :2] - anchor) * scale
        bl[0] = obs_c
        if torch.is_tensor(bl[1]):
            bl[1] = anchor + (bl[1] - anchor) * scale

    elif r < 0.65:


        if disable_c:


            pass
        else:
            T_pred = bl[1].shape[0] if torch.is_tensor(bl[1]) else 0
            if T_pred >= 4:
                gt      = bl[1].clone()
                max_deg = (torch.rand(1).item() - 0.5) * 40.0
                max_rad = max_deg * math.pi / 180.0
                pts  = torch.cat([anchor, gt], 0)
                disp = pts[1:] - pts[:-1]
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

                T_obs   = obs.shape[0]
                obs_aug = obs.clone()
                cp = math.cos(max_rad * 0.3); sp = math.sin(max_rad * 0.3)
                rp = torch.tensor([[cp, -sp], [sp, cp]], dtype=obs.dtype, device=device)
                for t_obs in range(max(1, T_obs - 3), T_obs):
                    d = obs_aug[t_obs, :, :2] - obs_aug[t_obs - 1, :, :2]
                    obs_aug[t_obs, :, :2] = obs_aug[t_obs - 1, :, :2] + (rp @ d.unsqueeze(-1)).squeeze(-1)
                bl[0] = obs_aug

    elif r < 0.90:

        pass

    else:

        obs_new = obs.clone()
        obs_new[..., :2] = obs[..., :2] + torch.randn_like(obs[..., :2]) * 0.003
        bl[0] = obs_new

    return bl


def compute_obs_attribution(model, batch_list, device: torch.device,
                             target_step: int = 11) -> torch.Tensor:
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
    all_deg  = _norm_to_deg(all_traj)
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


class TCFlowMatching(nn.Module):

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
        sigma_min:         float = 0.06,


        sigma_max:         float = 0.15,


        sigma_decay_start: int   = 5,
        sigma_decay_end:   int   = 100,


        lambda_reg:        float = 0.2,
        lambda_heading:    float = 0.07,
        lambda_momentum:   float = 0.0,
        lambda_calib:      float = 0.1,
        lambda_hard_reg:   float = 0.02,

        log_sigma_reg_min_clamp: float = -3.0,


        enable_horizon_nll: bool  = True,


        use_ot:            bool  = True,
        ot_epsilon:        float = 0.05,
        use_ema:           bool  = True,
        ema_decay:         float = 0.995,
        n_inference_steps: int   = 10,


        n_ensemble:        int   = 20,
        sigma_inference:   float = 0.04,
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
        self.lambda_momentum    = 0.0
        self.lambda_calib       = lambda_calib


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


        self.speed_correction_logits = nn.Parameter(torch.zeros(pred_len))


        self.reg_step_logits = nn.Parameter(torch.zeros(pred_len))


        self.log_b_horizon = nn.Parameter(torch.zeros(pred_len))


        self.hard_score_weight_logits = nn.Parameter(
            torch.log(torch.tensor([0.40, 0.30, 0.30, 0.15])))


        self.heading_step_logits = nn.Parameter(torch.zeros(pred_len))


        import math as _math
        self.log_sigma_reg     = nn.Parameter(torch.tensor(
            -0.5 * _math.log(2.0 * 0.20)))
        self.log_sigma_heading = nn.Parameter(torch.tensor(
            -0.5 * _math.log(2.0 * 0.07)))
        self.log_sigma_calib   = nn.Parameter(torch.tensor(
            -0.5 * _math.log(2.0 * 0.10)))

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
        if epoch < self.sigma_decay_start:
            return self.sigma_max
        if epoch < self.sigma_decay_end:
            span = max(self.sigma_decay_end - self.sigma_decay_start, 1)
            t = (epoch - self.sigma_decay_start) / span
            return self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + math.cos(math.pi * t))
        return self.sigma_min


    def _heading_loss_ms(self, pred_deg: torch.Tensor, obs_deg: torch.Tensor) -> torch.Tensor:
        if obs_deg.shape[0] < 2 or pred_deg.shape[0] < 1:
            return pred_deg.new_zeros(())

        ref_bear = _forward_azimuth(obs_deg[-2], obs_deg[-1])
        pts = torch.cat([obs_deg[-1:], pred_deg], 0)

        N = pred_deg.shape[0]
        sw = F.softmax(self.heading_step_logits[:N], dim=0)

        loss = pred_deg.new_zeros(())
        for t in range(N):
            pred_bear  = _forward_azimuth(pts[t], pts[t + 1])
            angle_diff = pred_bear - ref_bear
            loss       = loss + sw[t] * (1.0 - torch.cos(angle_diff)).mean()
            ref_bear   = pred_bear.detach()

        return loss


    def _reg_loss(self, x1_rel: torch.Tensor, last_obs: torch.Tensor,
                  cond: torch.Tensor,
                  hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x1_rel.shape
        device   = x1_rel.device
        x0   = torch.randn_like(x1_rel) * self.sigma_inference
        t0   = torch.zeros(B, device=device)
        v    = self.velocity(x0, t0, cond)
        x1_pred_abs = self._from_relative(x0 + v, last_obs)
        x1_gt_abs   = self._from_relative(x1_rel, last_obs)
        pred_deg = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))
        gt_deg   = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
        dist     = _haversine_deg(pred_deg, gt_deg)

        T_actual = dist.shape[0]

        if self.enable_horizon_nll:
            b_h  = torch.exp(self.log_b_horizon[:T_actual]).clamp(min=1e-2, max=1e4)
            nll_h = dist / b_h.unsqueeze(1) + torch.log(b_h).unsqueeze(1)
            weighted_dist = nll_h
        else:
            weighted_dist = dist

        sw = F.softmax(self.reg_step_logits[:T_actual], dim=0).unsqueeze(1)

        if hard_score is not None:
            sw_hard = (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)
        else:
            sw_hard = torch.ones(1, B, device=device, dtype=dist.dtype)


        return ((weighted_dist * sw) * sw_hard).mean() / 300.0


    def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
        obs_traj = batch_list[0]
        gt_traj  = batch_list[1]
        B        = obs_traj.shape[1]
        device   = obs_traj.device

        sigma    = self._sigma_schedule(epoch)
        x1_gt    = gt_traj.permute(1, 0, 2)
        last_obs = obs_traj[-1, :, :2]
        x1_rel   = self._to_relative(x1_gt, last_obs)


        h_score = hard_score_from_obs(obs_traj[:, :, :2],
                                       weight_logits=self.hard_score_weight_logits)
        cond    = self.encoder(batch_list, hard_score=h_score)


        hard_dist    = F.softmax(self.hard_score_weight_logits, dim=0)
        hard_uniform = hard_dist.new_full((4,), 0.25)
        l_hard_reg   = ((hard_dist - hard_uniform) ** 2).sum()


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


        if epoch < 10:     ramp_reg = 0.0
        elif epoch < 30:   ramp_reg = (epoch - 10) / 20.0
        else:              ramp_reg = 1.0

        l_reg = (self._reg_loss(x1_rel, last_obs, cond, h_score)
                 if ramp_reg > 0.0 else x0.new_zeros(()))


        if epoch < 5:      ramp_dir = 0.0
        elif epoch < 20:   ramp_dir = (epoch - 5) / 15.0
        else:              ramp_dir = 1.0

        if ramp_dir > 0.0:
            x0_h       = torch.randn_like(x1_rel) * self.sigma_inference
            v_h        = self.velocity(x0_h, torch.zeros(B, device=device), cond)
            x1_h_abs   = self._from_relative(x0_h + v_h, last_obs)
            pred_deg_h = _norm_to_deg(x1_h_abs.permute(1, 0, 2))
            obs_deg_h  = _norm_to_deg(obs_traj[:, :, :2])
            l_heading  = self._heading_loss_ms(pred_deg_h, obs_deg_h)
        else:
            l_heading = x0.new_zeros(())

        l_momentum = x0.new_zeros(())


        if epoch < 10:     ramp_calib = 0.0
        elif epoch < 30:   ramp_calib = (epoch - 10) / 20.0
        else:              ramp_calib = 1.0


        x0_c = torch.randn_like(x1_rel) * self.sigma_inference
        v_c  = self.velocity(x0_c, torch.zeros(B, device=device), cond)
        pred_c_abs = self._from_relative(x0_c + v_c, last_obs).permute(1, 0, 2)
        cal_abs    = self.speed_calibrate_pred(pred_c_abs, last_obs, obs_traj[:, :, :2])
        cal_deg    = _norm_to_deg(cal_abs)
        gt_c_deg   = _norm_to_deg(x1_gt.permute(1, 0, 2))

        l_calib    = _haversine_deg(cal_deg, gt_c_deg).mean() / 300.0


        HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)

        prec_reg     = torch.exp(-2.0 * self.log_sigma_reg.clamp(min=self.log_sigma_reg_min_clamp))
        prec_heading = torch.exp(-2.0 * self.log_sigma_heading.clamp(min=-3.0))
        prec_calib   = torch.exp(-2.0 * self.log_sigma_calib.clamp(min=-3.0))

        weighted_reg     = ramp_reg    * (0.5 * prec_reg     * l_reg     + self.log_sigma_reg.clamp(min=self.log_sigma_reg_min_clamp)     + HALF_LOG_2PI)
        weighted_heading = ramp_dir    * (0.5 * prec_heading * l_heading + self.log_sigma_heading.clamp(min=-3.0) + HALF_LOG_2PI)
        weighted_calib   = ramp_calib  * (0.5 * prec_calib   * l_calib   + self.log_sigma_calib.clamp(min=-3.0)   + HALF_LOG_2PI)


        total = (l_cfm + weighted_reg + weighted_heading + weighted_calib
                  + self.lambda_hard_reg * l_hard_reg)
        if not torch.isfinite(total):
            total = x0.new_zeros(())


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

            "l_fm": l_cfm.item(), "dpe": 0., "heading": 0., "vel_reg": 0.,
            "speed": 0., "accel": 0., "fm_mse": l_cfm.item(),
            "l_hard_total": 0., "n_hard": 0, "alpha_hard": 0.,
            "l_sel_total": 0., "speed_head_l": 0., "l_score": 0.,
            "l_speed_ratio": 0., "l_sigma_nll": 0.,
            "learned_lambda_speed_ratio": 0., "learned_sigma_infer": float(self.sigma_inference),
        }

    def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]


    def speed_calibrate_pred(self,
                              pred_abs_norm: torch.Tensor,
                              last_obs_norm: torch.Tensor,
                              obs_norm:      torch.Tensor) -> torch.Tensor:
        if obs_norm.shape[0] < 2 or pred_abs_norm.shape[0] < 2:
            return pred_abs_norm

        T = pred_abs_norm.shape[0]
        correction = (torch.sigmoid(self.speed_correction_logits[:T]) * 2.0
                      ).to(pred_abs_norm.dtype).view(T, 1, 1)

        pts  = torch.cat([last_obs_norm.unsqueeze(0), pred_abs_norm], 0)
        disp = pts[1:] - pts[:-1]
        disp_cal = disp * correction

        out = torch.empty_like(pred_abs_norm)
        cur = last_obs_norm
        for t in range(T):
            cur = cur + disp_cal[t]
            out[t] = cur
        return out


    @torch.no_grad()
    def sample(self, batch_list,
               num_ensemble:          Optional[int]  = None,
               ddim_steps:            Optional[int]  = None,
               return_xai:            bool           = False,
               return_attention:      bool           = False,
               use_speed_calibration: bool           = True,
               use_curvature_score:   bool           = False,
               **kwargs) -> Tuple:
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
            all_traj.append(x_abs.permute(1, 0, 2))

        scores  = torch.stack(
            [_physics_score(t, obs_norm, use_curvature_score=use_curvature_score)
             for t in all_traj], 0)
        all_t   = torch.stack(all_traj, 0)
        top_k   = min(3, K)
        top_idx = scores.topk(top_k, dim=0).indices

        pred_mean = torch.zeros_like(all_traj[0])
        for b in range(B):
            idx_b = top_idx[:, b]
            w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
            pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)


        if use_speed_calibration:
            pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

        if not return_xai:
            return pred_mean, torch.zeros_like(pred_mean), all_t


        xai = {}

        xai.update(compute_ensemble_uncertainty(all_t))

        _, hard_comps = hard_score_from_obs(obs_norm, return_components=True,
                                             weight_logits=self.hard_score_weight_logits)
        xai["hard_components"] = hard_comps

        pred_deg  = _norm_to_deg(pred_mean)
        obs_deg_x = _norm_to_deg(obs_norm)

        obs_spd_x    = _step_speeds_kmh(obs_deg_x)
        obs_spd_mu   = obs_spd_x.mean(0)
        if pred_deg.shape[0] >= 2:
            last_deg_x   = obs_deg_x[-1]
            pts_x        = torch.cat([last_deg_x.unsqueeze(0), pred_deg], 0)
            pred_spd_x   = _step_speeds_kmh(pts_x)
            pred_spd_mu  = pred_spd_x.mean(0)
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


        if pred_deg.shape[0] >= 2 and gt_deg_xai.shape[0] >= 2:
            T8       = min(pred_deg.shape[0], gt_deg_xai.shape[0])
            last_d   = obs_deg_x[-1]
            pts_pred = torch.cat([last_d.unsqueeze(0), pred_deg[:T8]], 0)
            pts_gt   = torch.cat([last_d.unsqueeze(0), gt_deg_xai[:T8]], 0)
            spd_pred = _step_speeds_kmh(pts_pred)
            spd_gt   = _step_speeds_kmh(pts_gt)
            ratio    = spd_pred / spd_gt.clamp(min=1.0)
            def _hz(s, lo, hi):
                hi = min(hi, s.shape[0])
                return float(s[lo:hi].mean()) if hi > lo else float("nan")
            r_mean = ratio.mean(1)
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


        obs_spd_cat = _step_speeds_kmh(obs_deg_x).mean(0)
        sm = obs_spd_cat < 8.0
        mm = (obs_spd_cat >= 8.0) & (obs_spd_cat < 15.0)
        fm = obs_spd_cat >= 15.0
        ade_per_storm = _haversine_deg(pred_deg[:gt_deg_xai.shape[0]],
                                        gt_deg_xai[:pred_deg.shape[0]]).mean(0)
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


        xai["learned_params"] = {
            "speed_correction":     (torch.sigmoid(self.speed_correction_logits) * 2.0).tolist(),
            "reg_step_weights":     F.softmax(self.reg_step_logits, dim=0).tolist(),
            "hard_score_weights":   F.softmax(self.hard_score_weight_logits, dim=0).tolist(),
            "b_horizon":            torch.exp(self.log_b_horizon).detach().tolist(),

            "heading_step_weights": F.softmax(self.heading_step_logits, dim=0).tolist(),
            "sigma_inf":            float(self.sigma_inference),
            "log_sigma_reg":        float(self.log_sigma_reg.detach()),
            "log_sigma_heading":    float(self.log_sigma_heading.detach()),
            "log_sigma_calib":      float(self.log_sigma_calib.detach()),
            "eff_lambda_reg":       float((0.5 * torch.exp(-2.0 * self.log_sigma_reg.clamp(min=self.log_sigma_reg_min_clamp))).detach()),
            "eff_lambda_heading":   float((0.5 * torch.exp(-2.0 * self.log_sigma_heading.clamp(min=-3.0))).detach()),
            "eff_lambda_calib":     float((0.5 * torch.exp(-2.0 * self.log_sigma_calib.clamp(min=-3.0))).detach()),
        }


        if return_attention:

            image_obs_x = batch_list[11]
            env_data_x  = batch_list[13]
            if image_obs_x.dim() == 4:
                image_obs_x = image_obs_x.unsqueeze(2)
            _, _, _, env_attn = self.encoder.env_enc(
                env_data_x, image_obs_x, return_attention=True)


            x_rel_probe = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference
            _, vel_attn = self.velocity(x_rel_probe, t0, cond, return_attention=True)

            xai["attention"] = {
                "env_net":              env_attn,
                "velocity_transformer": vel_attn,
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


TCDiffusion = TCFlowMatching