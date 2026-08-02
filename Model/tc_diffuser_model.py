from __future__ import annotations

"""
TC-Diffuser (velocity-space DDPM for TC track prediction, Trajectron++
lineage), re-implemented with the SAME multi-modal encoder used by
ST-Trans / paper LSTM-GRU-RNN / MMSTN / Phys-Diff baselines in this project
(PaperEncoder = FNO3D + Mamba + Env_net), for a fair comparison.

WHAT IS KEPT FAITHFUL to the original TC-Diffuser repo:
  - VarianceSchedule (models/diffusion.py): linear/cosine beta schedule,
    padded-betas convention (`torch.cat([zeros(1), betas])`), alpha_bars via
    cumulative log-sum, `sigmas_flex`/`sigmas_inflex` for flexibility-
    interpolated reverse variance. Copied verbatim, same formulas.
  - DiffusionTraj.get_loss: forward-noises x_0 (here: velocity, NOT
    position) via `c0*x_0 + c1*e_rand`, denoiser predicts e_theta, loss is
    per-timestep MSE(e_theta, e_rand) with the ORIGINAL per-step weighting
    `Wt` that upweights the first predicted step by 1.3x (`first_add`).
    Copied verbatim including this specific weighting scheme.
  - DiffusionTraj.sample: DDPM/DDIM reverse loop, `bestof` flag (sample
    from pure Gaussian noise vs. zeros), strided timestep loop. Copied
    verbatim.
  - ConcatSquashLinear (models/common.py): FiLM-style conditioning layer
    (gate + bias derived from context, applied multiplicatively/additively
    to a linear projection of x). Copied verbatim -- this is the core
    conditioning primitive of the denoiser.
  - TransformerConcatLinear structure: context+time embedding drives a
    stack of ConcatSquashLinear layers around a TransformerEncoder self-
    attention block over the prediction-horizon "tokens". Kept the same
    layer topology (expand to 2*context_dim -> transformer -> contract back
    down through two more ConcatSquashLinear stages -> final linear to
    point_dim), only the point_dim and the auxiliary per-quantity input
    branches changed (see #2 below).
  - SingleIntegrator.integrate_samples (models/encoders/dynamics/
    single_integrator.py): `torch.cumsum(v, dim=T_axis) * dt + p_0`.
    Copied verbatim -- the denoiser predicts VELOCITY noise, and positions
    are recovered by cumulative-sum integration from the last observed
    position, exactly as in the original repo. This is the "velocity
    diffusion" design that is core to TC-Diffuser and is preserved as
    requested (not flattened into direct position prediction).

WHAT CHANGED vs. the original TC-Diffuser:
  1. The original `encoder.get_latent()` (a full Trajectron++ scene-graph
     encoder: per-node history GRU, edge/social-pooling influence, robot-
     future encoder, map CNN) plus the auxiliary `EnvPredicter` branch
     (predicts wind/intensity-class/move-velocity/etc. from the encoded
     env vector as a secondary self-supervised task) are REMOVED and
     REPLACED by `PaperEncoder` (identical module used by
     STTrans/PaperBaseline/MMSTN/PhysDiff), which fuses obs_traj + obs_Me +
     Data3d (FNO3D) + env_data (Env_net transformer) into a single 512-d
     context vector. This vector plays the role of the original `context`
     (`x_gph`, 256-d Trajectron latent) fed into the denoiser; the
     `encoded_age` / `encoded_env_data` auxiliary conditioning branches
     (which fed the removed EnvPredicter's outputs back into the denoiser
     as extra FiLM context) are dropped since there is no more separate
     env-forecasting task -- PaperEncoder's context already encodes env
     information directly.
  2. Per your instruction, velocity is predicted for 2-D (lon, lat) ONLY;
     the original 4-D velocity (lon, lat, intensity, wind) and its 3
     separate per-quantity ConcatSquashLinear branches (trajectory /
     intensity / wind, each conditioned on `EnvPredicter` outputs) collapse
     to a single 2-D trajectory branch. The `Wt` first-step upweighting and
     the overall FiLM/transformer topology are otherwise untouched.
  3. Training loop restructured into the SAME epoch/early-stopping/
     metrics-CSV/checkpoint framework as the other three baselines.
  4. Evaluation NOTE (see train script): unlike the other three baselines
     (single-sample eval), TC-Diffuser's evaluate() uses the ORIGINAL
     repo's best-of-6 sampling (`generate(..., sample=6, bestof=True)` in
     tc_diffuser.py) since best-of-k stochastic sampling is a defining
     property of this diffusion-based baseline, not an incidental training
     detail -- flattening it to 1 sample would misrepresent what the model
     actually does. This is logged explicitly in the metrics CSV
     (`sampling` column) so it's never silently conflated with the other
     baselines' single-sample ADE.
"""

import math
from typing import Dict, List, Optional, Tuple

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
    HORIZON_STEPS,
)


# ══════════════════════════════════════════════════════════════════════════
#  Faithful copy: PositionalEncoding, ConcatSquashLinear (models/common.py)
# ══════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Verbatim from models/common.py."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class ConcatSquashLinear(nn.Module):
    """Verbatim from models/common.py: FiLM-style conditioning
    (gate * linear(x) + bias, gate/bias derived from context)."""

    def __init__(self, dim_in: int, dim_out: int, dim_ctx: int):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        a = self._layer(x)
        return a * gate + bias


# ══════════════════════════════════════════════════════════════════════════
#  Denoiser: TransformerConcatLinear, 2-D (coord-only) specialization of
#  models/diffusion.py TransformerConcatLinear.ptask_then_pshare
# ══════════════════════════════════════════════════════════════════════════

class TransformerConcatLinearCoordOnly(nn.Module):
    """Faithful to the original TransformerConcatLinear's
    `ptask_then_pshare` forward path (the one actually used, see
    models/diffusion.py `forward()`), specialized to point_dim=2 (lon,lat
    only) and a single PaperEncoder-derived context (no separate
    encoded_age / encoded_env_data auxiliary branches -- see module
    docstring change #1)."""

    def __init__(self, point_dim: int = 2, context_dim: int = 256, tf_layer: int = 3):
        super().__init__()
        self.context_dim = context_dim

        self.pos_emb = PositionalEncoding(d_model=2 * context_dim, dropout=0.1, max_len=64)

        # x (velocity noise, point_dim) -> 2*context_dim, conditioned on
        # [time_emb(3) ; context(context_dim)]  => dim_ctx = context_dim+3
        self.concat1 = ConcatSquashLinear(point_dim, 2 * context_dim, context_dim + 3)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=2 * context_dim, nhead=4, dim_feedforward=4 * context_dim,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=tf_layer)

        self.concat3 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
        self.concat4 = ConcatSquashLinear(context_dim, context_dim // 2, context_dim + 3)
        self.linear  = ConcatSquashLinear(context_dim // 2, point_dim, context_dim + 3)

    def forward(self, x: torch.Tensor, beta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, T, point_dim) -- noisy velocity
        beta:    (B,)               -- diffusion beta_t
        context: (B, context_dim)   -- PaperEncoder context vector
        Returns: (B, T, point_dim)  -- predicted noise
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)
        context = context.view(batch_size, 1, -1)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B,1,3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)                          # (B,1,context_dim+3)

        h = self.concat1(ctx_emb, x)                     # (B, T, 2*context_dim)
        h = h.permute(1, 0, 2)                              # (T, B, 2*context_dim)
        h = self.pos_emb(h)
        h = self.transformer_encoder(h).permute(1, 0, 2)   # (B, T, 2*context_dim)

        h = self.concat3(ctx_emb, h)                        # (B, T, context_dim)
        h = self.concat4(ctx_emb, h)                        # (B, T, context_dim//2)
        out = self.linear(ctx_emb, h)                        # (B, T, point_dim)
        return out


# ══════════════════════════════════════════════════════════════════════════
#  Faithful copy: VarianceSchedule (models/diffusion.py)
# ══════════════════════════════════════════════════════════════════════════

class VarianceSchedule(nn.Module):
    def __init__(self, num_steps: int, mode: str = "linear",
                 beta_1: float = 1e-4, beta_T: float = 5e-2, cosine_s: float = 8e-3):
        super().__init__()
        assert mode in ("linear", "cosine")
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        else:
            timesteps = torch.arange(num_steps + 1) / num_steps + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)  # padding at index 0

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size: int) -> List[int]:
        import numpy as np
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility: float):
        assert 0 <= flexibility <= 1
        return self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)


# ══════════════════════════════════════════════════════════════════════════
#  Faithful copy: DiffusionTraj (models/diffusion.py), 2-D specialization
# ══════════════════════════════════════════════════════════════════════════

class DiffusionTraj(nn.Module):
    def __init__(self, net: TransformerConcatLinearCoordOnly, var_sched: VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0: torch.Tensor, context: torch.Tensor, t: Optional[List[int]] = None) -> torch.Tensor:
        """x_0: (B, T, point_dim) -- ground-truth VELOCITY (not position)."""
        batch_size, T, point_dim = x_0.size()
        device = x_0.device
        if t is None:
            t = self.var_sched.uniform_sample_t(batch_size)

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].to(device)
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).to(device)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).to(device)
        e_rand = torch.randn_like(x_0)

        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)

        # Per-step weighting, generalized from the original repo's pattern.
        # IMPORTANT: the original models/diffusion.py only defines `Wt`
        # explicitly for T in {1, 2, 3, 4} (via if/elif branches on the
        # ORIGINAL repo's much shorter prediction horizon); for T=12 (this
        # project's pred_len) none of those branches match and the
        # original code would raise NameError (Wt undefined). Every
        # explicitly-defined branch follows the SAME pattern -- first step
        # weighted by `first_add=1.3`, every other step weighted 1.0 -- so
        # `Wt = [first_add] + [1.0]*(T-1)` is the unique natural extension
        # of that documented pattern to arbitrary T, not an invented
        # alternative weighting scheme.
        first_add = 1.3
        Wt = [first_add] + [1.0] * (T - 1)

        loss = 0.0
        for i in range(e_theta.size(1)):
            loss_i = Wt[i] * F.mse_loss(
                e_theta[:, i, :].reshape(-1, point_dim),
                e_rand[:, i, :].reshape(-1, point_dim),
                reduction="mean",
            )
            loss = loss + loss_i
        loss = loss / e_theta.size(1)
        return loss

    @torch.no_grad()
    def sample(
        self,
        num_points: int,
        context: torch.Tensor,
        sample: int,
        bestof: bool,
        point_dim: int = 2,
        flexibility: float = 0.0,
        ret_traj: bool = False,
        sampling: str = "ddpm",
        step: int = 1,
    ) -> torch.Tensor:
        """Returns (sample, B, num_points, point_dim) velocity samples.

        ⚠ IMPORTANT constraint on `step` (verbatim from the original repo's
        models/diffusion.py DiffusionTraj.sample): the 'ddpm' branch formula
        `x_next = c0*(x_t - c1*e_theta) + sigma*z` uses `alpha`/`alpha_bar`
        of the CURRENT t only (not a ratio between t and t-step), so it is
        mathematically valid ONLY when:
          - step == 1 (full adjacent-step reverse loop), or
          - step == self.var_sched.num_steps (one-shot: single denoising
            step directly from pure noise to t=0 -- this is what the
            original repo's own AutoEncoder.generate() uses by default,
            `step=100` with `num_steps=100`).
        Any intermediate step value (e.g. step=5 with num_steps=100) will
        silently use the wrong alpha/alpha_bar at each jump and produce
        samples that get WORSE, not better, as the denoiser is trained
        further -- this was a real bug found in an earlier version of this
        file's training script default. The 'ddim' branch (sampling="ddim")
        IS valid for arbitrary strides since it explicitly uses both
        alpha_bar and alpha_bar_next.
        """
        if sampling == "ddpm" and step not in (1, self.var_sched.num_steps):
            import warnings
            warnings.warn(
                f"DiffusionTraj.sample called with sampling='ddpm' and "
                f"step={step}, but num_steps={self.var_sched.num_steps}. "
                f"The 'ddpm' branch is only valid for step=1 (full reverse "
                f"loop) or step=num_steps (one-shot). An intermediate "
                f"stride will silently produce mathematically incorrect "
                f"samples. Use sampling='ddim' for arbitrary strides, or "
                f"set step=1 or step={self.var_sched.num_steps}.",
                RuntimeWarning,
            )
        device = context.device
        traj_list = []
        for _ in range(sample):
            batch_size = context.size(0)
            if bestof:
                x_T = torch.randn([batch_size, num_points, point_dim], device=device)
            else:
                x_T = torch.zeros([batch_size, num_points, point_dim], device=device)

            traj = {self.var_sched.num_steps: x_T}
            stride = step
            for t in range(self.var_sched.num_steps, 0, -stride):
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                alpha_bar_next = self.var_sched.alpha_bars[max(t - stride, 0)]
                sigma = self.var_sched.get_sigmas(t, flexibility)
                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                x_t = traj[t]
                beta = self.var_sched.betas[[t] * batch_size]
                e_theta = self.net(x_t, beta=beta, context=context)

                if sampling == "ddpm":
                    x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                elif sampling == "ddim":
                    x0_t = (x_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * e_theta
                else:
                    raise ValueError(sampling)

                traj[t - stride] = x_next.detach()
                traj[t] = traj[t].cpu()
                if not ret_traj:
                    del traj[t]

            traj_list.append(traj if ret_traj else traj[0])
        return torch.stack(traj_list) if not ret_traj else traj_list


# ══════════════════════════════════════════════════════════════════════════
#  Faithful copy: SingleIntegrator.integrate_samples
# ══════════════════════════════════════════════════════════════════════════

def integrate_samples(v: torch.Tensor, p_0: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """
    v:   (sample, B, T, 2) -- velocity samples
    p_0: (B, 2)            -- last observed position (normalized coords)
    Returns position samples (sample, B, T, 2), verbatim formula from
    models/encoders/dynamics/single_integrator.py SingleIntegrator.integrate_samples:
    cumulative sum of velocity * dt, offset by initial position.
    """
    p_0_exp = p_0.unsqueeze(0).unsqueeze(2)  # (1, B, 1, 2)
    return torch.cumsum(v, dim=2) * dt + p_0_exp


# ══════════════════════════════════════════════════════════════════════════
#  Top-level TCDiffuser model
# ══════════════════════════════════════════════════════════════════════════

class TCDiffuser(nn.Module):
    """
    Velocity-space DDPM for TC track prediction, sharing PaperEncoder with
    the other baselines. Interface mirrors STTrans/PaperBaseline/MMSTN/
    PhysDiff: forward-style loss via get_loss_breakdown(), plus sample()
    for evaluation (best-of-k, matching the original repo's design -- see
    module docstring point #4).
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        unet_in_ch: int = 13,
        context_dim: int = 256,
        tf_layer: int = 3,
        num_steps: int = 100,
        beta_T: float = 5e-2,
        beta_1: float = 1e-4,
        var_mode: str = "linear",
        dt: float = 1.0,
        best_k: int = 6,
        sample_steps_stride: int = 100,  # must be 1 or num_steps -- see DiffusionTraj.sample docstring
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.context_dim = context_dim
        self.dt = dt
        self.best_k = best_k
        self.sample_steps_stride = sample_steps_stride

        # Shared multi-modal context encoder (replaces Trajectron++
        # get_latent() + EnvPredicter auxiliary branch).
        self.encoder = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
        self.ctx_proj = nn.Sequential(
            nn.Linear(PaperEncoder.RAW_CTX_DIM, context_dim),
            nn.LayerNorm(context_dim),
        )

        self.denoiser = TransformerConcatLinearCoordOnly(
            point_dim=2, context_dim=context_dim, tf_layer=tf_layer,
        )
        self.diffusion = DiffusionTraj(
            net=self.denoiser,
            var_sched=VarianceSchedule(
                num_steps=num_steps, beta_T=beta_T, beta_1=beta_1, mode=var_mode,
            ),
        )

    def _context(self, batch_list) -> torch.Tensor:
        raw_ctx = self.encoder(batch_list)     # (B, 512)
        return self.ctx_proj(raw_ctx)           # (B, context_dim)

    @staticmethod
    def _velocity_from_traj(obs_last_pos: torch.Tensor, pred_traj: torch.Tensor, dt: float) -> torch.Tensor:
        """Ground-truth velocity sequence consistent with integrate_samples:
        v_t = (pos_t - pos_{t-1}) / dt, with pos_{-1} = last observed pos.
        obs_last_pos: (B, 2); pred_traj: (T_pred, B, 2) time-major.
        Returns: (B, T_pred, 2) batch-first velocity.
        """
        pred_bf = pred_traj.permute(1, 0, 2)              # (B, T, 2)
        prev = torch.cat([obs_last_pos.unsqueeze(1), pred_bf[:, :-1]], dim=1)  # (B, T, 2)
        return (pred_bf - prev) / dt

    def get_loss_breakdown(self, batch_list) -> Dict:
        obs_traj = batch_list[0]           # (T_obs, B, 2) time-major
        pred_traj = batch_list[1]          # (T_pred, B, 2) time-major gt

        context = self._context(batch_list)               # (B, context_dim)
        v_gt = self._velocity_from_traj(obs_traj[-1], pred_traj, self.dt)  # (B, T, 2)

        loss = self.diffusion.get_loss(v_gt, context)

        with torch.no_grad():
            # cheap single-sample DDPM-style forward-then-decode isn't
            # available without full reverse sampling; report ADE via a
            # quick 1-sample reverse pass (bestof=True, stride to keep it
            # affordable) purely for the training-loop progress log --
            # the authoritative ADE/ATE/CTE reported for early-stopping and
            # comparison use evaluate()/sample() in the train script.
            pred_pos = self._quick_sample(batch_list, context, num_samples=1)
            ade_m = compute_ade_per_horizon(pred_pos.detach(), pred_traj)
            atc_m = compute_ate_cte_per_horizon(pred_pos.detach(), pred_traj)

        out = dict(total=loss, diffusion_loss=loss.item())
        out.update(ade_m)
        out.update(atc_m)
        return out

    def get_loss(self, batch_list) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list)["total"]

    @torch.no_grad()
    def _quick_sample(self, batch_list, context: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Single-sample (or bestof-averaged) reverse pass -> (T_pred, B, 2)
        position prediction, time-major, using the strided DDPM loop."""
        obs_traj = batch_list[0]
        v_samples = self.diffusion.sample(
            num_points=self.pred_len, context=context,
            sample=num_samples, bestof=True, point_dim=2,
            flexibility=0.0, ret_traj=False, sampling="ddpm",
            step=self.sample_steps_stride,
        )  # (num_samples, B, T, 2)
        pos_samples = integrate_samples(v_samples, obs_traj[-1], dt=self.dt)  # (num_samples,B,T,2)
        pred_bf = pos_samples[0]                    # (B, T, 2) -- first sample
        return pred_bf.permute(1, 0, 2)               # (T, B, 2)

    @torch.no_grad()
    def sample(self, batch_list, num_ensemble: int = 1, **kwargs
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniform interface with other baselines: returns
        (pred[T,B,2], me_mean[T,B,2]-zeros, all_trajs[num_samples,T,B,2]).
        Uses best-of-`self.best_k` sampling (see module docstring #4): all
        `best_k` reverse samples are generated, each integrated to a
        position trajectory, and the one closest to the ground truth (by
        mean L2 over the horizon) is reported as `pred` -- this matches the
        original repo's own `generate(..., bestof=True)` + evaluation
        convention (best-of-k stochastic prediction).
        """
        obs_traj = batch_list[0]
        gt = batch_list[1]                                   # (T_pred, B, 2)
        device = obs_traj.device

        context = self._context(batch_list)
        v_samples = self.diffusion.sample(
            num_points=self.pred_len, context=context,
            sample=self.best_k, bestof=True, point_dim=2,
            flexibility=0.0, ret_traj=False, sampling="ddpm",
            step=self.sample_steps_stride,
        )  # (best_k, B, T, 2)
        pos_samples = integrate_samples(v_samples, obs_traj[-1], dt=self.dt)  # (best_k,B,T,2)

        # pick, per-sample-in-batch, the trajectory (among best_k) closest
        # to gt (mean L2 distance over the horizon) -- "best of k" selection.
        gt_bf = gt.permute(1, 0, 2).unsqueeze(0)               # (1, B, T, 2)
        dist = (pos_samples - gt_bf).norm(dim=-1).mean(dim=-1)  # (best_k, B)
        best_idx = dist.argmin(dim=0)                            # (B,)

        B = pos_samples.shape[1]
        pred_bf = pos_samples[best_idx, torch.arange(B, device=device)]  # (B, T, 2)
        pred = pred_bf.permute(1, 0, 2)                          # (T, B, 2)

        T = pred.shape[0]
        me_mean = torch.zeros(T, B, 2, device=device)
        all_trajs = pos_samples.permute(0, 2, 1, 3)               # (best_k, T, B, 2)
        return pred, me_mean, all_trajs