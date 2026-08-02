from __future__ import annotations

"""
MMSTN (Faiaz-style Social-GAN baseline for TC track prediction), re-implemented
with the SAME encoder pipeline used by ST-Trans / paper LSTM-GRU-RNN baselines
in this project (PaperEncoder = FNO3D + Mamba + Env_net), so that the four
compared baselines all consume identical inputs (env data, Data1d, Data3d)
for a fair comparison.

Everything about the ORIGINAL MMSTN GAN mechanics is kept faithful to the
official repo (mmstn/models.py, mmstn/losses.py):
  - Generator = Encoder (LSTM) -> [noise injection] -> autoregressive LSTM
    Decoder, where at every decode step the whole predicted-so-far relative
    trajectory is RE-ENCODED through a second small LSTM `Encoder` (this is
    the (odd but original) `self.encoders` call inside `Decoder.forward`
    in mmstn/models.py lines 142-186) to refresh the decoder hidden state.
  - Discriminator = LSTM Encoder + MLP real/fake classifier (`d_type='local'`,
    the default used by MMSTN's own train.py).
  - GAN losses: `gan_g_loss` / `gan_d_loss` = numerically-stable BCE with
    RANDOM LABEL SMOOTHING (`random.uniform(0.7,1.2)` for real/fake-target
    "1", `random.uniform(0,0.3)` for fake target "0"), exactly as in
    mmstn/losses.py.
  - `best_k` trick: Generator is sampled `best_k` times per batch (each time
    with fresh noise), L2 loss keeps only the MIN-error sample per sequence
    (variety loss, Social-GAN style) -- see generator_step() in train.py.
  - Discriminator trained `d_steps` times per Generator `g_steps` (2:1 by
    MMSTN's own defaults), alternating within an epoch -- kept in train_mmstn.py.

WHAT CHANGED vs. the original MMSTN:
  1. The original `Encoder` embeds a 4-D signal (lon, lat, pressure, wind)
     via `nn.Linear(4, embedding_dim)` and runs a *bare* LSTM over it --
     there is no env/Data3d/Data1d encoder at all in the official MMSTN.
     Here we REPLACE that initial context computation with `PaperEncoder`
     (identical to the one used by STTrans / PaperBaseline in this repo),
     producing a 512-dim context vector from (obs_traj, obs_Me, image_obs,
     env_data). This context is projected and used to (a) initialize the
     Generator-encoder's LSTM hidden state and (b) is concatenated at every
     decoder timestep, replacing MMSTN's "raw 4-D trajectory only" input.
  2. Per your instruction, we predict 2-D (lon, lat) ONLY -- pressure/wind
     ("Me") channels are dropped from both input and output, to match
     ST-Trans / PaperBaseline (also 2-D) for a fair comparison. The original
     4-D `spatial_embedding: Linear(4, ...)` / `hidden2pos: Linear(h, 4)`
     become Linear(2, ...) / Linear(h, 2).
  3. `seq_start_end`-based pooling (`PoolHiddenNet` / `SocialPooling`) is
     OMITTED. This matches MMSTN's own train.py defaults
     (`--pooling_type` defaults to `None`), so nothing is being weakened --
     the reference numbers everyone quotes from MMSTN's paper/repo were
     produced with pooling OFF. `pool_every_timestep` therefore is also off.
  4. Training loop is restructured from MMSTN's own iteration-count loop
     into the SAME epoch/early-stopping/CSV-logging/ADE-ATE-CTE-per-horizon
     framework used by train_st_trans.py / train_paper_baseline.py, so all
     four baselines produce directly comparable metrics.csv files. The GAN
     step mechanics themselves (d_steps/g_steps alternation, best_k, noise,
     label-smoothed BCE) are untouched.
"""

import random
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
#  Faithful helpers from mmstn/losses.py
# ══════════════════════════════════════════════════════════════════════════

def bce_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Numerically-stable BCE-with-logits, verbatim from mmstn/losses.py."""
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake: torch.Tensor) -> torch.Tensor:
    """Verbatim from mmstn/losses.py: random soft label in [0.7, 1.2]."""
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real: torch.Tensor, scores_fake: torch.Tensor) -> torch.Tensor:
    """Verbatim from mmstn/losses.py: random soft labels for real/fake."""
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


def l2_loss_masked(
    pred_traj: torch.Tensor,      # (seq_len, batch, 2)
    pred_traj_gt: torch.Tensor,   # (seq_len, batch, 2)
    loss_mask: torch.Tensor,      # (seq_len, batch) -- NOTE: this project's
                                   # seq_collate (Model/data/trajectoriesWithMe_
                                   # unet_training.py) produces mask_out with
                                   # shape (seq_len, batch), NOT (batch, seq_len)
                                   # as in the original MMSTN repo's own
                                   # dataset/collate. Semantics of l2_loss()
                                   # are otherwise preserved exactly (same
                                   # masked-MSE formula), just indexed on the
                                   # axis order this project's data actually
                                   # uses.
    mode: str = "raw",
) -> torch.Tensor:
    """Faithful to mmstn/losses.py l2_loss(), adapted for this project's
    (seq_len, batch) mask orientation instead of MMSTN's own (batch, seq_len)."""
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt - pred_traj) ** 2)       # (seq_len, batch, 2)
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == "raw":
        # per-sample scalar: sum over point-dim and time -> (batch,)
        return loss.sum(dim=2).sum(dim=0)
    raise ValueError(mode)


def get_noise(shape, noise_type: str, device) -> torch.Tensor:
    if noise_type == "gaussian":
        return torch.randn(*shape, device=device)
    elif noise_type == "uniform":
        return torch.rand(*shape, device=device).sub_(0.5).mul_(2.0)
    raise ValueError(f'Unrecognized noise type "{noise_type}"')


def relative_to_abs(rel_traj: torch.Tensor, start_pos: torch.Tensor) -> torch.Tensor:
    """torch.cumsum trick, verbatim from mmstn/utils.py."""
    return torch.cumsum(rel_traj, dim=0) + start_pos.unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════
#  Small LSTM "re-encoder" used INSIDE the decoder loop -- this reproduces
#  the (unusual, but original) `self.encoders(obs_traj_rel_new)` call inside
#  mmstn/models.py Decoder.forward(), which re-runs a fresh LSTM encode over
#  the growing relative-trajectory sequence at every decode step and uses
#  its final hidden state as the new decoder LSTM state.
# ══════════════════════════════════════════════════════════════════════════

class _RelEncoder(nn.Module):
    """Mirrors mmstn/models.py `Encoder` (the one instantiated as
    `self.encoders` inside `Decoder.__init__`), but with input dim = 2
    (lon, lat only) instead of the original 4 (lon, lat, pressure, wind).
    """

    def __init__(self, embedding_dim: int, h_dim: int, num_layers: int = 1):
        super().__init__()
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, h_dim, num_layers)

    def forward(self, traj_rel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # traj_rel: (T, B, 2)
        B = traj_rel.size(1)
        device = traj_rel.device
        emb = self.spatial_embedding(traj_rel.reshape(-1, 2))
        emb = emb.view(-1, B, self.embedding_dim)
        h0 = torch.zeros(self.num_layers, B, self.h_dim, device=device)
        c0 = torch.zeros(self.num_layers, B, self.h_dim, device=device)
        _, (h_n, c_n) = self.lstm(emb, (h0, c0))
        return h_n, c_n


# ══════════════════════════════════════════════════════════════════════════
#  Generator
# ══════════════════════════════════════════════════════════════════════════

class MMSTNGenerator(nn.Module):
    """
    Faithful re-implementation of mmstn/models.py::TrajectoryGenerator,
    with the trajectory-only `Encoder` replaced by PaperEncoder context
    injection, and restricted to 2-D (lon, lat) prediction.

    Original generator pipeline (kept):
      obs_traj_rel --Encoder(LSTM)--> final_h (context)
                    --[mlp_decoder_context]--> compressed context
                    --add_noise--> decoder_h0 (this is what makes MMSTN a GAN:
                                                different noise -> different
                                                plausible future tracks)
      decoder_h0, obs_traj[-1] --Decoder(autoregressive LSTM)--> pred_traj_rel

    Change: `final_h` (the LSTM encoder's output) is now obtained by fusing
    (a) `PaperEncoder`'s 512-d env/Data1d/Data3d context and (b) MMSTN's own
    kinematic LSTM encoder over obs_traj_rel, then projected the same way
    MMSTN projects its LSTM `final_h` into the decoder's initial state.
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        unet_in_ch: int = 13,
        embedding_dim: int = 32,
        encoder_h_dim: int = 64,
        decoder_h_dim: int = 64,
        num_layers: int = 1,
        noise_dim: Tuple[int, ...] = (16,),
        noise_type: str = "gaussian",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.num_layers = num_layers
        self.noise_dim = noise_dim
        self.noise_type = noise_type
        self.noise_first_dim = noise_dim[0] if noise_dim and noise_dim[0] > 0 else 0

        # ---- Shared multi-modal context encoder (replaces MMSTN's plain
        # trajectory-only Encoder as the SOURCE of context; identical module
        # used by STTrans / PaperBaseline for a fair comparison). ----------
        self.encoder = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)

        # ---- MMSTN's own kinematic LSTM encoder over obs_traj_rel (2-D),
        # kept because it is literally `mmstn.models.Encoder` -- we retain
        # it so the generator still "sees" fine-grained relative kinematics
        # the same way the original does, on top of the richer context. ---
        self.kin_encoder = _RelEncoder(embedding_dim, encoder_h_dim, num_layers)

        # Fuse PaperEncoder context (512-d) + kinematic LSTM h (encoder_h_dim)
        self.ctx_fuse = nn.Sequential(
            nn.Linear(PaperEncoder.RAW_CTX_DIM + encoder_h_dim, encoder_h_dim),
            nn.LayerNorm(encoder_h_dim),
            nn.ReLU(),
        )

        # mlp_decoder_context: encoder_h_dim -> (decoder_h_dim - noise_dim)
        # Present whenever noise_dim/pooling/dim-mismatch requires it --
        # MMSTN always needs it because noise_dim=(16,) by default.
        target_dim = decoder_h_dim - self.noise_first_dim
        self.mlp_decoder_context = nn.Sequential(
            nn.Linear(encoder_h_dim, 128),
            nn.ReLU(),
            nn.Linear(128, target_dim),
        )

        # ---- Decoder (autoregressive LSTM), faithful to mmstn Decoder ----
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.decoder_lstm = nn.LSTM(embedding_dim, decoder_h_dim, num_layers)
        self.hidden2pos = nn.Linear(decoder_h_dim, 2)
        # the odd "re-encode growing rel-traj every step" sub-module,
        # faithful to mmstn Decoder.__init__ self.encoders = Encoder(...)
        self.encoders = _RelEncoder(embedding_dim, decoder_h_dim, num_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def add_noise(self, ctx: torch.Tensor) -> torch.Tensor:
        """Per-pedestrian ("ped") noise mixing -- MMSTN default
        `noise_mix_type='ped'`: independent noise per sample in the batch.
        """
        if self.noise_first_dim == 0:
            return ctx
        B = ctx.size(0)
        z = get_noise((B, self.noise_first_dim), self.noise_type, ctx.device)
        return torch.cat([ctx, z], dim=1)

    def forward(self, batch_list) -> torch.Tensor:
        """
        batch_list indices (see Model/data/trajectoriesWithMe_unet_training.py
        seq_collate): 0=obs_traj(2D lon,lat) 2=obs_rel 7=obs_Me 11=img_obs
        13=env_data.
        Returns pred_traj_rel: (pred_len, B, 2)
        """
        obs_traj = batch_list[0]        # (T_obs, B, 2) abs, normalized
        obs_traj_rel = batch_list[2]    # (T_obs, B, 2) rel

        B = obs_traj.shape[1]
        device = obs_traj.device

        # 1) Multi-modal context (env + Data1d + Data3d), shared with
        #    ST-Trans / PaperBaseline.
        raw_ctx = self.encoder(batch_list)                      # (B, 512)

        # 2) MMSTN's own kinematic LSTM encoder over obs_traj_rel.
        kin_h, _ = self.kin_encoder(obs_traj_rel)                # (1, B, h)
        kin_h = kin_h.squeeze(0)                                 # (B, h)

        fused = self.ctx_fuse(torch.cat([raw_ctx, kin_h], dim=1))  # (B, enc_h)

        # 3) mlp_decoder_context + noise injection (the actual GAN part).
        dec_ctx = self.mlp_decoder_context(fused)                # (B, dec_h - noise)
        dec_ctx = self.add_noise(dec_ctx)                        # (B, dec_h)

        decoder_h = dec_ctx.unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
        decoder_c = torch.zeros(self.num_layers, B, self.decoder_h_dim, device=device)
        state = (decoder_h, decoder_c)

        last_pos = obs_traj[-1]           # (B, 2) abs
        last_pos_rel = obs_traj_rel[-1]   # (B, 2) rel

        decoder_input = self.spatial_embedding(last_pos_rel).view(1, B, self.embedding_dim)

        preds_rel = []
        obs_traj_rel_running = obs_traj_rel  # will grow each step, faithful to original

        for _ in range(self.pred_len):
            out, state = self.decoder_lstm(decoder_input, state)
            rel_pos = self.hidden2pos(out.view(-1, self.decoder_h_dim))  # (B, 2)
            curr_pos = rel_pos + last_pos

            # --- faithful re-encode step (mmstn Decoder.forward lines 171-176):
            # append the newly predicted rel step, then re-run a fresh LSTM
            # encode over the WHOLE growing relative trajectory, and use its
            # final hidden state as the new decoder state. ---
            rel_pos_t = rel_pos.unsqueeze(0)                       # (1, B, 2)
            obs_traj_rel_running = torch.cat([obs_traj_rel_running, rel_pos_t], dim=0)
            h_n, c_n = self.encoders(obs_traj_rel_running)
            state = (h_n, c_n)

            decoder_input = self.spatial_embedding(rel_pos_t.view(1, B, 2))
            preds_rel.append(rel_pos_t.view(B, 2))
            last_pos = curr_pos

        return torch.stack(preds_rel, dim=0)  # (pred_len, B, 2) relative


# ══════════════════════════════════════════════════════════════════════════
#  Discriminator -- faithful to mmstn/models.py TrajectoryDiscriminator,
#  d_type='local' (MMSTN train.py default), operating on 2-D (lon,lat).
# ══════════════════════════════════════════════════════════════════════════

class MMSTNDiscriminator(nn.Module):
    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        embedding_dim: int = 32,
        h_dim: int = 64,
        mlp_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.h_dim = h_dim
        self.encoder = _RelEncoder(embedding_dim, h_dim, num_layers)
        self.real_classifier = nn.Sequential(
            nn.Linear(h_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mlp_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, traj_rel: torch.Tensor) -> torch.Tensor:
        # traj_rel: (obs_len+pred_len, B, 2)
        h_n, _ = self.encoder(traj_rel)   # (1, B, h)
        final_h = h_n.squeeze(0)          # (B, h)
        scores = self.real_classifier(final_h)
        return scores.view(-1)


# ══════════════════════════════════════════════════════════════════════════
#  Top-level wrapper: bundles Generator + Discriminator, and exposes the
#  same get_loss_breakdown()/sample() interface as STTrans/PaperBaseline
#  so evaluate() / train loop code can stay uniform across all baselines.
#  NOTE: the actual GAN d_steps/g_steps alternation happens in
#  train_mmstn.py (it needs separate optimizers for G and D), NOT here.
# ══════════════════════════════════════════════════════════════════════════

class MMSTN(nn.Module):
    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        unet_in_ch: int = 13,
        embedding_dim: int = 32,
        encoder_h_dim_g: int = 64,
        decoder_h_dim_g: int = 64,
        encoder_h_dim_d: int = 64,
        mlp_dim: int = 128,
        num_layers: int = 1,
        noise_dim: Tuple[int, ...] = (16,),
        noise_type: str = "gaussian",
        dropout: float = 0.0,
        best_k: int = 6,
        l2_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.best_k = best_k
        self.l2_loss_weight = l2_loss_weight

        self.generator = MMSTNGenerator(
            obs_len=obs_len, pred_len=pred_len, unet_in_ch=unet_in_ch,
            embedding_dim=embedding_dim, encoder_h_dim=encoder_h_dim_g,
            decoder_h_dim=decoder_h_dim_g, num_layers=num_layers,
            noise_dim=noise_dim, noise_type=noise_type, dropout=dropout,
        )
        self.discriminator = MMSTNDiscriminator(
            obs_len=obs_len, pred_len=pred_len, embedding_dim=embedding_dim,
            h_dim=encoder_h_dim_d, mlp_dim=mlp_dim, num_layers=num_layers,
            dropout=dropout,
        )

    # -- convenience: run generator only (used by sample()/eval) ----------
    def forward(self, batch_list) -> torch.Tensor:
        obs_traj = batch_list[0]
        pred_traj_fake_rel = self.generator(batch_list)
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        return pred_traj_fake

    @torch.no_grad()
    def sample(self, batch_list, num_ensemble: int = 1, **kwargs
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniform interface with STTrans/PaperBaseline: returns
        (pred[T,B,2], me_mean[T,B,2]-zeros, all_trajs[1,T,B,2]).
        At eval time we use a SINGLE generator sample (noise still injected,
        since MMSTN's Generator always samples noise -- this mirrors how
        MMSTN's own check_accuracy() in train.py evaluates: one forward
        pass per batch, no explicit best-of-k at eval time).
        """
        pred = self.forward(batch_list)
        T, B, _ = pred.shape
        me_mean = torch.zeros(T, B, 2, device=pred.device)
        return pred, me_mean, pred.unsqueeze(0)

    # -- faithful GAN step losses, exposed as pure functions on this module
    #    so train_mmstn.py can call them without duplicating logic. --------

    def discriminator_step_loss(self, batch_list) -> Dict:
        """Mirrors mmstn train.py::discriminator_step (data_loss only;
        the (dead, commented-out in the original) triplet-loss branch is
        correctly omitted, as it is in the official code path).
        """
        obs_traj = batch_list[0]
        obs_traj_rel = batch_list[2]
        pred_traj_gt = batch_list[1]
        pred_traj_gt_rel = batch_list[3]

        with torch.no_grad():
            pred_traj_fake_rel = self.generator(batch_list)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

        scores_real = self.discriminator(traj_real_rel)
        scores_fake = self.discriminator(traj_fake_rel)

        d_loss = gan_d_loss(scores_real, scores_fake)
        return dict(total=d_loss, D_data_loss=d_loss.item())

    def generator_step_loss(self, batch_list) -> Dict:
        """Mirrors mmstn train.py::generator_step: best_k L2 (variety loss)
        + adversarial loss from discriminator, on freshly-sampled noise.
        """
        obs_traj = batch_list[0]
        obs_traj_rel = batch_list[2]
        pred_traj_gt = batch_list[1]
        pred_traj_gt_rel = batch_list[3]
        loss_mask_full = batch_list[5]          # (obs_len+pred_len, B) -- see
                                                   # l2_loss_masked docstring for
                                                   # why this axis order, not
                                                   # MMSTN's original (B, seq_len).
        loss_mask = loss_mask_full[self.obs_len:, :]  # (pred_len, B)

        B = obs_traj.shape[1]
        g_l2_loss_rel: List[torch.Tensor] = []
        pred_traj_fake_rel_last = None

        for _ in range(self.best_k):
            pred_traj_fake_rel = self.generator(batch_list)
            pred_traj_fake_rel_last = pred_traj_fake_rel
            if self.l2_loss_weight > 0:
                g_l2_loss_rel.append(
                    self.l2_loss_weight * l2_loss_masked(
                        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode="raw"
                    )
                )

        loss = pred_traj_gt.new_zeros(())
        g_l2_loss_sum_rel = pred_traj_gt.new_zeros(())
        if self.l2_loss_weight > 0:
            # (best_k, B) -> per-sample MIN over best_k samples (variety loss),
            # faithful to train.py generator_step (per seq_start_end group;
            # here each group has exactly 1 sequence, so this reduces to a
            # per-sample min, which is the correct specialization).
            stacked = torch.stack(g_l2_loss_rel, dim=1)              # (B, best_k)
            per_sample_min = torch.min(stacked, dim=1).values        # (B,)
            denom = loss_mask.sum(dim=0).clamp(min=1e-6)             # (B,) -- sum over time axis
            g_l2_loss_sum_rel = (per_sample_min / denom).sum()
            loss = loss + g_l2_loss_sum_rel

        pred_traj_fake = relative_to_abs(pred_traj_fake_rel_last, obs_traj[-1])
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel_last], dim=0)
        scores_fake = self.discriminator(traj_fake_rel)
        d_loss = gan_g_loss(scores_fake)
        loss = loss + d_loss

        with torch.no_grad():
            ade_m = compute_ade_per_horizon(pred_traj_fake.detach(), pred_traj_gt)
            atc_m = compute_ate_cte_per_horizon(pred_traj_fake.detach(), pred_traj_gt)

        out = dict(total=loss, G_l2_loss_rel=g_l2_loss_sum_rel.item(),
                   G_discriminator_loss=d_loss.item())
        out.update(ade_m)
        out.update(atc_m)
        return out

    # -- uniform loss-breakdown interface (used by evaluate()/val-loss code
    #    that is shared across baselines: reports G's variety+adv loss as
    #    "total" so val_dpe-style tracking still works the same way). ------
    def get_loss_breakdown(self, batch_list) -> Dict:
        return self.generator_step_loss(batch_list)

    def get_loss(self, batch_list) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list)["total"]