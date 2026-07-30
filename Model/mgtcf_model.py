"""
Model/mgtcf_model.py  -- MGTCF Multi-Generator GAN Baseline (v2, faithful port)
=================================================================================
THUAT TOAN GOC: Huang et al. (2023) "MGTCF: Multi-Generator Tropical
Cyclone Forecasting with Heterogeneous Meteorological Data", AAAI 2023,
va Huang et al. (2025) "Benchmark dataset and deep learning method for
global tropical cyclone forecasting" (TropiCycloneNet), Nat. Commun.

NGUON: code nay duoc VIET LAI TU CHINH source code goc cua tac gia
(TrajectoryGenerator/TrajectoryDiscriminator/Encoder/Decoder trong
models_prior_unet.py, cung train_github.py va losses.py, do nguoi dung
cung cap truc tiep), KHONG con la suy doan nhu ban v1. Moi thanh phan
duoi day duoc doi chieu TRUC TIEP tung dong voi code goc, sai khac
(neu co) duoc ghi chu ro rang.

KIEN TRUC (doi chieu voi code goc):
  Encoder (goc: class Encoder):
    - obs_traj_embedding = Linear(input_dim, embed_dim)(obs_traj)
    - obs_traj_embedding = obs_traj_embedding + img_embed_input  (CONG
      truc tiep anh da nhung vao track embedding, KHONG phai concat)
    - LSTM(embed_dim, h_dim, num_layers) -> final_h
    (code goc dung 2 Encoder rieng: 1 cho generator chinh dung anh du
    bao tuong lai tu Unet3D, 1 cho "encoder_env"/GC-Net dung anh quan
    trac that -- xem TrajectoryGenerator.forward)

  Decoder (goc: class Decoder):
    - LSTM stepwise, moi buoc: input = spatial_embedding(last_pos_rel)
      + decoder_img[step]  (CONG anh vao embedding vi tri, giong Encoder)
    - hidden2pos = Linear(h_dim, output_dim) -> rel_pos
    - curr_pos = rel_pos + last_pos  (autoregressive tren khong gian
      TUYET DOI, feed lai lam input cho buoc sau)

  TrajectoryGenerator (goc: class TrajectoryGenerator):
    - self.gs = ModuleList[Decoder x num_gs]  (K generator DOC LAP,
      MOI GENERATOR CO ENCODER RIENG -- khong share, dung 1
      state_tuple chung tu encoder chinh cho tat ca K generator, CHI
      khac o noise duoc them vao qua add_noise())
    - self.net_chooser = MLP(h_dim -> h_dim/2 -> h_dim/2 -> num_gs)
      (GC-Net: MLP 3 lop, KHONG PHAI Transformer/gi phuc tap, du input
      la dec_h_evn -- context TU MOT ENCODER RIENG dung anh QUAN TRAC
      THAT, khac voi context cua generator chinh dung anh DU BAO tu
      Unet3D)
    - forward(..., all_g_out: bool):
        all_g_out=True  -> chay TAT CA K generator (dung cho
                           net_chooser_step, KHONG sampling). Code goc
                           dung torch.no_grad() quanh mix_noise+cac
                           generator khi all_g_out=True, nghia la
                           GENERATOR KHONG duoc train o buoc nay, chi
                           GC-Net duoc train qua net_chooser_weights.
        all_g_out=False -> sample K_samples MAU qua GC-Net
                           (Categorical(logits=net_chooser_weights)),
                           MOI mau co the la generator KHAC nhau, dung
                           cho ca discriminator_step (1 mau) va
                           generator_step (best_k mau, Variety Loss)

  TrajectoryDiscriminator (goc: class TrajectoryDiscriminator):
    - Dung LAI class Encoder (share code, KHONG share weight) de encode
      CA traj that va traj gia (nhung 1 instance discriminator, khong
      phai 1 instance encoder rieng cho real/fake)
    - real_classifier = MLP(h_dim -> mlp_dim -> 1)  (KHONG co context
      rieng nao khac, chi dung final_h tu chinh LSTM encode trajectory)

3 LOSS (doi chieu train_github.py):
  discriminator_step: sample 1 mau qua GC-Net -> gan_d_loss (BCE that/gia)
  generator_step: sample best_k mau qua GC-Net -> Variety/Best-of-K L2
                   loss (MIN trong best_k) + gan_g_loss (danh lua D)
  net_chooser_step: chay TAT CA K generator (torch.no_grad cho generator,
                     grad CHI qua GC-Net) -> Winner-Takes-All + Cross-
                     Entropy (nhan = generator co L2 nho nhat)

CHIEN LUOC IMPLEMENT (giu tinh than "so sanh cong bang" da thong nhat):
  - GIU encoder cua ban (PaperEncoder-derived: FNO3D + Mamba + Env_net)
    THAY vi Unet3D/PredRNN + Encoder(LSTM) rieng cua code goc -- van
    la diem khac biet CO CHU DICH duy nhat, moi thanh phan KHAC deu
    bam sat dung code goc.
  - GIU 2 chieu output (lon,lat) thay vi 4 chieu (lon,lat,pressure,
    wind) -- theo dung quyet dinh cua nguoi dung, vi pressure/wind da
    duoc dua vao ENCODER lam context (qua obs_Me), khong can du bao
    lai qua Decoder, va giu tinh nhat quan voi 5 model khac trong
    nghien cuu deu chi du bao 2 chieu.
  - GIU DUNG K=6 generators (code goc: num_gs=6, khong phai 5 nhu ban
    v1 tu doan).
  - GIU DUNG 3-step training (discriminator_step / generator_step /
    net_chooser_step, moi buoc 1 optimizer.step() rieng) -- xem
    train_mgtcf.py.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from Model.paper_baseline_model import (
    PaperEncoder,
    _norm_to_deg,
    haversine_km,
    compute_ade_per_horizon,
    compute_ate_cte_per_horizon,
    HORIZON_STEPS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Encoder  (đúng class Encoder trong models_prior_unet.py gốc)
# ══════════════════════════════════════════════════════════════════════════════

class MGTCFEncoder(nn.Module):
    """
    Encode observed trajectory bằng LSTM, CỘNG (không phải concat) một
    embedding ảnh vào track embedding trước khi vào LSTM -- đúng dòng
    "obs_traj_embedding = obs_traj_embedding + img_embed_input" trong
    code gốc.
    """

    def __init__(self, input_dim: int = 2, embedding_dim: int = 64,
                 h_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.spatial_embedding = nn.Linear(input_dim, embedding_dim)

    def init_hidden(self, batch, device):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim, device=device),
            torch.zeros(self.num_layers, batch, self.h_dim, device=device),
        )

    def forward(self, obs_traj: torch.Tensor, img_embed_input: torch.Tensor) -> Dict:
        """
        obs_traj:        [T_obs, B, input_dim]
        img_embed_input: [T_obs, B, embedding_dim]
        -> {'final_h': (h_n, c_n), 'output': LSTM output}
        """
        batch = obs_traj.size(1)
        input_dim = obs_traj.size(2)
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, input_dim))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        obs_traj_embedding = obs_traj_embedding + img_embed_input

        state_tuple = self.init_hidden(batch, obs_traj.device)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        return {"final_h": state, "output": output}


# ══════════════════════════════════════════════════════════════════════════════
#  Decoder  (đúng class Decoder trong models_prior_unet.py gốc)
# ══════════════════════════════════════════════════════════════════════════════

class MGTCFDecoder(nn.Module):
    """
    LSTM decoder stepwise, autoregressive trên không gian TUYỆT ĐỐI
    (curr_pos = rel_pos + last_pos, feed lại làm input bước sau).
    """

    def __init__(self, seq_len: int, output_dim: int = 2, embedding_dim: int = 64,
                 h_dim: int = 128, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.spatial_embedding = nn.Linear(output_dim, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, output_dim)

    def forward(self, last_pos: torch.Tensor, state_tuple: Tuple[torch.Tensor, torch.Tensor],
                decoder_img: torch.Tensor, last_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        last_pos:    [B, output_dim]              vị trí tuyệt đối cuối cùng đã quan trắc
        state_tuple: (h, c) mỗi cái [num_layers, B, h_dim]
        decoder_img: [pred_len, B, embedding_dim]  ảnh đã nhúng cho từng bước dự báo
        last_img:    [B, embedding_dim]            ảnh nhúng của bước cuối obs
        -> pred_traj_fake_rel [seq_len, B, output_dim], final_h
        """
        batch = last_pos.size(0)
        decoder_input = self.spatial_embedding(torch.zeros_like(last_pos))
        decoder_input = decoder_input.view(-1, batch, self.embedding_dim)
        decoder_input = decoder_input + last_img.unsqueeze(0)

        pred_traj_fake_rel = []
        for i_step in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            rel_pos_unsq = rel_pos.unsqueeze(0)
            decoder_input = self.spatial_embedding(rel_pos)
            decoder_input = decoder_input.view(-1, batch, self.embedding_dim)
            decoder_input = decoder_input + decoder_img[i_step].unsqueeze(0)

            pred_traj_fake_rel.append(rel_pos_unsq.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


# ══════════════════════════════════════════════════════════════════════════════
#  Image branch: dùng FNO3DEncoder + interp thay Unet3D gốc (context cho
#  Encoder/Decoder, KHÔNG phải context tổng hợp của PaperEncoder)
# ══════════════════════════════════════════════════════════════════════════════

class _ImageEmbeddingBranch(nn.Module):
    """
    Thay thế vai trò `self.Unet` + `self.img_embedding` trong code gốc:
    code gốc dùng Unet3D để DỰ BÁO ảnh tương lai (image reconstruction,
    dùng cho image_loss + img_embedding cho decoder), rồi Linear(64*64,32)
    để nhúng mỗi frame ảnh thành vector 32-d, cộng vào track embedding.

    Ở đây, thay vì tự triển khai lại Unet3D (ngoài phạm vi bài toán --
    dự báo ảnh tương lai không phải mục tiêu chính của nghiên cứu này),
    ta dùng CHÍNH feature 3D đã pool theo thời gian từ FNO3DEncoder
    (giống PaperEncoder) làm "ảnh đã nhúng", lặp lại cho đủ pred_len
    bước (không dự báo ảnh tương lai thật, chỉ dùng ảnh quan trắc cuối
    cùng làm context cố định cho toàn bộ decoder — một xấp xỉ hợp lý
    khi không có Unet3D dự báo ảnh).
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, bot_pooled_TB128: torch.Tensor, pred_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        bot_pooled_TB128: [T_obs, B, 128]  (đã pool spatial, từ FNO3DEncoder)
        -> encoder_img [T_obs, B, embed_dim], decoder_img [pred_len, B, embed_dim]
        """
        encoder_img = self.proj(bot_pooled_TB128)         # [T_obs, B, embed_dim]
        last_frame = encoder_img[-1:].expand(pred_len, -1, -1)   # lặp lại frame cuối
        decoder_img = last_frame
        return encoder_img, decoder_img


# ══════════════════════════════════════════════════════════════════════════════
#  GC-Net (Generator Chooser Network) -- đúng self.net_chooser gốc
# ══════════════════════════════════════════════════════════════════════════════

class GCNet(nn.Module):
    """MLP 3 lớp, đúng cấu trúc self.net_chooser trong code gốc:
    Linear(h_dim, h_dim//2) -> ReLU -> Linear(h_dim//2, h_dim//2) -> ReLU
    -> Linear(h_dim//2, num_gs)."""

    def __init__(self, h_dim: int, n_generators: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(),
            nn.Linear(h_dim // 2, h_dim // 2),
            nn.ReLU(),
            nn.Linear(h_dim // 2, n_generators),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ══════════════════════════════════════════════════════════════════════════════
#  MGTCF Generator  (đúng class TrajectoryGenerator gốc)
# ══════════════════════════════════════════════════════════════════════════════

class MGTCFGenerator(nn.Module):
    """
    Multi-generator trajectory generator. GIỮ encoder chung (PaperEncoder)
    thay Unet3D/PredRNN gốc; mọi phần còn lại (Encoder, K Decoder, GC-Net,
    noise injection) bám sát code gốc.
    """

    def __init__(
        self,
        obs_len:      int   = 8,
        pred_len:     int   = 12,
        unet_in_ch:   int   = 13,
        n_generators: int   = 6,      # code gốc: num_gs=6
        embedding_dim: int  = 64,
        encoder_h_dim: int  = 64,
        decoder_h_dim: int  = 128,
        noise_dim:    int   = 8,
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.obs_len       = obs_len
        self.pred_len      = pred_len
        self.n_generators  = n_generators
        self.embedding_dim = embedding_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.noise_dim      = noise_dim

        # ── Shared encoder (thay Unet3D+PredRNN gốc) ────────────────────────
        self.encoder = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
        self.image_branch = _ImageEmbeddingBranch(embed_dim=embedding_dim)

        # ── Encoder chính (dùng ảnh "dự báo", ở đây = ảnh quan trắc lặp lại) ─
        self.track_encoder = MGTCFEncoder(
            input_dim=2, embedding_dim=embedding_dim,
            h_dim=encoder_h_dim, num_layers=1, dropout=dropout)

        # ── Encoder cho GC-Net (dùng ảnh quan trắc THẬT, đúng code gốc
        #    encoder_env dùng img_embedding_real) ────────────────────────────
        self.track_encoder_chooser = MGTCFEncoder(
            input_dim=2, embedding_dim=embedding_dim,
            h_dim=encoder_h_dim, num_layers=1, dropout=dropout)

        # feature2dech / feature2dech_env: kết hợp final_h với raw_ctx
        # (thay vai trò evn_feature_chooser trong code gốc)
        self.feature2dech     = nn.Linear(encoder_h_dim + PaperEncoder.RAW_CTX_DIM, decoder_h_dim)
        self.feature2dech_env = nn.Linear(encoder_h_dim + PaperEncoder.RAW_CTX_DIM, decoder_h_dim)

        # ── K generator độc lập (self.gs) ────────────────────────────────────
        self.gs = nn.ModuleList([
            MGTCFDecoder(seq_len=pred_len, output_dim=2, embedding_dim=embedding_dim,
                        h_dim=decoder_h_dim, num_layers=1, dropout=dropout)
            for _ in range(n_generators)
        ])

        # ── GC-Net ────────────────────────────────────────────────────────────
        self.net_chooser = GCNet(h_dim=decoder_h_dim, n_generators=n_generators)

        # ── Noise projection (mlp_decoder_context, đúng mlp_decoder_needed) ───
        self.mlp_decoder_context = nn.Sequential(
            nn.Linear(decoder_h_dim, decoder_h_dim - noise_dim),
        )

    def _encode_context(self, batch_list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        -> raw_ctx [B, 512], bot_pooled [T_obs, B, 128], obs_traj [T_obs, B, 2]
        """
        obs_traj = batch_list[0]
        raw_ctx = self.encoder(batch_list)   # [B, 512] -- dùng lại toàn bộ pipeline chung

        image_obs = batch_list[11]
        if image_obs.dim() == 4:
            image_obs = image_obs.unsqueeze(2)
        expected_ch = self.encoder.spatial_enc.in_channel
        if image_obs.shape[1] == 1 and expected_ch != 1:
            image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)
        bot, _ = self.encoder.spatial_enc.encode(image_obs)   # [B, 128, T, 4, 4]
        bot_pooled = F.adaptive_avg_pool3d(bot, (None, 1, 1)).squeeze(-1).squeeze(-1)
        bot_pooled = bot_pooled.permute(2, 0, 1)   # [T, B, 128]
        T_bot = bot_pooled.shape[0]
        T_obs = obs_traj.shape[0]
        if T_bot != T_obs:
            bot_pooled = F.interpolate(
                bot_pooled.permute(1, 2, 0), size=T_obs,
                mode="linear", align_corners=False).permute(2, 0, 1)

        return raw_ctx, bot_pooled, obs_traj

    def _add_noise(self, h: torch.Tensor) -> torch.Tensor:
        """Đúng add_noise() gốc (noise_mix_type='ped', mỗi sample 1 noise riêng)."""
        B = h.shape[0]
        device = h.device
        z = torch.randn(B, self.noise_dim, device=device)
        return torch.cat([h, z], dim=1)

    def _get_samples(self, dec_h_chooser: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, np.ndarray]:
        """Đúng get_samples() gốc: Categorical sampling từ GC-Net logits."""
        net_chooser_out = self.net_chooser(dec_h_chooser)
        dist = Categorical(logits=net_chooser_out)
        sampled_gen_idxs = dist.sample((num_samples,)).transpose(0, 1)   # [B, num_samples]
        return net_chooser_out, sampled_gen_idxs.detach().cpu().numpy()

    def forward(
        self,
        batch_list,
        num_samples: int = 1,
        all_g_out: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        all_g_out=True:  chạy TẤT CẢ K generator (dùng cho net_chooser_step).
            Generator chạy dưới torch.no_grad() (đúng code gốc — GC-Net
            được train ở bước này, KHÔNG phải generator).
            -> pred_traj_fake_rel_nums [pred_len, K, B, 2], net_chooser_out [B, K]

        all_g_out=False: sample `num_samples` mẫu qua GC-Net Categorical
            sampling, mỗi mẫu có thể dùng generator KHÁC nhau.
            -> pred_traj_fake_rel_nums [pred_len, num_samples, B, 2], net_chooser_out [B, K]
        """
        obs_traj = batch_list[0]
        B = obs_traj.shape[1]
        device = obs_traj.device

        raw_ctx, bot_pooled, _ = self._encode_context(batch_list)
        encoder_img, decoder_img = self.image_branch(bot_pooled, self.pred_len)
        last_img = encoder_img[-1]   # [B, embedding_dim]

        # ── Encoder chính (cho generator) ────────────────────────────────────
        final_encoder = self.track_encoder(obs_traj, encoder_img)
        final_encoder_h = final_encoder["final_h"][0][-1]   # [B, encoder_h_dim] (last layer)
        dec_h = self.feature2dech(torch.cat([final_encoder_h, raw_ctx], dim=1)).unsqueeze(0)

        # ── Encoder riêng cho GC-Net (dùng ảnh quan trắc thật -- ở đây
        #    dùng cùng encoder_img vì không có "ảnh dự báo" riêng biệt) ──────
        final_encoder_chooser = self.track_encoder_chooser(obs_traj, encoder_img)
        final_encoder_chooser_h = final_encoder_chooser["final_h"][0][-1]
        dec_h_chooser = self.feature2dech_env(
            torch.cat([final_encoder_chooser_h, raw_ctx], dim=1))

        last_pos = obs_traj[-1]   # [B, 2]

        if all_g_out:
            preds_rel = []
            with torch.no_grad():
                noise_input = self.mlp_decoder_context(dec_h.squeeze(0))
                decoder_h = self._add_noise(noise_input).unsqueeze(0)
                decoder_c = torch.zeros(1, B, self.decoder_h_dim, device=device)
                for g in self.gs:
                    state_tuple = (decoder_h.clone(), decoder_c.clone())
                    pred_traj_fake_rel, _ = g(last_pos, state_tuple, decoder_img, last_img)
                    preds_rel.append(pred_traj_fake_rel.unsqueeze(1))
            pred_traj_fake_rel_nums = torch.cat(preds_rel, dim=1)   # [pred_len, K, B, 2]
            net_chooser_out, sampled_gen_idxs = self._get_samples(dec_h_chooser, num_samples)
            return pred_traj_fake_rel_nums, net_chooser_out, sampled_gen_idxs

        else:
            with torch.no_grad():
                net_chooser_out, sampled_gen_idxs = self._get_samples(dec_h_chooser, num_samples)

            preds_rel = []
            for sample_i in range(num_samples):
                pred_traj_fake_rel_sample = torch.zeros(self.pred_len, B, 2, device=device)
                gs_index = sampled_gen_idxs[:, sample_i]
                for g_i in range(self.n_generators):
                    now_data_index = (gs_index == g_i)
                    n_sel = int(now_data_index.sum())
                    if n_sel < 1:
                        continue
                    idx_t = torch.from_numpy(np.where(now_data_index)[0]).to(device)

                    noise_input = self.mlp_decoder_context(dec_h.squeeze(0)[idx_t])
                    decoder_h_g = self._add_noise(noise_input).unsqueeze(0)
                    decoder_c_g = torch.zeros(1, n_sel, self.decoder_h_dim, device=device)

                    decoder = self.gs[g_i]
                    pred_traj_fake_rel, _ = decoder(
                        last_pos[idx_t], (decoder_h_g, decoder_c_g),
                        decoder_img[:, idx_t], last_img[idx_t])
                    pred_traj_fake_rel_sample[:, idx_t, :] = pred_traj_fake_rel

                preds_rel.append(pred_traj_fake_rel_sample.unsqueeze(1))
            pred_traj_fake_rel_nums = torch.cat(preds_rel, dim=1)   # [pred_len, num_samples, B, 2]
            return pred_traj_fake_rel_nums, net_chooser_out, sampled_gen_idxs


# ══════════════════════════════════════════════════════════════════════════════
#  MGTCF Discriminator  (đúng class TrajectoryDiscriminator gốc)
# ══════════════════════════════════════════════════════════════════════════════

class MGTCFDiscriminator(nn.Module):
    def __init__(self, embedding_dim: int = 64, h_dim: int = 64,
                 mlp_dim: int = 256, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.h_dim = h_dim
        self.encoder = MGTCFEncoder(
            input_dim=2, embedding_dim=embedding_dim,
            h_dim=h_dim, num_layers=num_layers, dropout=dropout)
        self.real_classifier = nn.Sequential(
            nn.Linear(h_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, 1))

    def forward(self, traj: torch.Tensor, img_embed: torch.Tensor) -> torch.Tensor:
        """
        traj:      [T, B, 2]  (obs concatenated với pred)
        img_embed: [T, B, embedding_dim]
        -> scores [B, 1]
        """
        final_h = self.encoder(traj, img_embed)
        final_h = final_h["final_h"][0][-1]     # [B, h_dim]
        return self.real_classifier(final_h)


# ══════════════════════════════════════════════════════════════════════════════
#  MGTCF Model  -- wrapper tổng hợp, interface nhất quán với các baseline khác
# ══════════════════════════════════════════════════════════════════════════════

class MGTCFModel(nn.Module):
    """
    Wrapper kết hợp MGTCFGenerator + MGTCFDiscriminator, cung cấp
    get_loss_breakdown()/sample() nhất quán với LSTM/GRU/RNN/ST-Trans/FM,
    nhưng training THẬT phải dùng riêng discriminator_step/generator_step/
    net_chooser_step (xem train_mgtcf.py) -- không thể gộp thành 1
    optimizer.step() duy nhất như các baseline khác, vì đây là bản chất
    của GAN 3-pha (đúng train_github.py gốc).
    """

    def __init__(
        self,
        obs_len:      int   = 8,
        pred_len:     int   = 12,
        unet_in_ch:   int   = 13,
        n_generators: int   = 6,      # code gốc: num_gs=6
        embedding_dim: int  = 64,
        encoder_h_dim: int  = 64,
        decoder_h_dim: int  = 128,
        noise_dim:    int   = 8,
        disc_h_dim:   int   = 64,
        disc_mlp_dim: int   = 256,
        dropout:      float = 0.0,
        best_k:       int   = 6,      # code gốc: args.best_k cho Variety Loss
    ):
        super().__init__()
        self.obs_len      = obs_len
        self.pred_len     = pred_len
        self.n_generators = n_generators
        self.best_k        = best_k

        self.generator = MGTCFGenerator(
            obs_len=obs_len, pred_len=pred_len, unet_in_ch=unet_in_ch,
            n_generators=n_generators, embedding_dim=embedding_dim,
            encoder_h_dim=encoder_h_dim, decoder_h_dim=decoder_h_dim,
            noise_dim=noise_dim, dropout=dropout,
        )
        self.discriminator = MGTCFDiscriminator(
            embedding_dim=embedding_dim, h_dim=disc_h_dim,
            mlp_dim=disc_mlp_dim, num_layers=1, dropout=dropout,
        )

    # ── Forward: 1 quỹ đạo duy nhất (dùng cho .forward() interface chung) ────

    def forward(self, batch_list) -> torch.Tensor:
        pred, _, _ = self.generator(batch_list, num_samples=1, all_g_out=False)
        return pred[:, 0, :, :]   # [pred_len, B, 2]

    # ── Loss breakdown (CHỈ DÙNG ĐỂ LOG/DIAGNOSTIC — training thật dùng
    #    3 hàm riêng trong train_mgtcf.py, KHÔNG dùng "total" của hàm này
    #    để backward trực tiếp như các baseline khác) ─────────────────────

    def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
        obs_traj = batch_list[0]
        traj_gt  = batch_list[1]
        T = min(self.pred_len, traj_gt.shape[0])
        gt = traj_gt[:T]

        with torch.no_grad():
            pred, _, _ = self.generator(batch_list, num_samples=1, all_g_out=False)
            pred = pred[:, 0, :T, :]
            pred_deg = _norm_to_deg(pred)
            gt_deg = _norm_to_deg(gt)
            l_dpe = haversine_km(pred_deg, gt_deg).mean()
            ade_m = compute_ade_per_horizon(pred.detach(), traj_gt)
            atc_m = compute_ate_cte_per_horizon(pred.detach(), traj_gt)

        return dict(total=l_dpe, dpe=l_dpe.item(), **ade_m, **atc_m)

    def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

    # ── Inference: best-of-K sampling (đúng chuẩn MGTCF-Ens/stochastic) ──────

    @torch.no_grad()
    def sample(
        self,
        batch_list,
        num_ensemble: int = 20,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample `num_ensemble` mẫu qua GC-Net Categorical sampling (đúng
        cơ chế Roulette thật của paper), chọn mẫu tốt nhất bằng
        ground-truth nếu có (giống "best of K generated trajectories"
        mà paper báo cáo), hoặc bằng độ mượt nếu không có ground-truth
        (inference thật, không leak label).
        """
        obs_traj = batch_list[0]
        B = obs_traj.shape[1]
        device = obs_traj.device

        pred, _, _ = self.generator(batch_list, num_samples=num_ensemble, all_g_out=False)
        # pred: [pred_len, num_ensemble, B, 2]
        all_trajs = pred.permute(1, 0, 2, 3)   # [num_ensemble, pred_len, B, 2]

        traj_gt = batch_list[1] if len(batch_list) > 1 and batch_list[1] is not None else None
        if traj_gt is not None and num_ensemble > 1:
            T = min(self.pred_len, traj_gt.shape[0])
            gt_deg = _norm_to_deg(traj_gt[:T])
            scores = []
            for k in range(num_ensemble):
                pred_k_deg = _norm_to_deg(all_trajs[k, :T])
                scores.append(haversine_km(pred_k_deg, gt_deg).mean(dim=0))
            scores = torch.stack(scores, dim=0)
            best_idx = scores.argmin(dim=0)
        else:
            smoothness = []
            for k in range(num_ensemble):
                traj = all_trajs[k]
                if traj.shape[0] >= 3:
                    vel = traj[1:] - traj[:-1]
                    accel = (vel[1:] - vel[:-1]).norm(dim=-1).mean(dim=0)
                else:
                    accel = torch.zeros(B, device=device)
                smoothness.append(accel)
            smoothness = torch.stack(smoothness, dim=0)
            best_idx = smoothness.argmin(dim=0)

        pred_mean = torch.stack(
            [all_trajs[best_idx[b], :, b, :] for b in range(B)], dim=1
        )
        me_mean = torch.zeros_like(pred_mean)
        return pred_mean, me_mean, all_trajs