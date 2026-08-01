"""
Model/physdiff_model.py  -- Phys-Diff Latent Diffusion Baseline
====================================================================
THUAT TOAN GOC: Liu, Yu, Chen, Huang, Liu, Zhao, Li (2026) "Phys-Diff:
A Physics-Inspired Latent Diffusion Model for Tropical Cyclone
Forecasting", ICASSP 2026 (arXiv:2603.00521). Code cong khai:
https://github.com/USTC-AI4EEE/Phys-Diff (MIT license).

NGUON: implement TRUC TIEP tu cong thuc Eq.(1)-(9) trong paper (Section
2.3-2.4), doi chieu voi log training THAT tu README repo (xac nhan dung
4 thanh phan loss: Diff, Coord, MSW, MLSP -- khop chinh xac L_diffusion
+ L_traj + L_wind + L_pres).

KIEN TRUC GOC (paper Section 2.3):
  Bai toan: H = {h_1..h_M}, h_i = (x_i, v_i, p_i)  (toa do, gio, ap suat)
  du bao F = {f_1..f_N} tuong lai.

  Latent Diffusion: encoder conv E anh xa x_0 (chuoi tuong lai) ->
    latent z_0 = E(x_0); hoc reverse diffusion process trong khong gian
    latent nay; decoder D anh xa z_0_hat -> x_0_hat.

  Forward process (Eq.1): q(z_t|z_0) = N(z_t; sqrt(alpha_bar_t) z_0,
                                          (1-alpha_bar_t) I)
  Reverse process (Eq.2): z_{t-1} = 1/sqrt(alpha_t) *
    (z_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta(z_t,t,c)) + sigma_t*w

  Conditional Denoising Network eps_theta = Conditional Encoder +
  Physics-Inspired Decoder:

    Conditional Encoder (Eq.3):
      c = TransformerEncoder([GRU(H), SwinTransformer([E_hist,E_fut]), t_emb])
      -- H_TC tu GRU(lich su), T_env tu Swin Transformer (anh moi truong).

    Physics-Inspired Decoder:
      Moi decoder block: self-attn(z_t) -> cross-attn(., c) -> PIGA -> FFN
      PIGA module (Eq.4-8):
        1. Decomposition: X_cross -> f_traj, f_wind, f_pres (3 nhanh rieng)
        2. Interaction: moi nhanh attend 2 nhanh con lai (cross-task attn)
           A_traj = Attention(Q=f_traj, K,V=[f_wind,f_pres])
        3. Gating: g_traj = sigmoid(MLP([f_traj, A_traj]))
                   f'_traj = (1-g_traj)*f_traj + g_traj*A_traj
        4. Fusion: X_PIGA = Conv1x1(Concat(f'_traj, f'_wind, f'_pres))

  Training Objective (Eq.9, uncertainty-weighted multi-task, Kendall
  et al. 2018):
    L_total = 1/(2*sigma_diff^2) * L_diffusion
            + 1/(2*sigma_recon^2) * L_recon
            + log(sigma_diff * sigma_recon)
    L_diffusion = E[||eps - eps_theta(z_t,t,c)||^2]
    L_recon = L_traj + L_wind + L_pres  (task-specific gradient routing:
              moi thanh phan CHI cap nhat projection layer tuong ung
              trong PIGA, dam bao feature disentanglement)

CHIEN LUOC IMPLEMENT (giu tinh than "so sanh cong bang" da thong nhat):
  - GIU encoder cua ban (PaperEncoder: FNO3D + Mamba + Env_net) THAY vi
    [GRU(H) + SwinTransformer(ERA5+FengWu)] cua paper goc -- diem khac
    biet CO CHU DICH duy nhat, giong het cach lam voi MGTCF/ST-Trans.
  - GIU DUNG latent diffusion (Eq.1-2), PIGA module (Eq.4-8), va
    uncertainty-weighted loss (Eq.9) -- day la 3 dong gop cot loi cua
    paper, bam sat cong thuc 100%.
  - GIU 3 nhanh du bao (trajectory + wind + pressure) NHU CODE GOC (vi
    PIGA module VON DI hoat dong tren su tuong tac giua 3 attribute --
    bo di 1 nhanh se pha vo chinh co che cross-task attention ma paper
    de xuat). Wind/pressure gia tri lay tu obs_Me (da co san trong
    pipeline du lieu), pred wind/pressure dung de tinh loss phu (khong
    bat buoc bao cao trong bang ket qua chinh cua ban, chi de giu dung
    kien truc PIGA hoat dong nhu thiet ke).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.paper_baseline_model import (
    PaperEncoder,
    _norm_to_deg,
    haversine_km,
    compute_ade_per_horizon,
    compute_ate_cte_per_horizon,
    HORIZON_STEPS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Latent Encoder/Decoder (E, D trong paper — nén chuỗi tương lai vào
#  không gian latent trước khi diffuse, đúng "Latent Diffusion" tinh thần)
# ══════════════════════════════════════════════════════════════════════════════

class LatentCodec(nn.Module):
    """
    Encoder E: x_0 [B, N, 4] (lon,lat,wind,pres) -> z_0 [B, N, D_embed]
    Decoder D: z_0_hat [B, N, D_embed] -> x_0_hat [B, N, 4]
    """

    def __init__(self, attr_dim: int = 4, d_embed: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(attr_dim, d_embed), nn.GELU(), nn.Linear(d_embed, d_embed))
        self.decoder = nn.Sequential(
            nn.Linear(d_embed, d_embed), nn.GELU(), nn.Linear(d_embed, attr_dim))

    def encode(self, x0: torch.Tensor) -> torch.Tensor:
        return self.encoder(x0)

    def decode(self, z0_hat: torch.Tensor) -> torch.Tensor:
        return self.decoder(z0_hat)


# ══════════════════════════════════════════════════════════════════════════════
#  Sinusoidal timestep embedding
# ══════════════════════════════════════════════════════════════════════════════

class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor, T_max: int) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float)
                         * (-math.log(10000.0) / max(half - 1, 1)))
        t_norm = t.float() / T_max * 1000.0
        emb = t_norm.unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ══════════════════════════════════════════════════════════════════════════════
#  Conditional Encoder (Eq. 3): c = TransformerEncoder([GRU(H), Swin(env), t_emb])
#  -- thay GRU(H) + Swin(env) bằng PaperEncoder theo quyết định đã thống nhất
# ══════════════════════════════════════════════════════════════════════════════

class ConditionalEncoder(nn.Module):
    def __init__(self, obs_len: int = 8, unet_in_ch: int = 13, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Thay [GRU(H), SwinTransformer([E_hist,E_fut])] bằng PaperEncoder
        self.paper_encoder = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
        self.ctx_proj = nn.Sequential(
            nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model), nn.LayerNorm(d_model))

        self.time_emb = TimestepEmbedding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def encode_static(self, batch_list) -> torch.Tensor:
        """
        [TỐI ƯU TỐC ĐỘ, không đổi công thức] Phần KHÔNG phụ thuộc t: chạy
        PaperEncoder (FNO3D+Mamba+Env_net, khá nặng) đúng 1 LẦN, dùng lại
        cho MỌI bước t trong reverse process (1000 bước) thay vì tính lại
        từ đầu mỗi bước -- vì batch_list (input quan trắc) không đổi khi
        t thay đổi, việc tính lại là lãng phí thuần tuý, KHÔNG ảnh hưởng
        kết quả toán học (đo thực nghiệm: ~0.56s/lần gọi PaperEncoder,
        không cache sẽ tốn thêm ~560s/sample chỉ riêng phần này).
        -> ctx_token [B, 1, d_model]
        """
        raw_ctx = self.paper_encoder(batch_list)              # [B, 512]
        return self.ctx_proj(raw_ctx).unsqueeze(1)             # [B, 1, d_model]

    def forward_with_ctx(self, ctx_token: torch.Tensor, t: torch.Tensor, T_max: int) -> torch.Tensor:
        """
        Phần PHỤ THUỘC t: nhận ctx_token đã cache từ encode_static(),
        chỉ tính lại timestep embedding + TransformerEncoder trộn 2 token
        -- đúng Eq.3, chỉ khác cách tổ chức lời gọi, không đổi phép tính.
        -> c [B, 2, d_model]
        """
        t_token = self.time_emb(t, T_max).unsqueeze(1)          # [B, 1, d_model]
        tokens = torch.cat([ctx_token, t_token], dim=1)        # [B, 2, d_model]
        c = self.transformer_encoder(tokens)                     # [B, 2, d_model]
        return c

    def forward(self, batch_list, t: torch.Tensor, T_max: int) -> torch.Tensor:
        """
        -> c [B, 2, d_model]  (context token từ PaperEncoder + timestep token,
                                "rich interaction between all conditioning
                                variables" qua TransformerEncoder, đúng Eq.3)

        [GIỮ NGUYÊN, dùng cho get_loss_breakdown() -- chỉ gọi 1 lần/batch
        nên không cần tối ưu cache]. Về mặt TOÁN HỌC, forward(bl, t, T_max)
        == forward_with_ctx(encode_static(bl), t, T_max) — hoàn toàn tương
        đương, đã verify bằng test số học (xem test_physdiff_cache_equiv).
        """
        ctx_token = self.encode_static(batch_list)
        return self.forward_with_ctx(ctx_token, t, T_max)


# ══════════════════════════════════════════════════════════════════════════════
#  PIGA Module (Physics-Inspired Gated Attention) — Eq. 4-8
# ══════════════════════════════════════════════════════════════════════════════

class PIGAModule(nn.Module):
    """
    Decompose X_cross vào 3 nhánh task-specific (trajectory, wind,
    pressure), cho mỗi nhánh attend tới 2 nhánh còn lại (cross-task
    attention), gate để cân bằng feature gốc/feature đã tương tác,
    fusion qua Conv1x1 -- đúng Eq. 4-8 trong paper.
    """

    def __init__(self, d_model: int, d_sub: Optional[int] = None, num_heads: int = 2):
        super().__init__()
        self.d_model = d_model
        # d_sub PHẢI chia hết cho num_heads (yêu cầu của nn.MultiheadAttention).
        # Mặc định d_sub = d_model // 3 (chia đều cho 3 nhánh task-specific,
        # đúng tinh thần Eq.4), nhưng làm tròn xuống bội số gần nhất của
        # num_heads để tránh lỗi "embed_dim must be divisible by num_heads"
        # với các giá trị d_model không chia hết đẹp cho 3*num_heads (ví dụ
        # d_model=64 → 64//3=21, không chia hết cho num_heads=2).
        raw_d_sub = d_sub or (d_model // 3)
        self.d_sub = max(num_heads, (raw_d_sub // num_heads) * num_heads)
        self.num_heads = num_heads

        self.proj_traj = nn.Linear(d_model, self.d_sub)
        self.proj_wind = nn.Linear(d_model, self.d_sub)
        self.proj_pres = nn.Linear(d_model, self.d_sub)

        # Cross-task attention: mỗi nhánh có 1 nn.MultiheadAttention riêng
        # (Q từ chính nó, K/V từ 2 nhánh còn lại nối lại)
        self.attn_traj = nn.MultiheadAttention(self.d_sub, num_heads=self.num_heads, batch_first=True)
        self.attn_wind = nn.MultiheadAttention(self.d_sub, num_heads=self.num_heads, batch_first=True)
        self.attn_pres = nn.MultiheadAttention(self.d_sub, num_heads=self.num_heads, batch_first=True)

        self.gate_traj = nn.Sequential(nn.Linear(self.d_sub * 2, self.d_sub), nn.Sigmoid())
        self.gate_wind = nn.Sequential(nn.Linear(self.d_sub * 2, self.d_sub), nn.Sigmoid())
        self.gate_pres = nn.Sequential(nn.Linear(self.d_sub * 2, self.d_sub), nn.Sigmoid())

        self.fusion = nn.Conv1d(self.d_sub * 3, d_model, kernel_size=1)

    def forward(self, x_cross: torch.Tensor) -> torch.Tensor:
        """x_cross [B, N, d_model] -> X_PIGA [B, N, d_model]"""
        # Eq.4: Decomposition
        f_traj = self.proj_traj(x_cross)   # [B, N, d_sub]
        f_wind = self.proj_wind(x_cross)
        f_pres = self.proj_pres(x_cross)

        # Eq.5: Interaction -- mỗi nhánh attend 2 nhánh còn lại
        kv_traj = torch.cat([f_wind, f_pres], dim=1)   # [B, 2N, d_sub]
        kv_wind = torch.cat([f_traj, f_pres], dim=1)
        kv_pres = torch.cat([f_traj, f_wind], dim=1)

        A_traj, _ = self.attn_traj(f_traj, kv_traj, kv_traj)
        A_wind, _ = self.attn_wind(f_wind, kv_wind, kv_wind)
        A_pres, _ = self.attn_pres(f_pres, kv_pres, kv_pres)

        # Eq.6-7: Gating
        g_traj = self.gate_traj(torch.cat([f_traj, A_traj], dim=-1))
        g_wind = self.gate_wind(torch.cat([f_wind, A_wind], dim=-1))
        g_pres = self.gate_pres(torch.cat([f_pres, A_pres], dim=-1))

        f_traj_p = (1 - g_traj) * f_traj + g_traj * A_traj
        f_wind_p = (1 - g_wind) * f_wind + g_wind * A_wind
        f_pres_p = (1 - g_pres) * f_pres + g_pres * A_pres

        # Eq.8: Fusion
        concat = torch.cat([f_traj_p, f_wind_p, f_pres_p], dim=-1)   # [B, N, 3*d_sub]
        x_piga = self.fusion(concat.permute(0, 2, 1)).permute(0, 2, 1)   # [B, N, d_model]

        return x_piga, (f_traj_p, f_wind_p, f_pres_p)


# ══════════════════════════════════════════════════════════════════════════════
#  Physics-Inspired Decoder block (self-attn -> cross-attn -> PIGA -> FFN)
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsDecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.piga = PIGAModule(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model))

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        # Self-attention
        h = self.norm1(z)
        sa, _ = self.self_attn(h, h, h)
        z = z + self.dropout(sa)

        # Cross-attention với context c
        h = self.norm2(z)
        ca, _ = self.cross_attn(h, c, c)
        x_cross = z + self.dropout(ca)

        # PIGA module (đúng vị trí "sau cross-attention layer", theo paper)
        h = self.norm3(x_cross)
        x_piga, task_feats = self.piga(h)
        z = x_cross + self.dropout(x_piga)

        # Feed-forward
        h = self.norm4(z)
        z = z + self.dropout(self.ffn(h))

        return z, task_feats


# ══════════════════════════════════════════════════════════════════════════════
#  Physics-Inspired Denoising Network eps_theta(z_t, t, c)
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsInspiredDenoiser(nn.Module):
    def __init__(self, pred_len: int = 12, d_model: int = 64, nhead: int = 4,
                 num_blocks: int = 3, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model

        self.latent_pos_emb = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            PhysicsDecoderBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_blocks)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, z_t: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        z_t: [B, N, d_model]  latent nhiễu (đã encode qua LatentCodec.encode
             rồi thêm noise theo Eq.1 -- xem PhysDiffModel.get_loss_breakdown)
        c:   [B, 2, d_model]  context từ ConditionalEncoder (Eq.3)
        -> eps_pred [B, N, d_model], danh sách task_feats mỗi block (dùng
           cho task-specific gradient routing, xem get_loss_breakdown)
        """
        z = z_t + self.latent_pos_emb
        all_task_feats = []
        for block in self.blocks:
            z, task_feats = block(z, c)
            all_task_feats.append(task_feats)
        eps_pred = self.out_norm(z)
        return eps_pred, all_task_feats


# ══════════════════════════════════════════════════════════════════════════════
#  DDPM schedule (fixed linear beta, giống DDPMSchedule đã dùng cho LT3P
#  trước đây nhưng nay áp dụng cho KHÔNG GIAN LATENT thay vì toạ độ thô)
# ══════════════════════════════════════════════════════════════════════════════

class LatentDiffusionSchedule:
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.T = T
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - torch.cat([alphas_cumprod.new_ones(1), alphas_cumprod[:-1]])) / (1.0 - alphas_cumprod)

    def to(self, device):
        for k in ("betas", "alphas", "alphas_cumprod", "sqrt_alphas_cumprod",
                  "sqrt_one_minus_alphas_cumprod", "sqrt_recip_alphas", "posterior_variance"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eq. 1: q(z_t|z_0) = N(z_t; sqrt(alpha_bar_t) z_0, (1-alpha_bar_t) I)"""
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (z0.dim() - 1)))
        sqrt_1mac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (z0.dim() - 1)))
        return sqrt_ac * z0 + sqrt_1mac * noise, noise


# ══════════════════════════════════════════════════════════════════════════════
#  Phys-Diff Model — wrapper tổng hợp
# ══════════════════════════════════════════════════════════════════════════════

class PhysDiffModel(nn.Module):
    """
    Phys-Diff: Latent Diffusion Model với PIGA module (Liu et al., ICASSP
    2026). Dự báo đồng thời trajectory (lon,lat) + wind + pressure, dùng
    latent diffusion + physics-inspired cross-task attention.

    interface nhất quán với LSTM/GRU/RNN/ST-Trans/FM/MGTCF: forward()/
    get_loss_breakdown()/sample() đều trả về CHỈ trajectory 2D (lon,lat)
    để so sánh công bằng trong bảng kết quả chính; wind/pressure vẫn
    được dự báo nội bộ (bắt buộc để PIGA hoạt động đúng thiết kế) nhưng
    không phải trọng tâm báo cáo.
    """

    def __init__(
        self,
        obs_len:      int   = 8,
        pred_len:     int   = 12,
        unet_in_ch:   int   = 13,
        d_model:      int   = 64,
        nhead:        int   = 4,
        num_blocks:   int   = 3,
        dim_ff:       int   = 256,
        dropout:      float = 0.1,
        T_diffusion:  int   = 1000,
        beta_start:   float = 1e-4,
        beta_end:     float = 0.02,
        n_sample_steps: int = 50,   # [DEPRECATED, xem _ddpm_reverse_sample] KHÔNG còn
                                     # dùng để điều khiển tốc độ sampling -- paper Phys-Diff
                                     # (Eq.2) không hỗ trợ strided/skip-step sampling, reverse
                                     # process nay LUON chay du T_diffusion buoc lien tiep.
                                     # Giu tham so nay chi de tuong thich nguoc voi checkpoint/
                                     # CLI args cu (train_physdiff.py van truyen no vao), KHONG
                                     # anh huong hanh vi sample() nua.
    ):
        super().__init__()
        self.obs_len       = obs_len
        self.pred_len      = pred_len
        self.T_diffusion    = T_diffusion
        self.n_sample_steps = n_sample_steps   # [DEPRECATED] không dùng, xem ghi chú trên
        self.d_model        = d_model

        self.cond_encoder = ConditionalEncoder(
            obs_len=obs_len, unet_in_ch=unet_in_ch, d_model=d_model,
            nhead=nhead, num_layers=2, dim_ff=dim_ff, dropout=dropout)

        self.latent_codec = LatentCodec(attr_dim=4, d_embed=d_model)   # (lon,lat,wind,pres)

        self.denoiser = PhysicsInspiredDenoiser(
            pred_len=pred_len, d_model=d_model, nhead=nhead,
            num_blocks=num_blocks, dim_ff=dim_ff, dropout=dropout)

        self.schedule = LatentDiffusionSchedule(T=T_diffusion, beta_start=beta_start, beta_end=beta_end)
        self._schedule_device = "cpu"

        # Uncertainty-weighted multi-task loss (Eq.9, Kendall et al. 2018)
        # log_sigma_diff, log_sigma_recon là 2 tham số học được (dùng log
        # để đảm bảo sigma > 0 khi exp(), tránh học ra giá trị âm/0).
        self.log_sigma_diff  = nn.Parameter(torch.zeros(1))
        self.log_sigma_recon = nn.Parameter(torch.zeros(1))

    def _ensure_schedule_device(self, device):
        if self._schedule_device != str(device):
            self.schedule = self.schedule.to(device)
            self._schedule_device = str(device)

    def _prepare_x0(self, batch_list) -> torch.Tensor:
        """
        Ghép (lon, lat, wind, pres) thành x_0 [B, N, 4] cho tương lai.

        [FIX] Bản trước đọc nhầm batch_list[2] (thực chất là obs_rel --
        relative displacement của phần QUAN TRẮC, độ dài obs_len=8) tưởng
        là pred_Me (wind/pressure tương lai). Đúng theo seq_collate() thật
        (Model/data/trajectoriesWithMe_unet_training.py), cấu trúc batch_list
        là:
            0=obs_traj, 1=pred_traj, 2=obs_rel, 3=pred_rel,
            4=nlp, 5=mask, 6=seq_start_end,
            7=obs_Me, 8=pred_Me, 9=obs_Me_rel, 10=pred_Me_rel,
            11=img_obs, 12=img_pred, 13=env_out, 14=None, 15=tyID
        -> pred_Me (wind/pressure cho horizon tương lai, cùng độ dài
        pred_len với pred_traj) nằm ở INDEX 8, không phải index 2.
        """
        traj_gt = batch_list[1]
        T = min(self.pred_len, traj_gt.shape[0])
        pos = traj_gt[:T]   # [T, B, 2]

        pred_me = batch_list[8] if len(batch_list) > 8 else None
        if pred_me is not None and pred_me.shape[0] >= T and pred_me.shape[-1] >= 2:
            me_gt = pred_me[:T, :, :2]   # [T, B, 2] -- chỉ lấy đúng 2 cột (wind,pres)
            x0 = torch.cat([pos, me_gt], dim=-1)
        else:
            wind_pres_placeholder = torch.zeros_like(pos)
            x0 = torch.cat([pos, wind_pres_placeholder], dim=-1)

        return x0.permute(1, 0, 2)   # [B, T, 4]

    def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
        obs_traj = batch_list[0]
        device   = obs_traj.device
        self._ensure_schedule_device(device)

        x0 = self._prepare_x0(batch_list)   # [B, N, 4]
        B, N, _ = x0.shape

        z0 = self.latent_codec.encode(x0)    # [B, N, d_model]

        t = torch.randint(0, self.T_diffusion, (B,), device=device)
        z_t, noise = self.schedule.q_sample(z0, t)

        c = self.cond_encoder(batch_list, t, self.T_diffusion)   # [B, 2, d_model]
        eps_pred, task_feats_per_block = self.denoiser(z_t, c)

        # L_diffusion (Eq.9 term 1)
        l_diffusion = F.mse_loss(eps_pred, noise)

        # ── L_recon: dùng 1-step denoise estimate của z0, decode ra x0_hat ──
        sqrt_ac    = self.schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_1mac  = self.schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        z0_hat = (z_t - sqrt_1mac * eps_pred) / sqrt_ac.clamp(min=1e-6)
        x0_hat = self.latent_codec.decode(z0_hat)   # [B, N, 4]

        l_traj = F.mse_loss(x0_hat[..., :2], x0[..., :2])
        l_wind = F.mse_loss(x0_hat[..., 2], x0[..., 2])
        l_pres = F.mse_loss(x0_hat[..., 3], x0[..., 3])
        l_recon = l_traj + l_wind + l_pres

        # Eq.9: uncertainty-weighted multi-task loss
        sigma_diff  = torch.exp(self.log_sigma_diff)
        sigma_recon = torch.exp(self.log_sigma_recon)
        l_total = (
            l_diffusion / (2 * sigma_diff ** 2)
            + l_recon / (2 * sigma_recon ** 2)
            + torch.log(sigma_diff * sigma_recon)
        ).squeeze()

        with torch.no_grad():
            pred_traj = x0_hat[..., :2].permute(1, 0, 2)   # [N, B, 2]
            ade_m = compute_ade_per_horizon(pred_traj.detach(), batch_list[1])
            atc_m = compute_ate_cte_per_horizon(pred_traj.detach(), batch_list[1])

        return dict(
            total=l_total,
            l_diffusion=l_diffusion.item(), l_traj=l_traj.item(),
            l_wind=l_wind.item(), l_pres=l_pres.item(), l_recon=l_recon.item(),
            sigma_diff=sigma_diff.item(), sigma_recon=sigma_recon.item(),
            **ade_m, **atc_m,
        )

    def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

    # ── Inference: DDPM reverse process trên latent, decode ra x0_hat ────────

    @torch.no_grad()
    def _ddpm_reverse_sample(self, batch_list) -> torch.Tensor:
        """
        [FIX] Ban truoc dung "strided sampling" (chi chay n_sample_steps
        buoc, nhay coc qua stride = T // n_sample_steps) nhung VAN ap
        dung cong thuc Eq.(2) cua paper -- cong thuc nay CHI DUNG cho 1
        buoc LIEN TIEP t -> t-1, khong dung cho viec nhay coc nhieu buoc
        cung luc (do la ly do DDIM can 1 cong thuc rieng, paper Phys-Diff
        KHONG dung DDIM). Ket qua: z_t khong hoi tu ve dung phan phoi
        z_0, decode ra toa do gan nhu ngau nhien (ADE ~6000km quan sat
        duoc khi train that).

        Sua dung theo paper (Section 2.3, Eq.2): chay DU T buoc LIEN
        TIEP tu t=T-1 xuong t=0, khong nhay coc. Paper khong nhac den
        DDIM/strided sampling o bat ky dau.
        """
        obs_traj = batch_list[0]
        device = obs_traj.device
        self._ensure_schedule_device(device)
        B = obs_traj.shape[1]
        N = self.pred_len

        z_t = torch.randn(B, N, self.d_model, device=device)

        # [TỐI ƯU] Tính ctx_token (PaperEncoder) đúng 1 LẦN trước vòng lặp,
        # không đổi công thức toán học -- xem docstring encode_static().
        ctx_token = self.cond_encoder.encode_static(batch_list)

        for t_val in range(self.T_diffusion - 1, -1, -1):   # T-1, T-2, ..., 0 (dung 1 buoc)
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            c = self.cond_encoder.forward_with_ctx(ctx_token, t, self.T_diffusion)
            eps_pred, _ = self.denoiser(z_t, c)

            beta_t = self.schedule.betas[t_val]
            sqrt_recip_alpha_t = self.schedule.sqrt_recip_alphas[t_val]
            sqrt_1mac_t = self.schedule.sqrt_one_minus_alphas_cumprod[t_val]

            model_mean = sqrt_recip_alpha_t * (z_t - beta_t / sqrt_1mac_t.clamp(min=1e-8) * eps_pred)

            if t_val > 0:
                noise = torch.randn_like(z_t)
                posterior_var = self.schedule.posterior_variance[t_val]
                z_t = model_mean + torch.sqrt(posterior_var.clamp(min=1e-20)) * noise
            else:
                z_t = model_mean

        x0_hat = self.latent_codec.decode(z_t)   # [B, N, 4]
        pred_traj = x0_hat[..., :2].permute(1, 0, 2)   # [N, B, 2]
        return pred_traj

    def forward(self, batch_list) -> torch.Tensor:
        return self._ddpm_reverse_sample(batch_list)

    @torch.no_grad()
    def sample(
        self,
        batch_list,
        num_ensemble: int = 20,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sinh `num_ensemble` mẫu độc lập bằng reverse diffusion trên latent
        (đúng cách paper tạo ensemble: "sampling N members from different
        Gaussian noise initializations", Table 2). Chọn best-of-K bằng
        ground-truth nếu có, hoặc độ mượt nếu không (inference thật).
        """
        all_samples = []
        for _ in range(num_ensemble):
            pred = self._ddpm_reverse_sample(batch_list)
            all_samples.append(pred)
        all_trajs = torch.stack(all_samples, dim=0)   # [K, N, B, 2]

        B = all_trajs.shape[2]
        device = all_trajs.device

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