from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight


def selective_scan_parallel(
    u:     torch.Tensor,
    delta: torch.Tensor,
    A:     torch.Tensor,
    B:     torch.Tensor,
    C:     torch.Tensor,
    D:     torch.Tensor,
) -> torch.Tensor:
    B_size, T, d_inner = u.shape
    d_state = A.shape[1]
    device  = u.device


    delta_clamped = delta.clamp(-10.0, 1.0)


    dA = torch.exp(
        torch.einsum("bti,is->btis", delta_clamped, A)
    )


    dB_u = torch.einsum("bti,bts->btis", delta_clamped, B) * u.unsqueeze(-1)

    h  = torch.zeros(B_size, d_inner, d_state, device=device)
    ys = []
    for t in range(T):
        h  = dA[:, t] * h + dB_u[:, t]
        y  = torch.einsum("bis,bs->bi", h, C[:, t])
        ys.append(y)

    out = torch.stack(ys, dim=1)
    return out + u * D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):

    def __init__(
        self,
        d_model:  int   = 128,
        d_state:  int   = 16,
        d_conv:   int   = 4,
        expand:   int   = 2,
        dt_rank:  int | str = "auto",
        dt_min:   float = 0.001,
        dt_max:   float = 0.1,
        dropout:  float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.expand  = expand
        self.d_inner = int(expand * d_model)
        dt_rank      = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)


        self.conv1d = nn.Conv1d(
            in_channels  = self.d_inner,
            out_channels = self.d_inner,
            kernel_size  = d_conv,
            padding      = d_conv - 1,
            groups       = self.d_inner,
            bias         = True,
        )

        self.x_proj  = nn.Linear(self.d_inner,
                                  dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)


        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(
            self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)


        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        B_sz, T, _ = x.shape

        xz  = self.in_proj(x)
        x_s = xz[:, :, :self.d_inner]
        z   = xz[:, :, self.d_inner:]


        x_conv_out = self.conv1d(x_s.permute(0, 2, 1))
        x_conv = F.silu(x_conv_out[:, :, :T]).permute(0, 2, 1)

        dt_rank = self.x_proj.out_features - 2 * self.d_state
        x_dbc   = self.x_proj(x_conv)
        dt_raw  = x_dbc[:, :, :dt_rank]
        B_ssm   = x_dbc[:, :, dt_rank:dt_rank + self.d_state]
        C_ssm   = x_dbc[:, :, dt_rank + self.d_state:]

        delta = F.softplus(self.dt_proj(dt_raw))
        A     = -torch.exp(self.A_log.float())

        y = selective_scan_parallel(x_conv, delta, A, B_ssm, C_ssm, self.D)
        y = y * F.silu(z)
        y = self.out_proj(y)

        return self.drop(y) + residual


class MambaEncoder(nn.Module):

    def __init__(
        self,
        input_dim:  int   = 192,
        hidden_dim: int   = 128,
        d_model:    int   = 128,
        n_layers:   int   = 3,
        d_state:    int   = 16,
        dropout:    float = 0.1,
        pool:       str   = "last",
    ):
        super().__init__()
        self.pool = pool

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        if self.pool == "last":
            h_out = h[:, -1, :]
        elif self.pool == "mean":
            h_out = h.mean(dim=1)
        else:
            h_out = h.max(dim=1).values
        return self.out_proj(h_out)


class DataEncoder1D_Mamba(nn.Module):

    def __init__(
        self,
        in_1d:       int   = 4,
        feat_3d_dim: int   = 128,
        mlp_h:       int   = 64,
        lstm_hidden: int   = 128,
        lstm_layers: int   = 3,
        dropout:     float = 0.1,
        d_state:     int   = 16,
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.feat_3d_dim = feat_3d_dim

        self.mlp_1d = nn.Sequential(
            nn.Linear(in_1d, mlp_h),
            nn.LayerNorm(mlp_h),
            nn.GELU(),
        )
        self.mlp_fusion = nn.Sequential(
            nn.Linear(feat_3d_dim + mlp_h, mlp_h * 2),
            nn.LayerNorm(mlp_h * 2),
            nn.GELU(),
        )
        self.mamba = MambaEncoder(
            input_dim  = mlp_h * 2,
            hidden_dim = lstm_hidden,
            d_model    = lstm_hidden,
            n_layers   = lstm_layers,
            d_state    = d_state,
            dropout    = dropout,
            pool       = "last",
        )

    def forward(
        self,
        obs_in:  torch.Tensor,
        feat_3d: torch.Tensor,
    ) -> torch.Tensor:

        T     = obs_in.shape[1]
        T_bot = feat_3d.shape[1]


        if T_bot != T:
            feat_3d = F.interpolate(
                feat_3d.permute(0, 2, 1),
                size=T, mode="linear", align_corners=False,
            ).permute(0, 2, 1)

        e_1d  = self.mlp_1d(obs_in)
        e_en  = self.mlp_fusion(torch.cat([feat_3d, e_1d], dim=-1))
        return self.mamba(e_en)