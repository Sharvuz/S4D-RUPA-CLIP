"""
S4D (Diagonal State Space Model) and Bidirectional Temporal Encoder.
Pure PyTorch implementation - zero external CUDA compilation required.
Runs seamlessly across Linux, Windows, Kaggle, Colab, CPU and GPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as OfficialMamba
    HAS_OFFICIAL_MAMBA = True
except ImportError:
    HAS_OFFICIAL_MAMBA = False


class DiagonalSSM(nn.Module):
    """
    Diagonal State Space Model (S4D style).
    Computes continuous-time state space kernel via diagonal decomposition
    and applies parallel FFT convolution over the temporal sequence.
    Complexity: O(T log T) or O(T), completely free of recurrence bottlenecks.
    """
    def __init__(self, d_model: int = 512, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Log time-step parameter (discretization step size dt)
        self.log_step = nn.Parameter(torch.log(torch.tensor(0.001)))

        # S4D-Lin / HiPPO diagonal initialization for A
        # Real part of A is initialized strictly negative to ensure stability
        self.A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))

        # Learnable observation matrix C and direct feedthrough D
        self.C = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.ones(d_model))

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, D)
        Returns:
            out: Convolved representation of shape (B, T, D)
        """
        B, T, H = x.shape
        dt = torch.exp(self.log_step)
        A = -torch.exp(self.A_real)  # Ensure negative eigenvalues for stability

        # Generate continuous-time convolution kernel K over sequence length T
        t = torch.arange(T, device=x.device, dtype=torch.float32)
        k = torch.einsum('hn,t->hnt', A * dt, t)
        K = torch.einsum('hn,hnt->ht', self.C, torch.exp(k)) * dt  # (H, T)

        # Parallel 1D temporal convolution via Fast Fourier Transform (FFT)
        x_t = x.transpose(1, 2)  # (B, H, T)
        n_fft = 2 * T
        K_f = torch.fft.rfft(K, n=n_fft)
        x_f = torch.fft.rfft(x_t, n=n_fft)
        y = torch.fft.irfft(x_f * K_f, n=n_fft)[..., :T]

        # Add skip connection D * x
        y = y + x_t * self.D.unsqueeze(-1)
        out = self.dropout(self.out_proj(y.transpose(1, 2)))
        return out


class PurePyTorchMamba(nn.Module):
    """
    Pure-PyTorch Selective State Space Model (Mamba).
    Uses input-dependent selection mechanism for data-controlled scanning.
    """
    def __init__(self, d_model: int = 512, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_act = F.silu(x_conv)

        ssm_proj = self.x_proj(x_act)
        delta_raw, B_param, C_param = torch.split(
            ssm_proj, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))
        A = -torch.exp(self.A_log)

        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = delta.unsqueeze(-1) * B_param.unsqueeze(2)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y_list = []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * x_act[:, t, :, None]
            y_t = torch.sum(h * C_param[:, t, None, :], dim=-1)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)
        y = y + x_act * self.D
        y = y * F.silu(z)
        out = self.dropout(self.out_proj(y))
        return out


class BiS4DBlock(nn.Module):
    """
    Bidirectional S4D / State Space Block.
    Processes temporal features forward and backward simultaneously,
    capturing both historical and future contextual cues with linear O(T) complexity.
    """
    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 64,
        ssm_type: str = 's4d',
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.ssm_type = ssm_type
        self.norm1 = nn.LayerNorm(d_model)

        if ssm_type == 'official' and HAS_OFFICIAL_MAMBA:
            self.fwd_ssm = OfficialMamba(d_model=d_model, d_state=16, d_conv=4, expand=expand)
            self.bwd_ssm = OfficialMamba(d_model=d_model, d_state=16, d_conv=4, expand=expand)
        elif ssm_type == 's4d':
            self.fwd_ssm = DiagonalSSM(d_model=d_model, d_state=d_state, dropout=dropout)
            self.bwd_ssm = DiagonalSSM(d_model=d_model, d_state=d_state, dropout=dropout)
        else:
            self.fwd_ssm = PurePyTorchMamba(d_model=d_model, d_state=16, expand=expand, dropout=dropout)
            self.bwd_ssm = PurePyTorchMamba(d_model=d_model, d_state=16, expand=expand, dropout=dropout)

        self.merge = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expand, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (B, T, D)
            padding_mask: (B, T) boolean mask where True indicates padding
        Returns:
            out: (B, T, D) context-aware temporal representations
        """
        residual = x
        x_norm = self.norm1(x)

        if padding_mask is not None:
            x_norm = x_norm.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Forward scan (past to future)
        fwd_out = self.fwd_ssm(x_norm)

        # Backward scan (future to past)
        x_rev = torch.flip(x_norm, dims=[1])
        bwd_out = self.bwd_ssm(x_rev)
        bwd_out = torch.flip(bwd_out, dims=[1])

        # Merge forward and backward temporal signals
        merged = self.merge(torch.cat([fwd_out, bwd_out], dim=-1))
        x = residual + merged

        # Position-wise Feed-Forward Network with residual connection
        x = x + self.ffn(self.norm2(x))
        return x


class BiS4DEncoder(nn.Module):
    """
    Multi-layer Bidirectional S4D Temporal Backbone.
    Replaces expensive GCNs and quadratic attention mechanisms.
    """
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 2,
        d_state: int = 64,
        ssm_type: str = 's4d',
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BiS4DBlock(
                d_model=d_model,
                d_state=d_state,
                ssm_type=ssm_type,
                expand=expand,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        x = self.final_norm(x)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x
