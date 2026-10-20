import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTCNBlock(nn.Module):
    """
    Multi-Scale Dilated Temporal Convolution Block.
    Extracts multi-granular temporal context (short-term, mid-term, long-term).
    """
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        branch_dim = d_model // 4
        self.conv_d1 = nn.Conv1d(d_model, branch_dim, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv1d(d_model, branch_dim, kernel_size=3, padding=2, dilation=2)
        self.conv_d4 = nn.Conv1d(d_model, branch_dim, kernel_size=3, padding=4, dilation=4)
        self.conv_id = nn.Conv1d(d_model, branch_dim, kernel_size=1)
        
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x_trans = x.transpose(1, 2)  # (B, D, T)
        
        c1 = F.gelu(self.conv_d1(x_trans))
        c2 = F.gelu(self.conv_d2(x_trans))
        c3 = F.gelu(self.conv_d4(x_trans))
        c4 = F.gelu(self.conv_id(x_trans))
        
        out = torch.cat([c1, c2, c3, c4], dim=1).transpose(1, 2)  # (B, T, D)
        out = self.dropout(self.out_proj(out))
        return self.norm(residual + out)


class TemporalMAE(nn.Module):
    """
    Temporal Masked Autoencoder for Video Snippet Inpainting.
    Learns normal temporal continuity by predicting masked snippets from visible context.
    Anomalous events (chaotic, sudden) have high inpainting error.
    """
    def __init__(self, in_dim: int = 512, d_model: int = 256, num_layers: int = 2, mask_ratio: float = 0.5):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.mask_ratio = mask_ratio

        self.mask_token = nn.Parameter(torch.randn(1, 1, in_dim))
        self.in_proj = nn.Linear(in_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, in_dim)

    def forward_train(self, x):
        """Randomly masks snippets during training."""
        B, T, D = x.shape
        num_masked = max(int(T * self.mask_ratio), 1)
        
        rand_indices = torch.rand(B, T, device=x.device).argsort(dim=-1)
        mask_idx = rand_indices[:, :num_masked]
        
        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        mask.scatter_(1, mask_idx, True)

        x_masked = x.clone()
        x_masked[mask] = self.mask_token.expand(B, T, D)[mask]

        h = self.in_proj(x_masked)
        h = self.encoder(h)
        x_rec = self.out_proj(h)
        return x_rec, mask

    def forward_infer(self, x):
        """Dual-Half Inpainting Pass (deterministic evaluation of all snippets)."""
        B, T, D = x.shape
        
        # Pass 1: Mask odd frames, inpaint from even
        mask_odd = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        mask_odd[:, 1::2] = True
        x_odd = x.clone()
        x_odd[mask_odd] = self.mask_token.expand(B, T, D)[mask_odd]
        x_rec_odd = self.out_proj(self.encoder(self.in_proj(x_odd)))

        # Pass 2: Mask even frames, inpaint from odd
        mask_even = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        mask_even[:, 0::2] = True
        x_even = x.clone()
        x_even[mask_even] = self.mask_token.expand(B, T, D)[mask_even]
        x_rec_even = self.out_proj(self.encoder(self.in_proj(x_even)))

        # Combine
        x_rec = torch.zeros_like(x)
        x_rec[:, 1::2] = x_rec_odd[:, 1::2]
        x_rec[:, 0::2] = x_rec_even[:, 0::2]
        return x_rec


class DMAEVAD(nn.Module):
    """
    DMAE-VAD: Decoupled Dual-Stream Masked Autoencoder for Video Anomaly Detection.

    Stream A: Multi-Scale Dilated TCN Discriminator -> S_det
    Stream B: Temporal Masked Autoencoder Inpainter -> S_rec
    Fusion: Adaptive Evidential Gated Network -> S_final
    """
    def __init__(
        self,
        in_dim: int = 512,
        d_model: int = 512,
        mae_dim: int = 256,
        num_tcn_layers: int = 2,
        num_mae_layers: int = 2,
        mask_ratio: float = 0.5,
        dropout: float = 0.1,
        gate_init_bias: float = -2.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model

        # --- Stream A: Discriminator (MS-TCN) ---
        self.in_proj_det = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.tcn_layers = nn.ModuleList([
            MultiScaleTCNBlock(d_model=d_model, dropout=dropout)
            for _ in range(num_tcn_layers)
        ])
        self.det_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # --- Stream B: Reconstructor (Temporal MAE) ---
        self.mae = TemporalMAE(
            in_dim=in_dim,
            d_model=mae_dim,
            num_layers=num_mae_layers,
            mask_ratio=mask_ratio
        )

        # --- Adaptive Gated Decision Network ---
        self.gate_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        nn.init.normal_(self.gate_mlp[-1].weight, std=0.01)
        nn.init.constant_(self.gate_mlp[-1].bias, gate_init_bias)

    def forward(self, x, padding_mask=None, force_infer_mae: bool = False):
        """
        Args:
            x: Input CLIP feature tensor (B, T, in_dim)
            padding_mask: BoolTensor (B, T) where True is padding
            force_infer_mae: Set True during testing to use deterministic dual-half inpainting

        Returns:
            dict containing S_final, S_det, S_rec, gate, x_rec, mask
        """
        B, T, D = x.shape

        # 1. Stream A: Discriminator
        h_a = self.in_proj_det(x)
        for layer in self.tcn_layers:
            h_a = layer(h_a)
        det_logits = self.det_head(h_a)               # (B, T, 1)
        S_det = torch.sigmoid(det_logits)              # (B, T, 1)

        # 2. Stream B: Temporal MAE Inpainting
        # If in training mode and not forced to infer: use random masking
        if self.training and not force_infer_mae:
            x_rec, mask = self.mae.forward_train(x)
        else:
            x_rec = self.mae.forward_infer(x)
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

        # Compute cosine reconstruction anomaly score: S_rec in [0, 1]
        x_norm = F.normalize(x, p=2, dim=-1)
        rec_norm = F.normalize(x_rec, p=2, dim=-1)
        cos_sim = torch.sum(x_norm * rec_norm, dim=-1, keepdim=True)  # (B, T, 1)
        S_rec = (1.0 - cos_sim) / 2.0                                # (B, T, 1)

        # 3. Adaptive Gated Fusion
        gate_inputs = torch.cat([
            S_det,
            S_rec,
            torch.abs(S_det - S_rec),
            S_det * S_rec
        ], dim=-1)  # (B, T, 4)

        g = torch.sigmoid(self.gate_mlp(gate_inputs))  # (B, T, 1)
        S_final = (1.0 - g) * S_det + g * S_rec

        if padding_mask is not None:
            mask_3d = padding_mask.unsqueeze(-1)
            S_final = S_final.masked_fill(mask_3d, 0.0)
            S_det = S_det.masked_fill(mask_3d, 0.0)
            S_rec = S_rec.masked_fill(mask_3d, 0.0)

        return {
            'final_score': S_final,
            'det_score': S_det,
            'rec_score': S_rec,
            'gate': g,
            'x_rec': x_rec,
            'mask': mask,
            'features': h_a
        }
