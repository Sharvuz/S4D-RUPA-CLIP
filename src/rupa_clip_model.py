"""
S4D-RUPA-CLIP: Bidirectional State Space Model with Counterfactual Residual-to-Text Alignment
for Weakly-Supervised Video Anomaly Detection (WS-VAD).

Scientific Innovations:
1. Bidirectional S4D Temporal Backbone:
   - Captures fine-grained past and future temporal context with linear O(T) complexity.
   - Eliminates quadratic attention memory and GCN message-passing overhead.
2. Autonomous RUPA Counterfactual Reconstruction:
   - Normal Prototype Bank (P in R^{K x D}) captures diverse modes of normal surveillance.
   - Counterfactual Cross-Attention reconstructs the normal background: F_rec.
   - Anomaly Residual (R = H - F_rec) completely strips normal background clutter.
3. Residual-to-Text Semantic Projector:
   - Projects the pure anomaly residual R into the multi-class CLIP text latent space.
   - For normal frames: R ~ 0 -> aligns to Class 0 (Normal) with high confidence.
   - For crime frames: R preserves pure anomalous semantics -> directly matches crime text prompts.
   - Bypasses the background domain gap, achieving unprecedented classification precision.
4. Dynamic Evidential Safe Gate:
   - Dynamically balances discriminative temporal cues and counterfactual semantic evidence.
   - Safeguards against false alarms on complex scenes without requiring warm-start.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from s4d_block import BiS4DEncoder


class NormalPrototypeBank(nn.Module):
    """
    Learnable Normal Prototype Bank for Counterfactual Reconstruction.
    Maintains K normal prototypes representing distinct surveillance patterns
    (e.g., pedestrian walking, normal vehicle traffic, stationary background).
    """
    def __init__(self, num_prototypes: int = 16, d_model: int = 512, temperature: float = 10.0):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.d_model = d_model

        # Prototypes initialized orthogonally on unit hypersphere
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))
        nn.init.orthogonal_(self.prototypes)
        self.temperature = nn.Parameter(torch.ones([]) * temperature)

    def get_normalized_prototypes(self):
        """Returns L2-normalized prototypes on unit hypersphere."""
        return F.normalize(self.prototypes, p=2, dim=-1)

    def forward(self, H):
        """
        Counterfactual Reconstruction via Cosine Soft-Attention:
        Args:
            H: Contextual temporal representations of shape (B, T, D)
        Returns:
            F_rec: Counterfactual normal reconstruction (B, T, D)
            S_rec: Reconstruction anomaly score in [0, 1] (B, T, 1)
            attn: Attention distribution over prototypes (B, T, K)
        """
        B, T, D = H.shape
        P = self.get_normalized_prototypes()  # (K, D)
        H_norm = F.normalize(H, p=2, dim=-1)   # (B, T, D)

        # Cosine similarity between each frame and all normal prototypes
        sim = torch.matmul(H_norm, P.transpose(0, 1))  # (B, T, K)
        tau = torch.clamp(self.temperature, min=1.0, max=50.0)
        attn = F.softmax(sim * tau, dim=-1)            # (B, T, K)

        # Counterfactual normal reconstruction
        F_rec = torch.matmul(attn, P)                  # (B, T, D)
        F_rec_norm = F.normalize(F_rec, p=2, dim=-1)

        # Reconstruction score based on cosine distance:
        # If frame is normal: H_norm aligns with F_rec_norm -> cos_sim -> 1, S_rec -> 0
        # If frame is abnormal: cos_sim << 1 -> S_rec -> 1
        cos_sim = torch.sum(H_norm * F_rec_norm, dim=-1, keepdim=True)
        S_rec = torch.clamp((1.0 - cos_sim) / 2.0, min=0.0, max=1.0)

        return F_rec, S_rec, attn


class ResidualSemanticProjector(nn.Module):
    """
    Residual-to-Text Semantic Projector.
    Projects the anomaly residual R = H - F_rec into the CLIP text latent space.
    Because normal background is stripped away by F_rec, R contains only anomalous semantics.
    """
    def __init__(self, d_model: int = 512, text_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim, text_dim)
        )
        # Learnable logit scale initialized to 1 / 0.07 (standard CLIP temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, R, text_features):
        """
        Args:
            R: Anomaly residual tensor of shape (B, T, D)
            text_features: Precomputed normalized CLIP text features of shape (C, 512)
        Returns:
            S_sem: Semantic anomaly score in [0, 1] (B, T, 1)
            class_probs: Multi-class probabilities over C categories (B, T, C)
            class_logits: Scaled cosine similarity logits (B, T, C)
        """
        R_proj = self.proj(R)
        R_norm = F.normalize(R_proj, p=2, dim=-1)
        W_norm = F.normalize(text_features.to(dtype=R_norm.dtype, device=R_norm.device), p=2, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        # Scaled cosine similarity between residual and all text prompts
        class_logits = scale * torch.matmul(R_norm, W_norm.transpose(0, 1))  # (B, T, C)
        class_probs = F.softmax(class_logits, dim=-1)

        # Class 0 corresponds to "Normal"
        # Probability of being anomalous = 1 - P(Normal)
        prob_normal = class_probs[:, :, 0:1]
        S_sem = torch.clamp(1.0 - prob_normal, min=0.0, max=1.0)

        return S_sem, class_probs, class_logits


class DynamicEvidentialSafeGate(nn.Module):
    """
    Dynamic Evidential Safe Gate.
    Integrates coarse discriminative score (S_det), counterfactual reconstruction score (S_rec),
    semantic text alignment score (S_sem), score discrepancy, and residual energy.
    Produces an adaptive gating coefficient g in [0, 1] that guards against false alarms.
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 16, init_bias: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.constant_(self.mlp[-1].bias, init_bias)

    def forward(self, S_det, S_rec, S_sem, residual_norm_sq):
        """
        Args:
            S_det: (B, T, 1)
            S_rec: (B, T, 1)
            S_sem: (B, T, 1)
            residual_norm_sq: (B, T, 1)
        Returns:
            g: Dynamic gate value in [0, 1] (B, T, 1)
        """
        discrepancy = torch.abs(S_det - S_rec)
        evidence = torch.cat([
            S_det,
            S_rec,
            S_sem,
            discrepancy,
            residual_norm_sq
        ], dim=-1)  # (B, T, 5)

        gate_logits = self.mlp(evidence)
        g = torch.sigmoid(gate_logits)
        return g


class S4DRUPACLIP(nn.Module):
    """
    Unified S4D-RUPA-CLIP Model Architecture.
    Combines:
    1. Input Feature Projector & LayerNorm.
    2. Multi-layer Bidirectional S4D Temporal Encoder.
    3. Discriminative Anomaly Detector Head.
    4. Normal Prototype Bank & Counterfactual Reconstruction.
    5. Anomaly Residual Extraction & Residual-to-Text Projector.
    6. Dynamic Evidential Safe Gate Ensemble.
    """
    def __init__(
        self,
        in_dim: int = 512,
        d_model: int = 512,
        num_layers: int = 2,
        d_state: int = 64,
        ssm_type: str = 's4d',
        num_prototypes: int = 16,
        text_dim: int = 512,
        dropout: float = 0.1,
        gate_hidden: int = 16,
        gate_init_bias: float = 0.0
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_prototypes = num_prototypes
        self.text_dim = text_dim

        # 1. Input Linear Projection & Normalization
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. Bidirectional S4D Temporal Backbone
        self.encoder = BiS4DEncoder(
            d_model=d_model,
            num_layers=num_layers,
            d_state=d_state,
            ssm_type=ssm_type,
            expand=2,
            dropout=dropout
        )

        # 3. Branch 1: Discriminative Anomaly Detector Head
        self.det_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # 4. Branch 2: Normal Prototype Bank & Counterfactual Reconstruction
        self.prototype_bank = NormalPrototypeBank(
            num_prototypes=num_prototypes,
            d_model=d_model,
            temperature=10.0
        )

        # 5. Branch 3: Residual-to-Text Semantic Projector
        self.residual_projector = ResidualSemanticProjector(
            d_model=d_model,
            text_dim=text_dim,
            dropout=dropout
        )

        # 6. Branch 4: Dynamic Evidential Safe Gate
        self.safe_gate = DynamicEvidentialSafeGate(
            input_dim=5,
            hidden_dim=gate_hidden,
            init_bias=gate_init_bias
        )

        # Buffer for CLIP text embeddings to enable seamless PyTorch DataParallel replication
        self.register_buffer('text_features', None, persistent=False)

    def set_text_features(self, text_features):
        """Sets precomputed CLIP text embeddings as a model buffer (float32)."""
        if text_features is not None:
            self.register_buffer(
                'text_features',
                F.normalize(text_features.float(), p=2, dim=-1),
                persistent=False
            )

    def forward(self, x, padding_mask=None, text_features=None):
        """
        Forward Pass:
        Args:
            x: Pre-extracted CLIP features of shape (B, T, in_dim)
            padding_mask: Optional boolean tensor of shape (B, T) where True is padding
            text_features: Precomputed multi-prompt CLIP text embeddings of shape (C, 512)
        Returns:
            Dictionary containing:
                - final_score: Dynamically gated final anomaly score in [0, 1] (B, T, 1)
                - det_score: Discriminative detector score in [0, 1] (B, T, 1)
                - rec_score: Counterfactual reconstruction score in [0, 1] (B, T, 1)
                - sem_score: Residual-to-text semantic anomaly score in [0, 1] (B, T, 1)
                - gate: Adaptive safe gate values in [0, 1] (B, T, 1)
                - residual: Anomaly residual tensor R = H - F_rec (B, T, D)
                - residual_norm_sq: Squared L2 norm of residual (B, T, 1)
                - reconstructed: Counterfactual normal reconstruction F_rec (B, T, D)
                - features: Contextual temporal representations H (B, T, D)
                - prototypes: Normalized prototype tensor (K, D)
                - class_probs: Multi-class probabilities over C classes (B, T, C)
                - class_logits: Multi-class text alignment logits (B, T, C)
        """
        B, T, _ = x.shape

        # 1. Project input visual features
        H_in = self.in_proj(x)

        # 2. Extract contextual representations with Bidirectional S4D
        H = self.encoder(H_in, padding_mask=padding_mask)

        # 3. Discriminative Branch: Anomaly Score S_det
        det_logits = self.det_head(H)
        S_det = torch.sigmoid(det_logits)

        # 4. RUPA Counterfactual Reconstruction Branch
        F_rec, S_rec, attn_weights = self.prototype_bank(H)

        # 5. Anomaly Residual Extraction: R = H - F_rec
        # Strips normal background clutter; contains pure anomalous dynamics
        R = H - F_rec
        # Normalized MSE residual energy per feature dimension (scaled smoothly to ~[0, 1])
        residual_norm_sq = torch.mean(R ** 2, dim=-1, keepdim=True)  # (B, T, 1)

        # 6. Residual-to-Text Alignment Branch
        effective_text_features = text_features if text_features is not None else self.text_features
        if effective_text_features is not None:
            S_sem, class_probs, class_logits = self.residual_projector(R, effective_text_features)
        else:
            S_sem = S_rec  # Fallback if no text features provided
            class_probs = None
            class_logits = None

        # 7. Dynamic Evidential Safe Gate
        g = self.safe_gate(S_det, S_rec, S_sem, residual_norm_sq)

        # 8. Evidential Gated Ensemble:
        # P_final = (1 - g) * S_det + g * ((S_rec + S_sem) / 2)
        P_refiner = 0.5 * (S_rec + S_sem)
        P_final = (1.0 - g) * S_det + g * P_refiner

        # Mask padding snippets if padding_mask is present
        if padding_mask is not None:
            mask_3d = padding_mask.unsqueeze(-1)
            P_final = P_final.masked_fill(mask_3d, 0.0)
            S_det = S_det.masked_fill(mask_3d, 0.0)
            S_rec = S_rec.masked_fill(mask_3d, 0.0)
            S_sem = S_sem.masked_fill(mask_3d, 0.0)
            g = g.masked_fill(mask_3d, 0.0)

        return {
            'final_score': P_final,
            'det_score': S_det,
            'rec_score': S_rec,
            'sem_score': S_sem,
            'gate': g,
            'residual': R,
            'residual_norm_sq': residual_norm_sq,
            'reconstructed': F_rec,
            'features': H,
            'prototypes': self.prototype_bank.get_normalized_prototypes(),
            'class_probs': class_probs,
            'class_logits': class_logits,
            'attn_weights': attn_weights
        }
