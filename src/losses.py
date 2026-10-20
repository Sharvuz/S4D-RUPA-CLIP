import torch
import torch.nn as nn
import torch.nn.functional as F


class DMAEVADLoss(nn.Module):
    """
    Loss Formulation for DMAE-VAD:
    1. Robust Top-K MIL Ranking Loss (L_mil)
    2. Self-Supervised Masked Feature Inpainting Loss (L_mae)
    3. Reconstruction Error Contrastive Margin Loss (L_rec_contrast)
    4. Temporal Smoothness Loss (L_smooth)
    5. Anomaly Sparsity Loss (L_sparse)
    """
    def __init__(
        self,
        mil_margin: float = 1.0,
        rec_margin: float = 0.2,
        loss_mae_weight: float = 1.0,
        loss_rec_contrast_weight: float = 0.5,
        loss_smooth_weight: float = 8e-5,
        loss_sparse_weight: float = 8e-5,
        loss_aux_det_weight: float = 0.2,
        k_ratio: float = 16.0,
    ):
        super().__init__()
        self.mil_margin = mil_margin
        self.rec_margin = rec_margin
        self.loss_mae_weight = loss_mae_weight
        self.loss_rec_contrast_weight = loss_rec_contrast_weight
        self.loss_smooth_weight = loss_smooth_weight
        self.loss_sparse_weight = loss_sparse_weight
        self.loss_aux_det_weight = loss_aux_det_weight
        self.k_ratio = k_ratio

    def get_topk_mean(self, scores):
        """Compute mean of top-k scores per video in batch."""
        if scores.dim() == 3:
            scores = scores.squeeze(-1)
        B, T = scores.shape
        k = max(int(T / self.k_ratio) + 1, 1)
        topk_scores, _ = torch.topk(scores, k, dim=-1)
        return torch.mean(topk_scores, dim=-1)

    def forward(self, abn_out, nor_out, abn_orig_feat, nor_orig_feat, current_epoch: int = 0):
        """
        Args:
            abn_out: Output dict from abnormal videos
            nor_out: Output dict from normal videos
            abn_orig_feat: Original unmasked features for abnormal batch (B/2, T, D)
            nor_orig_feat: Original unmasked features for normal batch (B/2, T, D)
        """
        abn_final = abn_out['final_score'].squeeze(-1)
        nor_final = nor_out['final_score'].squeeze(-1)

        abn_det = abn_out['det_score'].squeeze(-1)
        nor_det = nor_out['det_score'].squeeze(-1)

        abn_rec = abn_out['rec_score'].squeeze(-1)
        nor_rec = nor_out['rec_score'].squeeze(-1)

        # 1. Top-K MIL Ranking Loss
        abn_topk = self.get_topk_mean(abn_final)
        nor_topk = self.get_topk_mean(nor_final)
        loss_mil = torch.mean(F.relu(self.mil_margin - abn_topk + nor_topk))

        # Auxiliary loss for detector
        abn_det_topk = self.get_topk_mean(abn_det)
        nor_det_topk = self.get_topk_mean(nor_det)
        loss_aux_det = torch.mean(F.relu(self.mil_margin - abn_det_topk + nor_det_topk))

        # 2. Self-Supervised Masked Feature Inpainting Loss (L_mae)
        # Compute MSE only on masked positions
        abn_mask = abn_out['mask']
        nor_mask = nor_out['mask']
        
        loss_mae_abn = F.mse_loss(abn_out['x_rec'][abn_mask], abn_orig_feat[abn_mask])
        loss_mae_nor = F.mse_loss(nor_out['x_rec'][nor_mask], nor_orig_feat[nor_mask])
        loss_mae = (loss_mae_abn + loss_mae_nor) / 2.0

        # 3. Reconstruction Error Contrastive Margin
        # Normal snippets should reconstruct better (lower rec score) than abnormal peaks
        abn_rec_topk = self.get_topk_mean(abn_rec)
        nor_rec_topk = self.get_topk_mean(nor_rec)
        loss_rec_contrast = torch.mean(F.relu(self.rec_margin - abn_rec_topk + nor_rec_topk))

        # 4. Temporal Smoothness
        loss_smooth = (
            torch.mean((abn_final[:, 1:] - abn_final[:, :-1]) ** 2) +
            torch.mean((nor_final[:, 1:] - nor_final[:, :-1]) ** 2)
        ) / 2.0

        # 5. Anomaly Sparsity
        loss_sparse = torch.mean(abn_final)

        total_loss = (
            loss_mil +
            self.loss_aux_det_weight * loss_aux_det +
            self.loss_mae_weight * loss_mae +
            self.loss_rec_contrast_weight * loss_rec_contrast +
            self.loss_smooth_weight * loss_smooth +
            self.loss_sparse_weight * loss_sparse
        )

        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_mil': loss_mil.item(),
            'loss_mae': loss_mae.item(),
            'loss_rec_contrast': loss_rec_contrast.item(),
            'loss_smooth': loss_smooth.item(),
            'loss_sparse': loss_sparse.item(),
            'abn_topk': abn_topk.mean().item(),
            'nor_topk': nor_topk.mean().item(),
            'gate_mean': abn_out['gate'].mean().item(),
        }

        return total_loss, loss_dict
