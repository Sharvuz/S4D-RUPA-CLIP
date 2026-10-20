"""
Non-Saturating Multi-Task Loss Functions for S4D-RUPA-CLIP.

Components:
1. Non-Saturating DeepMIL BCE Loss:
   - Replaces fragile hinge ranking margins with smooth, non-vanishing video-level BCE.
   - Operates on Top-K temporal means of both coarse detector S_det and evidential ensemble P_final.
2. Residual-to-Text Semantic Alignment Loss:
   - Cross-Entropy alignment: normal videos must align to Class 0 ("Normal").
   - Abnormal videos must align to the ground truth crime class on Top-K anomaly frames.
3. Counterfactual Reconstruction & Prototype Margin Loss:
   - Compactness on normal videos (normal snippets must be tightly reconstructed).
   - Push loss on abnormal videos (top-k crime snippets must exceed normal prototype boundary).
   - Orthogonality regularization to prevent prototype redundancy.
4. Background Residual Suppression Regularization:
   - Penalizes residual magnitude ||R||^2 on normal videos to crush false alarms on complex scenes.
5. Temporal Smoothness & Sparsity Constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class S4DRUPALoss(nn.Module):
    """
    Unified Loss Module for S4D-RUPA-CLIP.
    """
    def __init__(
        self,
        loss_rec_weight: float = 0.5,
        loss_sem_weight: float = 0.5,
        loss_bg_weight: float = 0.1,
        loss_ortho_weight: float = 0.1,
        loss_smooth_weight: float = 1e-4,
        loss_sparse_weight: float = 5e-4,
        k_ratio: float = 16.0,
        margin: float = 0.2,
    ):
        super().__init__()
        self.loss_rec_weight = loss_rec_weight
        self.loss_sem_weight = loss_sem_weight
        self.loss_bg_weight = loss_bg_weight
        self.loss_ortho_weight = loss_ortho_weight
        self.loss_smooth_weight = loss_smooth_weight
        self.loss_sparse_weight = loss_sparse_weight
        self.k_ratio = k_ratio
        self.margin = margin

    def get_topk_mean(self, scores):
        """
        Computes the mean score of the top-k anomalous frames for each video in the batch.
        Args:
            scores: Tensor of shape (B, T) or (B, T, 1)
        Returns:
            topk_mean: Tensor of shape (B,)
        """
        if scores.dim() == 3:
            scores = scores.squeeze(-1)
        B, T = scores.shape
        k = max(int(T / self.k_ratio) + 1, 1)
        topk_scores, _ = torch.topk(scores, k, dim=-1)
        return torch.mean(topk_scores, dim=-1)

    def compute_bce_mil_loss(self, abn_scores, nor_scores):
        """
        Non-saturating DeepMIL BCE Loss on Top-K predictions.
        """
        abn_topk = self.get_topk_mean(abn_scores)  # (B_abn,)
        nor_topk = self.get_topk_mean(nor_scores)  # (B_nor,)

        eps = 1e-6
        # Abnormal videos: maximize score -> minimize -log(score)
        loss_abn = -torch.mean(torch.log(abn_topk.clamp(min=eps, max=1.0)))
        # Normal videos: minimize score -> minimize -log(1 - score)
        loss_nor = -torch.mean(torch.log((1.0 - nor_topk).clamp(min=eps, max=1.0)))

        return 0.5 * (loss_abn + loss_nor)

    def compute_reconstruction_loss(self, abn_out, nor_out):
        """
        Counterfactual Prototype Reconstruction Loss:
        - Compactness: Normal snippets must be close to at least one normal prototype.
        - Push: Top-K abnormal snippets must be distant from all normal prototypes.
        - Orthogonality: Prototypes must span diverse normal surveillance modes.
        """
        prototypes = nor_out['prototypes']  # (K, D)
        K, D = prototypes.shape

        # 1. Compactness Loss on Normal Videos
        nor_feat = F.normalize(nor_out['features'], p=2, dim=-1).view(-1, D)  # (N_nor, D)
        sim_nor = torch.matmul(nor_feat, prototypes.transpose(0, 1))           # (N_nor, K)
        max_sim_nor, _ = torch.max(sim_nor, dim=-1)
        loss_compact = torch.mean(1.0 - max_sim_nor)

        # 2. Push Loss on Top-K Abnormal Snippets
        abn_det = abn_out['det_score'].squeeze(-1)  # (B_abn, T)
        B_abn, T_abn, _ = abn_out['features'].shape
        k = max(int(T_abn / self.k_ratio) + 1, 1)

        _, topk_idx = torch.topk(abn_det, k, dim=-1)
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, D)

        abn_feat_norm = F.normalize(abn_out['features'], p=2, dim=-1)
        abn_topk_feat = torch.gather(abn_feat_norm, 1, topk_idx_exp).view(-1, D)  # (B_abn * k, D)

        sim_abn = torch.matmul(abn_topk_feat, prototypes.transpose(0, 1))  # (B_abn * k, K)
        loss_push = torch.mean(F.relu(sim_abn - self.margin))

        # 3. Prototype Orthogonality Loss
        gram = torch.matmul(prototypes, prototypes.transpose(0, 1))
        identity = torch.eye(K, device=prototypes.device)
        loss_ortho = torch.mean((gram - identity) ** 2)

        total_rec_loss = loss_compact + loss_push + self.loss_ortho_weight * loss_ortho
        return total_rec_loss, loss_compact, loss_push, loss_ortho

    def compute_semantic_loss(self, abn_out, nor_out, abn_class_indices):
        """
        Residual-to-Text Semantic Alignment Loss:
        - Normal videos: ALL frames must align to Class 0 (Normal).
        - Abnormal videos: Top-K anomalous frames must align to the target crime class.
        """
        abn_probs = abn_out.get('class_probs')
        nor_probs = nor_out.get('class_probs')

        if abn_probs is None or nor_probs is None or abn_class_indices is None:
            zero = torch.tensor(0.0, device=abn_out['det_score'].device)
            return zero

        eps = 1e-6
        # 1. Normal Video Alignment: Entire video is normal -> Class 0
        loss_sem_nor = -torch.mean(torch.log(nor_probs[:, :, 0].clamp(min=eps, max=1.0)))

        # 2. Abnormal Video Alignment: Top-K frames align to specific crime label
        B_abn, T_abn, C = abn_probs.shape
        k = max(int(T_abn / self.k_ratio) + 1, 1)
        abn_det = abn_out['det_score'].squeeze(-1)
        _, topk_idx = torch.topk(abn_det, k, dim=-1)

        event_losses = []
        for b in range(B_abn):
            target_cls = int(abn_class_indices[b].item())
            b_topk = topk_idx[b]
            if 0 < target_cls < C:
                target_prob = abn_probs[b, b_topk, target_cls]
            else:
                # Any non-normal class
                target_prob = abn_probs[b, b_topk, 1:].sum(dim=-1)
            event_losses.append(-torch.mean(torch.log(target_prob.clamp(min=eps, max=1.0))))

        loss_sem_abn = torch.stack(event_losses).mean() if event_losses else torch.tensor(0.0, device=abn_probs.device)
        return 0.5 * (loss_sem_nor + loss_sem_abn)

    def forward(self, abn_out, nor_out, abn_class_indices=None):
        """
        Computes total loss across abnormal and normal batch outputs.
        """
        abn_det = abn_out['det_score'].squeeze(-1)
        nor_det = nor_out['det_score'].squeeze(-1)
        abn_final = abn_out['final_score'].squeeze(-1)
        nor_final = nor_out['final_score'].squeeze(-1)

        # 1. DeepMIL Non-Saturating BCE on both coarse and final ensemble scores
        loss_bce_det = self.compute_bce_mil_loss(abn_det, nor_det)
        loss_bce_final = self.compute_bce_mil_loss(abn_final, nor_final)
        loss_mil = loss_bce_det + loss_bce_final

        # 2. Counterfactual Prototype Reconstruction Loss
        loss_rec, loss_compact, loss_push, loss_ortho = self.compute_reconstruction_loss(abn_out, nor_out)

        # 3. Residual-to-Text Semantic Alignment Loss
        loss_sem = self.compute_semantic_loss(abn_out, nor_out, abn_class_indices)

        # 4. Background Residual Suppression Loss (||R_nor||^2 -> 0)
        # Normal frames must leave near-zero residual, crushing false alarms
        nor_res_sq = nor_out['residual_norm_sq']
        loss_bg = torch.mean(nor_res_sq)

        # 5. Temporal Regularizations (Smoothness & Sparsity)
        diff_abn = abn_det[:, 1:] - abn_det[:, :-1]
        loss_smooth = torch.mean(diff_abn ** 2)
        loss_sparse = torch.mean(abn_det)

        # Total Weighted Multi-Task Loss
        total_loss = (
            loss_mil
            + self.loss_rec_weight * loss_rec
            + self.loss_sem_weight * loss_sem
            + self.loss_bg_weight * loss_bg
            + self.loss_smooth_weight * loss_smooth
            + self.loss_sparse_weight * loss_sparse
        )

        return {
            'total_loss': total_loss,
            'loss_mil': loss_mil,
            'loss_rec': loss_rec,
            'loss_sem': loss_sem,
            'loss_bg': loss_bg,
            'loss_compact': loss_compact,
            'loss_push': loss_push,
            'loss_ortho': loss_ortho,
            'loss_smooth': loss_smooth,
            'loss_sparse': loss_sparse
        }
