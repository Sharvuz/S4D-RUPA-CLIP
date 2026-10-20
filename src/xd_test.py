"""
Evaluation script for S4D-RUPA-CLIP on XD-Violence.
Computes frame-level ROC-AUC, Average Precision (AP), and temporal mAP across IoU thresholds.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from rupa_clip_model import S4DRUPACLIP
from dataset import XDDataset
from tools import get_batch_mask
from xd_detectionMAP import getDetectionMAP as dmAP
from options import get_parser
from text_prompts import load_or_encode_text_features


def evaluate_xd(model, test_loader, maxlen, gt, gtsegments, gtlabels, device, text_features=None):
    """
    Evaluates S4D-RUPA-CLIP on the XD-Violence test set.
    """
    model.eval()
    all_scores = []
    element_logits_stack = []

    with torch.no_grad():
        for i, item in enumerate(test_loader):
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length

            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            num_chunks = int(np.ceil(len_cur / maxlen))
            lengths = torch.zeros(num_chunks, dtype=torch.long)
            rem = len_cur
            for j in range(num_chunks):
                if rem >= maxlen:
                    lengths[j] = maxlen
                    rem -= maxlen
                else:
                    lengths[j] = rem

            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            out = model(visual, padding_mask=padding_mask, text_features=text_features)
            final_scores = out['final_score'].reshape(-1)
            video_scores = final_scores[:len_cur].cpu().numpy()

            all_scores.append(video_scores)

            pred_abn = video_scores[:, np.newaxis]
            pred_nor = 1.0 - pred_abn
            dist_2class = np.concatenate([pred_nor, pred_abn], axis=1)
            dist_expanded = np.repeat(dist_2class, 16, axis=0)
            element_logits_stack.append(dist_expanded)

    all_scores_flat = np.concatenate(all_scores, axis=0)
    frame_scores = np.repeat(all_scores_flat, 16)

    auc = roc_auc_score(gt[:len(frame_scores)], frame_scores) * 100.0
    ap = average_precision_score(gt[:len(frame_scores)], frame_scores) * 100.0

    print(f"\n==================== XD-VIOLENCE EVALUATION (S4D-RUPA-CLIP) ====================")
    print(f"Average Precision (AP): {ap:.2f}% | Frame ROC-AUC: {auc:.2f}%")

    try:
        dmap, iou = dmAP(element_logits_stack, gtsegments, gtlabels, excludeNormal=False)
        avg_map = 0.0
        for i in range(len(iou)):
            print(f"mAP@{iou[i]:.1f} = {dmap[i]:.2f}%")
            avg_map += dmap[i]
        avg_map /= len(iou)
        print(f"Mean Average Precision (mAP): {avg_map:.2f}%")
    except Exception as e:
        print(f"Note: Temporal mAP evaluation skipped: {e}")

    print(f"=================================================================================\n")
    return auc, ap


if __name__ == '__main__':
    parser = get_parser(dataset='xd')
    parser.add_argument('--model-path', default='models/s4d_rupa_clip_xd_best.pth', type=str)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Running XD-Violence evaluation on device: {device}")

    # Load pre-encoded text embeddings
    text_features = None
    if args.use_text_prompt:
        try:
            text_features = load_or_encode_text_features(device=device, dataset='xd')
        except Exception as e:
            print(f"[Warning] Could not load text features: {e}. Running in visual-only mode.")

    test_dataset = XDDataset(args.visual_length, args.test_list, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = S4DRUPACLIP(
        in_dim=args.in_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        d_state=args.d_state,
        ssm_type=args.ssm_type,
        num_prototypes=args.num_prototypes,
        text_dim=args.text_dim,
        dropout=args.dropout,
        gate_hidden=args.gate_hidden,
        gate_init_bias=args.gate_init_bias
    ).to(device)

    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded checkpoint: {args.model_path}")
    else:
        print(f"Warning: Checkpoint not found at {args.model_path}. Running with random initialization.")

    evaluate_xd(model, test_loader, args.visual_length, gt, gtsegments, gtlabels, device, text_features=text_features)
