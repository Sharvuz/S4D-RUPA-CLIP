"""
Evaluation script for S4D-RUPA-CLIP on UCF-Crime.
Computes frame-level ROC-AUC, Average Precision (AP), and temporal mAP across IoU thresholds.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from rupa_clip_model import S4DRUPACLIP
from dataset import UCFDataset
from tools import get_batch_mask
from ucf_detectionMAP import getDetectionMAP as dmAP
from options import get_parser, apply_model_scale
from text_prompts import load_or_encode_text_features


def evaluate_ucf(model, test_loader, maxlen, gt, gtsegments, gtlabels, device, text_features=None):
    """
    Evaluates S4D-RUPA-CLIP on the complete UCF-Crime test set.
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

            # S4D-RUPA-CLIP Forward evaluation
            out = model(visual, padding_mask=padding_mask, text_features=text_features)
            final_scores = out['final_score'].reshape(-1)
            video_scores = final_scores[:len_cur].cpu().numpy()

            all_scores.append(video_scores)

            if out.get('class_probs') is not None:
                probs = out['class_probs'].squeeze(0)[:len_cur].cpu().numpy()  # (T, 14)
                dist_expanded = np.repeat(probs, 16, axis=0)
            else:
                pred_abn = video_scores[:, np.newaxis]
                pred_nor = 1.0 - pred_abn
                dist_2class = np.concatenate([pred_nor, pred_abn], axis=1)
                dist_expanded = np.repeat(dist_2class, 16, axis=0)
            element_logits_stack.append(dist_expanded)

            if (i + 1) % 50 == 0 or (i + 1) == len(test_loader):
                print(f"  [Validation] Processing test video [{i+1:03d}/{len(test_loader):03d}]...", flush=True)

    all_scores_flat = np.concatenate(all_scores, axis=0)
    frame_scores = np.repeat(all_scores_flat, 16)

    auc = roc_auc_score(gt[:len(frame_scores)], frame_scores) * 100.0
    ap = average_precision_score(gt[:len(frame_scores)], frame_scores) * 100.0

    print(f"\n==================== UCF-CRIME EVALUATION (S4D-RUPA-CLIP) ====================", flush=True)
    print(f"Frame ROC-AUC: {auc:.2f}% | Average Precision (AP): {ap:.2f}%", flush=True)

    try:
        dmap, iou = dmAP(element_logits_stack, gtsegments, gtlabels, excludeNormal=False)
        avg_map = 0.0
        for i in range(len(iou)):
            print(f"mAP@{iou[i]:.1f} = {dmap[i]:.2f}%", flush=True)
            avg_map += dmap[i]
        avg_map /= len(iou)
        print(f"Mean Average Precision (mAP): {avg_map:.2f}%", flush=True)
    except Exception as e:
        print(f"Note: Temporal mAP evaluation skipped: {e}", flush=True)

    print(f"===============================================================================\n", flush=True)
    return auc, ap


if __name__ == '__main__':
    parser = get_parser(dataset='ucf')
    parser.add_argument('--model-path', default='models/s4d_rupa_clip_ucf_best.pth', type=str)
    args = apply_model_scale(parser.parse_args())

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Running UCF-Crime evaluation on device: {device}")

    # Load pre-encoded text embeddings
    text_features = None
    if args.use_text_prompt:
        try:
            text_features = load_or_encode_text_features(device=device, dataset='ucf').float()
        except Exception as e:
            print(f"[Warning] Could not load text features: {e}. Running in visual-only mode.")

    test_dataset = UCFDataset(args.visual_length, args.test_list, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    checkpoint_state = None
    if os.path.exists(args.model_path):
        try:
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.model_path, map_location=device)
        checkpoint_state = checkpoint.get('model_state_dict', checkpoint)
        saved_args = checkpoint.get('args', {})
        if saved_args:
            args.num_layers = saved_args.get('num_layers', args.num_layers)
            args.d_state = saved_args.get('d_state', args.d_state)
            args.ssm_type = saved_args.get('ssm_type', args.ssm_type)
            args.expand = saved_args.get('expand', args.expand)
            args.num_prototypes = saved_args.get('num_prototypes', args.num_prototypes)
            args.gate_hidden = saved_args.get('gate_hidden', args.gate_hidden)
            print(f"[Auto-Config] Loaded architecture from checkpoint: layers={args.num_layers}, d_state={args.d_state}, expand={args.expand}, prototypes={args.num_prototypes}")

    model = S4DRUPACLIP(
        in_dim=args.in_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        d_state=args.d_state,
        ssm_type=args.ssm_type,
        expand=args.expand,
        num_prototypes=args.num_prototypes,
        text_dim=args.text_dim,
        dropout=args.dropout,
        gate_hidden=args.gate_hidden,
        gate_init_bias=args.gate_init_bias
    ).to(device)

    if checkpoint_state is not None:
        model.load_state_dict(checkpoint_state)
        print(f"Successfully loaded checkpoint: {args.model_path}")
    else:
        print(f"Warning: Checkpoint not found at {args.model_path}. Running with random initialization.")

    evaluate_ucf(model, test_loader, args.visual_length, gt, gtsegments, gtlabels, device, text_features=text_features)
