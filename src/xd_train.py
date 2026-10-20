"""
Training script for S4D-RUPA-CLIP on XD-Violence from scratch.
End-to-End training with Non-Saturating DeepMIL BCE + Multi-Prompt Residual Alignment.
Primary benchmark target: Average Precision (AP) >= 87%.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rupa_clip_model import S4DRUPACLIP
from losses import S4DRUPALoss
from dataset import XDDataset
from options import get_parser, apply_model_scale
from text_prompts import load_or_encode_text_features, get_class_index
from xd_test import evaluate_xd


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(
    epoch: int,
    model: nn.Module,
    normal_loader: DataLoader,
    abnormal_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: S4DRUPALoss,
    device: torch.device,
    args
):
    model.train()
    total_steps = min(len(normal_loader), len(abnormal_loader))
    normal_iter = iter(normal_loader)
    abnormal_iter = iter(abnormal_loader)
    print(f"  [Epoch {epoch:02d}] Initializing batch iterator ({total_steps} steps total)...", flush=True)

    running_loss = 0.0
    running_mil = 0.0
    running_sem = 0.0
    running_rec = 0.0
    running_bg = 0.0
    running_gate = 0.0

    for step in range(total_steps):
        abn_feat, abn_label_str, _ = next(abnormal_iter)
        nor_feat, _, _ = next(normal_iter)

        abn_feat = abn_feat.to(device)
        nor_feat = nor_feat.to(device)

        abn_class_indices = torch.tensor(
            [get_class_index(lbl, dataset='xd') for lbl in abn_label_str],
            dtype=torch.long,
            device=device
        )

        optimizer.zero_grad()

        abn_out = model(abn_feat)
        nor_out = model(nor_feat)

        loss_dict = criterion(abn_out, nor_out, abn_class_indices=abn_class_indices)
        loss = loss_dict['total_loss']

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            abn_topk = criterion.get_topk_mean(abn_out['final_score']).mean().item()
            nor_topk = criterion.get_topk_mean(nor_out['final_score']).mean().item()
            gate_val = abn_out['gate'].mean().item()

        running_loss += loss.item()
        running_mil += loss_dict['loss_mil'].item()
        running_sem += loss_dict['loss_sem'].item()
        running_rec += loss_dict['loss_rec'].item()
        running_bg += loss_dict['loss_bg'].item()
        running_gate += gate_val

        if step < 5 or step % 10 == 0 or step == total_steps - 1:
            print(
                f"Epoch [{epoch:02d}/{args.max_epoch:02d}] Step [{step:03d}/{total_steps:03d}] | "
                f"Loss: {loss.item():.4f} (MIL: {loss_dict['loss_mil'].item():.4f}, "
                f"Sem: {loss_dict['loss_sem'].item():.4f}, "
                f"Rec: {loss_dict['loss_rec'].item():.4f}, "
                f"BG: {loss_dict['loss_bg'].item():.4f}) | "
                f"Gate: {gate_val:.3f} | TopK Abn/Nor: {abn_topk:.3f}/{nor_topk:.3f}",
                flush=True
            )

    return running_loss / total_steps


def main():
    parser = get_parser(dataset='xd')
    args = apply_model_scale(parser.parse_args())

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    num_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0

    print("=" * 65)
    print("  S4D-RUPA-CLIP: Training from Scratch on XD-Violence")
    print("  Decoupled Residual-to-Text Alignment | Standalone (No DSANet)")
    print(f"  Device: {device} (GPUs: {num_gpus}) | Batch Size: {args.batch_size} | Epochs: {args.max_epoch}")
    print(f"  Architecture: Layers={args.num_layers} | d_state={args.d_state} | expand={args.expand} | Prototypes={args.num_prototypes}")
    print("=" * 65)

    os.makedirs(args.save_dir, exist_ok=True)

    text_features = None
    if args.use_text_prompt:
        text_features = load_or_encode_text_features(device=device, dataset='xd').float()
        print(f"[TextPrompts] Active text embedding matrix shape: {text_features.shape}")

    half_batch = max(1, args.batch_size // 2)
    normal_dataset = XDDataset(args.visual_length, args.train_list, test_mode=False, normal=True)
    abnormal_dataset = XDDataset(args.visual_length, args.train_list, test_mode=False, normal=False)

    normal_loader = DataLoader(
        normal_dataset,
        batch_size=half_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    abnormal_loader = DataLoader(
        abnormal_dataset,
        batch_size=half_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )

    test_dataset = XDDataset(args.visual_length, args.test_list, test_mode=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    print(f"Loaded Datasets: Normal Train={len(normal_dataset)} | Abnormal Train={len(abnormal_dataset)} | Test={len(test_dataset)}")

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

    if text_features is not None:
        model.set_text_features(text_features)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {trainable_params:,}")

    if num_gpus > 1:
        print(f"Activating Dual-GPU DataParallel across {num_gpus} GPUs!")
        model = nn.DataParallel(model)

    criterion = S4DRUPALoss(
        loss_rec_weight=args.loss_rec_weight,
        loss_sem_weight=args.loss_sem_weight,
        loss_bg_weight=args.loss_bg_weight,
        loss_ortho_weight=args.loss_ortho_weight,
        loss_smooth_weight=args.loss_smooth_weight,
        loss_sparse_weight=args.loss_sparse_weight,
        k_ratio=args.k_ratio,
        margin=args.margin
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.warmup_epochs > 0 and args.max_epoch > args.warmup_epochs:
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(args.max_epoch - args.warmup_epochs),
            eta_min=1e-6
        )
    else:
        cosine_scheduler = None

    best_ap = 0.0
    best_auc = 0.0
    best_epoch = 0

    best_model_path = os.path.join(args.save_dir, "s4d_rupa_clip_xd_best.pth")
    latest_model_path = os.path.join(args.save_dir, "s4d_rupa_clip_xd_latest.pth")

    print("\nStarting Training Pipeline...")
    for epoch in range(1, args.max_epoch + 1):
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * (epoch / float(args.warmup_epochs))
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            current_lr = warmup_lr
        else:
            if cosine_scheduler is not None:
                cosine_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

        print(f"\n--- Epoch [{epoch:02d}/{args.max_epoch:02d}] (LR: {current_lr:.6f}) ---")
        train_loss = train_epoch(
            epoch=epoch,
            model=model,
            normal_loader=normal_loader,
            abnormal_loader=abnormal_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            args=args
        )

        if epoch % args.eval_interval == 0:
            print(f"\n[Validation] Evaluating Epoch {epoch} on XD-Violence test set...")
            eval_model = model.module if isinstance(model, nn.DataParallel) else model
            auc, ap = evaluate_xd(
                model=eval_model,
                test_loader=test_loader,
                maxlen=args.visual_length,
                gt=gt,
                gtsegments=gtsegments,
                gtlabels=gtlabels,
                device=device,
                text_features=text_features
            )

            torch.save({
                'epoch': epoch,
                'model_state_dict': eval_model.state_dict(),
                'auc': auc,
                'ap': ap,
                'args': vars(args)
            }, latest_model_path)

            # Primary metric for XD-Violence is Average Precision (AP)
            if ap > best_ap:
                best_ap = ap
                best_auc = auc
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': eval_model.state_dict(),
                    'auc': auc,
                    'ap': ap,
                    'args': vars(args)
                }, best_model_path)
                print(f"⭐ NEW BEST XD-VIOLENCE MODEL! Epoch {epoch}: AP = {best_ap:.2f}% | ROC-AUC = {best_auc:.2f}%")

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print(f"  Best Epoch: {best_epoch:02d}")
    print(f"  Best AP: {best_ap:.2f}% | Best ROC-AUC: {best_auc:.2f}%")
    print(f"  Best Model Saved At: {best_model_path}")
    print("=" * 65)


if __name__ == '__main__':
    main()
