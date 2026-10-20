import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import DMAEVAD
from dataset import XDDataset
from losses import DMAEVADLoss
from options import get_parser
from xd_test import evaluate_xd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_xd():
    parser = get_parser(dataset='xd')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"\n=======================================================", flush=True)
    print(f"  DMAE-VAD: Decoupled Dual-Stream Masked Autoencoder", flush=True)
    print(f"  Training from Scratch on XD-Violence (End-to-End)", flush=True)
    print(f"  Device: {device} | Batch Size: {args.batch_size} | Epochs: {args.max_epoch}", flush=True)
    print(f"=======================================================\n", flush=True)

    train_abn_dataset = XDDataset(args.visual_length, args.train_list, test_mode=False, normal=False)
    train_nor_dataset = XDDataset(args.visual_length, args.train_list, test_mode=False, normal=True)

    train_abn_loader = DataLoader(
        train_abn_dataset, batch_size=args.batch_size // 2, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    train_nor_loader = DataLoader(
        train_nor_dataset, batch_size=args.batch_size // 2, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )

    test_dataset = XDDataset(args.visual_length, args.test_list, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = DMAEVAD(
        in_dim=args.in_dim,
        d_model=args.d_model,
        mae_dim=args.mae_dim,
        num_tcn_layers=args.num_tcn_layers,
        num_mae_layers=args.num_mae_layers,
        mask_ratio=args.mask_ratio,
        dropout=args.dropout,
        gate_init_bias=args.gate_init_bias
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}\n", flush=True)

    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"🔥 KÍCH HOẠT CHẾ ĐỘ CHẠY KÉP TRÊN {torch.cuda.device_count()} GPUs (DataParallel)!\n", flush=True)
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-6)

    criterion = DMAEVADLoss(
        mil_margin=args.mil_margin,
        rec_margin=args.rec_margin,
        loss_mae_weight=args.loss_mae_weight,
        loss_rec_contrast_weight=args.loss_rec_contrast_weight,
        loss_smooth_weight=args.loss_smooth_weight,
        loss_sparse_weight=args.loss_sparse_weight,
        loss_aux_det_weight=args.loss_aux_det_weight,
        k_ratio=args.k_ratio
    )

    best_ap = 0.0
    best_auc = 0.0

    for epoch in range(args.max_epoch):
        model.train()
        nor_iter = iter(train_nor_loader)
        epoch_loss = 0.0
        steps = 0

        for step, abn_batch in enumerate(train_abn_loader):
            try:
                nor_batch = next(nor_iter)
            except StopIteration:
                nor_iter = iter(train_nor_loader)
                nor_batch = next(nor_iter)

            abn_feat = abn_batch[0].to(device)
            nor_feat = nor_batch[0].to(device)

            abn_out = model(abn_feat)
            nor_out = model(nor_feat)

            total_loss, loss_dict = criterion(
                abn_out, nor_out,
                abn_orig_feat=abn_feat,
                nor_orig_feat=nor_feat,
                current_epoch=epoch
            )

            optimizer.zero_grad()
            total_loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            epoch_loss += total_loss.item()
            steps += 1

            if step % 20 == 0:
                print(
                    f"Epoch [{epoch+1:02d}/{args.max_epoch:02d}] Step [{step:03d}/{len(train_abn_loader):03d}] | "
                    f"Loss: {total_loss.item():.4f} (MIL: {loss_dict['loss_mil']:.4f}, "
                    f"MAE: {loss_dict['loss_mae']:.4f}, RecContrast: {loss_dict['loss_rec_contrast']:.4f}) | "
                    f"Gate: {loss_dict['gate_mean']:.3f} | TopK Abn/Nor: {loss_dict['abn_topk']:.3f}/{loss_dict['nor_topk']:.3f}",
                    flush=True
                )

        scheduler.step()
        avg_loss = epoch_loss / max(steps, 1)
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n--> Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}\n", flush=True)

        if (epoch + 1) % args.eval_interval == 0:
            print(f"--- Running Validation at Epoch {epoch+1} ---", flush=True)
            auc, ap = evaluate_xd(model, test_loader, args.visual_length, gt, gtsegments, gtlabels, device)

            raw_model = model.module if hasattr(model, 'module') else model
            save_state_dict = raw_model.state_dict()

            is_best = ap > best_ap
            if is_best:
                best_ap = ap
                best_auc = auc
                best_path = os.path.join(args.save_dir, 'dmae_vad_xd_best.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': save_state_dict,
                    'best_ap': best_ap,
                    'best_auc': best_auc,
                    'args': vars(args)
                }, best_path)
                print(f"🌟 NEW BEST MODEL SAVED: AP = {best_ap:.2f}% | AUC = {best_auc:.2f}% at {best_path}\n", flush=True)

            latest_path = os.path.join(args.save_dir, 'dmae_vad_xd_latest.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': save_state_dict,
                'latest_ap': ap,
                'latest_auc': auc
            }, latest_path)

    print(f"\n=======================================================", flush=True)
    print(f"  Training Finished! Best AP: {best_ap:.2f}% | Best AUC: {best_auc:.2f}%", flush=True)
    print(f"=======================================================\n", flush=True)


if __name__ == '__main__':
    train_xd()
