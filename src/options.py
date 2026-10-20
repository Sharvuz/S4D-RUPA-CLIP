"""
Configuration and Hyperparameter Options for S4D-RUPA-CLIP.
"""

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser(dataset='ucf'):
    parser = argparse.ArgumentParser(description=f'S4D-RUPA-CLIP for {dataset.upper()}')

    # General configuration
    parser.add_argument('--seed', default=234, type=int, help='Random seed for reproducibility')
    parser.add_argument('--dataset', default=dataset, choices=['ucf', 'xd'])
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num-workers', default=0, type=int)

    # Architecture: S4D & RUPA
    parser.add_argument('--in-dim', default=512, type=int, help='Input feature dimension (CLIP ViT-B/16)')
    parser.add_argument('--d-model', default=512, type=int, help='S4D backbone feature dimension')
    parser.add_argument('--num-layers', default=2, type=int, help='Number of Bidirectional S4D layers')
    parser.add_argument('--d-state', default=64, type=int, help='S4D state dimension')
    parser.add_argument('--expand', default=2, type=int, help='FFN expansion factor in BiS4D blocks')
    parser.add_argument('--model-scale', default='base', choices=['base', 'ultra'], help='Model capacity: base or ultra')
    parser.add_argument('--ssm-type', default='s4d', choices=['s4d', 'pure', 'official'], help='SSM backend')
    parser.add_argument('--num-prototypes', default=16, type=int, help='Number of normal prototypes in RUPA bank')
    parser.add_argument('--text-dim', default=512, type=int, help='CLIP text embedding dimension')
    parser.add_argument('--use-text-prompt', default=True, type=str2bool, help='Enable residual-to-text semantic alignment')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--gate-hidden', default=16, type=int, help='Hidden size for evidential gate MLP')
    parser.add_argument('--gate-init-bias', default=0.0, type=float, help='Initial bias for safe gate')

    # Data paths & parameters
    parser.add_argument('--visual-length', default=256, type=int)
    if dataset == 'ucf':
        parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv', type=str)
        parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv', type=str)
        parser.add_argument('--gt-path', default='list/gt_ucf.npy', type=str)
        parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy', type=str)
        parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy', type=str)
        parser.add_argument('--batch-size', default=64, type=int)
        parser.add_argument('--lr', default=5e-5, type=float)
        parser.add_argument('--max-epoch', default=15, type=int)
        parser.add_argument('--k-ratio', default=16.0, type=float)
    else:
        parser.add_argument('--train-list', default='list/xd_CLIP_rgb.csv', type=str)
        parser.add_argument('--test-list', default='list/xd_CLIP_rgbtest.csv', type=str)
        parser.add_argument('--gt-path', default='list/gt.npy', type=str)
        parser.add_argument('--gt-segment-path', default='list/gt_segment.npy', type=str)
        parser.add_argument('--gt-label-path', default='list/gt_label.npy', type=str)
        parser.add_argument('--batch-size', default=64, type=int)
        parser.add_argument('--lr', default=5e-5, type=float)
        parser.add_argument('--max-epoch', default=20, type=int)
        parser.add_argument('--k-ratio', default=16.0, type=float)

    # Training & Optimization
    parser.add_argument('--warmup-epochs', default=2, type=int, help='Number of warmup epochs')
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--grad-clip', default=1.0, type=float)

    # Loss Weights
    parser.add_argument('--loss-rec-weight', default=0.5, type=float, help='Weight of RUPA prototype reconstruction loss')
    parser.add_argument('--loss-sem-weight', default=0.5, type=float, help='Weight of residual semantic alignment loss')
    parser.add_argument('--loss-bg-weight', default=0.1, type=float, help='Weight of background residual suppression loss')
    parser.add_argument('--loss-ortho-weight', default=0.1, type=float, help='Weight of prototype orthogonality loss')
    parser.add_argument('--loss-smooth-weight', default=1e-4, type=float, help='Weight of temporal smoothness loss')
    parser.add_argument('--loss-sparse-weight', default=5e-4, type=float, help='Weight of anomaly sparsity loss')
    parser.add_argument('--margin', default=0.2, type=float, help='Margin for abnormal prototype push loss')

    # Checkpoint and evaluation
    parser.add_argument('--save-dir', default='models', type=str)
    parser.add_argument('--eval-interval', default=1, type=int)

    return parser


def apply_model_scale(args):
    """
    If model-scale is 'ultra', automatically scales up architecture and batch parameters
    to achieve maximum GPU utilization and SOTA performance, unless explicitly overridden.
    """
    import sys
    if getattr(args, 'model_scale', 'base') == 'ultra':
        argv_str = ' '.join(sys.argv)
        if '--num-layers' not in argv_str:
            args.num_layers = 4
        if '--expand' not in argv_str:
            args.expand = 4
        if '--d-state' not in argv_str:
            args.d_state = 128
        if '--num-prototypes' not in argv_str:
            args.num_prototypes = 32
        if '--gate-hidden' not in argv_str:
            args.gate_hidden = 64
        if args.dataset == 'xd':
            if '--batch-size' not in argv_str:
                args.batch_size = 96
            if '--max-epoch' not in argv_str:
                args.max_epoch = 15
        elif args.dataset == 'ucf':
            if '--batch-size' not in argv_str:
                args.batch_size = 64
            if '--max-epoch' not in argv_str:
                args.max_epoch = 15
        print(f"[Model Scale: ULTRA] Initializing heavyweight architecture: "
              f"layers={args.num_layers}, d_state={args.d_state}, expand={args.expand}, "
              f"prototypes={args.num_prototypes}, gate_hidden={args.gate_hidden}, "
              f"batch_size={args.batch_size}, epochs={args.max_epoch}")
    return args
