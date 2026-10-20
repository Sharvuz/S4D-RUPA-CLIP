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
    parser = argparse.ArgumentParser(description=f'DMAE-VAD for {dataset.upper()}')

    # General
    parser.add_argument('--seed', default=234, type=int, help='Random seed')
    parser.add_argument('--dataset', default=dataset, choices=['ucf', 'xd'])
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num-workers', default=2, type=int)

    # Architecture
    parser.add_argument('--in-dim', default=512, type=int)
    parser.add_argument('--d-model', default=512, type=int, help='Discriminator hidden dimension')
    parser.add_argument('--mae-dim', default=256, type=int, help='MAE inpainting hidden dimension')
    parser.add_argument('--num-tcn-layers', default=2, type=int)
    parser.add_argument('--num-mae-layers', default=2, type=int)
    parser.add_argument('--mask-ratio', default=0.5, type=float, help='Temporal snippet masking ratio')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--gate-init-bias', default=-2.5, type=float)

    # Data
    parser.add_argument('--visual-length', default=256, type=int)
    if dataset == 'ucf':
        parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv', type=str)
        parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv', type=str)
        parser.add_argument('--gt-path', default='list/gt_ucf.npy', type=str)
        parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy', type=str)
        parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy', type=str)
        parser.add_argument('--batch-size', default=64, type=int)
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--max-epoch', default=15, type=int)
        parser.add_argument('--k-ratio', default=16.0, type=float)
    else:
        parser.add_argument('--train-list', default='list/xd_CLIP_rgb.csv', type=str)
        parser.add_argument('--test-list', default='list/xd_CLIP_rgbtest.csv', type=str)
        parser.add_argument('--gt-path', default='list/gt.npy', type=str)
        parser.add_argument('--gt-segment-path', default='list/gt_segment.npy', type=str)
        parser.add_argument('--gt-label-path', default='list/gt_label.npy', type=str)
        parser.add_argument('--batch-size', default=64, type=int)
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--max-epoch', default=20, type=int)
        parser.add_argument('--k-ratio', default=16.0, type=float)

    # Training
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--grad-clip', default=1.0, type=float)

    # Losses
    parser.add_argument('--mil-margin', default=1.0, type=float)
    parser.add_argument('--rec-margin', default=0.2, type=float)
    parser.add_argument('--loss-mae-weight', default=1.0, type=float)
    parser.add_argument('--loss-rec-contrast-weight', default=0.5, type=float)
    parser.add_argument('--loss-smooth-weight', default=8e-5, type=float)
    parser.add_argument('--loss-sparse-weight', default=8e-5, type=float)
    parser.add_argument('--loss-aux-det-weight', default=0.2, type=float)

    # Checkpoint
    parser.add_argument('--save-dir', default='models', type=str)
    parser.add_argument('--eval-interval', default=1, type=int)

    return parser
