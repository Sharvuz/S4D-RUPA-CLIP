#!/bin/bash
# ========================================================
#   DMAE-VAD: Train from Scratch on XD-Violence (Linux/Kaggle)
# ========================================================

python src/xd_train.py \
    --dataset xd \
    --batch-size 64 \
    --lr 1e-4 \
    --max-epoch 20 \
    --num-tcn-layers 2 \
    --num-mae-layers 2 \
    --mask-ratio 0.5 \
    --save-dir models
