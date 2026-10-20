#!/bin/bash
# ========================================================
#   DMAE-VAD: Train from Scratch on UCF-Crime (Linux/Kaggle)
# ========================================================

python src/ucf_train.py \
    --dataset ucf \
    --batch-size 64 \
    --lr 1e-4 \
    --max-epoch 15 \
    --num-tcn-layers 2 \
    --num-mae-layers 2 \
    --mask-ratio 0.5 \
    --save-dir models
