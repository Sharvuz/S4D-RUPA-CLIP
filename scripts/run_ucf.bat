@echo off
REM ========================================================
REM   S4D-RUPA-CLIP: Train from Scratch on UCF-Crime (Windows)
REM ========================================================

python src/ucf_train.py ^
    --dataset ucf ^
    --batch-size 64 ^
    --lr 5e-5 ^
    --max-epoch 15 ^
    --warmup-epochs 2 ^
    --num-layers 2 ^
    --d-state 64 ^
    --ssm-type s4d ^
    --num-prototypes 16 ^
    --use-text-prompt True ^
    --loss-rec-weight 0.5 ^
    --loss-sem-weight 0.5 ^
    --loss-bg-weight 0.1 ^
    --save-dir models
