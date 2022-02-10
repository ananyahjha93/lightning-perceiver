#!/bin/sh

python src/tasks/image/classification.py \
    --datamodule cifar10 \
    --data_dir . \
    --accelerator gpu \
    --num_accelerators 8 \
    --img_size 32 \
    --batch_size 128 \
    --num_workers 8 \
    --optimizer lamb \
    --scheduler no_warmup \
    --learning_rate 0.001 \
    --weight_decay 0.1 \
    --training_epochs 1600 \
    --decay_start_from 50 \
    --num_classes 10 \
    --num_blocks 8 \
    --z_index_dim 128 \
    --num_z_channels 256 \
    --num_self_attends_per_block 6 \
    --fourier_enc_bands 32 \
    --gradient_clip_val 1.
