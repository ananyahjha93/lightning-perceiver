#!/bin/sh

python src/tasks/images/classification.py \
    --datamodule cifar10 \
    --accelerator gpu \
    --num_accelerators 8 \
    --img_size 32 \
    --batch_size 256 \
    --num_workers 8 \
    --optimizer lamb \
    --scheduler no_warmup \
    --learning_rate 0.002 \
    --weight_decay 0.1 \
    --training_epochs 110 \
    --num_classes 10 \
    --num_blocks 4 \
    --z_index_dim 128 \
    --num_z_channels 256 \
    --num_self_attends_per_block 4 \
    --fourier_enc_bands 16

