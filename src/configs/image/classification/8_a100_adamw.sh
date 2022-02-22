#!/bin/sh

python src/tasks/image/classification.py \
    --datamodule imagenet \
    --accelerator gpu \
    --num_accelerators 8 \
    --batch_size 16 \
    --num_workers 4 \
    --optimizer lamb \
    --scheduler multi_step \
    --learning_rate 0.000125 \
    --weight_decay 0.1 \
    --training_epochs 120 \
    --milestones 84 102 114 \
    --decay_coeff 0.1 \
    --data_dir /home/ananya/imagenet/data/imagenet_2012
