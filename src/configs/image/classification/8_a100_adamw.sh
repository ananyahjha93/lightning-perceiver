#!/bin/sh

python src/tasks/images/classification.py \
    --accelerator gpu \
    --num_accelerators 8 \
    --batch_size 16 \
    --num_workers 4 \
    --optimizer adamw \
    --scheduler warmup \
    --learning_rate 0.0001 \
    --weight_decay 0.05 \
    --training_epochs 300 \
    --warmup_epochs 5 \
    --data_dir /home/ananya/imagenet/data/imagenet_2012
