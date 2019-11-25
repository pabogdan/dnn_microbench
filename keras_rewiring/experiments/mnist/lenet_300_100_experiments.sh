#!/bin/bash

mkdir experiment_text
mkdir experiment_text/lenet_300_100

exp=experiment_text/lenet_300_100

nohup python train_and_test_lenet_300_100.py :lenet_300_100 \
    --epochs 10 --batch 10 \
    --optimizer ns --suffix DENSE \
    > $exp/dense.out 2>&1 &

python train_and_test_lenet_300_100.py :lenet_300_100 \
    --epochs 10 --batch 10 --sparse_layers \
    --optimizer ns --suffix HARD\
    > $exp/hard.out 2>&1 &

python train_and_test_lenet_300_100.py :lenet_300_100 \
    --epochs 10 --batch 10 --sparse_layers --soft_rewiring \
    --optimizer ns --suffix SOFT \
    > $exp/soft.out 2>&1 &

python train_and_test_lenet_300_100.py :lenet_300_100 \
    --epochs 10 --batch 10 --sparse_layers --conn_decay \
    --optimizer ns --suffix DECAY \
    > $exp/decay.out 2>&1 &

python train_and_test_lenet_300_100.py :lenet_300_100 \
    --epochs 10 --batch 10 --sparse_layers --disable_rewiring \
    --optimizer ns --suffix SPARSE_NO_REWIRING \
    > $exp/no_rew.out 2>&1 &
