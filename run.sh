#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 ./train_dual.py --model_name ASFEN --dataset twitter17 --seed 1000 --bert_lr 1e-5 --num_epoch 10 --hidden_dim 1024 --max_length 100 --cuda 0,1
python -m torch.distributed.launch --nproc_per_node=2 ./train_dual.py --model_name ASFEN --dataset twitter15 --seed 1000 --bert_lr 1e-5 --num_epoch 10 --hidden_dim 1024 --max_length 100 --cuda 0,1

