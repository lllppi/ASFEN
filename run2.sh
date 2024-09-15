#!/bin/bash
 python ./train_single.py --model_name ASFEN --dataset twitter15 --seed 1000 --bert_lr 1e-5 --num_epoch 10 --hidden_dim 1024 --max_length 100 --cuda 0