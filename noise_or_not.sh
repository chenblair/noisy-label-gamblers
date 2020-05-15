#!/bin/bash

for noise_rate in 0.2 0.4 0.6 0.8
do
    python3 main.py --seed 10 --dataset mnist --noise_rate $noise_rate --lr 0.001 --batch_size 128 --noise_type symmetric --n_epoch 50 --lambda_type gmblers --eps 9.9 --result_dir results/noise_or_not
done

# python3 main.py --result_dir results/grid_search --dataset mnist --noise_rate 0.0 --smoothing 1.0 --lr 0.1 --batch_size 128 --noise_type symmetric --n_epoch 50 --lambda_type nll
# python3 main.py --result_dir results/grid_search --dataset cifar10 --noise_rate 0.0 --smoothing 1.0 --batch_size 128 --noise_type symmetric --n_epoch 50 --lambda_type nll