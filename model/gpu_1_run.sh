#!/bin/bash
mkdir -p result

python3 preprocess-train.py
python3 preprocess-eval.py
CUDA_VISIBLE_DEVICES=1 python3 -u train.py --model_type=VAE3 --CC=1 || exit 1;
python3 -u convert.py --model_type=VAE3 --model_path=model/VAE3_CC --convert_path=result/VAE3_cycle || exit 1;