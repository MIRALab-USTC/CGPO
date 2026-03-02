#!/bin/bash
gpu_list=$1


cd /home/xzliang/General-Reasoner/utils

python vllm-inference.py --bench mt-bench-v2 --temp 0.75 --topp 0.95 --gpus ${gpu_list} 