#!/bin/bash


###### 要改
original_model_path=/home/shared/xzliang/Qwen2.5-7B-Instruct
exp_name=CGPO-replr1.2-mot1of8-raiden-sft-ckpt5600-math-code-creative-easy_251114130750
ckpt=610


#######
ckpt_path=/home/xzliang/General-Reasoner/checkpoint
local_dir=${ckpt_path}/${exp_name}/global_step_${ckpt}/actor
merged_model_path=${ckpt_path}/merged/${exp_name}-ckpt${ckpt}

python /home/xzliang/General-Reasoner/verl/scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path ${original_model_path} \
    --local_dir ${local_dir} \
    --target_dir ${merged_model_path}

cp ${original_model_path}/merges.txt ${merged_model_path}/
cp ${original_model_path}/tokenizer.json ${merged_model_path}/
cp ${original_model_path}/tokenizer_config.json ${merged_model_path}/
cp ${original_model_path}/vocab.json ${merged_model_path}/