#!/bin/bash


###### 要改
original_model_path=/home/shared/xzliang/Qwen2.5-7B-Instruct
ckpt=159

for exp_name in REPTILE-lr1e-6-replr1.2-qwen2.5-7b-instruct-guru-math-codegen-stem-creative-easy_250909232344 REPTILE-lr1e-6-replr0.9-qwen2.5-7b-instruct-guru-math-codegen-stem-creative-easy_250908203249 OMNITHINKER-qwen2.5-7b-instruct-guru-math-codegen-stem-creative-easy_250917011459 SELFPACEDCL-qwen2.5-7b-instruct-guru-math-codegen-stem-creative-easy_250917152158; do
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
done

ckpt=160
for exp_name in JOINT-qwen2.5-7b-instruct-guru-math-codegen-stem-creative6k-easy_250829023041; do
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
done



###### 要改
original_model_path=/home/shared/xzliang/Qwen2.5-3B-Instruct
ckpt=159

for exp_name in JOINT-qwen2.5-3b-instruct-guru-math-codegen-stem-creative-easy_250918233010 OMNITHINKER-qwen2.5-3b-instruct-guru-math-codegen-stem-creative-easy_250920152817 SELFPACEDCL-qwen2.5-3b-instruct-guru-math-codegen-stem-creative-easy_250921013017 REPTILE-lr1e-6-replr1.5-qwen2.5-3b-instruct-guru-math-codegen-stem-creative-easy_250921231355; do
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
done