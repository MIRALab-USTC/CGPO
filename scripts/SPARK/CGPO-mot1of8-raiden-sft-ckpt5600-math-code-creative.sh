#!/bin/bash

USER_ENV=`whoami`

set -x

gpu_list=$1
gpu_count=$(echo "$gpu_list" | tr ',' ' ' | wc -w)
cpu_count=$(nproc)
timestamp="${2:-$(TZ='Asia/Shanghai' date +"%y%m%d%H%M%S")}"


if [ "$gpu_count" -lt 2 ]; then
    tensor_model_parallel_size=1
else
    tensor_model_parallel_size=2
fi

ray stop --force

CUDA_VISIBLE_DEVICES=${gpu_list} ray start \
    --head \
    --num-gpus ${gpu_count} \
    --port=6391 \
    --dashboard-port=8251 \
    --ray-debugger-external

export NCCL_DEBUG=DEBUG
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=1
# sandbox
export CODER1_EXEC="sandboxfusion"
export SANDBOX_FUSION_SERVERS="localhost"

export PROJECT_NAME="Multi-domain-RL"
export WANDB_API_KEY="a454837ba515fc31d2c8a3088353b017183f47dd"
export WANDB_OFFICIAL=1
# export VLLM_ATTENTION_BACKEND=XFORMERS
export HDFS_DATA_PATH="/home/shared/xzliang/data"
export HDFS__MODELPATH="/home/xzliang/General-Reasoner/checkpoint/merged"
export HDFS_CHECKPOINT_PATH="/home/xzliang/General-Reasoner/checkpoint"
export HDFS_LOG_PATH="/home/xzliang/General-Reasoner/log"
export HDFS_QUESTION_AND_RESPONSE_PATH="/home/xzliang/General-Reasoner/question_and_response"
export HDFS_LLM_JUDGEMENT_PATH="/home/xzliang/General-Reasoner/llm_judgement"

export DATASET_NAME="guru-RL-92k"
export TRAIN_DATA_DIR="$HDFS_DATA_PATH/$DATASET_NAME/train"
export OFFLINE_TEST_DATA_DIR="$HDFS_DATA_PATH/$DATASET_NAME/offline_eval"
export ONLINE_TEST_DATA_DIR="$HDFS_DATA_PATH/$DATASET_NAME/online_eval"

### 数据集
# Math (train)
math_train_path=${TRAIN_DATA_DIR}/math__combined_54.4k.parquet
math_easy_train_path=${TRAIN_DATA_DIR}/math__combined_easy_6.25k.parquet
math_14k_train_path=${TRAIN_DATA_DIR}/math__combined_easy_14k.parquet
# Math (online_test)
math_online_test_path=${ONLINE_TEST_DATA_DIR}/math__math_500.parquet
aime_online_test_path=${ONLINE_TEST_DATA_DIR}/math__aime_repeated_8x_240.parquet
amc_online_test_path=${ONLINE_TEST_DATA_DIR}/math__amc_repeated_4x_332.parquet
# Math (offline_test)
math_offline_test_path=${OFFLINE_TEST_DATA_DIR}/math__math_500.parquet
aime_offline_test_path=${OFFLINE_TEST_DATA_DIR}/math__aime_repeated_8x_240.parquet
amc_offline_test_path=${OFFLINE_TEST_DATA_DIR}/math__amc_repeated_4x_332.parquet

# Code (train)
leetcode_train_path=${TRAIN_DATA_DIR}/codegen__leetcode2k_1.3k.parquet
leetcode_easy_train_path=${TRAIN_DATA_DIR}/codegen__leetcode2k_easy_0.9k.parquet
livecodebench_train_path=${TRAIN_DATA_DIR}/codegen__livecodebench_440.parquet
livecodebench_easy_train_path=${TRAIN_DATA_DIR}/codegen__livecodebench_easy_145.parquet
primeintellect_train_path=${TRAIN_DATA_DIR}/codegen__primeintellect_7.5k.parquet
primeintellect_easy_train_path=${TRAIN_DATA_DIR}/codegen__primeintellect_easy_2k.parquet
taco_train_path=${TRAIN_DATA_DIR}/codegen__taco_8.8k.parquet
taco_easy_train_path=${TRAIN_DATA_DIR}/codegen__taco_easy_1.5k.parquet
# Code (online_test)
humaneval_online_test_path=${ONLINE_TEST_DATA_DIR}/codegen__humaneval_164.parquet
mbpp_online_test_path=${ONLINE_TEST_DATA_DIR}/codegen__mbpp_200.parquet
livecodebench_online_test_path=${ONLINE_TEST_DATA_DIR}/codegen__livecodebench_279.parquet
# Code (offline_test)
humaneval_offline_test_path=${OFFLINE_TEST_DATA_DIR}/codegen__humaneval_164.parquet
mbpp_offline_test_path=${OFFLINE_TEST_DATA_DIR}/codegen__mbpp_500.parquet
livecodebench_offline_test_path=${OFFLINE_TEST_DATA_DIR}/codegen__livecodebench_279.parquet

# Logic (train)
arcagi1_train_path=${TRAIN_DATA_DIR}/logic__arcagi1_111.parquet
arcagi2_train_path=${TRAIN_DATA_DIR}/logic__arcagi2_190.parquet
barc_train_path=${TRAIN_DATA_DIR}/logic__barc_1.6k.parquet
graph_train_path=${TRAIN_DATA_DIR}/logic__graph_logical_1.2k.parquet
ordering_train_path=${TRAIN_DATA_DIR}/logic__ordering_puzzle_1.9k.parquet
zebra_train_path=${TRAIN_DATA_DIR}/logic__zebra_puzzle_1.3k.parquet
# Logic (online_test)
zebralogic_online_test_path=${ONLINE_TEST_DATA_DIR}/logic__zebra_puzzle_dataset_200.parquet
ordering_puzzle_online_test_path=${ONLINE_TEST_DATA_DIR}/logic__ordering_puzzle_dataset_100.parquet
# Logic (offline_test)
zebralogic_offline_test_path=${OFFLINE_TEST_DATA_DIR}/logic__zebra_puzzle_dataset_300.parquet
ordering_puzzle_offline_test_path=${OFFLINE_TEST_DATA_DIR}/logic__ordering_puzzle_dataset_150.parquet

# Simulation (train)
codeio_train_path=${TRAIN_DATA_DIR}/simulation__codeio_3.7k.parquet
# Simulation (online_test)
codeio_online_test_path=${ONLINE_TEST_DATA_DIR}/simulation__codeio_200.parquet
arcagi1_online_test_path=${ONLINE_TEST_DATA_DIR}/simulation__arcagi1_200.parquet
# Simulation (offline_test)
codeio_offline_test_path=${OFFLINE_TEST_DATA_DIR}/simulation__codeio_500.parquet
arcagi1_offline_test_path=${OFFLINE_TEST_DATA_DIR}/simulation__arcagi1_200.parquet

# Table (train)
hitab_train_path=${TRAIN_DATA_DIR}/table__hitab_4.3k.parquet
multihier_train_path=${TRAIN_DATA_DIR}/table__multihier_1.5k.parquet
# Table (online_test)
multihier_online_test_path=${ONLINE_TEST_DATA_DIR}/table__multihier_200.parquet
hitab_online_test_path=${ONLINE_TEST_DATA_DIR}/table__hitab_200.parquet
# Table (offline_test)
multihier_offline_test_path=${OFFLINE_TEST_DATA_DIR}/table__multihier_300.parquet
hitab_offline_test_path=${OFFLINE_TEST_DATA_DIR}/table__hitab_300.parquet

# Stem (train)
webinstruct_train_path=${TRAIN_DATA_DIR}/stem__web_3.6k.parquet
# Stem (online_test)
supergpqa_online_test_path=${ONLINE_TEST_DATA_DIR}/stem__supergpqa_200.parquet
# Stem (offline_test)
gpqa_diamond_offline_test_path=${OFFLINE_TEST_DATA_DIR}/stem__gpqa_diamond_198.parquet
supergpqa_offline_test_path=${OFFLINE_TEST_DATA_DIR}/stem__supergpqa_200.parquet

# Creative (train)
sharegpt_train_path=${TRAIN_DATA_DIR}/creative__sharegpt_2.0k.parquet
wildchat_train_path=${TRAIN_DATA_DIR}/creative__wildchat_2.0k.parquet
litbench_train_path=${TRAIN_DATA_DIR}/creative__litbench_2.0k.parquet

# 训练数据和测试数据
# train_files="['${math_easy_train_path}','${leetcode_easy_train_path}','${livecodebench_easy_train_path}','${primeintellect_easy_train_path}','${taco_easy_train_path}','${sharegpt_train_path}','${wildchat_train_path}','${litbench_train_path}']"
train_files="['${math_easy_train_path}','${leetcode_easy_train_path}','${livecodebench_easy_train_path}','${primeintellect_easy_train_path}','${taco_easy_train_path}','${sharegpt_train_path}','${wildchat_train_path}','${litbench_train_path}']"
# train_files="['${gsm8k_train_path}']"
val_files="['${math_online_test_path}']"


if [ -z "$RUN_NAME" ]; then
    RUN_NAME=CGPO-replr1.2-mot1of8-raiden-sft-ckpt5600-math-code-creative-easy
fi

# Default values
TRAIN_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=64
# PPO_MICRO_BATCH_SIZE_PER_GPU=8
# LOG_PROB_MICRO_BATCH_SIZE=8

MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=8192
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=$(( (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2 ))
CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU=$(( (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 4 ))

VAL_BATCH_SIZE=500
LEARNING_RATE=1e-6
REP_LEARNING_RATE=1.2
SAVE_FREQ=10
TEST_FREQ=20000

# per GPU
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_LOW=0.2

KL_LOSS_COEF=0.001
KL_COEF=0.001
ENTROPY_COEFFIENT=0.000

KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
ROLLOUT_N=8

TOTAL_EPOCHS=10

ROLLOUT_GPU_MEMORY_UTIL=0.6
ACTOR_OPTIMIZER_OFFLOAD=False
ACTOR_PARAMETER_OFFLOAD=False
MODEL_NAME=qwen2.5-7b-instruct-mot8192-1of8-raiden-max8192-sft-gbs32-lr2e-5-ckpt5600
VERIFIER_NAME=general-verifier

generate_suffix() {
  local suffix=""
  local dataset_provided=false
  local model_provided=false

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --val_batch_size) suffix+="_valbatch$2"; shift 2 ;;
      --max_prompt_length) suffix+="_max_prompt$2"; shift 2 ;;
      --max_response_length) suffix+="_max_response$2"; shift 2 ;;
      --learning_rate) suffix+="_lr$2"; shift 2 ;;
      --ppo_mini_batch_size) suffix+="_ppomini$2"; shift 2 ;;
      --kl_loss_coef) suffix+="_klcoef$2"; shift 2 ;;
      --entropy_coeffient) suffix+="_entcoef$2"; shift 2 ;;
      --clip_ratio_high) suffix+="_clipratio_high$2"; shift 2 ;;
      --clip_ratio_low) suffix+="_clipratio_low$2"; shift 2 ;;
      --kl_loss_type) suffix+="_kltype$2"; shift 2 ;;
      --temperature) suffix+="_temp$2"; shift 2 ;;
      --rollout_n) suffix+="_rollout$2"; shift 2 ;;
      --kl_coef) suffix+="_klcontrol$2"; shift 2 ;;
      --total_epochs) suffix+="_epochs$2"; shift 2 ;;
      --rollout_gpu_memory_util) shift 2 ;;
      --actor_optimizer_offload) shift 2 ;;
      --actor_parameter_offload) shift 2 ;;
      --dataset_name) suffix+="_$2"; dataset_provided=true; shift 2 ;;
      --model_name) suffix+="_$2"; model_provided=true; shift 2 ;;
      *) shift ;;
    esac
  done

  if [ "$dataset_provided" = false ]; then
    suffix+="_$DATASET_NAME"
  fi

  if [ "$model_provided" = false ]; then
    suffix+="_$MODEL_NAME"
  fi

  echo "$suffix"
}

echo "Arguments received: $@"

# Generate a unique suffix based on the input arguments
SUFFIX=$(generate_suffix "$@")
# RUN_NAME="$RUN_NAME$SUFFIX"
RUN_NAME+="_$timestamp"
LOG_FILE_PATH="$HDFS_LOG_PATH/$RUN_NAME.log"
QUESTION_AND_RESPONSE_PATH="$HDFS_QUESTION_AND_RESPONSE_PATH/$RUN_NAME"
LLM_JUDGEMENT_PATH="$HDFS_LLM_JUDGEMENT_PATH/$RUN_NAME"


echo "Training with the following parameters:"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Val Batch Size: $VAL_BATCH_SIZE"
echo "Max Prompt Length: $MAX_PROMPT_LENGTH"
echo "Max Response Length: $MAX_RESPONSE_LENGTH"
echo "Learning Rate: $LEARNING_RATE"
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE"
echo "KL Loss Coefficient: $KL_LOSS_COEF"
echo "KL Loss Type: $KL_LOSS_TYPE"
echo "Temperature: $TEMPERATURE"
echo "Rollout N: $ROLLOUT_N"
echo "KL Coefficient: $KL_COEF"
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Dataset Name: $DATASET_NAME"
echo "Model Name: $MODEL_NAME"
echo "LOG FILE PATH: $LOG_FILE_PATH"
echo "Question and response path: $QUESTION_AND_RESPONSE_PATH"

max_num_batched_tokens=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 1000)
echo -e "Training with the following parameters:\nTrain Batch Size: $TRAIN_BATCH_SIZE\nVal Batch Size: $VAL_BATCH_SIZE\nMax Prompt Length: $MAX_PROMPT_LENGTH\nMax Response Length: $MAX_RESPONSE_LENGTH\nLearning Rate: $LEARNING_RATE\nPPO Mini Batch Size: $PPO_MINI_BATCH_SIZE\nKL Loss Coefficient: $KL_LOSS_COEF\nKL Loss Type: $KL_LOSS_TYPE\nTemperature: $TEMPERATURE\nRollout N: $ROLLOUT_N\nKL Coefficient: $KL_COEF\nTotal Epochs: $TOTAL_EPOCHS\nDataset Name: $DATASET_NAME\nModel Name: $MODEL_NAME"

CUDA_VISIBLE_DEVICES=${gpu_list} HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 \
    python3 -m main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.enable=True \
    reward_model.model.tokenizer_path=/home/xzliang/General-Reasoner/checkpoint/qwen2.5-7b-instruct-mot8192-1of8-raiden-max8192-sft-gbs32-lr2e-5/checkpoint-5600 \
    reward_model.model.path=$HDFS_MODEL_PATH/$VERIFIER_NAME \
    reward_model.strategy=verifier \
    reward_model.reward_manager=naive \
    reward_model.llm_judgement_path="$LLM_JUDGEMENT_PATH" \
    reward_model.need_verifier=False \
    data.train_files=${train_files} \
    data.val_files=${val_files} \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/home/xzliang/General-Reasoner/checkpoint/qwen2.5-7b-instruct-mot8192-1of8-raiden-max8192-sft-gbs32-lr2e-5/checkpoint-5600 \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.reptile_lr=$REP_LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_c=10 \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAMETER_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=${gpu_count} \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=False \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
    trainer.default_question_and_response_path=$QUESTION_AND_RESPONSE_PATH \
    trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee -a $LOG_FILE_PATH
  
# CUDA_VISIBLE_DEVICES=${gpu_list} HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 \
#     python3 -m main_ppo \
#     algorithm.adv_estimator=grpo \
#     reward_model.enable=True \
#     reward_model.model.tokenizer_path=$HDFS_MODEL_PATH/$MODEL_NAME \
#     reward_model.model.path=$HDFS_MODEL_PATH/$VERIFIER_NAME \
#     reward_model.strategy=verifier \
#     reward_model.reward_manager=naive \
#     reward_model.llm_judgement_path="$LLM_JUDGEMENT_PATH" \
#     reward_model.need_verifier=False \
#     reward_model.micro_batch_size=0 \
#     data.train_files=${train_files} \
#     data.val_files=${val_files} \
#     data.train_batch_size=$TRAIN_BATCH_SIZE \
#     data.val_batch_size=$VAL_BATCH_SIZE \
#     data.max_prompt_length=$MAX_PROMPT_LENGTH \
#     data.max_response_length=$MAX_RESPONSE_LENGTH \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
#     actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
#     actor_rollout_ref.actor.reptile_lr=$REP_LEARNING_RATE \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
#     actor_rollout_ref.actor.use_dynamic_bsz=True \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
#     actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
#     actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
#     actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
#     actor_rollout_ref.actor.clip_ratio_c=10 \
#     actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAMETER_OFFLOAD \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
#     actor_rollout_ref.rollout.temperature=$TEMPERATURE \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
#     actor_rollout_ref.rollout.n=$ROLLOUT_N \
#     actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     actor_rollout_ref.rollout.enforce_eager=False \
#     actor_rollout_ref.rollout.free_cache_engine=False \
#     actor_rollout_ref.rollout.enable_chunked_prefill=False \
#     algorithm.kl_ctrl.kl_coef=$KL_COEF \
#     trainer.critic_warmup=0 \
#     trainer.logger=['console','wandb'] \
#     trainer.project_name=$PROJECT_NAME \
#     trainer.experiment_name=$RUN_NAME \
#     trainer.n_gpus_per_node=${gpu_count} \
#     trainer.nnodes=1 \
#     trainer.save_freq=$SAVE_FREQ \
#     trainer.test_freq=$TEST_FREQ \
#     trainer.val_before_train=False \
#     trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
#     trainer.default_question_and_response_path=$QUESTION_AND_RESPONSE_PATH \
#     trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee -a $LOG_FILE_PATH

# CUDA_VISIBLE_DEVICES=${gpu_list} HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 \
#     python3 -m main_ppo \
#     algorithm.adv_estimator=grpo \
#     custom_reward_function.path=./compute_score.py \
#     reward_model.enable=True \
#     reward_model.model.path=$HDFS_MODEL_PATH/$VERIFIER_NAME \
#     reward_model.strategy=verifier \
#     reward_model.reward_manager=naive \
#     reward_model.micro_batch_size=0 \
#     data.train_files=${train_files} \
#     data.val_files=${val_files} \
#     data.train_batch_size=$TRAIN_BATCH_SIZE \
#     data.val_batch_size=$VAL_BATCH_SIZE \
#     data.max_prompt_length=$MAX_PROMPT_LENGTH \
#     data.max_response_length=$MAX_RESPONSE_LENGTH \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
#     actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
#     actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
#     actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
#     actor_rollout_ref.actor.clip_ratio_c=10 \
#     actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAMETER_OFFLOAD \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
#     actor_rollout_ref.rollout.temperature=$TEMPERATURE \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
#     actor_rollout_ref.rollout.n=$ROLLOUT_N \
#     actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     actor_rollout_ref.rollout.enforce_eager=False \
#     actor_rollout_ref.rollout.free_cache_engine=False \
#     actor_rollout_ref.rollout.enable_chunked_prefill=False \
#     algorithm.kl_ctrl.kl_coef=$KL_COEF \
#     critic.ppo_micro_batch_size_per_gpu=8 \
#     trainer.critic_warmup=0 \
#     trainer.logger=['console','wandb'] \
#     trainer.project_name=$PROJECT_NAME \
#     trainer.experiment_name=$RUN_NAME \
#     trainer.n_gpus_per_node=${gpu_count} \
#     trainer.nnodes=1 \
#     trainer.save_freq=20 \
#     trainer.test_freq=5 \
#     trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
#     trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee -a $LOG_FILE_PATH
