#!/bin/bash

CUDA_VISIBLE_DEVICES=6,7 # set your CUDA_VISIBLE_DEVICES here

# --- Model Settings ---
MODEL=/home/ruomeng/model/Qwen/Qwen3-8B # Set your model path
TEMP=0.0
TOP_P=1.0
MAX_TOKENS=4096
BATCH_SIZE=32 # Larger if possible


for dataset in "Ultra-Real" 
do
    # --- Path Configurations ---
    TASK="helpfulness"
    BENCH_DIR="./data/${TASK}/${dataset}/${dataset}-Bench"
    TARGET_DIR="./data/${TASK}/${dataset}/${dataset}-Target"
    RUBRICS_DIR="./rubrics_selection/rubrics/${TASK}/${dataset}/${dataset}_selected_rubrics.jsonl"
    RESULT_DIR="./data/dpo/${dataset}"

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python rubrics_selection/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path $RUBRICS_DIR \
        --data_path $BENCH_DIR/dpo_20k.jsonl \
        --output_dir $RESULT_DIR \
        --dataset_name bench --subset dpo_20k --debug
    
     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python rubrics_selection/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path $RUBRICS_DIR \
        --data_path $TARGET_DIR/dpo_20k.jsonl \
        --output_dir $RESULT_DIR \
        --dataset_name target --subset dpo_20k --debug
done

for dataset in "Anthropic-SafeRLHF" 
do
    # --- Path Configurations ---
    TASK="harmlessness"
    BENCH_DIR="./data/${TASK}/${dataset}/${dataset}-Bench"
    TARGET_DIR="./data/${TASK}/${dataset}/${dataset}-Target"
    RUBRICS_DIR="./rubrics_selection/rubrics/${TASK}/${dataset}/${dataset}_selected_rubrics.jsonl"
    RESULT_DIR="./data/dpo/${dataset}"

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python rubrics_selection/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path $RUBRICS_DIR \
        --data_path $BENCH_DIR/dpo_20k.jsonl \
        --output_dir $RESULT_DIR \
        --dataset_name bench --subset dpo_20k --debug
    
     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python rubrics_selection/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path $RUBRICS_DIR \
        --data_path $TARGET_DIR/dpo_20k.jsonl \
        --output_dir $RESULT_DIR \
        --dataset_name target --subset dpo_20k --debug
done