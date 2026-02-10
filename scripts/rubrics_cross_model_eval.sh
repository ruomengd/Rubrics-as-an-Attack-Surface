#!/bin/bash
export DEEPSEEK_API_KEY=your_api_key # set your DEEPSEEK_API_KEY here

# --- Model Settings ---
MODEL=deepseek-chat # Set your model name
TEMP=0.0
TOP_P=1.0
MAX_TOKENS=1024
BATCH_SIZE=10 # set the concurrency

for dataset in "Ultra-Real" "Ultra-Creative" 
do

    # --- Path Configurations ---
    TASK="helpfulness"
    RUBRICS_DIR="./rubrics_selection/rubrics/${TASK}/${dataset}"
    RESULT_DIR="./rubrics_selection/results/cross_model_eval/${MODEL}/${TASK}/${dataset}"
    BENCH_DATA="./data/${TASK}/${dataset}/${dataset}-Bench/test.jsonl"
    TARGET_DATA="./data/${TASK}/${dataset}/${dataset}-Target/test.jsonl"

    python rubrics_selection/eval_api.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics_selection/rubrics/${TASK}/${dataset}/${dataset}_selected_rubrics.jsonl \
        --data_path $BENCH_DATA \
        --output_dir $RESULT_DIR \
        --dataset_name bench --subset test

    python rubrics_selection/eval_api.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics_selection/rubrics/${TASK}/${dataset}/${dataset}_selected_rubrics.jsonl \
        --data_path $TARGET_DATA \
        --output_dir $RESULT_DIR \
        --dataset_name target --subset test
done

for dataset in "SafeRLHF-RMB" "Anthropic-SafeRLHF"
do

    # --- Path Configurations ---
    TASK="harmlessness"
    RUBRICS_DIR="./rubrics_selection/rubrics/${TASK}/${dataset}"
    RESULT_DIR="./rubrics_selection/results/cross_model_eval/${MODEL}/${TASK}/${dataset}"
    BENCH_DATA="./data/${TASK}/${dataset}/${dataset}-Bench/test.jsonl"
    TARGET_DATA="./data/${TASK}/${dataset}/${dataset}-Target/test.jsonl"

    python rubrics_selection/eval_api.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics_selection/rubrics/${TASK}/${dataset}/${dataset}_selected_rubrics.jsonl \
        --data_path $BENCH_DATA \
        --output_dir $RESULT_DIR \
        --dataset_name bench --subset test

    python rubrics_selection/eval_api.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics_selection/rubrics/${TASK}/${dataset}/${dataset}_selected_rubrics.jsonl \
        --data_path $TARGET_DATA \
        --output_dir $RESULT_DIR \
        --dataset_name target --subset test
done