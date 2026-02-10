#!/bin/bash
CUDA_VISIBLE_DEVICES=6,7 # set your CUDA_VISIBLE_DEVICES here
# --- Model Settings ---
MODEL=Qwen/Qwen3-14B # Set your model path
TEMP=0.0
TOP_P=1.0
MAX_TOKENS=4096
BATCH_SIZE=64

for dataset in "Ultra-Real" "Ultra-Creative" 
do

    # --- Path Configurations ---
    TASK="helpfulness"
    MODE="min_target_acc"
    TARGET_DIR="./data/${TASK}/${dataset}/${dataset}-Target"
    BENCH_DIR="./data/${TASK}/${dataset}/${dataset}-Bench"
    RESULT_DIR="./rubrics_selection/results/${TASK}/${dataset}"


    python rubrics_selection/selection.py \
        --task $TASK --dataset $dataset --mode $MODE

    echo "Evaluating final rubrics (Target-Test)..."
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python rubrics_selection/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics_selection/rubrics/${TASK}/${dataset}/${dataset}_selected_rubrics.jsonl \
        --data_path $TARGET_DIR/test.jsonl \
        --output_dir $RESULT_DIR \
        --dataset_name target --subset test 

    echo "Evaluating final rubrics (Bench-Test)..." 
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python rubrics_selection/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics_selection/rubrics/${TASK}/${dataset}/${dataset}_selected_rubrics.jsonl \
        --data_path $BENCH_DIR/test.jsonl \
        --output_dir $RESULT_DIR \
        --dataset_name bench --subset test
done


for dataset in "SafeRLHF-RMB" "Anthropic-SafeRLHF"
do

    # --- Path Configurations ---
    TASK="harmlessness"
    MODE="min_target_acc"
    TARGET_DIR="./data/${TASK}/${dataset}/${dataset}-Target"
    BENCH_DIR="./data/${TASK}/${dataset}/${dataset}-Bench"
    RESULT_DIR="./rubrics_selection/results/${TASK}/${dataset}"

    python rubrics_selection/selection.py \
        --task $TASK --dataset $dataset --mode $MODE

    echo "Evaluating final rubrics (Target-Test)..."
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python rubrics_selection/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics_selection/rubrics/${TASK}/${dataset}/${dataset}_selected_rubrics.jsonl \
        --data_path $TARGET_DIR/test.jsonl \
        --output_dir $RESULT_DIR \
        --dataset_name target --subset test

    echo "Evaluating final rubrics (Bench-Test)..." 
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python rubrics_selection/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics_selection/rubrics/${TASK}/${dataset}/${dataset}_selected_rubrics.jsonl \
        --data_path $BENCH_DIR/test.jsonl \
        --output_dir $RESULT_DIR \
        --dataset_name bench --subset test
done
