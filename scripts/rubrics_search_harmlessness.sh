#!/bin/bash
export DEEPSEEK_API_KEY=your_api_key # set your DEEPSEEK_API_KEY here
CUDA_VISIBLE_DEVICES=6,7 # set your CUDA_VISIBLE_DEVICES here
# --- Searching Settings ---
OPTIMIZE_TIMES=5
SAMPLE_SIZE=200
BATCH_SIZE=64 
REFINE_NUMS=4

# --- Model Settings ---
MODEL=Qwen/Qwen3-14B # Set your model path
TEMP=0.0
TOP_P=1.0
MAX_TOKENS=4096


########################################################
# Harmlessness
########################################################
for dataset in "SafeRLHF-RMB" "Anthropic-SafeRLHF"
do

    # --- Path Configurations ---
    TASK="harmlessness"
    RUBRICS_DIR="./rubrics_search/rubrics/${TASK}/${dataset}"
    RESULT_DIR="./rubrics_search/results/${TASK}/${dataset}"
    BENCH_DIR="./data/${TASK}/${dataset}/${dataset}-Bench"
    TARGET_DIR="./data/${TASK}/${dataset}/${dataset}-Target"
   

    # Execute the Python script
    echo "Starting the search..."
    echo "RUBRICS_DIR: $RUBRICS_DIR"
    echo "RESULT_DIR: $RESULT_DIR"
    echo "BENCH_DIR: $BENCH_DIR"
    echo "TARGET_DIR: $TARGET_DIR"
    echo "OPTIMIZE_TIMES: $OPTIMIZE_TIMES"
    echo "SAMPLE_SIZE: $SAMPLE_SIZE"
    echo "BATCH_SIZE: $BATCH_SIZE"
    echo "REFINE_NUMS: $REFINE_NUMS"
    echo "MODEL: $MODEL"
    echo "TEMP: $TEMP"
    echo "TOP_P: $TOP_P"
    echo "MAX_TOKENS: $MAX_TOKENS"

    echo "Starting the search..."
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python rubrics_search/search/main.py \
        --task $TASK \
        --rubrics_dir "$RUBRICS_DIR" \
        --result_dir "$RESULT_DIR" \
        --bench_data "$BENCH_DIR/train.jsonl" \
        --target_data "$TARGET_DIR/train.jsonl" \
        --optimize_times $OPTIMIZE_TIMES \
        --sample_size $SAMPLE_SIZE \
        --batch_size $BATCH_SIZE \
        --refine_nums $REFINE_NUMS \
        --model_name "$MODEL" \
        --temp $TEMP \
        --top_p $TOP_P \
        --max_tokens $MAX_TOKENS

    echo "Selecting top-k rubrics (Target-Train)..."
    python rubrics_search/search/select_rubrics.py --folder ${dataset} --topk 10 --task ${TASK}

    echo "Evaluating top-k rubrics (Target-Val)..."
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python  rubrics_search/search/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics_search/rubrics/${TASK}/${dataset}/top_selected.jsonl \
        --data_path $TARGET_DIR/val.jsonl \
        --output_dir $RESULT_DIR \
        --dataset_name target --subset val 

    echo "Selecting final rubrics (Target-Val)..."
    python rubrics_search/search/select_final.py --folder ${dataset} --topk 10 --task ${TASK}

    echo "Evaluating final rubrics (Bench-Val)..."
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python rubrics_search/search/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics_search/rubrics/${TASK}/${dataset}/final.jsonl \
        --data_path $BENCH_DIR/val.jsonl \
        --output_dir $RESULT_DIR \
        --dataset_name bench --subset val 
done