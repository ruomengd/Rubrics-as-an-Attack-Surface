#!/bin/bash
cd ./downstream_eval/eval

# we only show the sample command here, you may set corresponding parameters for your evaluations
harm_bench_data="your eval bench data path"
harm_target_data="your eval target data path"
help_bench_data="your eval bench data path"
help_target_data="your eval target data path"


# STEP1: generate responses from dpoed policy model (same for helpfulness and harmlessness)
# after generating the data, we can score them and analyze the result, as well as selecting bon for 3rd party evaluation

## full, ori means untrained model
CUDA_VISIBLE_DEVICES=0 python generate.py \
  --datasets ${harm_bench_data} ${harm_target_data} \
  --model name=ori,path=sirev/Gemma-2b-Uncensored-v1,tokenizer=sirev/Gemma-2b-Uncensored-v1 \
  --model name=name1,path=path1 \
  --model name=name2,path=path2 \
  --bon_n 8 --tensor_parallel_size 1 --gpu_memory_utilization 0.85 \
  --out_dir result_safe_2b

## lora
CUDA_VISIBLE_DEVICES=0 python generate.py \
  --datasets ${harm_bench_data} ${harm_target_data} \
  --lora name=name1,base=dphn/Dolphin3.0-Llama3.1-8B,adapter=path1 \
  --lora name=name2,base=dphn/Dolphin3.0-Llama3.1-8B,adapter=path2 \
  --bon_n 8 --tensor_parallel_size 2 --gpu_memory_utilization 0.85 \
  --out_dir result_safe_8b


# STEP2: eval using corresponding reward model / cost model

## for harmlessnes, use beaver-7b-unified-cost with lower scores better
CUDA_VISIBLE_DEVICES=0 python scores.py \
  --out_dir ./result_safe_2b \
  --datasets ${harm_target_data}  \
  --scorer beaver \
  --score_direction lower \
  --batch_size 32 \
  --max_length 2048 --models tgt_opt

## for helpfulness, use Skywork-Reward-V2-Llama-3.1-8B with higher scores better
CUDA_VISIBLE_DEVICES=0 python scores.py \
  --out_dir ./result_8b \
  --datasets ${help_bench_data}  \
  --scorer skywork \
  --score_direction higher \
  --batch_size 32 \
  --max_length 3000 --models name1 name2 ...


# STEP3: analyze result from bon scores, same for each setting
python analyze.py \
  --out_dir ./result_8b \
  --datasets ${help_bench_data} \
  --select_n 4 \
  --subset_mode first 


# STEP4: select best bon results, and store them for api model evaluation, same for each setting
python select_best.py \
  --out_dir result_8b \
  --datasets ${help_bench_data} ${help_target_data} \
  --select_n 4 --subset_mode first


# STEP5: eval the bon responses of 2 models through 3rd API
export DEEPSEEK_API_KEY="your api key"

## helpfulness
python help_eval.py \
  --file_a file1 \
  --file_b file2 \
  --concurrency 15 \
  --seed 123 --out_root path

## harmlessness
python safety_eval.py \
  --file_a file1 \
  --file_b file2 \
  --concurrency 15 \
  --seed 123 --out_root path

