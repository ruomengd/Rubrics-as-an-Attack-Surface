#!/bin/bash

cd ./downstream_eval/train


# harmlessness

# 2b full finetune
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 train_dpo_deepspeed.py \
  --data_paths   your_dpo_data_path \
  --judge_paths  your_label_path \
  --deepspeed ds_zero2_bf16.json --model_name_or_path sirev/Gemma-2b-Uncensored-v1 \
  --bf16 --gradient_checkpointing \
  --max_length 2048 --max_prompt_length 2048 \
  --learning_rate 1e-6 --max_grad_norm 1 --beta 0.1 \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
  --evaluation_strategy steps --eval_steps 100 --per_device_eval_batch_size 8 \
  --eval_ratio 0.04 --eval_max_length_cap 2048 --eval_max_samples 1000 \
  --wandb_project dpo_safety_2b --wandb_run_name name --output_dir path

# 8b LORA
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 train_dpo_deepspeed.py \
  --data_paths   your_dpo_data_path \
  --judge_paths  your_label_path \
  --deepspeed ds_zero2_bf16.json --model_name_or_path dphn/Dolphin3.0-Llama3.1-8B \
  --bf16 --gradient_checkpointing \
  --max_length 2048 --max_prompt_length 2048 \
  --learning_rate 1e-4 --max_grad_norm 1 --beta 0.1 \
  --use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
  --evaluation_strategy steps --eval_steps 100 --per_device_eval_batch_size 16 \
  --eval_ratio 0.04 --eval_max_length_cap 2048 --eval_max_samples 1000 \
  --wandb_project dpo_safety_doph --wandb_run_name name --output_dir path


# helpfulness

# 2b full finetune
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 train_dpo_deepspeed.py \
  --model_name_or_path google/gemma-2-2b-it \
  --data_paths   your_dpo_data_path \
  --judge_paths  your_label_path \
  --deepspeed ds_zero2_bf16.json \
  --bf16 --gradient_checkpointing \
  --max_length 2048 --max_prompt_length 2048 \
  --learning_rate 1e-6 --max_grad_norm 1 --beta 0.1 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --evaluation_strategy steps --eval_steps 100 --per_device_eval_batch_size 4 \
  --eval_ratio 0.05 --eval_max_length_cap 2048 --eval_max_samples 1000 \
  --wandb_project dpo_2b --wandb_run_name name --output_dir path

# 8b LORA
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 train_dpo_deepspeed.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --data_paths   your_dpo_data_path \
  --judge_paths  your_label_path \
  --deepspeed ds_zero2_bf16.json \
  --bf16 --gradient_checkpointing \
  --max_length 2048 --max_prompt_length 2048 \
  --learning_rate 1e-4 --max_grad_norm 1 --beta 0.1 \
  --use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --evaluation_strategy steps --eval_steps 100 --per_device_eval_batch_size 4 \
  --eval_ratio 0.04 --eval_max_length_cap 2048 --eval_max_samples 1000 \
  --wandb_project dpo_8b --wandb_run_name name --output_dir path