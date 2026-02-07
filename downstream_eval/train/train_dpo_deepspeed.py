#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DPO training script for google/gemma-2-2b-it using TRL DPOTrainer.

Key features:
- Merge data JSONL + judge JSONL line-by-line (decision A/B -> chosen/rejected)
- Build Gemma IT chat-style prompts via tokenizer chat template
- Filter samples by token length (hard cap)
- Optional QLoRA / LoRA
- W&B logging
- SAFE, low-memory evaluation:
  * fixed eval_ratio (default 0.02)
  * eval length cap (default 2048 on max(prompt+chosen, prompt+rejected))
  * eval max samples cap (default 512)
  * per_device_eval_batch_size=1
  * prediction_loss_only=True
"""

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer
import random

# ----------------------------
# Distributed device binding
# ----------------------------
if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)


# ----------------------------
# Utilities: length stats
# ----------------------------
def add_length_stats(batch, tokenizer):
    prompt_lens = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in batch["prompt"]]
    chosen_lens = [len(tokenizer(c, add_special_tokens=False).input_ids) for c in batch["chosen"]]
    rejected_lens = [len(tokenizer(r, add_special_tokens=False).input_ids) for r in batch["rejected"]]

    chosen_total = [p + c for p, c in zip(prompt_lens, chosen_lens)]
    rejected_total = [p + r for p, r in zip(prompt_lens, rejected_lens)]
    pair_max_total = [max(ct, rt) for ct, rt in zip(chosen_total, rejected_total)]

    return {
        "len_prompt": prompt_lens,
        "len_chosen_total": chosen_total,
        "len_rejected_total": rejected_total,
        "len_pair_max_total": pair_max_total,
    }


def print_length_summary(ds, key, name):
    arr = np.array(ds[key], dtype=np.int32)
    qs = [50, 75, 90, 95, 97, 99, 99.5]
    print(f"\n[length summary] {name} ({key})")
    print(f"  count={arr.size}  mean={arr.mean():.1f}  std={arr.std():.1f}  min={arr.min()}  max={arr.max()}")
    for q in qs:
        print(f"  p{q:>4} = {np.percentile(arr, q):.0f}")


def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def merge_data_with_judge(
    data_path: str,
    judge_path: str,
    decision_key: str = "decision",
    instruction_key: str = "instruction",
    r1_key: str = "response1",
    r2_key: str = "response2",
) -> List[Tuple[str, str, str]]:
    """
    Returns list of tuples: (instruction, chosen_response, rejected_response)
    Filters out any rows where decision is not 'A' or 'B'.
    Assumes the judge file is aligned line-by-line with the data file.
    """
    data_rows = read_jsonl(data_path)
    judge_rows = read_jsonl(judge_path)

    n = min(len(data_rows), len(judge_rows))
    pairs: List[Tuple[str, str, str]] = []

    for i in range(n):
        d = data_rows[i]
        j = judge_rows[i]

        decision = j.get(decision_key, None)
        if decision not in ("A", "B"):
            continue

        instr = d.get(instruction_key, "")
        r1 = d.get(r1_key, "")
        r2 = d.get(r2_key, "")

        if not isinstance(instr, str) or not isinstance(r1, str) or not isinstance(r2, str):
            continue

        instr = instr.strip()
        r1 = r1.strip()
        r2 = r2.strip()
        if not instr or not r1 or not r2:
            continue

        if decision == "A":
            chosen, rejected = r1, r2
        else:
            chosen, rejected = r2, r1

        pairs.append((instr, chosen, rejected))

    return pairs


def build_prompt_with_chat_template(tokenizer, instruction: str) -> str:
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def token_length(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def prepare_dpo_dataset(
    tokenizer,
    merged_pairs: List[Tuple[str, str, str]],
    context_window: int = 8192,
    max_length_cap: int = 4096,
) -> Dataset:
    """
    Creates Dataset with columns: prompt, chosen, rejected.
    Drops samples where (prompt+chosen) or (prompt+rejected) exceeds hard_limit:
      hard_limit = min(context_window, max_length_cap) if max_length_cap>0 else context_window
    """
    hard_limit = context_window
    if max_length_cap and max_length_cap > 0:
        hard_limit = min(context_window, max_length_cap)

    out = {"prompt": [], "chosen": [], "rejected": []}
    kept, dropped = 0, 0

    for instr, chosen, rejected in merged_pairs:
        prompt = build_prompt_with_chat_template(tokenizer, instr)

        lc = token_length(tokenizer, prompt + chosen)
        lr = token_length(tokenizer, prompt + rejected)

        if lc > hard_limit or lr > hard_limit:
            dropped += 1
            continue

        out["prompt"].append(prompt)
        out["chosen"].append(chosen)
        out["rejected"].append(rejected)
        kept += 1

    ds = Dataset.from_dict(out)
    print(f"[dataset] kept={kept} dropped_by_len={dropped} hard_limit={hard_limit}")
    return ds


# ----------------------------
# SAFE eval builder (low-memory)
# ----------------------------
def build_safe_eval_split(
    train_ds: Dataset,
    tokenizer,
    eval_ratio: float,
    seed: int,
    eval_max_length_cap: int,
    eval_max_samples: int,
) -> Tuple[Dataset, Dataset]:
    """
    Split train/eval using eval_ratio, then make eval safe:
    - filter by pair_max_len <= eval_max_length_cap
    - cap number of eval samples to eval_max_samples
    """
    split = train_ds.train_test_split(test_size=eval_ratio, seed=seed, shuffle=True)
    train_out = split["train"]
    eval_out = split["test"]

    def compute_pair_max_len(batch):
        p_lens = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in batch["prompt"]]
        c_lens = [len(tokenizer(c, add_special_tokens=False).input_ids) for c in batch["chosen"]]
        r_lens = [len(tokenizer(r, add_special_tokens=False).input_ids) for r in batch["rejected"]]
        pair_max = [max(p + c, p + r) for p, c, r in zip(p_lens, c_lens, r_lens)]
        return {"pair_max_len": pair_max}

    eval_out = eval_out.map(compute_pair_max_len, batched=True, batch_size=256, desc="Eval: computing lengths")
    before = len(eval_out)
    eval_out = eval_out.filter(lambda x: x["pair_max_len"] <= eval_max_length_cap)
    after = len(eval_out)
    eval_out = eval_out.remove_columns(["pair_max_len"])

    if eval_max_samples and eval_max_samples > 0 and len(eval_out) > eval_max_samples:
        eval_out = eval_out.shuffle(seed=seed).select(range(eval_max_samples))

    print(f"[eval] ratio={eval_ratio} raw={before} after_len_cap={after} final={len(eval_out)} cap={eval_max_length_cap} max_samples={eval_max_samples}")
    return train_out, eval_out


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data_paths", type=str, nargs="+", required=True)
    p.add_argument("--judge_paths", type=str, nargs="+", required=True)
    p.add_argument("--decision_key", type=str, default="decision")
    p.add_argument("--instruction_key", type=str, default="instruction")
    p.add_argument("--response1_key", type=str, default="response_a")
    p.add_argument("--response2_key", type=str, default="response_b")

    # Model
    p.add_argument("--model_name_or_path", type=str, default="google/gemma-2-2b-it")
    p.add_argument("--context_window", type=int, default=8192)

    # Training caps
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--max_prompt_length", type=int, default=2048)

    # DPO
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--loss_type", type=str, default="sigmoid")

    # Optim / train
    p.add_argument("--output_dir", type=str, default="gemma2_2b_it_dpo_out")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)

    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=1000)

    # SAFE eval controls (defaults chosen to be stable)
    p.add_argument("--evaluation_strategy", type=str, default="steps", help="no/steps/epoch")
    p.add_argument("--eval_steps", type=int, default=500, help="Only used if evaluation_strategy=steps")
    p.add_argument("--eval_ratio", type=float, default=0.02, help="Fraction of data used for eval (default 0.02)")
    p.add_argument("--eval_max_length_cap", type=int, default=2048, help="Hard max length for eval samples")
    p.add_argument("--eval_max_samples", type=int, default=512, help="Cap eval set size to avoid OOM/slow eval")
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--eval_accumulation_steps", type=int, default=1)

    p.add_argument("--seed", type=int, default=42)

    # Precision / memory
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--use_4bit", action="store_true")

    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.set_defaults(use_lora=False)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")

    # W&B
    p.add_argument("--wandb_project", type=str, default="gemma2_dpo")
    p.add_argument("--wandb_run_name", type=str, default="gemma2_2b_it_dpo")

    # DeepSpeed
    p.add_argument("--deepspeed", type=str, default=None)

    p.add_argument("--samples_per_file", type=int, default=30000)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    assert len(args.data_paths) == len(args.judge_paths), "data_paths and judge_paths must have same length."

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_RUN_NAME", args.wandb_run_name)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Merge data + judge
    merged_pairs: List[Tuple[str, str, str]] = []
    samples_per_file = args.samples_per_file
    for dp, jp in zip(args.data_paths, args.judge_paths):
        pairs = merge_data_with_judge(
            data_path=dp,
            judge_path=jp,
            decision_key=args.decision_key,
            instruction_key=args.instruction_key,
            r1_key=args.response1_key,
            r2_key=args.response2_key,
        )

        if samples_per_file and samples_per_file > 0 and len(pairs) > samples_per_file:
            rng = random.Random(args.seed)
            pairs = rng.sample(pairs, k=samples_per_file)

        merged_pairs.extend(pairs)
        # merged_pairs.extend(
        #     merge_data_with_judge(
        #         data_path=dp,
        #         judge_path=jp,
        #         decision_key=args.decision_key,
        #         instruction_key=args.instruction_key,
        #         r1_key=args.response1_key,
        #         r2_key=args.response2_key,
        #     )
        # )
    print(f"[data] merged preference pairs: {len(merged_pairs)}")

    # Build dataset with length filtering
    ds_all = prepare_dpo_dataset(
        tokenizer=tokenizer,
        merged_pairs=merged_pairs,
        context_window=args.context_window,
        max_length_cap=args.max_length,
    )

    # Optional: print train length stats (costs some CPU time, not GPU)
    tmp = ds_all.map(
        lambda b: add_length_stats(b, tokenizer),
        batched=True,
        batch_size=256,
        desc="Computing token length stats",
    )
    print_length_summary(tmp, "len_prompt", "prompt")
    print_length_summary(tmp, "len_chosen_total", "prompt+chosen")
    print_length_summary(tmp, "len_rejected_total", "prompt+rejected")
    print_length_summary(tmp, "len_pair_max_total", "max(prompt+chosen, prompt+rejected)")

    # Shuffle once (important)
    ds_all = ds_all.shuffle(seed=args.seed)

    # Build safe eval split
    train_ds, eval_ds = ds_all, None
    if args.evaluation_strategy != "no" and args.eval_ratio and args.eval_ratio > 0:
        train_ds, eval_ds = build_safe_eval_split(
            train_ds=ds_all,
            tokenizer=tokenizer,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
            eval_max_length_cap=args.eval_max_length_cap,
            eval_max_samples=args.eval_max_samples,
        )
    else:
        print("[eval] disabled (evaluation_strategy=no or eval_ratio=0)")

    # Quant config
    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        quantization_config=quant_config,
        attn_implementation="sdpa",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.use_lora:
        target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        if args.use_4bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        print("[train] Full fine-tuning (no LoRA).")

    # DPO config
    dpo_args = DPOConfig(
        deepspeed=args.deepspeed,

        output_dir=args.output_dir,
        beta=args.beta,
        loss_type=args.loss_type,

        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,

        logging_steps=args.logging_steps,
        save_strategy="no",
        save_steps=args.save_steps,

        # disk saving: keep tiny checkpoints
        save_total_limit=2,
        save_only_model=True,

        # SAFE eval
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        prediction_loss_only=True,

        bf16=args.bf16,
        fp16=(args.fp16 if not args.bf16 else False),

        report_to=["wandb"],
        remove_unused_columns=False,

        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    print("[dpo_config]", asdict(dpo_args))

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
