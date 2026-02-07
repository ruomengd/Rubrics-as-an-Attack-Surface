#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate-only (BoN) script with CLI model specs (no config file).

Output layout:
out_dir/<dataset_name>/
  instructions.jsonl
  gens/
    <model_name>.jsonl

Gen file format:
  {"instruction": "...", "responses": ["cand1", "cand2", ...]}

Supports:
- Multiple base models: --model name=...,path=...,tokenizer=...
- Multiple LoRA models: --lora  name=...,base=...,adapter=...,tokenizer=...

Instruction set is fixed per dataset:
- If instructions.jsonl exists, reuse it by default.
- Use --force_overwrite_instructions true to overwrite.

Example:
python gen_bon_cli.py \
  --datasets data1.jsonl data2.jsonl \
  --out_dir ./exp_out \
  --model name=A,path=meta-llama/Llama-3.1-8B-Instruct,tokenizer=meta-llama/Llama-3.1-8B-Instruct \
  --lora  name=B_lora,base=dphn/Dolphin3.0-Llama3.1-8B,adapter=/path/to/adapter,tokenizer=dphn/Dolphin3.0-Llama3.1-8B \
  --bon_n 8 --seed 42 --sample_size 1000 --max_inst_tokens 1024
"""

import argparse
import gc
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# -----------------------------
# Basic IO helpers
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Bad JSON at {path}:{ln}: {e}")
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_set_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# -----------------------------
# Instruction sampling
# -----------------------------
def get_token_len(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def sample_instructions_from_dataset(
    dataset_rows: List[Dict[str, Any]],
    tokenizer_for_len: AutoTokenizer,
    max_inst_tokens: int,
    sample_size: int,
    seed: int,
) -> List[str]:
    insts: List[str] = []
    for r in dataset_rows:
        inst = r.get("instruction", None)
        if not isinstance(inst, str):
            continue
        if get_token_len(tokenizer_for_len, inst) <= max_inst_tokens:
            insts.append(inst)

    rnd = random.Random(seed)
    rnd.shuffle(insts)
    if sample_size > 0:
        insts = insts[: min(sample_size, len(insts))]
    return insts


def chunked(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def build_chat_prompts(tokenizer: AutoTokenizer, instructions: List[str]) -> List[str]:
    if getattr(tokenizer, "chat_template", None):
        prompts = []
        for inst in instructions:
            messages = [{"role": "user", "content": inst}]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return prompts
    return instructions


# -----------------------------
# CLI model spec parsing
# -----------------------------
def parse_kv_spec(spec: str) -> Dict[str, str]:
    """
    Parse "k=v,k2=v2" into dict. Values may contain ':' or '/' etc (avoid commas).
    """
    out: Dict[str, str] = {}
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad spec segment (missing '='): {p} in spec: {spec}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_model_specs(models: List[str], loras: List[str]) -> List[Dict[str, Any]]:
    """
    Returns list of model dicts:
      base model: {"name","tokenizer","mode":"base","path":...}
      lora model: {"name","tokenizer","mode":"lora","base":...,"adapter":...}
    """
    out: List[Dict[str, Any]] = []

    for s in models:
        d = parse_kv_spec(s)
        for req in ("name", "path"):
            if req not in d:
                raise ValueError(f"--model missing '{req}': {s}")
        out.append(
            {
                "name": d["name"],
                "tokenizer": d["path"],
                "mode": "base",
                "path": d["path"],
            }
        )

    for s in loras:
        d = parse_kv_spec(s)
        for req in ("name", "base", "adapter"):
            if req not in d:
                raise ValueError(f"--lora missing '{req}': {s}")
        out.append(
            {
                "name": d["name"],
                "tokenizer": d["base"],
                "mode": "lora",
                "base": d["base"],
                "adapter": d["adapter"],
            }
        )

    # ensure unique names
    names = [m["name"] for m in out]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate model names found: {names}")
    return out


# -----------------------------
# vLLM BoN generation
# -----------------------------
def make_sampling_params(
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    bon_n: int,
) -> SamplingParams:
    if do_sample:
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            seed=seed,
            n=bon_n,
        )
    else:
        return SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            seed=seed,
            n=bon_n,
        )


def vllm_generate_bon_for_model(
    *,
    out_path: str,
    instructions: List[str],
    tokenizer_path: str,
    mode: str,
    model_path_or_base: str,
    lora_adapter_path: str,
    # gen params
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    bon_n: int,
    # engine params
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    chunk_size: int,
) -> None:
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    safe_set_pad_token(tok)

    samp = make_sampling_params(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        bon_n=bon_n,
    )

    if mode == "base":
        llm = LLM(
            model=model_path_or_base,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
        )
        lora_req = None
    elif mode == "lora":
        llm = LLM(
            model=model_path_or_base,  # base
            enable_lora=True,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
        )
        lora_req = LoRARequest("adapter", 1, lora_adapter_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    rows_out: List[Dict[str, Any]] = []

    chunks = list(chunked(instructions, chunk_size))
    for inst_chunk in tqdm(chunks, desc=f"generate chunks ({os.path.basename(out_path)})"):
        prompts = build_chat_prompts(tok, inst_chunk)
        outs = llm.generate(prompts, samp, lora_request=lora_req)

        for inst, o in zip(inst_chunk, outs):
            cands: List[str] = []
            if not o.outputs:
                cands = ["" for _ in range(bon_n)]
            else:
                for j in range(min(bon_n, len(o.outputs))):
                    cands.append((o.outputs[j].text or "").strip())
                while len(cands) < bon_n:
                    cands.append("")
            rows_out.append({"instruction": inst, "responses": cands})

    write_jsonl(out_path, rows_out)

    del llm
    del tok
    cleanup_gpu()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--datasets", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # repeatable model args
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        help="Base model spec: name=...,path=...,tokenizer=... (repeatable)",
    )
    ap.add_argument(
        "--lora",
        action="append",
        default=[],
        help="LoRA model spec: name=...,base=...,adapter=...,tokenizer=... (repeatable)",
    )

    # instruction sampling
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_inst_tokens", type=int, default=1024)
    ap.add_argument("--sample_size", type=int, default=1000)
    ap.add_argument("--force_overwrite_instructions", type=str, default="false")

    # generation
    ap.add_argument("--bon_n", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--do_sample", type=str, default="true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)

    # vLLM engine params
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--max_model_len", type=int, default=2222)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--chunk_size", type=int, default=500)

    # optional: skip if exists
    ap.add_argument("--skip_if_exists", type=str, default="true")

    args = ap.parse_args()

    do_sample = args.do_sample.strip().lower() in {"1", "true", "yes", "y"}
    overwrite_inst = args.force_overwrite_instructions.strip().lower() in {"1", "true", "yes", "y"}
    skip_if_exists = args.skip_if_exists.strip().lower() in {"1", "true", "yes", "y"}

    models = parse_model_specs(args.model, args.lora)
    if not models:
        raise ValueError("You must provide at least one --model or --lora")

    ensure_dir(args.out_dir)

    # tokenizer for length filtering: use the first model tokenizer
    tok_len = AutoTokenizer.from_pretrained(models[0]["tokenizer"], use_fast=True)
    safe_set_pad_token(tok_len)

    for ds_path in args.datasets:
        ds_name = os.path.splitext(os.path.basename(ds_path))[0]
        if "bench" in ds_path.lower():
            ds_name = "bench"
        elif "target" in ds_path.lower():
            ds_name = "target"
        ds_out = os.path.join(args.out_dir, ds_name)
        ensure_dir(ds_out)

        gens_dir = os.path.join(ds_out, "gens")
        ensure_dir(gens_dir)

        inst_path = os.path.join(ds_out, "instructions.jsonl")

        # Prepare instruction list (fixed)
        if os.path.isfile(inst_path) and not overwrite_inst:
            inst_rows = read_jsonl(inst_path)
            instructions = [r["instruction"] for r in inst_rows]
            print(f"\n[GEN] Dataset={ds_name} reuse instructions: {inst_path} (n={len(instructions)})")
        else:
            rows = read_jsonl(ds_path)
            instructions = sample_instructions_from_dataset(
                dataset_rows=rows,
                tokenizer_for_len=tok_len,
                max_inst_tokens=args.max_inst_tokens,
                sample_size=args.sample_size,
                seed=args.seed,
            )
            if not instructions:
                print(f"\n[GEN] Dataset={ds_name}: no valid instructions after filtering, skip.")
                continue
            write_jsonl(inst_path, [{"instruction": x} for x in instructions])
            print(f"\n[GEN] Dataset={ds_name} saved instructions: {inst_path} (n={len(instructions)})")

        # Generate for each model
        for m in models:
            name = m["name"]
            out_path = os.path.join(gens_dir, f"{name}.jsonl")

            if skip_if_exists and os.path.isfile(out_path):
                print(f"[GEN] Exists, skip: {out_path}")
                continue

            if m["mode"] == "base":
                print(f"[GEN] Model={name} (base) bon_n={args.bon_n} -> {out_path}")
                vllm_generate_bon_for_model(
                    out_path=out_path,
                    instructions=instructions,
                    tokenizer_path=m["tokenizer"],
                    mode="base",
                    model_path_or_base=m["path"],
                    lora_adapter_path="",
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    seed=args.seed,
                    bon_n=args.bon_n,
                    tensor_parallel_size=args.tensor_parallel_size,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_num_seqs=args.max_num_seqs,
                    chunk_size=args.chunk_size,
                )
            else:
                print(f"[GEN] Model={name} (lora) bon_n={args.bon_n} -> {out_path}")
                vllm_generate_bon_for_model(
                    out_path=out_path,
                    instructions=instructions,
                    tokenizer_path=m["tokenizer"],
                    mode="lora",
                    model_path_or_base=m["base"],
                    lora_adapter_path=m["adapter"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    seed=args.seed,
                    bon_n=args.bon_n,
                    tensor_parallel_size=args.tensor_parallel_size,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_num_seqs=args.max_num_seqs,
                    chunk_size=args.chunk_size,
                )

    print("\n[GEN] Done.")


if __name__ == "__main__":
    main()
