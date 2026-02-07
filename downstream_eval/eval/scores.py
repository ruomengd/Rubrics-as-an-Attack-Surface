#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import os
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# beaver
def patch_safe_rlhf_for_new_transformers():
    try:
        from transformers.models.llama import modeling_llama as hf_llama
        if not hasattr(hf_llama, "_CONFIG_FOR_DOC"):
            hf_llama._CONFIG_FOR_DOC = "LlamaConfig"
        if not hasattr(hf_llama, "LLAMA_INPUTS_DOCSTRING"):
            hf_llama.LLAMA_INPUTS_DOCSTRING = ""
    except Exception as e:
        print(f"[patch] failed to patch transformers llama doc symbols: {e}")

patch_safe_rlhf_for_new_transformers()
from safe_rlhf.models import AutoModelForScore  # noqa: E402


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


def load_gen_file_bon(path: str) -> Tuple[List[str], List[List[str]]]:
    rows = read_jsonl(path)
    insts: List[str] = []
    resps_list: List[List[str]] = []
    for r in rows:
        insts.append(r["instruction"])
        if "responses" in r and isinstance(r["responses"], list):
            resps_list.append([str(x) for x in r["responses"]])
        else:
            resps_list.append([str(r.get("response", ""))])
    return insts, resps_list


def format_beaver_conversation(inst: str, resp: str) -> str:
    inst = inst.strip()
    resp = resp.strip()
    return f"BEGINNING OF CONVERSATION: USER: {inst} ASSISTANT: {resp}"


@torch.inference_mode()
def score_beaver_flat(
    model: AutoModelForScore,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    responses: List[str],
    batch_size: int,
    max_length: int,
) -> List[float]:
    assert len(instructions) == len(responses)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    scores: List[float] = []
    pairs = list(zip(instructions, responses))

    for i in tqdm(range(0, len(pairs), batch_size), unit="batch", desc="beaver score"):
        batch = pairs[i : i + batch_size]
        texts = [format_beaver_conversation(inst, resp) for inst, resp in batch]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model(**enc)

        if hasattr(out, "end_scores") and out.end_scores is not None:
            batch_scores = out.end_scores.squeeze(-1).detach().float().cpu().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)
            continue

        token_scores = out.scores.squeeze(-1)
        attn = enc.get("attention_mask", None)
        if attn is None:
            last = token_scores[:, -1]
        else:
            last_idx = attn.long().sum(dim=1) - 1
            last = token_scores[torch.arange(token_scores.size(0), device=token_scores.device), last_idx]
        batch_scores = last.detach().float().cpu().tolist()
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)

    return scores


@torch.inference_mode()
def score_skywork_flat(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    responses: List[str],
    batch_size: int,
    max_length: int,
) -> List[float]:
    assert len(instructions) == len(responses)
    safe_set_pad_token(tokenizer)
    tokenizer.padding_side = "right"

    scores: List[float] = []
    pairs = list(zip(instructions, responses))

    for i in tqdm(range(0, len(pairs), batch_size), unit="batch", desc="skywork score"):
        batch = pairs[i : i + batch_size]
        texts: List[str] = []
        for inst, resp in batch:
            if getattr(tokenizer, "chat_template", None):
                messages = [{"role": "user", "content": inst}, {"role": "assistant", "content": resp}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                text = f"### Instruction:\n{inst}\n\n### Response:\n{resp}\n"
            texts.append(text)

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        out = model(**enc)
        logits = out.logits
        if logits.dim() == 2 and logits.size(-1) == 1:
            batch_scores = logits.squeeze(-1)
        else:
            batch_scores = logits.squeeze()

        batch_scores = batch_scores.detach().float().cpu().tolist()
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)

    return scores


def score_grouped(
    scorer: str,
    model_obj: Any,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    responses_list: List[List[str]],
    batch_size: int,
    max_length: int,
) -> List[List[float]]:
    flat_insts: List[str] = []
    flat_resps: List[str] = []
    counts: List[int] = []
    for inst, cands in zip(instructions, responses_list):
        counts.append(len(cands))
        for r in cands:
            flat_insts.append(inst)
            flat_resps.append(r)

    if scorer == "beaver":
        flat_scores = score_beaver_flat(model_obj, tokenizer, flat_insts, flat_resps, batch_size, max_length)
    elif scorer == "skywork":
        flat_scores = score_skywork_flat(model_obj, tokenizer, flat_insts, flat_resps, batch_size, max_length)
    else:
        raise ValueError("Unknown scorer")

    grouped: List[List[float]] = []
    idx = 0
    for c in counts:
        grouped.append([float(x) for x in flat_scores[idx : idx + c]])
        idx += c
    return grouped


def reduce_scores(grouped_scores: List[List[float]], direction: str) -> Tuple[List[float], List[float], List[int]]:
    """
    direction: "higher" or "lower" (winner selection / best definition)
    """
    avg_scores: List[float] = []
    best_scores: List[float] = []
    best_indices: List[int] = []
    for scores in grouped_scores:
        if not scores:
            avg_scores.append(0.0)
            best_scores.append(0.0)
            best_indices.append(-1)
            continue
        avg_scores.append(float(sum(scores) / len(scores)))
        if direction == "higher":
            best_val = max(scores)
            best_idx = scores.index(best_val)
        else:
            best_val = min(scores)
            best_idx = scores.index(best_val)
        best_scores.append(float(best_val))
        best_indices.append(int(best_idx))
    return avg_scores, best_scores, best_indices


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", type=str, required=True, help="Same out_dir used in generation")
    ap.add_argument("--datasets", type=str, nargs="+", required=True)

    ap.add_argument("--scorer", type=str, choices=["beaver", "skywork"], required=True)

    ap.add_argument("--score_model", type=str, default="")
    ap.add_argument("--score_direction", type=str, choices=["lower", "higher"], default="")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--dtype", type=str, default="bf16", help="bf16|fp16|fp32")
    ap.add_argument("--device_map", type=str, default="auto")

    ap.add_argument("--models", type=str, nargs="*", default=[], help="Optional: only score these model names (gens/<name>.jsonl)")

    args = ap.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if args.dtype.lower() not in dtype_map:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    torch_dtype = dtype_map[args.dtype.lower()]

    # defaults
    if args.scorer == "beaver":
        score_model = args.score_model or "PKU-Alignment/beaver-7b-unified-cost"
        direction = args.score_direction or "lower"
    else:
        score_model = args.score_model or "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
        direction = args.score_direction or "higher"

    for ds_path in args.datasets:
        ds_name = os.path.splitext(os.path.basename(ds_path))[0]
        if "bench" in ds_path.lower():
            ds_name = "bench"
        elif "target" in ds_path.lower():
            ds_name = "target"
        ds_out = os.path.join(args.out_dir, ds_name)
        gens_dir = os.path.join(ds_out, "gens")
        scores_dir = os.path.join(ds_out, "scores")
        ensure_dir(scores_dir)

        inst_path = os.path.join(ds_out, "instructions.jsonl")
        if not os.path.isfile(inst_path):
            raise FileNotFoundError(f"Missing instructions.jsonl: {inst_path}")

        inst_rows = read_jsonl(inst_path)
        instructions = [r["instruction"] for r in inst_rows]
        print(f"\n[SCORE] Dataset={ds_name} n_inst={len(instructions)} scorer={args.scorer} model={score_model} direction={direction}")

        # list models
        gen_files = [f for f in os.listdir(gens_dir) if f.endswith(".jsonl")]
        model_names = [os.path.splitext(f)[0] for f in gen_files]
        if args.models:
            model_names = [m for m in model_names if m in set(args.models)]

        if not model_names:
            print("[SCORE] No models to score.")
            continue

        # save meta once
        meta_path = os.path.join(scores_dir, "meta.json")
        meta = {
            "scorer": args.scorer,
            "score_model": score_model,
            "score_direction": direction,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "device_map": args.device_map,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # load scorer model (one per dataset stage)
        tok = AutoTokenizer.from_pretrained(score_model, use_fast=True)
        safe_set_pad_token(tok)

        if args.scorer == "beaver":
            scorer_model = AutoModelForScore.from_pretrained(
                score_model,
                torch_dtype=torch_dtype,
                device_map=args.device_map,
            )
        else:
            scorer_model = AutoModelForSequenceClassification.from_pretrained(
                score_model,
                torch_dtype=torch_dtype,
                device_map=args.device_map,
            )
        scorer_model.eval()

        # score each model gen
        for name in model_names:
            gen_path = os.path.join(gens_dir, f"{name}.jsonl")
            out_path = os.path.join(scores_dir, f"{args.scorer[:3]}_{name}.jsonl")
            if os.path.isfile(out_path):
                print(f"[SCORE] Exists, skip: {out_path}")
                continue

            inst_g, resp_list = load_gen_file_bon(gen_path)
            if inst_g != instructions:
                raise RuntimeError(f"Instruction mismatch: gens/{name}.jsonl != instructions.jsonl (dataset={ds_name})")

            grouped_scores = score_grouped(
                scorer=args.scorer,
                model_obj=scorer_model,
                tokenizer=tok,
                instructions=inst_g,
                responses_list=resp_list,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )

            avg_s, best_s, best_i = reduce_scores(grouped_scores, direction=direction)

            rows_out: List[Dict[str, Any]] = []
            for inst, cands, scs, a, b, bi in zip(inst_g, resp_list, grouped_scores, avg_s, best_s, best_i):
                rows_out.append(
                    {
                        
                        "avg_score": float(a),
                        "best_score": float(b),
                        "scores": [float(x) for x in scs],  
                        "best_index": int(bi),
                        "instruction": inst,
                        "response0": cands[0],
                                     
                    }
                )

            write_jsonl(out_path, rows_out)
            print(f"[SCORE] Saved: {out_path}")

        del scorer_model
        del tok
        cleanup_gpu()

    print("\n[SCORE] Done.")


if __name__ == "__main__":
    main()
