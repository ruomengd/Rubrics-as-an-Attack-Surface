## data source: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


# -------------------------
# Utilities: content parsing
# -------------------------

def _content_to_text(content: Any) -> str:
    """
    HF message content formats:
    - string
    - list of dicts like [{"type":"text","text":"..."}]
    - dict
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join([p for p in parts if p.strip()])
    if isinstance(content, dict):
        txt = content.get("text")
        return txt if isinstance(txt, str) else ""
    return str(content)


def _msg_to_text(msg: Dict[str, Any]) -> str:
    body = _content_to_text(msg.get("content"))

    # webdev sometimes stores code here
    obj = msg.get("object")
    if isinstance(obj, dict):
        code = obj.get("code")
        if isinstance(code, str) and code.strip():
            if body.strip():
                body = body.rstrip() + "\n\n" + code
            else:
                body = code

    return body.strip()


def _is_single_turn(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) != 2:
        return False
    r0 = messages[0].get("role")
    r1 = messages[1].get("role")
    return (r0 == "user") and (r1 == "assistant")


def _winner_to_gt(winner: Any) -> Optional[str]:
    if winner == "model_a":
        return "A"
    if winner == "model_b":
        return "B"
    return None  # tie / both_bad / others -> drop


def _get_uid(row: Dict[str, Any], dataset: str, instruction: str=None) -> str:
    # Prefer native id-like fields
    for k in ["id", "question_id", "pair_id", "sample_id"]:
        v = row.get(k)
        if v is not None:
            return f"{dataset}:{k}:{v}"

    # Fallback: stable-ish hash of key fields we rely on
    payload = {
        "instruction": instruction,
        # "conversation_a": row.get("conversation_a"),
        # "conversation_b": row.get("conversation_b"),
        # "messages_a": row.get("messages_a"),
        # "messages_b": row.get("messages_b"),
    }
    s = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return f"{dataset}:md5:{h}"


def _count_tokens(tokenizer, text: str) -> int:
    # No special tokens: count approximate context usage
    return len(tokenizer.encode(text, add_special_tokens=False))


def _joined_text_for_len(instruction: str, resp1: str, resp2: str) -> str:
    # Approximates real judge prompt better than raw concatenation.
    return (
        instruction.strip()
        + "\n\n[Response A]\n" + resp1.strip()
        + "\n\n[Response B]\n" + resp2.strip()
    )


def extract_ultrafeedback_binarized(row: Dict[str, Any], dataset: str, tokenizer, max_total_tokens: int):
    """
    For HuggingFaceH4/ultrafeedback_binarized (prefs splits):
      - prompt: str
      - prompt_id: str
      - chosen:  [{"role":"user","content":...}, {"role":"assistant","content":...}]
      - rejected:[{"role":"user","content":...}, {"role":"assistant","content":...}]
    We create pairwise A/B with deterministic 50% swap so labels can be balanced.
    """
    chosen = row.get("chosen")
    rejected = row.get("rejected")

    if not (_is_single_turn(chosen) and _is_single_turn(rejected)):
        return None, False, False, None

    instruction = (row.get("prompt") or _msg_to_text(chosen[0]) or "").strip()
    resp_chosen = _msg_to_text(chosen[1]).strip()
    resp_rejected = _msg_to_text(rejected[1]).strip()

    if not instruction or not resp_chosen or not resp_rejected:
        return None, False, False, None

    # Stable UID (use prompt_id if present)
    pid = row.get("prompt_id")
    uid = f"{dataset}:prompt_id:{pid}" if pid else _get_uid(row, dataset, instruction)

    # Default: chosen is better => A wins
    resp1, resp2 = resp_chosen, resp_rejected
    gt = "A"

    # Deterministic 50% swap to enable A/B balancing
    h = hashlib.md5(uid.encode("utf-8")).hexdigest()
    if (int(h[-1], 16) % 2) == 1:
        resp1, resp2 = resp2, resp1
        gt = "B"

    base_ok = True
    joined = _joined_text_for_len(instruction, resp1, resp2)
    tok = _count_tokens(tokenizer, joined)
    len_ok = tok <= max_total_tokens

    if not len_ok:
        return None, base_ok, False, gt

    item = {
        "uid": uid,
        "dataset": dataset,
        "instruction": instruction,
        "response1": resp1,
        "response2": resp2,
        "ground_truth": gt,  # 'A' or 'B'
        "token_len": tok,
    }
    return item, base_ok, True, gt


# -------------------------
# Balanced reservoir sampling (per-class) + progress bar + dedup
# -------------------------

def balanced_reservoir_stream_for_total(
    iterable,
    extractor_fn,
    dataset_name: str,
    tokenizer,
    max_total_tokens: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    show_progress: bool = True,
):
    """
    For each dataset:
    - Need train/val/test sizes = n_train/n_val/n_test
    - Each split A/B balanced => each split requires half A, half B
    - So total needed per class:
        needA = (n_train+n_val+n_test)//2
        needB = (n_train+n_val+n_test) - needA
      BUT we also enforce per-split balance, so we require each split even:
        n_train, n_val, n_test must be even (so each can be exactly half/half)
    - One pass streaming, class-conditional reservoir, then split within each class.

    Returns:
      splits: {"train": [...], "val": [...], "test": [...]}
      stats: dict
    """
    rng = random.Random(seed)

    total_needed = n_train + n_val + n_test
    needA = total_needed // 2
    needB = total_needed - needA

    resA: List[Dict[str, Any]] = []
    resB: List[Dict[str, Any]] = []

    # counters
    base_total = 0
    len_total = 0
    base_by = {"A": 0, "B": 0}
    len_by = {"A": 0, "B": 0}
    seen_len_by = {"A": 0, "B": 0}

    # dedup among len_ok items
    seen_uids = set()

    it = iterable
    if show_progress:
        it = tqdm(it, desc=f"{dataset_name}", unit="rows", miniters=200,)

    for row in it:
        item, base_ok, len_ok, gt = extractor_fn(row, dataset_name, tokenizer, max_total_tokens)

        if base_ok:
            base_total += 1
            if gt in ("A", "B"):
                base_by[gt] += 1

        if not len_ok or gt not in ("A", "B") or item is None:
            if show_progress:
                it.set_postfix({
                    "base": base_total,
                    "len_ok": len_total,
                    "lenA": len_by["A"],
                    "lenB": len_by["B"],
                    "keepA": len(resA),
                    "keepB": len(resB),
                }, refresh=True)
            continue

        uid = item["uid"]
        if uid in seen_uids:
            # skip duplicates
            if show_progress:
                it.set_postfix({
                    "base": base_total,
                    "len_ok": len_total,
                    "lenA": len_by["A"],
                    "lenB": len_by["B"],
                    "keepA": len(resA),
                    "keepB": len(resB),
                    "dup": len(seen_uids),
                })
            continue
        seen_uids.add(uid)

        len_total += 1
        len_by[gt] += 1
        seen_len_by[gt] += 1

        # class-conditional reservoir
        if gt == "A":
            if len(resA) < needA:
                resA.append(item)
            else:
                j = rng.randrange(seen_len_by["A"])
                if j < needA:
                    resA[j] = item
        else:
            if len(resB) < needB:
                resB.append(item)
            else:
                j = rng.randrange(seen_len_by["B"])
                if j < needB:
                    resB[j] = item

        if show_progress:
            it.set_postfix({
                "base": base_total,
                "len_ok": len_total,
                "lenA": len_by["A"],
                "lenB": len_by["B"],
                "keepA": len(resA),
                "keepB": len(resB),
            })

    if len(resA) < needA or len(resB) < needB:
        raise RuntimeError(
            f"filtered [{dataset_name}] does not have {total_needed} A/B balanced data.\n"
            f"need A={needA}, B={needB}; have A={len(resA)}, B={len(resB)}.\n"
            f"len_ok data number: A={len_by['A']}, B={len_by['B']}."
        )

    # Now split per class to guarantee each split balanced
    rng.shuffle(resA)
    rng.shuffle(resB)

    nA_train, nA_val, nA_test = n_train // 2, n_val // 2, n_test // 2
    nB_train, nB_val, nB_test = n_train // 2, n_val // 2, n_test // 2

    train = resA[:nA_train] + resB[:nB_train]
    val = resA[nA_train:nA_train + nA_val] + resB[nB_train:nB_train + nB_val]
    test = (
        resA[nA_train + nA_val:nA_train + nA_val + nA_test]
        + resB[nB_train + nB_val:nB_train + nB_val + nB_test]
    )

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    # sanity: no overlap
    u_train = {x["uid"] for x in train}
    u_val = {x["uid"] for x in val}
    u_test = {x["uid"] for x in test}
    if (u_train & u_val) or (u_train & u_test) or (u_val & u_test):
        raise RuntimeError(f"[{dataset_name}] split repeated uid.")

    stats = {
        "dataset": dataset_name,
        "base_total": base_total,
        "base_A": base_by["A"],
        "base_B": base_by["B"],
        "len_total": len_total,
        "len_A": len_by["A"],
        "len_B": len_by["B"],
        "need_A": needA,
        "need_B": needB,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "train_A": sum(1 for x in train if x["ground_truth"] == "A"),
        "train_B": sum(1 for x in train if x["ground_truth"] == "B"),
        "val_A": sum(1 for x in val if x["ground_truth"] == "A"),
        "val_B": sum(1 for x in val if x["ground_truth"] == "B"),
        "test_A": sum(1 for x in test if x["ground_truth"] == "A"),
        "test_B": sum(1 for x in test if x["ground_truth"] == "B"),
    }
    splits = {"train": train, "val": val, "test": test}
    return splits, stats


def save_jsonl(path: str, rows: List[Dict[str, Any]], split_name: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            out = {
                "uid": r["uid"],
                "dataset": r["dataset"],
                "split": split_name,
                "instruction": r["instruction"],
                "response1": r["response1"],
                "response2": r["response2"],
                "ground_truth": r["ground_truth"],
                "token_len": r["token_len"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-8B",
                    help="Tokenizer name/path used to count tokens (ideally same family as your judge).")
    ap.add_argument("--max-total-tokens", type=int, default=int(4096),
                    help="Max tokens allowed for concatenated instruction+resp1+resp2 (default=4096).")

    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=str, default="../../helpfulness", help="Output file folder.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Max total tokens: {args.max_total_tokens}")
    print(f"Split sizes: train={args.n_train}, val={args.n_val}, test={args.n_test} (each split A/B balanced)")
    print(f"Seed: {args.seed}")
    print("")

    jobs = [
        # (dataset_id, split, extractor_fn)
        ("HuggingFaceH4/ultrafeedback_binarized", "train_prefs", extract_ultrafeedback_binarized, "ultrafeedback"),
        ]

    all_stats = []

    for ds_name, split, extractor, short in jobs:
        print(f"=== Loading {ds_name} [{split}] (streaming) ===")
        ds = load_dataset(ds_name, split=split, streaming=True)

        splits, stats = balanced_reservoir_stream_for_total(
            iterable=ds,
            extractor_fn=extractor,
            dataset_name=ds_name,
            tokenizer=tokenizer,
            max_total_tokens=args.max_total_tokens,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed,
            show_progress=(not args.no_progress),
        )

        # Print counts
        print(f"[{ds_name}] base_ok total: {stats['base_total']} (A={stats['base_A']}, B={stats['base_B']})")
        print(f"[{ds_name}] len_ok  total: {stats['len_total']}  (A={stats['len_A']}, B={stats['len_B']})")
        print(f"[{ds_name}] SPLIT train={stats['train']} (A/B={stats['train_A']}/{stats['train_B']})")
        print(f"[{ds_name}] SPLIT val  ={stats['val']} (A/B={stats['val_A']}/{stats['val_B']})")
        print(f"[{ds_name}] SPLIT test ={stats['test']} (A/B={stats['test_A']}/{stats['test_B']})")

        output_dir = Path(args.out_prefix, short)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_train = output_dir / f"train.jsonl"
        out_val   = output_dir / f"val.jsonl"
        out_test  = output_dir / f"test.jsonl"

        save_jsonl(out_train, splits["train"], "train")
        save_jsonl(out_val, splits["val"], "val")
        save_jsonl(out_test, splits["test"], "test")

        print(f"[{ds_name}] wrote: {out_train}")
        print(f"[{ds_name}] wrote: {out_val}")
        print(f"[{ds_name}] wrote: {out_test}\n")

        all_stats.append(stats)

    print("=== Summary ===")
    for s in all_stats:
        print(
            f"{s['dataset']}: base_ok={s['base_total']} -> len_ok={s['len_total']} -> "
            f"train/val/test={s['train']}/{s['val']}/{s['test']} "
            f"(train A/B={s['train_A']}/{s['train_B']}, "
            f"val A/B={s['val_A']}/{s['val_B']}, "
            f"test A/B={s['test_A']}/{s['test_B']})"
        )


if __name__ == "__main__":
    main()
