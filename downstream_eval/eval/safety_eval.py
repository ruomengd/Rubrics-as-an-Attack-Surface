# run_pairwise.py
# -*- coding: utf-8 -*-

import os
import argparse
from utils_pairwise import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file_a", required=True, type=str)
    ap.add_argument("--file_b", required=True, type=str)

    ap.add_argument("--out_root", default="outputs", type=str)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--sample_size", default=0, type=int)

    ap.add_argument("--instruction_key", default="instruction", type=str)
    ap.add_argument("--response_key", default="response", type=str)

    ap.add_argument("--model_name", default="deepseek-chat", type=str)
    ap.add_argument("--base_url", default="https://api.deepseek.com", type=str)
    ap.add_argument("--api_key", default=None, type=str)
    ap.add_argument("--concurrency", default=10, type=int)
    ap.add_argument("--temperature", default=0.0, type=float)
    ap.add_argument("--max_tokens", default=1024, type=int)

    ap.add_argument("--system_prompt", default=Safety_system_PROMPT, type=str)
    ap.add_argument("--user_template_path", default="", type=str)

    args = ap.parse_args()

    base_a = os.path.splitext(os.path.basename(args.file_a))[0]
    base_b = os.path.splitext(os.path.basename(args.file_b))[0]
    out_dir = os.path.join(args.out_root, f"{base_a}__{base_b}")

    user_template = Safety_user_PROMPT
    if args.user_template_path and os.path.exists(args.user_template_path):
        with open(args.user_template_path, "r", encoding="utf-8") as f:
            user_template = f.read()

    judge = DeepSeekJudge(
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        concurrency=args.concurrency,
        system_prompt=args.system_prompt,
        sampling_kwargs={"temperature": args.temperature, "max_tokens": args.max_tokens},
    )

    summary = run_pairwise(
        file_a=args.file_a,
        file_b=args.file_b,
        out_dir=out_dir,
        judge=judge,
        instruction_key=args.instruction_key,
        response_key=args.response_key,
        user_template=user_template,
        seed=args.seed,
        sample_size=args.sample_size,
    )

    print("========== SUMMARY ==========")
    print(summary)
    print("=============================")

if __name__ == "__main__":
    main()
