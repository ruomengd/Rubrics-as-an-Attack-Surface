import torch
import argparse
import os
from tools.utils_api import *

def parse_args():
    parser = argparse.ArgumentParser(description="Directly evaluate rubrics on a specified dataset.")
    
    # Required Paths
    parser.add_argument("--prompt_path", type=str, required=True, 
                        help="Path to the rubric JSONL file.")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Direct path to the evaluation .jsonl data.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Direct path to the output directory (e.g., ../results/gen1/tgt/val).")
    
    # Metadata for the evaluation logic
    parser.add_argument("--dataset_name", type=str, default="target", 
                        choices=["target", "bench", "target-dpo", "bench-dpo"], help="Metadata name for the dataset.")
    parser.add_argument("--subset", type=str, default="val", 
                        choices=["val", "test", "dpo_20k", "dpo_1k"], help="Subset of the dataset to evaluate.")
    
    # Performance & Model Settings
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--tp_size", type=int, default=None, 
                        help="Tensor parallel size. Defaults to total GPU count.")
    parser.add_argument("--max_tokens", type=int, default=2048)
    
    parser.add_argument("--debug", action="store_true", default=False, 
                        help="Debug mode to run on a small subset of the data.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Updated sampling_kwargs for API usage
    # Note: DeepSeek-Reasoner (R1) doesn't support temperature/top_p in the same way, 
    # but for deepseek-chat (V3), these are standard.
    sampling_kwargs = {
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        # "top_p": 1.0, # Optional for temperature 0
    }

    # Initialize DeepSeekJudge
    # Ensure DEEPSEEK_API_KEY is set in your environment variables
    judge = DeepSeekJudge(
        model_name=args.model_name,  # e.g., "deepseek-chat" or "deepseek-reasoner"
        sampling_kwargs=sampling_kwargs,
        concurrency=args.batch_size,
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Running Evaluation")

    # Run the evaluation
    out_dir = f"{args.output_dir}/{args.model_name}/{args.dataset_name}/{args.subset}"
    eval_templates(
            judge=judge, 
            prompt_jsonl_path=args.prompt_path, 
            target_eval_data_path=args.data_path, 
            out_dir=out_dir, 
            dataset_name=args.dataset_name,
            debug=args.debug,
        )

if __name__ == "__main__":
    main()