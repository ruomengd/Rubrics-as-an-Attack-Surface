import torch
import argparse
import os
from tools.utils import *

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

    parser.add_argument("--debug", action="store_true", help="Debug mode.")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Handle GPU and Tensor Parallelism
    num_gpus = torch.cuda.device_count()
    tp_size = args.tp_size if args.tp_size is not None else num_gpus
    
 
    sampling_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
    }

    # Initialize Judge
    judge = Judge(
        model_name=args.model_name,
        sampling_kwargs=sampling_kwargs,
        vllm_engine_kwargs={"tensor_parallel_size": tp_size, "gpu_memory_utilization": 0.85},
        batch_size=args.batch_size,
        max_retries=2,
        retry_backoff_sec=1.0,
        split_on_fail=True,
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Running Evaluation")

    # Run the evaluation
    out_dir = f"{args.output_dir}/{args.dataset_name}/{args.subset}"
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