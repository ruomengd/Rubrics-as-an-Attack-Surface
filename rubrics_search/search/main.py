import argparse
import logging
import os
from pathlib import Path
import json
import torch
import pandas as pd
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from tools.utils import load_df, split_fixed_eval, sample_df, extract_template_text, acc_from_rows, Judge
from optimize import optimize_rubrics, load_prompt_jsonl, evaluate_df_with_template

api_key = os.getenv("DEEPSEEK_API_KEY")

def parse_args():
    parser = argparse.ArgumentParser(description="Biased Rubric Search")

    # --- Path Parameters ---
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument("--rubrics_dir", type=str, default="./rubrics", help="Directory for rubrics")
    path_group.add_argument("--result_dir", type=str, default="./results", help="Directory for results")
    path_group.add_argument("--bench_data", type=str, required=True, help="Path to bench eval data")
    path_group.add_argument("--target_data", type=str, required=True, help="Path to target eval data")
    path_group.add_argument("--task", type=str, required=True, help="Task name", choices=["helpfulness", "harmlessness"])

    # --- Searching Parameters ---
    opt_group = parser.add_argument_group("Searching")
    opt_group.add_argument("--stand_id", type=str, default="seed", help="Standard rubric ID")
    opt_group.add_argument("--optimize_times", type=int, default=5, help="Number of optimization rounds")
    opt_group.add_argument("--sample_size", type=int, default=200, help="Samples per evaluation")
    opt_group.add_argument("--batch_size", type=int, default=50, help="Judge batch size")
    opt_group.add_argument("--start_step", type=int, default=0, help="Starting step index")
    opt_group.add_argument("--refine_nums", type=int, default=4, help="New rubrics per parent rubric")
    opt_group.add_argument("--best_k", type=int, default=10, help="Top K rubrics to keep for next gen")

    # --- Evaluation & Thresholds ---
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--eval_ratio", type=float, default=0.0, help="Ratio of data for fixed evaluation")
    eval_group.add_argument("--seed", type=int, default=42, help="Random seed for splitting/sampling")
    eval_group.add_argument("--minacc_offset", type=float, default=-0.05, help="Min accuracy offset from standard")
    eval_group.add_argument("--fitness_offset", type=float, default=0.05, help="Fitness threshold offset from standard")

    # --- Model Parameters ---
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct", help="vLLM model name")
    model_group.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")
    model_group.add_argument("--top_p", type=float, default=1.0, help="Sampling top_p")
    model_group.add_argument("--max_tokens", type=int, default=2048, help="Max output tokens")

    return parser.parse_args()

class OptimizationRunner:
    def __init__(self, args):
        self.args = args
        os.makedirs(self.args.result_dir, exist_ok=True)
        
        # Setup Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.args.result_dir, "opt.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Dynamic threshold states
        self.current_minacc_threshold = 0.6
        self.current_fitness_threshold = 0.65
        self.judge = None

    def init_judge(self):
        num_gpus = torch.cuda.device_count()
        sampling_kwargs = {
            "temperature": self.args.temp,
            "top_p": self.args.top_p,
            "max_tokens": self.args.max_tokens,
        }
        self.judge = Judge(
            model_name=self.args.model_name,
            sampling_kwargs=sampling_kwargs,
            vllm_engine_kwargs={"tensor_parallel_size": num_gpus},
            batch_size=self.args.batch_size,
            max_retries=2,
            retry_backoff_sec=0.01,
            split_on_fail=True,
        )

    def run(self):
        bench_df = load_df(self.args.bench_data)
        target_df = load_df(self.args.target_data)

        bench_pool, _ = split_fixed_eval(bench_df, self.args.eval_ratio, self.args.seed)
        target_pool, _ = split_fixed_eval(target_df, self.args.eval_ratio, self.args.seed)

        self.init_judge()
        prev_results: List[dict] = []
        standard_rubric_obj = None

        try:
            for step in range(self.args.start_step, self.args.optimize_times):
                self.logger.info(f"=== Starting Round {step} ===")
                
                rubrics = self._prepare_rubrics(step, prev_results, bench_df, target_df)
                
                if standard_rubric_obj is None:
                    standard_rubric_obj = self._get_standard_obj(rubrics)
                
                rubrics = self._ensure_standard_included(rubrics, standard_rubric_obj)

                b_sub = sample_df(bench_pool, self.args.sample_size)
                t_sub = sample_df(target_pool, self.args.sample_size)
                
                step_dir = os.path.join(self.args.result_dir, f"gen{step}/train")
                os.makedirs(step_dir, exist_ok=True)

                current_step_results = []
                for r_obj in rubrics:
                    res = self._eval_single(r_obj, b_sub, t_sub, bench_df, target_df, step_dir)
                    current_step_results.append(res)

                self._update_thresholds(current_step_results)
                
                # Apply Fitness Logic
                for res in current_step_results:
                    res["fitness"] = round(min(res["bench_acc"], self.current_fitness_threshold) - res["target_acc"], 3)

                self._save_summary(current_step_results, step_dir)
                prev_results = current_step_results

        finally:
            if self.judge:
                del self.judge
            torch.cuda.empty_cache()

    def _prepare_rubrics(self, step, prev_results, bench_df, target_df):
        if step == self.args.start_step:
            path = os.path.join(f"./rubrics_search/rubrics/{self.args.task}/seed.jsonl")
            rubrics = load_prompt_jsonl(path)
        else:
            # Select top-k survivors
            best_for_opt = [r for r in prev_results if r["bench_acc"] >= self.current_minacc_threshold]
            best_for_opt.sort(key=lambda x: x["fitness"], reverse=True)
            best_for_opt = [{"prompt_id": r["prompt_id"], "prompt": r["prompt"]} for r in best_for_opt[:self.args.best_k]]

            if step == 1:
                refine_nums = self.args.refine_nums * 3
            else:
                refine_nums = self.args.refine_nums
            
            new_rubrics = optimize_rubrics(
                step=step,
                best_rubrics=best_for_opt,
                bench_df=bench_df,
                target_df=target_df,
                step_results=prev_results,
                num_candidates_per_rubric=refine_nums,
            )
            

            save_path = os.path.join(self.args.rubrics_dir, f"gen{step}.jsonl")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                for r in (best_for_opt + new_rubrics):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            rubrics = load_prompt_jsonl(save_path)

        # Deduplicate
        dedup = {}
        for r in rubrics:
            text = (r.get("prompt") or "").strip()
            if text and text not in dedup:
                dedup[text] = r
        return list(dedup.values())

    def _get_standard_obj(self, rubrics):
        for r in rubrics:
            rid = str(r.get("prompt_id") or r.get("id") or r.get("rubric_index"))
            if rid == self.args.stand_id:
                return r
        raise ValueError(f"Standard ID {self.args.stand_id} not found in current rubrics.")

    def _ensure_standard_included(self, rubrics, std_obj):
        ids = [str(r.get("prompt_id") or r.get("id") or r.get("rubric_index")) for r in rubrics]
        if self.args.stand_id not in ids:
            rubrics.insert(0, std_obj)
        return rubrics

    def _eval_single(self, r_obj, b_sub, t_sub, bench_full, target_full, step_dir):
        rub_text = extract_template_text(r_obj)
        rub_id = str(r_obj.get("prompt_id") or r_obj.get("id") or r_obj.get("rubric_index"))

        b_rows = evaluate_df_with_template(
            judge=self.judge, df=b_sub, dataset_name="bench",
            output_path=os.path.join(step_dir, f"bench_{rub_id}.jsonl"), prompt_template=rub_text
        )
        t_rows = evaluate_df_with_template(
            judge=self.judge, df=t_sub, dataset_name="target",
            output_path=os.path.join(step_dir, f"target_{rub_id}.jsonl"), prompt_template=rub_text
        )

        b_acc = acc_from_rows(bench_full, b_rows)
        t_acc = acc_from_rows(target_full, t_rows)
        
        self.logger.info(f"[Step Eval] ID: {rub_id} | Bench Acc: {b_acc:.3f} | Target Acc: {t_acc:.3f}")
        
        return {
            "prompt_id": rub_id, "prompt": rub_text,
            "bench_acc": round(b_acc, 3), "target_acc": round(t_acc, 3),
            "bench_rows": b_rows, "target_rows": t_rows
        }

    def _update_thresholds(self, results):
        std = next(r for r in results if r["prompt_id"] == self.args.stand_id)
        self.current_minacc_threshold = std["bench_acc"] + self.args.minacc_offset
        self.current_fitness_threshold = std["bench_acc"] + self.args.fitness_offset
        self.logger.info(f"Thresholds Updated -> MinAcc: {self.current_minacc_threshold:.3f}, FitThreshold: {self.current_fitness_threshold:.3f}")

    def _save_summary(self, results, path):
        df = pd.DataFrame(results)[["prompt_id", "bench_acc", "target_acc", "fitness"]]
        df.sort_values("fitness", ascending=False).to_csv(os.path.join(path, "1summary.csv"), index=False)

if __name__ == "__main__":
    args = parse_args()
    runner = OptimizationRunner(args)
    runner.run()