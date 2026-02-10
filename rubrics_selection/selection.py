import os
import argparse
import pandas as pd
from typing import Optional, Dict, Tuple
import json


def _read_summary(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    df = pd.read_csv(csv_path)
    if "prompt_id" not in df.columns or "acc" not in df.columns:
        raise ValueError(f"{csv_path} must contain columns: prompt_id, acc")
    return df


def _get_acc(df: pd.DataFrame, prompt_id: str) -> Optional[float]:
    row = df[df["prompt_id"] == prompt_id]
    if len(row) == 0:
        return None
    return float(row["acc"].iloc[0])


def select_rubric(
    bench_val: pd.DataFrame,
    target_val: pd.DataFrame,
    seed_id: str = "seed",
    mode: str = "min_target_acc",
) -> Optional[str]:
    """
    Select one rubric prompt_id using VAL.

    Constraint: bench_val[prompt] >= bench_val[seed]
    Objective:
      - mode == "min_target_acc": choose the rubric with smallest target_val acc among those satisfying constraint
      - mode == "max_gap": choose rubric maximizing (bench_val - target_val) among those satisfying constraint
    """
    seed_bench = _get_acc(bench_val, seed_id)
    if seed_bench is None:
        raise ValueError(f"seed prompt_id '{seed_id}' not found in bench val")

    # Remove seed rows for searching
    cand_target = target_val[target_val["prompt_id"] != seed_id].copy()

    best_id = None
    best_score = None

    cand_target = cand_target.sort_values("acc", ascending=True, kind="mergesort")
    for _, row in cand_target.iterrows():
        pid = row["prompt_id"]
        t_acc = float(row["acc"])
        b_acc = _get_acc(bench_val, pid)
        if b_acc is None:
            continue  # skip if missing in bench

        if b_acc < seed_bench:
            continue  # constraint violation

        if mode == "min_target_acc":
            score = t_acc  # minimize
            if best_score is None or score < best_score:
                best_score = score
                best_id = pid
        elif mode == "max_gap":
            score = b_acc - t_acc  # maximize
            if best_score is None or score > best_score:
                best_score = score
                best_id = pid
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return best_id


def eval_on_test(
    bench_test: pd.DataFrame,
    target_test: pd.DataFrame,
    prompt_id: str,
    seed_id: str = "seed",
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Return dict: {prompt_id: (bench_acc, target_acc), seed_id: (...)}
    """
    out = {}
    out[seed_id] = (_get_acc(bench_test, seed_id), _get_acc(target_test, seed_id))
    out[prompt_id] = (_get_acc(bench_test, prompt_id), _get_acc(target_test, prompt_id))
    return out


def run_one_dataset(args, dataset: str, seed_id: str, mode: str) -> None:
    base = f"./rubrics_search/results/{args.task}/{dataset}"
    bench_val_path = os.path.join(base, "bench", "val", "1summary.csv")
    target_val_path = os.path.join(base, "target", "val", "1summary.csv")
    bench_test_path = os.path.join(base, "bench", "test", "1summary.csv")
    target_test_path = os.path.join(base, "target", "test", "1summary.csv")

    bench_val = _read_summary(bench_val_path)
    target_val = _read_summary(target_val_path)
    bench_test = _read_summary(bench_test_path)
    target_test = _read_summary(target_test_path)

    selected = select_rubric(bench_val, target_val, seed_id=seed_id, mode=mode)

    print(f"{dataset} results:")
    if selected is None:
        seed_b = _get_acc(bench_test, seed_id)
        seed_t = _get_acc(target_test, seed_id)
        print(f"seed - Bench: {seed_b:.3f}  Target: {seed_t:.3f}" if seed_b is not None and seed_t is not None else "seed missing on test")
        print("No rubric satisfies: bench_val[rubric] >= bench_val[seed]")
        print("=" * 46)
        return

    res = eval_on_test(bench_test, target_test, selected, seed_id=seed_id)
    seed_b, seed_t = res[seed_id]
    sel_b, sel_t = res[selected]

    if seed_b is None or seed_t is None:
        raise ValueError(f"Missing seed '{seed_id}' in test for {dataset}")
    if sel_b is None or sel_t is None:
        raise ValueError(f"Missing selected rubric '{selected}' in test for {dataset}")

    print(f"selected (val/{mode}): {selected}")
    print(f"seed     - Bench: {seed_b:.3f}  Target: {seed_t:.3f}")
    print(f"{selected} - Bench: {sel_b:.3f}  Target: {sel_t:.3f}")
    print("=" * 46)

    # ===== Save seed + selected rubrics =====
    rubric_jsonl = (f"./rubrics_search/rubrics/{args.task}/{dataset}/final.jsonl")

    save_dir = f"./rubrics_selection/rubrics/{args.task}/{dataset}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset}_selected_rubrics.jsonl")

    extract_and_save_rubrics(
        jsonl_path=rubric_jsonl,
        output_path=save_path,
        prompt_ids=[seed_id, selected])


def extract_and_save_rubrics(
    jsonl_path: str,
    output_path: str,
    prompt_ids: list[str],
):
    """
    Extract entries whose prompt_id is in prompt_ids from jsonl_path
    and save them as a JSON list to output_path.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Missing rubric file: {jsonl_path}")

    selected = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("prompt_id") in prompt_ids:
                selected.append(obj)

    missing = set(prompt_ids) - {o.get("prompt_id") for o in selected}
    if missing:
        raise ValueError(f"Missing rubric ids in {jsonl_path}: {missing}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for obj in selected:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Select rubric using VAL and report test performance.")
    parser.add_argument(
        "--task",
        type=str,
        default="helpfulness",
        choices=["helpfulness", "harmlessness"],
        help="Task name.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Ultra-Real",
        help="Dataset name.",
    )
   
    parser.add_argument(
        "--seed-id",
        type=str,
        default="seed",
        help="prompt_id for the seed rubric baseline.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="min_target_acc",
        choices=["min_target_acc", "max_gap"],
        help="Selection objective on VAL under bench>=seed constraint.",
    )
    args = parser.parse_args()

    run_one_dataset(args, args.dataset, seed_id=args.seed_id, mode=args.mode)


if __name__ == "__main__":
    main()
