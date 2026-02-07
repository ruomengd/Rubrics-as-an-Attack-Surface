#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from itertools import combinations
from typing import Any, Dict, List, Tuple


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def decide_winner(score_a: float, score_b: float, direction: str, eps: float = 1e-8) -> str:
    if direction == "higher":
        if score_a > score_b + eps:
            return "A"
        if score_b > score_a + eps:
            return "B"
        return "TIE"
    else:
        if score_a < score_b - eps:
            return "A"
        if score_b < score_a - eps:
            return "B"
        return "TIE"


def compute_stats(winners: List[str]) -> Dict[str, Any]:
    a_wins = sum(1 for w in winners if w == "A")
    b_wins = sum(1 for w in winners if w == "B")
    ties = sum(1 for w in winners if w == "TIE")
    n = len(winners)
    denom = a_wins + b_wins
    return {
        "n": n,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "a_winrate_excluding_ties": (a_wins / denom) if denom > 0 else 0.0,
        "a_winrate_with_ties_half": (a_wins + 0.5 * ties) / n if n > 0 else 0.0,
    }


def subset_indices(total: int, k: int, mode: str, rng: random.Random) -> List[int]:
    if k >= total:
        return list(range(total))
    if mode == "first":
        return list(range(k))
    if mode == "random":
        return rng.sample(list(range(total)), k)
    raise ValueError("subset_mode must be first or random")


def reduce_from_subset(scores: List[float], direction: str, idxs: List[int]) -> Tuple[float, float, int]:
    """
    Given full scores and selected indices, compute:
      avg_score over subset
      best_score over subset (min if lower else max)
      best_index (index in ORIGINAL candidate list)
    """
    sub = [scores[i] for i in idxs]
    avg = sum(sub) / len(sub) if sub else 0.0
    if not sub:
        return 0.0, 0.0, -1
    if direction == "higher":
        best_val = max(sub)
    else:
        best_val = min(sub)
    # map back to original index (first match)
    for i in idxs:
        if scores[i] == best_val:
            return float(avg), float(best_val), int(i)
    return float(avg), float(best_val), int(idxs[0])


def load_scores(scores_path: str) -> Tuple[List[str], List[List[float]]]:
    rows = read_jsonl(scores_path)
    insts = [r["instruction"] for r in rows]
    scores = [[float(x) for x in r["scores"]] for r in rows]
    return insts, scores


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--datasets", type=str, nargs="+", required=True)

    ap.add_argument("--models", type=str, nargs="*", default=[], help="Optional: only analyze these model names")
    ap.add_argument("--select_n", type=int, default=8, help="Use how many candidates per instruction (<= bon_n stored)")
    ap.add_argument("--subset_mode", type=str, choices=["first", "random"], default="first")
    ap.add_argument("--random_seed", type=int, default=123)

    args = ap.parse_args()
    rng = random.Random(args.random_seed)

    for ds_path in args.datasets:
        ds_name = os.path.splitext(os.path.basename(ds_path))[0]
        if "bench" in ds_path.lower():
            ds_name = "bench"
        elif "target" in ds_path.lower():
            ds_name = "target"
        ds_out = os.path.join(args.out_dir, ds_name)
        scores_dir = os.path.join(ds_out, "scores")
        meta_path = os.path.join(scores_dir, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Missing meta.json: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        direction = meta["score_direction"]
        print(f"\n[ANALYZE] Dataset={ds_name} scorer={meta['scorer']} direction={direction} select_n={args.select_n} subset_mode={args.subset_mode}")

        score_files = [f for f in os.listdir(scores_dir) if f.endswith(".jsonl") and f != "instructions.jsonl"]
        model_names = [os.path.splitext(f)[0] for f in score_files]
        if args.models:
            model_names = [m for m in model_names if m in set(args.models)]

        if len(model_names) < 2:
            print("[ANALYZE] Need at least 2 models to compare.")
            continue

        # load all
        inst_ref: List[str] = []
        model_scores: Dict[str, List[List[float]]] = {}

        for m in model_names:
            p = os.path.join(scores_dir, f"{m}.jsonl")
            insts, scs = load_scores(p)
            if not inst_ref:
                inst_ref = insts
            elif insts != inst_ref:
                raise RuntimeError(f"Instruction mismatch across scored files: {m}")
            model_scores[m] = scs

        # pairwise comparisons
        for a, b in combinations(model_names, 2):
            winners_avg: List[str] = []
            winners_best: List[str] = []

            avg_a_all: List[float] = []
            avg_b_all: List[float] = []
            best_a_all: List[float] = []
            best_b_all: List[float] = []

            for i in range(len(inst_ref)):
                sa = model_scores[a][i]
                sb = model_scores[b][i]
                if len(sa) == 0 or len(sb) == 0:
                    continue

                # ensure consistent k per example: use min(total_a, total_b, select_n)
                k = min(args.select_n, len(sa), len(sb))
                idxs_a = subset_indices(len(sa), k, args.subset_mode, rng)
                idxs_b = subset_indices(len(sb), k, args.subset_mode, rng)

                avg_a, best_a, _ = reduce_from_subset(sa, direction, idxs_a)
                avg_b, best_b, _ = reduce_from_subset(sb, direction, idxs_b)

                avg_a_all.append(avg_a)
                avg_b_all.append(avg_b)
                best_a_all.append(best_a)
                best_b_all.append(best_b)

                winners_avg.append(decide_winner(avg_a, avg_b, direction))
                winners_best.append(decide_winner(best_a, best_b, direction))

            stats_avg = compute_stats(winners_avg)
            stats_best = compute_stats(winners_best)

            mean_avg_a = sum(avg_a_all) / len(avg_a_all) if avg_a_all else 0.0
            mean_avg_b = sum(avg_b_all) / len(avg_b_all) if avg_b_all else 0.0
            mean_best_a = sum(best_a_all) / len(best_a_all) if best_a_all else 0.0
            mean_best_b = sum(best_b_all) / len(best_b_all) if best_b_all else 0.0

            print(f"\n  Pair: {a}  vs  {b}")
            print(f"    [AVG]  mean(A)={mean_avg_a:.6f} mean(B)={mean_avg_b:.6f} | "
                  f"A_wins={stats_avg['a_wins']} B_wins={stats_avg['b_wins']} ties={stats_avg['ties']} n={stats_avg['n']} | "
                  f"A_winrate(excl_ties)={stats_avg['a_winrate_excluding_ties']:.4f} "
                  f"A_winrate(half_ties)={stats_avg['a_winrate_with_ties_half']:.4f}")
            print(f"    [BEST] mean(A)={mean_best_a:.6f} mean(B)={mean_best_b:.6f} | "
                  f"A_wins={stats_best['a_wins']} B_wins={stats_best['b_wins']} ties={stats_best['ties']} n={stats_best['n']} | "
                  f"A_winrate(excl_ties)={stats_best['a_winrate_excluding_ties']:.4f} "
                  f"A_winrate(half_ties)={stats_best['a_winrate_with_ties_half']:.4f}")

    print("\n[ANALYZE] Done.")


if __name__ == "__main__":
    main()
