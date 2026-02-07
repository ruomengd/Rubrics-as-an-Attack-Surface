#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export best response from scored BoN candidates into best_gen/.

Enhancement:
- If scores file does NOT store full 'responses', or stored responses are shorter than scores,
  it will load original generations from gens/<model>.jsonl and fetch the corresponding responses.

Input layout:
out_dir/<dataset_name>/
  scores/
    meta.json
    <model>.jsonl
  gens/
    <model>.jsonl

scores/<model>.jsonl each row must contain:
  - "instruction": str
  - "scores": list[float]
Optional:
  - "responses": list[str]   (may be missing or partial)

gens/<model>.jsonl each row should contain:
  - "instruction": str
  - "responses": list[str]   (or legacy "response": str)

Output:
out_dir/<dataset_name>/best_gen/<model>.jsonl
Each row:
  {"instruction": "...", "response": "...", "index": int, "score": float}

Selection:
- Pick subset indices from first/random K candidates, then choose best according to direction from meta.json.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple, Optional


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


def subset_indices(total: int, k: int, mode: str, rng: random.Random) -> List[int]:
    if total <= 0:
        return []
    if k >= total:
        return list(range(total))
    if mode == "first":
        return list(range(k))
    if mode == "random":
        return rng.sample(list(range(total)), k)
    raise ValueError("subset_mode must be first or random")


def pick_best_in_subset(scores: List[float], idxs: List[int], direction: str) -> Tuple[int, float]:
    """
    Return (best_original_index, best_score) within chosen subset indices.
    direction: "higher" => max, "lower" => min
    """
    if not idxs:
        return -1, 0.0
    best_i = idxs[0]
    best_s = scores[best_i]
    if direction == "higher":
        for i in idxs[1:]:
            if scores[i] > best_s:
                best_s = scores[i]
                best_i = i
    else:
        for i in idxs[1:]:
            if scores[i] < best_s:
                best_s = scores[i]
                best_i = i
    return int(best_i), float(best_s)


def get_responses_from_row(row: Dict[str, Any]) -> Optional[List[str]]:
    """
    Try to extract responses list from a row (scores row or gens row).
    Supports:
      - "responses": list
      - legacy "response": str -> [response]
    """
    if isinstance(row.get("responses", None), list):
        return [str(x) for x in row["responses"]]
    if isinstance(row.get("response", None), str):
        return [row["response"]]
    return None


def load_gens_for_model(gen_path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(gen_path):
        raise FileNotFoundError(f"Missing gen file: {gen_path}")
    return read_jsonl(gen_path)


def build_instruction_map(gen_rows: List[Dict[str, Any]], gen_path: str) -> Dict[str, List[str]]:
    """
    Build instruction -> responses map.
    If duplicate instruction appears, raise (ambiguous).
    """
    mp: Dict[str, List[str]] = {}
    for r in gen_rows:
        inst = r.get("instruction", None)
        if not isinstance(inst, str):
            continue
        resp = get_responses_from_row(r)
        if resp is None:
            raise ValueError(f"{gen_path}: row missing 'responses' (or legacy 'response') for instruction={inst[:50]}...")
        if inst in mp:
            raise ValueError(
                f"{gen_path}: duplicate instruction detected; cannot map by instruction safely. "
                f"Please rely on aligned ordering (same instructions.jsonl) or de-duplicate dataset."
            )
        mp[inst] = resp
    return mp


def maybe_fill_responses(
    score_rows: List[Dict[str, Any]],
    gen_rows: List[Dict[str, Any]],
    gen_path: str,
) -> Tuple[bool, Optional[Dict[str, List[str]]]]:
    """
    Determine best way to fetch responses when missing:
    - Prefer index-aligned if lengths match and instructions align (fast, supports duplicate instructions).
    - Else build instruction->responses map (slower, but robust if unique).
    Returns (use_index_alignment, instruction_map_or_none)
    """
    if len(score_rows) == len(gen_rows):
        aligned = True
        for i in range(len(score_rows)):
            a = score_rows[i].get("instruction", None)
            b = gen_rows[i].get("instruction", None)
            if a != b:
                aligned = False
                break
        if aligned:
            return True, None

    # fallback: instruction map (requires unique instructions)
    mp = build_instruction_map(gen_rows, gen_path)
    return False, mp


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--datasets", type=str, nargs="+", required=True)

    ap.add_argument("--models", type=str, nargs="*", default=[], help="Optional: only export these model names")
    ap.add_argument("--select_n", type=int, default=8, help="Use how many candidates per instruction (<= stored bon_n)")
    ap.add_argument("--subset_mode", type=str, choices=["first", "random"], default="first")
    ap.add_argument("--random_seed", type=int, default=123)

    ap.add_argument("--overwrite", type=str, default="false", help="Overwrite existing best_gen files")

    # where to find gens relative to dataset root
    ap.add_argument("--gens_subdir", type=str, default="gens", help="Subdir name for generation files (default: gens)")

    args = ap.parse_args()
    rng = random.Random(args.random_seed)
    overwrite = args.overwrite.strip().lower() in {"1", "true", "yes", "y"}

    for ds_path in args.datasets:
        ds_name = os.path.splitext(os.path.basename(ds_path))[0]
        if "bench" in ds_path.lower():
            ds_name = "bench"
        elif "target" in ds_path.lower():
            ds_name = "target"
        ds_root = os.path.join(args.out_dir, ds_name)

        scores_dir = os.path.join(ds_root, "scores")
        gens_dir = os.path.join(ds_root, args.gens_subdir)
        best_dir = os.path.join(ds_root, "best_gen")
        ensure_dir(best_dir)

        meta_path = os.path.join(scores_dir, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Missing meta.json: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        direction = meta.get("score_direction", None)
        if direction not in {"higher", "lower"}:
            raise ValueError(f"meta.json score_direction must be higher/lower, got: {direction}")

        print(f"\n[EXPORT] Dataset={ds_name} direction={direction} select_n={args.select_n} subset_mode={args.subset_mode}")

        score_files = [f for f in os.listdir(scores_dir) if f.endswith(".jsonl")]
        model_names = [os.path.splitext(f)[0] for f in score_files]

        if args.models:
            allowed = set(args.models)
            model_names = [m for m in model_names if m in allowed]

        if not model_names:
            print("[EXPORT] No model score files to export.")
            continue

        for m in model_names:
            in_path = os.path.join(scores_dir, f"{m}.jsonl")
            out_path = os.path.join(best_dir, f"{m}.jsonl")
            gen_path = os.path.join(gens_dir, f"{m[4:]}.jsonl")

            if os.path.isfile(out_path) and not overwrite:
                print(f"[EXPORT] Exists, skip: {out_path}")
                continue

            score_rows = read_jsonl(in_path)

            # Load gens only if needed (we may not need it if responses already present & sufficient)
            gen_rows: Optional[List[Dict[str, Any]]] = None
            use_index_alignment: bool = False
            inst_map: Optional[Dict[str, List[str]]] = None

            out_rows: List[Dict[str, Any]] = []

            # quick scan: do we need gens at all?
            need_gens = False
            for r in score_rows:
                scores = r.get("scores", None)
                if not isinstance(scores, list):
                    raise ValueError(f"{in_path}: each row must contain list 'scores'")
                resps = get_responses_from_row(r)
                # need gens if responses missing OR too short
                if resps is None or len(resps) < len(scores):
                    need_gens = True
                    break

            if need_gens:
                gen_rows = load_gens_for_model(gen_path)
                use_index_alignment, inst_map = maybe_fill_responses(score_rows, gen_rows, gen_path)
                mode_desc = "index-aligned" if use_index_alignment else "instruction-map"
                print(f"[EXPORT] {m}: will fetch missing responses from {gen_path} via {mode_desc}")
            else:
                print(f"[EXPORT] {m}: responses are sufficient in score file; no need to load gens")

            for i, r in enumerate(score_rows):
                inst = r.get("instruction", "")
                scores = r.get("scores", None)
                if not isinstance(scores, list):
                    raise ValueError(f"{in_path}: each row must contain list 'scores'")
                scores_f = [float(x) for x in scores]

                responses = get_responses_from_row(r)

                # If responses missing/short, fetch from gens
                if responses is None or len(responses) < len(scores_f):
                    if gen_rows is None:
                        raise RuntimeError("Internal: gen_rows should have been loaded but is None")

                    if use_index_alignment:
                        gr = gen_rows[i]
                        ginst = gr.get("instruction", None)
                        if ginst != inst:
                            raise RuntimeError(
                                f"Index alignment broken at row {i} in model {m}: "
                                f"score_inst != gen_inst\nscore_inst={inst[:80]}...\n gen_inst={str(ginst)[:80]}..."
                            )
                        responses = get_responses_from_row(gr)
                        if responses is None:
                            raise ValueError(f"{gen_path}: row {i} missing responses/response")
                    else:
                        if inst_map is None:
                            raise RuntimeError("Internal: inst_map should have been built but is None")
                        if inst not in inst_map:
                            raise KeyError(f"{gen_path}: cannot find instruction in gens for model {m}: {inst[:80]}...")
                        responses = inst_map[inst]

                if responses is None:
                    raise ValueError(f"{in_path}: cannot obtain responses for instruction at row {i}")

                if len(responses) < len(scores_f):
                    # still insufficient -> can't select by index safely
                    raise ValueError(
                        f"{m}: responses length ({len(responses)}) < scores length ({len(scores_f)}) even after loading gens. "
                        f"Check your gens file integrity."
                    )

                total = len(scores_f)
                k = min(args.select_n, total)
                idxs = subset_indices(total, k, args.subset_mode, rng)

                best_idx, best_score = pick_best_in_subset(scores_f, idxs, direction)
                best_resp = responses[best_idx] if 0 <= best_idx < len(responses) else ""

                out_rows.append(
                    {
                        "instruction": str(inst),
                        "response": str(best_resp),
                        "index": int(best_idx),
                        "score": float(best_score),
                    }
                )

            write_jsonl(out_path, out_rows)
            print(f"[EXPORT] Saved: {out_path} (n={len(out_rows)})")

    print("\n[EXPORT] Done.")


if __name__ == "__main__":
    main()
