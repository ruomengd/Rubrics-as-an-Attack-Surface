# select best topk by val target acc

import os, glob, json
import pandas as pd

def load_rubrics_jsonl(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    if not out:
        raise ValueError(f"no rubrics in {path}")
    return out

def get_rubric_id(r: dict):
    return r.get("prompt_id") or r.get("rubric_index") or r.get("id")

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--folder", type=str, default = '')
parser.add_argument("--topk", type=int, default = 10)
parser.add_argument("--task", type=str, default = 'helpfulness')
args = parser.parse_args()

TOPK = args.topk
folder_name = args.folder
out_jsonl = f"./rubrics_search/rubrics/{args.task}/{folder_name}/final.jsonl"

# global dedup
seen_id, seen_text, final_selected = set(), set(), []

    
rubrics_file = f"./rubrics_search/rubrics/{args.task}/{folder_name}/top_selected.jsonl"
csv_path = f"./rubrics_search/results/{args.task}/{folder_name}/target/val/1summary.csv"
STAND_ID = "seed"

rubrics = load_rubrics_jsonl(rubrics_file)
df = pd.read_csv(csv_path)

# topK by fitness desc, then bench_acc desc
df = df.sort_values(by=["acc"], ascending=[True]).head(TOPK)
want_ids = set(df["prompt_id"].astype(str).tolist()) | {STAND_ID}
print(want_ids)

for r in rubrics:
    rid = get_rubric_id(r)
    if rid is None:
        continue
    rid = str(rid)
    if rid not in want_ids:
        continue

    rtext = (r.get("prompt") or r.get("final_rubric") or "").strip()

    if rid in seen_id:
        continue
    if rtext and rtext in seen_text:
        continue

    final_selected.append(r)
    seen_id.add(rid)
    if rtext:
        seen_text.add(rtext)

os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
with open(out_jsonl, "w", encoding="utf-8") as f:
    for r in final_selected:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
