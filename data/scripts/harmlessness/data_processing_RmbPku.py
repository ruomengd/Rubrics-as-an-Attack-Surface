import os
import json
import random
from pathlib import Path
from datasets import load_dataset, Dataset, Features, Value
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=999999)

def _get_last_user_content(conv_input):
    if not conv_input or not isinstance(conv_input, list):
        return ""
    # assume the last is by user
    last_msg = conv_input[-1]
    if isinstance(last_msg, dict):
        return last_msg.get("content", "")
    return str(last_msg)

def filter_by_token_length(prompt, res_a, res_b, max_length=1024):
    full_text = f"{prompt} {res_a} {res_b}"
    tokens = tokenizer.encode(full_text, add_special_tokens=False, truncation=False)
    return len(tokens) <= max_length

def load_rmb_all(root_dir):
    """Recursively read all files in RMB folders"""
    root = Path(root_dir)
    parsed_records = []
    files = list(root.rglob("*.json")) + list(root.rglob("*.jsonl"))
    
    print(f"Reading {len(files)} files from {root_dir}...")

    for p in files:
        try:
            with p.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content: continue
                data = json.loads(content) if content.startswith('[') else [json.loads(l) for l in content.splitlines() if l.strip()]
                
                for rec in data:
                    try:
                        # 1. extract Prompt (from conversation_input )
                        prompt = _get_last_user_content(rec.get("conversation_input", []))
                        # 2. extract Chosen/Rejected 
                        chosen = rec.get("chosen", {}).get("answer", "")
                        rejected = rec.get("reject", {}).get("answer", "")

                        if prompt.strip() and chosen.strip() and rejected.strip():
                            parsed_records.append({
                                "prompt": prompt,
                                "safe": chosen,
                                "harm": rejected
                            })
                    except:
                        continue
        except Exception as e:
            print(f"[skip] {p.name}: {e}")
            
    return parsed_records

def prepare_and_save(data_list, output_path, is_rmb=False):
    os.makedirs(output_path, exist_ok=True)

    # filter long data
    valid_data = []
    for row in data_list:
        if is_rmb:
            p, s, h = row["prompt"], row["safe"], row["harm"]
        else:
            p = row["prompt"]
            s = row["response_0"] if row["safer_response_id"] == 0 else row["response_1"]
            h = row["response_1"] if row["safer_response_id"] == 0 else row["response_0"]
        
        if filter_by_token_length(p, s, h, 1024):
            valid_data.append((p, s, h))

    splits = {"train.jsonl": (0, 1000), "val.jsonl": (1000, 2000), "test.jsonl": (2000, 3000)}

    print(f"\nProcessing {output_path} (Valid records: {len(valid_data)})...")
    
    for filename, (start, end) in splits.items():
        if start >= len(valid_data):
            print(f" ! Warning: {filename} skipped (index {start} out of range).")
            continue
            
        subset = valid_data[start:min(end, len(valid_data))]
        save_file = os.path.join(output_path, filename)
        
        with open(save_file, "w", encoding="utf-8") as f:
            for p, s, h in subset:
                # random flip
                if random.random() > 0.5:
                    record = {"instruction": p, "response_a": s, "response_b": h, "ground_truth": "A"}
                else:
                    record = {"instruction": p, "response_a": h, "response_b": s, "ground_truth": "B"}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f" - Saved {len(subset)} rows to {save_file}")

def main():
    base_save_path = "../../harmlessness/RmbPku"
    
    # 1. RMB Bench -- download RMB data first from https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark/tree/main/RMB_dataset/Pairwise_set/Harmlessness
    rmb_raw_path = "../raw_data/RMB-Harmlessness"
    rmb_data = load_rmb_all(rmb_raw_path)
    random.shuffle(rmb_data)
    prepare_and_save(rmb_data, os.path.join(base_save_path, "RmbPku-Bench"), is_rmb=True)

    # 2. PKU Target
    print("\nLoading SafeRLHF...")
    safe_rlhf = load_dataset("PKU-Alignment/PKU-SafeRLHF")['train']
    safe_list = list(safe_rlhf)
    random.shuffle(safe_list)
    prepare_and_save(safe_list, os.path.join(base_save_path, "RmbPku-Target"), is_rmb=False)

def seed_everything(seed=42):
    # 1. Standard Python random
    random.seed(seed)
    
    # 2. OS environment (important for internal Python hashing)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy (if used now or in the future)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

if __name__ == "__main__":
    seed_everything(42)
    main()