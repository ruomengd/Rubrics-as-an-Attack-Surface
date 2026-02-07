import os
import json
import random
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

# Global Tokenizer for length filtering
tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=999999)

def is_valid_length(prompt, res_a, res_b, max_length=1024):
    """Checks if the combined prompt and responses are within token limits."""
    full_text = f"{prompt} {res_a} {res_b}"
    tokens = tokenizer.encode(full_text, add_special_tokens=False, truncation=False)
    return len(tokens) <= max_length

def parse_hh_rlhf_example(example):
    """
    Parses Anthropic HH-RLHF format.
    Ensures safe_content/harm_content only contain the final assistant response.
    """
    def extract_last_assistant_response(text):
        # 1. Split at the last "Assistant: " marker
        parts = text.rsplit("\n\nAssistant: ", 1)
        if len(parts) < 2:
            return "", ""
        
        prompt_history = parts[0].strip()
        last_response = parts[1].strip()
        
        # 2. Safety check: If the response contains a subsequent "Human:" turn, 
        # (which can happen in some multi-turn versions), strip it.
        if "Human:" in last_response:
            last_response = last_response.split("Human:")[0].strip()
            
        return prompt_history, last_response

    # Extract safe content (chosen)
    prompt, safe_content = extract_last_assistant_response(example["chosen"])
    
    # Extract harm content (rejected)
    # Note: We use the prompt derived from the 'chosen' path as the standard 
    # instruction, as the history prefix is identical.
    _, harm_content = extract_last_assistant_response(example["rejected"])

    return {
        "prompt": prompt,
        "safe_content": safe_content,
        "harm_content": harm_content
    }

def prepare_and_save(data_iterable, output_path, is_hh=False):
    """
    Unified function that filters by length and saves datasets.
    Iterates until target counts are met for each split.
    """
    os.makedirs(output_path, exist_ok=True)

    # Target counts for Train, Val, and Test
    splits = {
        "train.jsonl": 1000,
        "val.jsonl": 1000,
        "test.jsonl": 1000
    }

    data_iter = iter(data_iterable)

    print(f"\nProcessing {output_path}...")
    for filename, target_count in splits.items():
        save_file = os.path.join(output_path, filename)
        count = 0
        
        with open(save_file, "w", encoding="utf-8") as f:
            while count < target_count:
                try:
                    row = next(data_iter)
                except StopIteration:
                    print(f" ! Warning: Ran out of data for {filename}")
                    break

                # --- 1. Parsing Logic ---
                if is_hh:
                    parsed = parse_hh_rlhf_example(row)
                    prompt = parsed["prompt"]
                    safe_content = parsed["safe_content"]
                    harm_content = parsed["harm_content"]
                else:
                    # SafeRLHF Logic
                    prompt = row["prompt"]
                    if row["safer_response_id"] == 0:
                        safe_content, harm_content = row["response_0"], row["response_1"]
                    else:
                        safe_content, harm_content = row["response_1"], row["response_0"]

                # --- 2. Length Filtering & Validation ---
                if not (prompt and safe_content and harm_content):
                    continue
                
                if not is_valid_length(prompt, safe_content, harm_content, 1024):
                    continue

                # --- 3. Positional Bias Randomization ---
                if random.random() > 0.5:
                    record = {
                        "instruction": prompt,
                        "response_a": safe_content,
                        "response_b": harm_content,
                        "ground_truth": "A"
                    }
                else:
                    record = {
                        "instruction": prompt,
                        "response_a": harm_content,
                        "response_b": safe_content,
                        "ground_truth": "B"
                    }
                
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                
        print(f" - Saved {count} rows to {save_file}")

def main():
    # Base save path
    base_path = "../../harmlessness/AnthropicPku"

    # 1. Load Datasets
    print("Loading datasets from Hugging Face...")
    safe_rlhf = load_dataset("PKU-Alignment/PKU-SafeRLHF")['train']
    hh_rlhf = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")['train']

    # 2. Shuffle
    safe_rlhf = safe_rlhf.shuffle(seed=42)
    hh_rlhf = hh_rlhf.shuffle(seed=42)

    print(f"SafeRLHF length: {len(safe_rlhf)}")
    print(f"HH-RLHF length: {len(hh_rlhf)}")
    

    # 3. Process and Save
    # Bench: Anthropic HH-RLHF
    prepare_and_save(hh_rlhf, os.path.join(base_path, "AnthropicPku-Bench"), is_hh=True)
    # Target: SafeRLHF
    prepare_and_save(safe_rlhf, os.path.join(base_path, "AnthropicPku-Target"), is_hh=False)
    
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