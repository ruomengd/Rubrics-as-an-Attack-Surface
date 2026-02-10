import csv
import glob
import json
import re
import time
import contextlib
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
import asyncio
from openai import AsyncOpenAI


# =========================
# Constants
# =========================
DEFAULT_SYSTEM_PROMPT = "you are a helpful assistant and will work as an impartial judge."


def split_fixed_eval(df: pd.DataFrame, eval_ratio=0.2, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed) 
    n_eval = int(len(df) * eval_ratio)
    eval_df = df.iloc[:n_eval]
    pool_df = df.iloc[n_eval:]
    return pool_df, eval_df

# =========================
# IO helpers
# =========================
@contextlib.contextmanager
def suppress_vllm_output():
    """Suppress stdout/stderr during vLLM generation to keep logs clean."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_prompt_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load prompt templates from a jsonl file.

    Expected fields (best-effort):
      - id / prompt_id / rubric_id / template_id
      - prompt / template / text / content
    """
    out = load_jsonl(path)
    if not out:
        raise ValueError("No prompt templates found in jsonl.")
    return out


# =========================
# Formatting / parsing
# =========================
def safe_format(template: str, **kwargs) -> str:
    """
    Safer formatter:
    - If template contains unknown placeholders, they remain unchanged.
    """
    class SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    return template.format_map(SafeDict(**kwargs))


def to_chatml(tok: AutoTokenizer, user_content: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """
    Build chat-formatted prompt using transformers' apply_chat_template.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


_VERDICT_RE = re.compile(r"\[\[\s*([abAB])\s*\]\]")


def parse_verdict(text: str) -> Optional[str]:
    """
    Parse verdict from model output. Expected format: [[A]] or [[B]] (case-insensitive).
    Returns "A" or "B" if found, else None.
    """
    if not text:
        return None
    m = _VERDICT_RE.search(text)
    if not m:
        return None
    v = m.group(1).upper()
    return "A" if v == "A" else "B"


def fallback_decision_from_text(text: str) -> str:
    """
    Fallback decision parser if [[A]]/[[B]] is missing.
    """
    t = (text or "").strip().lower()
    if "tie" in t:
        return "tie"
    # try to detect standalone A/B
    if re.search(r"\bA\b", text, flags=re.IGNORECASE):
        return "A"
    if re.search(r"\bB\b", text, flags=re.IGNORECASE):
        return "B"
    return "tie"


def extract_template_id(obj: Dict[str, Any], default_idx: int) -> str:
    for k in ("id", "prompt_id", "template_id", "rubric_id", "rubric_index"):
        v = obj.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return str(default_idx)


def extract_template_text(obj: Dict[str, Any]) -> str:
    for k in ("prompt", "template", "text", "content"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # last resort: dump the object
    return json.dumps(obj, ensure_ascii=False)


# =========================
# vLLM Engine with retry / split
# =========================
class VLLMEngine:
    """
    Official vLLM adapter.

    If you have an in-house engine, inject it into Judge as long as it provides:
        generate_text(prompts: List[str], sampling_kwargs: Dict[str, Any], ...) -> List[str]
    """

    def __init__(self, model: str, **engine_kwargs):
        if LLM is None:
            raise ImportError("vllm is not available. Install vllm or inject your in-house engine.")
        self.llm = LLM(model=model, **engine_kwargs)

    def _generate_once(self, prompts: List[str], sampling_kwargs: Dict[str, Any]) -> List[str]:
        if SamplingParams is None:
            raise ImportError("vllm SamplingParams is not available.")
        params = SamplingParams(**sampling_kwargs)
        outputs = self.llm.generate(prompts, params)

        texts: List[str] = []
        for out in outputs:
            if out.outputs and len(out.outputs) > 0:
                texts.append(out.outputs[0].text)
            else:
                texts.append("")

        # Ensure output length matches input length
        if len(texts) != len(prompts):
            if len(texts) < len(prompts):
                texts.extend([""] * (len(prompts) - len(texts)))
            else:
                texts = texts[: len(prompts)]
        return texts

    def generate_text(
        self,
        prompts: List[str],
        sampling_kwargs: Dict[str, Any],
        max_retries: int = 2,
        retry_backoff_sec: float = 1.0,
        split_on_fail: bool = True,
        max_split_depth: int = 4,
    ) -> List[str]:
        """
        Robust generation:
        - Retry on failure.
        - If still failing and split_on_fail=True, split batch into halves recursively.
        - Never raises to caller; returns an error marker string per failed item.
        """

        def _gen_with_retry(batch: List[str], depth: int) -> List[str]:
            last_err: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return self._generate_once(batch, sampling_kwargs)
                except Exception as e:
                    last_err = e
                    # if attempt < max_retries:
                    #     time.sleep(retry_backoff_sec * (attempt + 1))

            # retries failed
            if split_on_fail and len(batch) > 1 and depth < max_split_depth:
                mid = len(batch) // 2
                left = _gen_with_retry(batch[:mid], depth + 1)
                right = _gen_with_retry(batch[mid:], depth + 1)
                return left + right

            err_msg = f"[vLLM generation failed] {type(last_err).__name__}: {last_err}" if last_err else "[vLLM generation failed]"
            return [err_msg for _ in batch]

        return _gen_with_retry(prompts, depth=0)


def fill_last_only(template: str, **kwargs) -> str:
    result = template
    for key, value in kwargs.items():
        placeholder = f"{{{key}}}"
        idx = result.rfind(placeholder)
        if idx != -1:
            result = (
                result[:idx]
                + str(value)
                + result[idx + len(placeholder):]
            )
    return result


class DeepSeekJudge:
    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        concurrency: int = 10,  
        system_prompt: str = "You are a professional judge evaluating model responses."
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)
        self.concurrency = concurrency
        self.system_prompt = system_prompt
        self.batch_size = concurrency 
        
        self.sampling_kwargs = sampling_kwargs or {
            "temperature": 0.0,
            "max_tokens": 1024,
        }

    async def _call_api_async(self, prompt: str, sem: asyncio.Semaphore) -> str:
        async with sem:
            for attempt in range(3):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        **self.sampling_kwargs
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if attempt == 2: return f"Error: {e}"
                    await asyncio.sleep(2 * (attempt + 1))
            return ""

    def eval(
        self,
        prompt_template: str,
        questions: List[str],
        answers_a: List[str],
        answers_b: List[str],
    ) -> Tuple[List[str], List[str]]:
        async def _run():
            sem = asyncio.Semaphore(self.concurrency)
            tasks = []
            for q, a, b in zip(questions, answers_a, answers_b):
                # format the prompt template
                filled = fill_last_only(prompt_template,
                    question=q, answer_a=a, answer_b=b,
                    instruction=q, response_a=a, response_b=b
                )
                tasks.append(self._call_api_async(filled, sem))
            return await asyncio.gather(*tasks)

        raw_outputs = asyncio.run(_run())
        decisions = [self._parse_verdict(raw) for raw in raw_outputs]
        return raw_outputs, decisions

    def _parse_verdict(self, text: str) -> str:
        text = text.upper()
        if "[[A]]" in text: return "A"
        if "[[B]]" in text: return "B"
        match = re.search(r"\[([AB])\]", text)
        if match: return match.group(1)
        return "TIE"

# =========================
# Data utilities
# =========================
def read_parquet_glob(pat: str) -> pd.DataFrame:
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(pat)
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_from_jsonl(path: str) -> pd.DataFrame:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return pd.DataFrame(data)


def load_df(path: str) -> pd.DataFrame:
    if path.endswith(".jsonl") or path.endswith(".json"):
        return load_from_jsonl(path)
    return read_parquet_glob(path)


def sample_df(df: pd.DataFrame, size: int) -> pd.DataFrame:
    if size is None or size <= 0 or size >= len(df):
        return df
    return df.sample(n=size, replace=False, random_state=None)


P_COLS = ["instruction", "question"]
A_COLS = ["response1", "answer_a", "response_a"]
B_COLS = ["response2", "answer_b", "response_b"]


def detect_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    p = next((c for c in P_COLS if c in df.columns), None)
    a = next((c for c in A_COLS if c in df.columns), None)
    b = next((c for c in B_COLS if c in df.columns), None)
    if not (p and a and b):
        raise ValueError(f"Need question + A + B; got {list(df.columns)}")
    return p, a, b


# =========================
# Evaluation loop per template id
# =========================
def evaluate_df_with_template(
    judge: DeepSeekJudge,
    df: pd.DataFrame,
    dataset_name: str,
    output_path: str,
    prompt_template: str,
) -> List[Dict[str, Any]]:
    done_indices = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    done_indices.add(data["data_index"])
                except:
                    continue
    

    full_idx_list = df.index.tolist()
    todo_df = df[~df.index.isin(done_indices)]
    
    if len(todo_df) == 0:
        print(f"Skipping: All rows for {dataset_name} are already processed.")
        return [] 

    print(f"Resuming {dataset_name}: {len(todo_df)} rows remaining (total {len(df)}).")

    pcol, acol, bcol = detect_cols(todo_df)
    questions = todo_df[pcol].astype(str).tolist()
    a_list = todo_df[acol].astype(str).tolist()
    b_list = todo_df[bcol].astype(str).tolist()
    idx_list = todo_df.index.tolist()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_rows: List[Dict[str, Any]] = []

    batch_size = judge.batch_size 
    
    for s in tqdm(range(0, len(questions), batch_size), desc=f"Eval {dataset_name}", leave=False):
        e = min(len(questions), s + batch_size)

        feedbacks, decisions = judge.eval(
            prompt_template=prompt_template,
            questions=questions[s:e],
            answers_a=a_list[s:e],
            answers_b=b_list[s:e],
        )

        with open(output_path, "a", encoding="utf-8") as f:
            for i, (fb, dc) in enumerate(zip(feedbacks, decisions)):
                row = {
                    "dataset": dataset_name,
                    "data_index": int(idx_list[s + i]),
                    "decision": dc,
                    "feedback": fb,
                }
                new_rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return new_rows


def acc_from_rows(df_full: pd.DataFrame, rows: List[Dict[str, Any]]) -> float:
    """
    Accuracy against df_full['ground_truth'] which should be "A" or "B".
    If decision is not A/B (e.g., tie), count as 0.5.
    """
    if not rows:
        return 0.0

    correct = 0.0
    total = 0

    for r in rows:
        idx = r.get("data_index")
        if idx is None:
            continue
        try:
            idx = int(idx)
        except Exception:
            continue
        if idx not in df_full.index:
            continue

        gold = df_full.loc[idx, "ground_truth"]
        total += 1

        decision = r.get("decision")
        if decision not in ("A", "B"):
            correct += 0.5
        elif decision == gold:
            correct += 1.0

    return float(correct / total) if total > 0 else 0.0


def eval_templates(
    judge: DeepSeekJudge,
    prompt_jsonl_path: str,
    target_eval_data_path: str,
    out_dir: str,
    dataset_name: str,
    sample_size: int = 0,
    summary_filename: str = "1summary.csv",
    debug: bool = False,
) -> None:
    templates = load_prompt_jsonl(prompt_jsonl_path)
    df = load_df(target_eval_data_path)
    
    if sample_size and sample_size > 0:
        df = df.head(sample_size)
    if debug:
        df = df.head(4)

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, summary_filename)
    

    for i, obj in enumerate(templates):
        template_id = extract_template_id(obj, i)
        template_text = extract_template_text(obj)

        out_path = os.path.join(out_dir, f"{template_id}_eval.jsonl")

        evaluate_df_with_template(
            judge=judge,
            df=df,
            dataset_name=dataset_name,
            output_path=out_path,
            prompt_template=template_text,
        )

        all_rows_for_template = []
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                all_rows_for_template.append(json.loads(line))
        
        acc = acc_from_rows(df, all_rows_for_template)
        print(f"Template {template_id} finished. Total ACC: {acc:.3f}")

        file_exists = os.path.isfile(summary_path)
        with open(summary_path, "a", encoding="utf-8", newline="") as fsum:
            writer = csv.writer(fsum)
            if not file_exists:
                writer.writerow(["prompt_id", "acc"])
            writer.writerow([template_id, round(acc, 3)])