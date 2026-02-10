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



# =========================
# Constants
# =========================
DEFAULT_SYSTEM_PROMPT = "you are a helpful assistant and will work as an impartial judge."


def split_fixed_eval(df: pd.DataFrame, eval_ratio=0.2, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed)  # 不 reset index：保留原 index
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


# =========================
# Judge
# =========================

class Judge:
    """
    Pairwise judge using Qwen2.5-32B-Instruct + vLLM (or your in-house vLLM engine).

    Each template is treated as a single user prompt (no system/user splitting from template).
    Output format is expected to be [[A]] or [[B]] (case-insensitive).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-32B-Instruct",
        vllm_engine_kwargs: Optional[Dict[str, Any]] = None,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 16,
        engine: Optional[Any] = None,
        max_retries: int = 2,
        retry_backoff_sec: float = 1.0,
        split_on_fail: bool = True,
    ):
        self.model_name = model_name
        self.vllm_engine_kwargs = vllm_engine_kwargs or {}
        self.batch_size = batch_size

        self.sampling_kwargs = sampling_kwargs or {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2048,
        }

        if "gpu_memory_utilization" not in self.vllm_engine_kwargs:
            self.vllm_engine_kwargs["gpu_memory_utilization"] = 0.9
        if "max_model_len" not in self.vllm_engine_kwargs:
            self.vllm_engine_kwargs["max_model_len"] = 16384

        self.engine = engine or VLLMEngine(model=self.model_name, **self.vllm_engine_kwargs)

        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.split_on_fail = split_on_fail

        tok_name = model_name
        self.tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)


    def _replace_last(self, text: str, placeholder: str, value: str) -> str:
        idx = text.rfind(placeholder)
        if idx == -1:
            return text
        return text[:idx] + value + text[idx + len(placeholder):]


    def _build_filled_prompt(
        self,
        template: str,
        question: str,
        answer_a: str,
        answer_b: str,
    ) -> str:
        data = {
            "question": question,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "instruction": question,
            "response_a": answer_a,
            "response_b": answer_b,
        }

        result = template
        for key, value in data.items():
            placeholder = f"{{{key}}}"
            result = self._replace_last(result, placeholder, str(value))

        return result


    def eval(
        self,
        prompt_template: str,
        questions: List[str],
        answers_a: List[str],
        answers_b: List[str],
        params_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Returns:
          - feedbacks: List[str]   (raw model output for debugging)
          - decisions: List[str]   ("A" | "B" | "tie")
        """
        if not (len(questions) == len(answers_a) == len(answers_b)):
            raise ValueError("questions, answers_a, answers_b must have the same length.")

        sampling_kwargs: Dict[str, Any] = dict(self.sampling_kwargs)
        if params_override:
            sampling_kwargs.update(params_override)

        all_feedbacks: List[str] = []
        all_decisions: List[str] = []

        n = len(questions)
        for s in range(0, n, self.batch_size):
            e = min(s + self.batch_size, n)

            batch_q = questions[s:e]
            batch_a = answers_a[s:e]
            batch_b = answers_b[s:e]

            vllm_inputs: List[str] = []
            for q, a, b in zip(batch_q, batch_a, batch_b):
                filled = self._build_filled_prompt(prompt_template, q, a, b)
                vllm_inputs.append(to_chatml(self.tok, filled, system_prompt=DEFAULT_SYSTEM_PROMPT))

            with suppress_vllm_output():
                raw_texts = self.engine.generate_text(
                    vllm_inputs,
                    sampling_kwargs=sampling_kwargs,
                    max_retries=self.max_retries,
                    retry_backoff_sec=self.retry_backoff_sec,
                    split_on_fail=self.split_on_fail,
                )

            for raw in raw_texts:
                verdict = parse_verdict(raw)
                if verdict in ("A", "B"):
                    decision = verdict
                else:
                    decision = fallback_decision_from_text(raw)

                all_decisions.append(decision)
                all_feedbacks.append(raw or "")

        return all_feedbacks, all_decisions


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
    judge: Judge,
    df: pd.DataFrame,
    dataset_name: str,
    output_path: str,
    prompt_template: str,
) -> List[Dict[str, Any]]:
    """
    Evaluate a dataframe with a single prompt template.
    Writes jsonl rows with: dataset, data_index, decision, feedback
    """
    pcol, acol, bcol = detect_cols(df)

    questions = df[pcol].astype(str).tolist()
    a_list = df[acol].astype(str).tolist()
    b_list = df[bcol].astype(str).tolist()
    idx_list = [int(i) for i in df.index.tolist()]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for s in tqdm(range(0, len(questions), judge.batch_size), desc=f"{dataset_name} eval", unit="batch"):
        e = min(len(questions), s + judge.batch_size)

        feedbacks, decisions = judge.eval(
            prompt_template=prompt_template,
            questions=questions[s:e],
            answers_a=a_list[s:e],
            answers_b=b_list[s:e],
        )

        for i, (fb, dc) in enumerate(zip(feedbacks, decisions)):
            global_idx = idx_list[s + i]
            rows.append(
                {
                    "dataset": dataset_name,
                    "data_index": int(global_idx),
                    "decision": str(dc),
                    "feedback": fb,
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return rows



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
    judge: Judge,
    prompt_jsonl_path: str,
    target_eval_data_path: str,
    out_dir: str,
    dataset_name: str,
    sample_size: int = 0,
    summary_filename: str = "1summary.csv",
    debug: bool = False,
) -> None:
    """
    Load templates from jsonl, evaluate each template once, and store results by template id.
    """
    templates = load_prompt_jsonl(prompt_jsonl_path)

    df = load_df(target_eval_data_path)
    if debug:
        df = df.head(4)
    if sample_size and sample_size > 0:
        df = sample_df(df, sample_size)

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, summary_filename)
    

    for i, obj in enumerate(templates):
        template_id = extract_template_id(obj, i)
        template_text = extract_template_text(obj)

        out_path = os.path.join(out_dir, f"{template_id}_eval.jsonl")

        rows = evaluate_df_with_template(
            judge=judge,
            df=df,
            dataset_name=dataset_name,
            output_path=out_path,
            prompt_template=template_text,
        )

        acc = acc_from_rows(df, rows)
        print(f"{template_id}  acc={acc:.3f}")

        file_exists = os.path.isfile(summary_path)
        with open(summary_path, "a", encoding="utf-8", newline="") as fsum:
            writer = csv.writer(fsum)
            if not file_exists:
                writer.writerow(["prompt_id", "acc"])
            writer.writerow([template_id, round(acc, 3)])



