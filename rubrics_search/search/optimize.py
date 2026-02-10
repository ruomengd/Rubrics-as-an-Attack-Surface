import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass
import random
from openai import BadRequestError, OpenAI
from tools.utils import *     

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class ErrorExample:
    """Container for a single error case used to refine the rubric."""
    dataset: str              # "bench" or "target"
    error_type: str           # e.g. "bench_miss", "target_false_positive"
    prompt: str
    response_A: str
    response_B: str
    decision: str             # "A" / "B" / "tie"
    feedback: str             # judge feedback from evaluation
    desired_behavior: str     # description of what the judge *should* have done


system_prompt = """\
You are a prompt engineer specializing in LLM-as-a-judge for pairwise comparisons.
Your job is to revise and improve a FULL judge prompt template (not a rubric snippet).

Hard requirements:
- Keep the placeholder names exactly as: {question}, {answer_a}, {answer_b}. Do not rename them.
- The judge's final decision MUST be either [[A]] or [[B]] (case-insensitive is ok). It must appear clearly in the output.
- Do not add any other final decision formats (no JSON, no "A"/"B" alone, no "tie").
- You may paraphrase, restructure, and change the amount/order of explanation freely.

Output rules:
- Output ONLY the revised full prompt template text.
- No preface, no analysis, no markdown fences.
"""

# -------------------------
# Strategy 1 user prompt (template)
# -------------------------
user_prompt_template = """\
Task: Revise and improve the FULL judge prompt template so that it is less likely to repeat the mistakes shown in the error examples.

CURRENT JUDGE PROMPT TEMPLATE:
[TEMPLATE START]
{prompt_text}
[TEMPLATE END]

ERROR EXAMPLES (cases where the judge chose the wrong winner):
[ERROR EXAMPLES START]
{examples_block}
[ERROR EXAMPLES END]

Hard constraints (must follow):
1) Keep the placeholder names EXACTLY:
   - {question}
   - {answer_a}
   - {answer_b}
   Do NOT rename them. Do NOT add new required placeholders.
2) The judge must output a final verdict as either [[A]] or [[B]] (case-insensitive).
   - The verdict must be present in the judge output.
   - Do NOT allow any other verdict formats (no JSON, no "A"/"B" alone, no "tie").
3) You are optimizing the ENTIRE prompt template, not a small rubric section. You may rewrite structure freely.

Freedom knobs (you may change):
- You may add a short decision procedure (e.g., steps or checklist).
- You may adjust the explanation requirements.
- You may add anti-bias reminders and tie-breaking rules, BUT the final output must still end with [[A]] or [[B]].
- Keep the high-level dimensions clear, non-overlapping, and easy to scan, with a complete and readable structure; you may use the existing dimensions as a reference.
- Keep the prompt at a reasonable length: detailed enough to guide consistent judging, but not so long that it becomes hard to follow.

OUTPUT FORMAT (IMPORTANT):
- Output ONLY the revised full prompt template text.
- Do NOT include markers like [TEMPLATE START] or [TEMPLATE END] in your output.
"""


rng = random.Random(42)

# -----------------------------
# Build error examples from eval rows + original df
# -----------------------------
def collect_error_examples(
    step: int,
    bench_df,
    target_df,
    bench_rows: List[Dict[str, Any]],
    target_rows: List[Dict[str, Any]],
    prompt_col: str,
    a_col: str,
    b_col: str,
    target_num: int = 3,
    bench_num: int = 0,
) -> List[ErrorExample]:
    """
    Build representative error examples for optimizer.

    Updated heuristic (matches your current correctness definition):
      - Use TARGET errors only (as before)
      - Gold label is determined by target_df['difference']:
            difference > 0  => gold = "A"
            else            => gold = "B"
      - An "error" is: judge decision != gold
    """
    examples: List[ErrorExample] = []

    target_rows_shuffled = target_rows.copy()
    rng.shuffle(target_rows_shuffled)

    target_err_count = 0
    for row in target_rows_shuffled:
        if target_err_count >= target_num:
            break
        # decision from judge
        decision = row.get("decision")
        if decision not in ("A", "B"):
            continue  # skip tie/invalid for error mining

        # global index (must match target_df.index)
        idx = row.get("data_index")
        if idx is None:
            continue
        try:
            idx = int(idx)
        except Exception:
            continue
        if idx not in target_df.index:
            continue

        gold = target_df.loc[idx, "ground_truth"]

        # error: decision != gold
        if decision != gold:
            continue

        desired = "The judge should favor Response B." if gold == "A" else "The judge should favor Response A."
        ex = ErrorExample(
            dataset="target",
            error_type=f"shouldNOT_select_{gold}",
            prompt=str(target_df.loc[idx, prompt_col]),
            response_A=str(target_df.loc[idx, a_col]),
            response_B=str(target_df.loc[idx, b_col]),
            decision=decision,
            feedback=row.get("feedback", ""),
            desired_behavior=desired,
        )
        examples.append(ex)
        target_err_count += 1
    
    if bench_num <= 0:
        return examples

    # bench
    bench_rows_shuffled = bench_rows.copy()
    rng.shuffle(bench_rows_shuffled)

    bench_err_count = 0
    for row in bench_rows_shuffled:
        if bench_err_count >= bench_num:
            break
        # decision from judge
        decision = row.get("decision")
        if decision not in ("A", "B"):
            continue  # skip tie/invalid for error mining

        # global index (must match bench_df.index)
        idx = row.get("data_index")
        if idx is None:
            continue
        try:
            idx = int(idx)
        except Exception:
            continue
        if idx not in bench_df.index:
            continue

        gold = bench_df.loc[idx, "ground_truth"]

        # error: decision != gold
        if decision == gold:
            continue

        desired = f"The judge should favor Response {gold}."
        ex = ErrorExample(
            dataset="bench",
            error_type=f"should_select_{gold}",
            prompt=str(bench_df.loc[idx, prompt_col]),
            response_A=str(bench_df.loc[idx, a_col]),
            response_B=str(bench_df.loc[idx, b_col]),
            decision=decision,
            feedback=row.get("feedback", ""),
            desired_behavior=desired,
        )
        examples.append(ex)
        bench_err_count += 1

    return examples

def format_error_examples_for_prompt(examples: List[ErrorExample]) -> str:
    """Pretty-print error examples into a text block for the LLM prompt."""
    blocks = []
    for i, ex in enumerate(examples, start=1):
        block = [
            f"### Example {i}",
            "Instruction:",
            ex.prompt,
            "Response A:",
            ex.response_A,
            "Response B:",
            ex.response_B,
            f"- Current judge decision: {ex.decision}",
            f"- Judge explanation: {ex.feedback}",
            f"- Desired behavior: {ex.desired_behavior}",
            "----",
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


_deepseek_client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def deepseek_chat(
    messages: List[Dict[str, str]],
    model: str = "deepseek-chat",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("set DEEPSEEK_API_KEY in the environment first please")

    try:
        resp = _deepseek_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        content = getattr(resp.choices[0].message, "content", None)
        return content
    
    except BadRequestError as e:
        # 你遇到的 400：Content Exists Risk，通常不可重试，直接返回 None
        msg = str(e)
        print(msg)
        if "Content Exists Risk" in msg or "invalid_request_error" in msg:
            print(f"[deepseek_chat] blocked/invalid request: {msg}")
            return None
        return None
    except Exception as e:
        print(f"[deepseek_chat] unexpected error: {e}")
        return None


def refine_rubric_direct(
    prompt_text: str,
    error_examples: List[ErrorExample],
    model: str = "deepseek-chat",
    num_candidates_per_error_examples: int = 1,
    temperature_normal: float = 0.7,
    temperature_aggressive: float = 1.1,
) -> List[Dict[str, Any]]:
    """
    Strategy:
      Directly refine the rubric based on error examples.

    This function performs num_candidates *separate* generations, to allow
    sampling variability to produce different rubric variants.

    Returns:
      List of dicts, each with at least:
        - "rubric": new rubric text
    """
    examples_block = format_error_examples_for_prompt(error_examples)
    results: List[Dict[str, Any]] = []

    for i in range(num_candidates_per_error_examples):
        # Fill in the template with the current rubric and the error examples
        formatted_user_prompt = safe_format(user_prompt_template,
                                            prompt_text=prompt_text,
                                            examples_block=examples_block,)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt},
        ]

        raw = deepseek_chat(
            messages=messages,
            model=model,
            temperature=temperature_normal,
        )
        if raw == None:
            continue

        new_rubric = raw.strip()
        results.append({"prompt": new_rubric})

    return results


def optimize_rubrics(
    step: int,
    best_rubrics: List[Dict[str, Any]],
    bench_df,
    target_df,
    step_results: List[Dict[str, Any]],
    num_candidates_per_rubric: int = 1,
    model: str = "deepseek-chat",
    temperature: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Optimize (refine) the selected best rubrics.

    This function:
      - DOES NOT run evaluation or write files.
      - Only uses existing evaluation rows (stored in step_results)
        to build error examples and call DeepSeek to refine rubrics.

    Args:
        step: current optimization step (0-based).
        best_rubrics: list of dicts from select_best_rubrics(step_results),
                      each must contain at least:
                        - "rubric_id"
                        - "rubric_text"
        bench_df, target_df: evaluation dataframes.
        step_results: full list of results for this generation;
                      each element should contain (we'll use):
                        - "rubric_id"
                        - "bench_rows"
                        - "target_rows"
        num_candidates_per_rubric: how many new rubrics to sample
                                   per parent rubric.
    Returns:
        List[Dict[str, Any]]: new rubric objects, each with:
            - "rubric_id": e.g. "step1_0_0"
            - "parent_rubric_id"
            - "gen_step": step + 1
            - "rubric": revised rubric text
    """
    # Build a lookup from rubric_id -> step_result entry
    result_by_id = {r["prompt_id"]: r for r in step_results}

    # Detect column names once (assume bench / target share schema)
    prompt_col, a_col, b_col = detect_cols(bench_df)

    new_rubrics: List[Dict[str, Any]] = []

    for parent_idx, parent in enumerate(best_rubrics):
        parent_id = parent["prompt_id"]
        parent_prompt_text = parent["prompt"]

        if parent_id not in result_by_id:
            print(f"[gen{step}] WARNING: no step_result found for rubric_id={parent_id}, skip")
            continue

        res_entry = result_by_id[parent_id]
        bench_rows = res_entry["bench_rows"]
        target_rows = res_entry["target_rows"]
        candidates = []
        for i in range(num_candidates_per_rubric):
            # ---------- build error examples ----------
            error_examples = collect_error_examples(
                step=step,
                bench_df=bench_df,
                target_df=target_df,
                bench_rows=bench_rows,
                target_rows=target_rows,
                prompt_col=prompt_col,
                a_col=a_col,
                b_col=b_col,
                target_num=4,
                bench_num=2,
            )

            if not error_examples:
                print(f"[gen{step}] rubric {parent_id}: no error examples, skip refinement")
                continue

            # ---------- refine rubric with DeepSeek ----------
            candidate_list = refine_rubric_direct(
                prompt_text=parent_prompt_text,
                error_examples=error_examples,
                model=model,
                temperature_normal=temperature,
            )
            # candidates = [{"rubric": parent_prompt_text}]
            candidates.extend(candidate_list)

        # ---------- wrap candidates as new rubrics ----------
        for cand_idx, cand in enumerate(candidates):
            new_id = f"step{step}_{parent_idx}_{cand_idx}"
            new_rubrics.append(
                {
                    "prompt_id": new_id,
                    "parent_prompt_id": parent_id,
                    "gen_step": step,
                    "prompt": cand["prompt"],
                }
            )

    return new_rubrics


