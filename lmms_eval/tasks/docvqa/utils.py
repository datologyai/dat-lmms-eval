"""
Datology DocVQA customizations

Deviations from upstream lmms-eval:
- Prompt: mirror our default instruction suffix "Answer the question using a single word or phrase." via YAML.
- Scoring (val): add a per-doc process_results that extracts a concise final answer
  (prefers last "Answer:"), then computes ANLS against the provided answers. Uses the
  official `anls` package when available; otherwise a spec-aligned fallback.
- Test split: keep submission behavior unchanged.

Original upstream only provided a test submission process_results; we add a val-time
scorer to match our evaluation flow which calls Task.process_results per doc.
"""

import json
import os
import re

from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def docvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def docvqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def docvqa_process_results(doc, results):
    """Val-time scoring: extract final answer and compute ANLS vs answers.

    Fallback ANLS mirrors standard: lowercase + whitespace normalize, Levenshtein-based,
    zero if distance ratio > 0.5.
    """
    pred_raw = results[0] if results else ""

    # Extract concise final answer (prefer last 'Answer:')
    def _extract_final(text: str) -> str:
        if not isinstance(text, str):
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in reversed(lines):
            m = re.match(r"(?i)^\s*(?:final\s+answer|answer)\s*:\s*(.+)$", ln)
            if m:
                ans = m.group(1).strip()
                return re.sub(r"\s+", " ", ans).strip()
        return re.sub(r"\s+", " ", (text or "")).strip()

    pred_answer = _extract_final(pred_raw)
    answers = doc.get("answers") or []

    # Compute ANLS
    score = 0.0
    try:
        from anls import anls_score

        score = anls_score(prediction=pred_answer, gold_labels=answers, threshold=0.5)
    except ImportError:
        def _norm(s: str) -> str:
            s = (s or "").strip().lower()
            return " ".join(s.split())
        def _lev(a: str, b: str) -> int:
            if len(a) < len(b):
                a, b = b, a
            if len(b) == 0:
                return len(a)
            prev = list(range(len(b) + 1))
            for i, c1 in enumerate(a):
                cur = [i + 1]
                for j, c2 in enumerate(b):
                    ins = prev[j + 1] + 1
                    dele = cur[j] + 1
                    sub = prev[j] + (c1 != c2)
                    cur.append(min(ins, dele, sub))
                prev = cur
            return prev[-1]
        det = _norm(pred_answer)
        for gt in answers:
            gtN = _norm(gt)
            if not gtN and not det:
                score = max(score, 1.0)
                continue
            if not gtN or not det:
                continue
            dist = _lev(det, gtN)
            ratio = dist / max(len(det), len(gtN))
            if ratio <= 0.5:
                score = max(score, 1.0 - ratio)

    return {"anls": score}


def docvqa_test_process_results(doc, results):
    pred = results[0]
    questionId = doc["questionId"]
    return {"anls": {"questionId": int(questionId), "answer": pred}, "submission": {"questionId": int(questionId), "answer": pred}}


def docvqa_test_aggregate_results(results, args):
    # save results as json
    path = generate_submission_file("docvqa_test_for_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    logger.info(f"Results saved to {path}")
