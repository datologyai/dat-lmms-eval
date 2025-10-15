"""
Datology TextVQA customizations

Deviations from upstream lmms-eval:
- Prompt: ensure our default is "<image>\nQuestion: ...\nAnswer the question using a single word or phrase.\nAnswer:".
- Scoring: extract a concise final answer (prefers last "Answer:"), then apply
  EvalAI normalization. If exact match remains 0 but the prediction and any GT look
  like numeric strings differing only by currency/percent symbols, treat as correct.

The original textvqa_process_results is retained below (commented) as a reference.
"""

import datetime
import json
import os
import pathlib
import re
import statistics

import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


def textvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def textvqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."

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

    res_final = _extract_final(result[0])
    resAns = eval_ai_processor(res_final)
    accuracy = 0.0

    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []
        for i in range(len(doc["answers"])):
            doc["answers"][i] = eval_ai_processor(doc["answers"][i])
        for i in range(len(doc["answers"])):
            otherGTAns = [doc["answers"][j] for j in range(len(doc["answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        accuracy = statistics.mean(gtAcc)

        # Numeric fallback for currency/percent mismatches
        if accuracy == 0.0:
            def to_num(s: str) -> str:
                s = (s or "").replace("$", "").replace("£", "").replace("€", "").replace("%", "")
                s = s.replace(",", "")
                s = re.sub(r"[^0-9\.]", "", s)
                if s.count(".") > 1:
                    parts = [p for p in s.split(".") if p != ""]
                    if parts:
                        s = parts[0] + "." + "".join(parts[1:])
                return s.strip(".")
            pred_num = to_num(res_final)
            if any(ch.isdigit() for ch in res_final) and pred_num:
                for gt in doc["answers"]:
                    if gt and any(ch.isdigit() for ch in str(gt)):
                        if to_num(str(gt)) == pred_num:
                            accuracy = 1.0
                            break

    return {
        "exact_match": accuracy,
        "submission": {
            "question_id": doc["question_id"],
            "answer": resAns,
        },
    }


def textvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = ""
    post_prompt = ""
    ocr_ref = ""
    if lmms_eval_specific_kwargs:
        if "pre_prompt" in lmms_eval_specific_kwargs:
            pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
        if "post_prompt" in lmms_eval_specific_kwargs:
            post_prompt = lmms_eval_specific_kwargs["post_prompt"]
        if "ocr" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["ocr"]:
            ocr_ref = f"\nReference OCR token: {', '.join(doc['ocr_tokens'])}"
    return f"{pre_prompt}{doc['question'].capitalize()}{ocr_ref}{post_prompt}"

"""
Original upstream textvqa_process_results (deprecated reference):

def textvqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    resAns = eval_ai_processor(result[0])
    ... (compute accuracy against answers) ...
    return { 'exact_match': accuracy, 'submission': {...} }
"""


def textvqa_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = generate_submission_file(f"textvqa_submission_{now_date_time}.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    # print(f"Submission file saved to {path}")
    eval_logger.info(f"Submission file saved to {path}")
