import json
import os
import re
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mathvista.mathvista_evals import MathVistaEvaluator

# Optional import: use our ERMA-style final answer extractor if available
try:  # pragma: no cover
    from datvlmeval.datasets.evaluation_utils.erma_utils import (
        extract_final_answer as _extract_final_answer,
    )
except Exception:  # pragma: no cover
    _extract_final_answer = None

with open(Path(__file__).parent / "mathvista.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


mathvista_evaluator = MathVistaEvaluator()


def mathvista_doc_to_visual(doc):
    """Return a PIL image for MathVista samples.

    Some dataset variants provide a pre-decoded image under 'decoded_image'; others expose
    an 'image' field. Fallback to 'image' when 'decoded_image' is not present.
    """
    img = doc.get("decoded_image") or doc.get("image")
    try:
        return [img.convert("RGB")] if img is not None else []
    except Exception:
        return []


def mathvista_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    problem = {
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "question": doc["question"],
        "unit": doc["unit"] if "unit" in doc else "",
        "caption": doc["caption"] if "caption" in doc else "",
        "ocr": doc["ocr"] if "ocr" in doc else "",
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "precision": doc["precision"] if "precision" in doc else 0,
    }
    query_prompt = mathvista_evaluator.create_one_query(
        problem,
        shot_num=lmms_eval_specific_kwargs["shot"],
        shot_type=lmms_eval_specific_kwargs["shot_type"],
        use_caption=lmms_eval_specific_kwargs["use_caption"],
        use_ocr=lmms_eval_specific_kwargs["use_ocr"],
    )
    return query_prompt


def mathvista_process_results(doc, results):
    prediction_raw = (results[0] or "").strip()

    # Build problem dict
    question_type = doc.get("question_type", "free_form")
    answer_type = doc.get("answer_type")
    choices = doc.get("choices", []) or []
    all_choices = doc.get("all_choices") or [chr(ord("A") + i) for i in range(len(choices))]
    ground_truth = doc.get("answer")

    # Scoring logic adapted from datvlmeval/datasets/mathvista.py
    parsed_response = None
    is_correct = False

    if question_type == "multi_choice":
        # Build index2ans mapping
        index2ans = {all_choices[i]: choices[i] for i in range(min(len(all_choices), len(choices)))}
        pred_idx = _parse_multi_choice_response(prediction_raw, all_choices, index2ans)
        parsed_response = pred_idx
        is_correct = _eval_multi_choice(ground_truth, pred_idx)
    else:
        parsed_list = _parse_open_response(prediction_raw)
        parsed_response = parsed_list
        is_correct = _eval_open(ground_truth, parsed_list)

    # Prepare result structure consistent with upstream
    result = {
        "question_id": doc.get("pid", doc.get("question_id")),
        "query": doc.get("query", doc.get("question", "")),
        "choices": choices,
        "answer": ground_truth,
        "extraction": parsed_response,
        "prediction": prediction_raw,
        "true_false": bool(is_correct),
        "question_type": question_type,
        "answer_type": answer_type,
        "precision": doc.get("precision", 0),
        "metadata": doc.get("metadata", {}),
    }

    return {
        "llm_as_judge_eval": result,
        "submission": result,
        "exact_match": 1.0 if is_correct else 0.0,
    }


# ------------------------
# Parsing / evaluation utils (ported)
# ------------------------

def _parse_multi_choice_response(response: str, all_choices, index2ans):
    try:
        lines = [ln.strip() for ln in str(response).splitlines() if ln.strip()]
    except Exception:
        lines = []
    if lines:
        pat = re.compile(r"^(?:final\s+answer|answer)\s*:\s*([A-E])\b", re.IGNORECASE)
        for ln in reversed(lines):
            m = pat.match(ln)
            if m:
                return m.group(1).upper()

    # Trim trivial punctuation at ends
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    lone = response.strip().upper()
    if lone in all_choices:
        return lone

    m = re.findall(r"\b([A-E])\b", response, flags=re.IGNORECASE)
    if m:
        for ch in reversed(m):
            chU = ch.upper()
            if chU in all_choices:
                return chU

    index_ans = True
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
    if not candidates:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)
    if not candidates:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)
    if not candidates and len(response.split()) > 5 and index2ans:
        for idx, ans in index2ans.items():
            try:
                if ans and ans.lower() in response.lower():
                    candidates.append(idx)
                    index_ans = False
            except Exception:
                pass

    if not candidates:
        # Fuzzy option text matching
        def _norm(s: str) -> str:
            s = (s or "").lower()
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        respN = _norm(response)
        if index2ans:
            best = None
            best_score = 0.0
            for idx, ans in index2ans.items():
                ansN = _norm(ans)
                if not ansN:
                    continue
                set_a = set(ansN.split())
                set_b = set(respN.split())
                if not set_a:
                    continue
                jacc = len(set_a & set_b) / float(len(set_a))
                if jacc > best_score:
                    best_score = jacc
                    best = idx
            if best is not None and best_score >= 0.8 and best in all_choices:
                return best
        return None
    elif len(candidates) > 1:
        # Choose the candidate with minimum distance to the last indicator occurrence
        start_indexes = []
        for choice in candidates:
            last_index = response.rfind(choice)
            if last_index != -1:
                start_indexes.append((last_index, choice))
        if start_indexes:
            start_indexes.sort(key=lambda x: x[0])
            return start_indexes[-1][1]
        return candidates[-1]
    else:
        return candidates[0]


def _parse_open_response(response: str):
    def get_key_subresponses(resp: str):
        key_responses = []
        indicators = [
            ": ",
            " is ",
            " are ",
            " was ",
            " were ",
            " be ",
            ":",
            "=",
        ]
        shortest_key_response = None
        for indicator in indicators:
            if indicator in resp:
                parts = resp.split("\n")
                for line in reversed(parts):
                    if indicator in line:
                        cand = line.split(indicator)[-1].strip()
                        if not shortest_key_response or (
                            len(cand) < len(shortest_key_response)
                        ):
                            shortest_key_response = cand
                if shortest_key_response:
                    if shortest_key_response.strip() not in [
                        ":",
                        ",",
                        ".",
                        "!",
                        "?",
                        ";",
                        ":",
                        "'",
                    ]:
                        key_responses.append(shortest_key_response)
        if not key_responses:
            return [resp]
        return key_responses

    short = None
    if _extract_final_answer is not None:
        try:
            short = _extract_final_answer(response)
        except Exception:
            short = None

    key_responses = [short] if short else get_key_subresponses(response)
    pred_list = list(key_responses)
    for resp in key_responses:
        pred_list.extend(_extract_numbers(resp))

    tmp_pred_list = []
    for item in pred_list:
        tmp_pred_list.extend(_normalize_str(item))
    pred_list = list(set(tmp_pred_list))
    return pred_list


def _extract_numbers(string: str):
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
    numbers_with_commas = re.findall(pattern_commas, string)
    numbers_scientific = re.findall(pattern_scientific, string)
    numbers_simple = re.findall(pattern_simple, string)
    return numbers_with_commas + numbers_scientific + numbers_simple


def _check_is_number(string: str) -> bool:
    try:
        float(str(string).replace(",", ""))
        return True
    except Exception:
        return False


def _normalize_str(string: str):
    s = str(string).strip()
    if _check_is_number(s):
        s = s.replace(",", "")
        try:
            val = float(s)
            return [round(val, 2)]
        except Exception:
            pass
    # normalize text
    s = s.lower()
    if len(s) == 1:
        return [" " + s, s + " "]
    return [s]


def _eval_multi_choice(gold_i, pred_i) -> bool:
    if isinstance(gold_i, list):
        return any(ans == pred_i for ans in gold_i)
    return gold_i == pred_i


def _eval_open(gold_i, pred_i_list) -> bool:
    if isinstance(gold_i, list):
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(_normalize_str(answer))
    else:
        norm_answers = _normalize_str(gold_i)
    for pred in pred_i_list:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    return True
        else:
            if pred in norm_answers:
                return True
    return False


def mathvista_aggregate_results(results, args, *, calculate_gain=False, random_scores=None):
    split_flag = results[0]["metadata"]["split"]
    full_pids = [result["question_id"] for result in results]
    total = len(results)
    correct = sum(1 for idx, pid in enumerate(full_pids) if results[idx]["true_false"])
    accuracy = round(correct / total * 100, 2)
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    for result in results:
        result.update(result.pop("metadata"))

    results_dict = {result["question_id"]: result for result in results}
    df = pd.DataFrame(results_dict).T
    target_keys = ["question_type", "answer_type", "language", "source", "category", "task", "context", "grade", "skills"]

    for key in target_keys:
        values = df[key].explode().unique() if key == "skills" else df[key].unique()
        scores[key] = {}
        for value in values:
            correct, total, acc = mathvista_evaluator.get_acc_with_contion(df, key, value)
            if total > 0:
                scores[key][value] = {"accuracy": acc, "correct": correct, "total": total}
        scores[key] = dict(sorted(scores[key].items(), key=lambda item: float(item[1]["accuracy"]), reverse=True))

    if calculate_gain:
        for key in scores:
            if key == "average":
                gain = round(float(scores[key]["accuracy"]) - float(random_scores[key]["accuracy"]), 2)
                scores[key]["acc_gain"] = gain
            else:
                for sub_key in scores[key]:
                    gain = round(float(scores[key][sub_key]["accuracy"]) - float(random_scores[key][sub_key]["accuracy"]), 2)
                    scores[key][sub_key]["acc_gain"] = gain

    if scores["average"]["accuracy"] == 0:
        return None
    return scores["average"]["accuracy"]
