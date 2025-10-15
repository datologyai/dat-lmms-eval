"""
Datology AI2D customizations

Deviations from upstream lmms-eval:
- Prompt: we use an explicit instruction and trailing "Answer:" consistent with our
  historical runs: "Answer with exactly one uppercase letter (A/B/C/D) only...\nAnswer:".
- Scoring: instead of relying solely on regex filters, we parse the final answer letter
  using a robust MMMU-style extractor (prefers last "Answer:" line, handles standalone
  letters, bracketed forms, and content matches), then exact-match against the GT.

The original filter-based behavior remains available via the filter classes below and
is effectively superseded by ai2d_process_results for clearer, single-point scoring.
"""

import re

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter


def ai2d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if lmms_eval_specific_kwargs["prompt_format"] == "mcq":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"
    elif lmms_eval_specific_kwargs["prompt_format"] == "qa":
        options = "\n".join(choices)
        return f"{pre_prompt}{question}{options}{post_prompt}"
    elif lmms_eval_specific_kwargs["prompt_format"] == "mcq_xcomposer":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = " ".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{pre_prompt}{question}\nContext: N/A\n{choices_str}{post_prompt}"
    else:
        raise ValueError(f"Unknown prompt format: {lmms_eval_specific_kwargs['prompt_format']}")


def ai2d_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def ai2d_doc_to_target(doc, model_specific_target_kwargs):
    if model_specific_target_kwargs == "mcq":
        len_choices = len(doc["options"])
        options = [chr(ord("A") + i) for i in range(len_choices)]
        return options[int(doc["answer"])]
    elif model_specific_target_kwargs == "qa":
        return doc["options"][int(doc["answer"])]


def _parse_mc_letter(response: str, answer_choices: list[str]) -> str | None:
    try:
        lines = [ln.strip() for ln in str(response).splitlines() if ln.strip()]
    except Exception:
        lines = []
    if lines:
        pat = re.compile(r"^(?:final\s+answer|answer)\s*:\s*([A-D])\b", re.IGNORECASE)
        for ln in reversed(lines):
            m = pat.match(ln)
            if m:
                return m.group(1).upper()
    # Clean and pad for token-boundary search
    s = " " + (str(response) or "").strip() + " "
    lone = s.strip().upper()
    if lone in {"A","B","C","D"}:
        return lone
    m = re.findall(r"\b([A-D])\b", s, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()
    # bracketed or dotted
    for ch in ["A","B","C","D"]:
        if f"({ch})" in s:
            return ch
    for ch in ["A","B","C","D"]:
        if f"{ch}." in s:
            return ch
    # weak content match against choice texts if provided
    if answer_choices:
        def _norm(t: str) -> str:
            t = (t or "").lower()
            t = re.sub(r"[^a-z0-9\s]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t
        sN = _norm(s)
        best = None; best_score = 0.0
        for idx, choice in enumerate(answer_choices):
            cN = _norm(choice)
            if not cN:
                continue
            sa = set(cN.split()); sb = set(sN.split())
            if not sa:
                continue
            j = len(sa & sb) / float(len(sa))
            if j > best_score:
                best_score = j; best = idx
        if best is not None and best_score >= 0.8 and 0 <= best < len(answer_choices):
            return chr(ord('A') + best)
    return None


def ai2d_process_results(doc, results):
    """Parse a robust MC letter and exact-match against GT index.

    - Prefers last "Answer:" line; handles standalone letters and common patterns.
    - Falls back to fuzzy content match against choice texts.
    """
    pred_raw = results[0] if results else ""
    choices = doc.get("options", [])
    letter = _parse_mc_letter(pred_raw, choices)
    if letter is None:
        return {"exact_match": 0.0}
    pred_idx = ord(letter) - ord('A')
    try:
        gt_idx = int(doc.get("answer", 0))
    except Exception:
        gt_idx = 0
    return {"exact_match": 1.0 if pred_idx == gt_idx else 0.0}


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            # Regex to directly extract the option letter from the model response
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")

            # Process each response
            filtered = []
            for resp in r:
                # Try to match the option letter at the start of the response
                match = option_letter_regex.match(resp)
                if match:
                    # If a match is found, append the matched letter
                    filtered.append(match.group(1))
                else:
                    # If no match, return the original response
                    filtered.append(resp)

            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])

        return filtered_resps
