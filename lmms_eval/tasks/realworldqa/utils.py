"""
Datology RealWorldQA customizations

Deviations from upstream lmms-eval:
- Prompt: we keep the raw multiple-choice question text (which already includes choices)
  and do not append extra instructions by default.
- Scoring: we extract a concise final answer (prefers last "Answer:"), map number words
  to digits, robustly parse a final choice letter from either the final answer line
  or the full output, and finally exact-match against the lowercase GT.

The original minimal logic (lower/strip and exact match) is kept below as a
commented reference and considered deprecated.
"""

import re

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

REPLACE_PROMPT = "Please answer directly with only the letter of the correct option and nothing else."


def realworldqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def realworldqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
        question = question.replace(REPLACE_PROMPT, "")
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


# number_words_to_digits = {
#     "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
#     "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
#     "ten": "10"
# }


def realworldqa_process_results(doc, results):
    pred_raw = results[0] if results else ""
    gt_ans = str(doc.get("answer", "")).lower().strip()

    # 1) Prefer concise final answer line
    def _extract_final(text: str) -> str:
        if not isinstance(text, str):
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in reversed(lines):
            m = re.match(r"(?i)^\s*answer\s*:\s*(.+)$", ln)
            if m:
                return m.group(1).strip()
        # fallback: last short non-empty line
        for ln in reversed(lines):
            if len(ln.split()) <= 10:
                return ln
        return str(text).strip()

    final_pred = _extract_final(pred_raw)

    # 2) Map number words to digits
    num_map = {
        "zero": "0","one": "1","two": "2","three": "3","four": "4",
        "five": "5","six": "6","seven": "7","eight": "8","nine": "9","ten": "10",
    }
    filtered = num_map.get(final_pred.lower().strip(), final_pred)

    # 3) Parse letter from final line or full output; or fuzzy map choice text
    qtext = str(doc.get("question", ""))
    def _parse_choices_from_question(question_text: str):
        choices = {}
        for line in (question_text or "").splitlines():
            m = re.match(r"\s*([A-Ea-e])\s*\.\s*(.+)$", line)
            if m:
                letter = m.group(1).lower(); txt = m.group(2).strip()
                # normalize for fuzzy match
                txtN = re.sub(r"[^a-z0-9\s]", " ", txt.lower())
                txtN = re.sub(r"\s+", " ", txtN).strip()
                choices[letter] = txtN
        return choices

    choices_map = _parse_choices_from_question(qtext)

    def _norm_letter(s: str):
        s = (s or "").strip().upper()
        return s if s in {"A","B","C","D","E"} else None

    # Try explicit letter in concise final answer
    letter = _norm_letter(filtered)
    if not letter:
        # Try extracting from raw output
        lines = [ln.strip() for ln in str(pred_raw).splitlines() if ln.strip()]
        for ln in reversed(lines):
            m = re.match(r"(?i)^\s*answer\s*:\s*(.+)$", ln)
            if m:
                ans = m.group(1).strip()
                m1 = re.search(r"\b([A-Ea-e])\b", ans)
                if m1:
                    letter = m1.group(1).upper(); break
                m2 = re.search(r"(?:^|\s)[\(\[]?([A-Ea-e])[\)\]\.!?,]?(?:\s|$)", ans)
                if m2:
                    letter = m2.group(1).upper(); break
        if not letter:
            # Look for last short line letter
            for ln in reversed(lines):
                if len(ln.split()) <= 10:
                    m1 = re.search(r"\b([A-Ea-e])\b", ln)
                    if m1:
                        letter = m1.group(1).upper(); break
                    m2 = re.search(r"(?:^|\s)[\(\[]?([A-Ea-e])[\)\]\.!?,]?(?:\s|$)", ln)
                    if m2:
                        letter = m2.group(1).upper(); break

    # Fuzzy map choice text if no letter
    if not letter and choices_map:
        def _norm(s: str) -> str:
            s = (s or "").lower()
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s
        predN = _norm(filtered)
        best = None; best_score = 0.0
        for lett, txt in choices_map.items():
            sa = set(txt.split()); sb = set(predN.split())
            if not sa:
                continue
            j = len(sa & sb) / float(len(sa))
            if j > best_score:
                best_score = j; best = lett
        if best is not None and best_score >= 0.8:
            letter = best.upper()

    # 4) Normalize and exact-match
    pred_norm = (letter or filtered or "").lower().strip().rstrip('.')
    score = 1.0 if pred_norm == gt_ans and pred_norm != "" else 0.0
    return {"exact_match": score}

"""
Deprecated upstream logic for reference:

def realworldqa_process_results(doc, results):
    pred = results[0].lower().strip().rstrip('.')
    gt_ans = doc['answer'].lower().strip()
    return { 'exact_match': 1.0 if pred == gt_ans else 0.0 }
"""


class NumberWordsToDigitsFilter(MapFilter):
    def __init__(self) -> None:
        mapping_dict = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        super().__init__(mapping_dict, default_value=None)

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp.lower(), resp) for resp in inst]

        return [filter_set(resp) for resp in resps]


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
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            # Regex to extract multiple choice options from the question
            multiple_choices_regex = re.compile(r"\b([A-Z])\.\s+([^\n]*)")
            matches = multiple_choices_regex.findall(doc["question"])

            # Build regex patterns and mappings for each choice
            for m in matches:
                choice_text = m[1].strip()
                fallback_regexes.append(f"{re.escape(choice_text)}")
                choice_to_alpha[choice_text] = next_alpha

                next_alpha = chr(ord(next_alpha) + 1)

            # Compile regex to match any of the extracted choices
            fallback_regex = re.compile("|".join(fallback_regexes))

            # Process each response
            filtered = []
            for resp in r:
                # Remove any punctuation and extra spaces
                cleaned_resp = re.sub(r"[^\w\s]", "", resp).strip()
                # Try to match cleaned response with the choice text
                match = fallback_regex.search(cleaned_resp)
                if match and match.group() in choice_to_alpha:
                    # Map the matched choice text back to its corresponding letter
                    filtered.append(choice_to_alpha[match.group()])
                else:
                    # If no match, return the cleaned response
                    filtered.append(cleaned_resp)

            filtered_resps.append(filtered[0])

        return filtered_resps
