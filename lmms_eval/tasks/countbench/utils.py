"""
Datology CountBench task

Prompt and scoring aligned with our datvlmeval implementation:
- Prompt: "<image>\n{question}\nAnswer the question using a single word or phrase.\nAnswer:"
- Scoring: extract a single number from model output (handles CoT-style final lines,
  common sentence patterns, digits, and number words) and exact-match to GT.

Notes:
- This adds a concrete per-doc process_results path for lmms-eval integration. The
  original code in datvlmeval also computes dataset-level MAE/WithinK stats; here we
  expose a per-sample exact_match metric compatible with lmms-eval aggregation.
"""

import re
from typing import Dict, List, Optional
from PIL import Image


def countbench_doc_to_visual(doc):
    """Return image as PIL. Supports either an in-memory image or an img_path string."""
    img = doc.get("image")
    if img is not None and hasattr(img, "convert"):
        return [img.convert("RGB")]
    # Fallback to img_path
    path = doc.get("img_path") or doc.get("image_path")
    if isinstance(path, str):
        try:
            return [Image.open(path).convert("RGB")]
        except Exception:
            return []
    return []


def countbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre = (lmms_eval_specific_kwargs or {}).get("pre_prompt", "")
    post = (lmms_eval_specific_kwargs or {}).get("post_prompt", "")
    return f"{pre}{doc['question']}{post}"


_WORD_TO_NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
}


def _extract_final_number(text: str) -> Optional[int]:
    if not isinstance(text, str):
        return None
    t = text.strip()

    # Strategy 1: Prefer final concise line (Answer: ... or similar)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        # Common CoT patterns
        final_patterns = [
            r"^(?:final\s+answer|answer)\s*:\s*(\w+)",
            r"^(?:therefore|thus|hence|so),?\s*(?:the\s+)?(?:number\s+of\s+\w+\s+is\s+|answer\s+is\s+)(\w+)",
            r"^(?:the\s+)?(?:total\s+)?(?:number\s+of\s+\w+\s+is\s+|count\s+is\s+)(\w+)",
            r"^(?:i\s+count\s+|there\s+are\s+)(\w+)",
            r"^(?:the\s+)?answer\s+is\s+(\w+)",
            r"^(?:final\s+answer:?\s*)(\w+)",
            r"^(?:result:?\s*)(\w+)",
        ]
        for ln in reversed(lines):
            for pat in final_patterns:
                m = re.search(pat, ln, flags=re.IGNORECASE)
                if m:
                    tok = m.group(1)
                    if tok.isdigit():
                        return int(tok)
                    lw = tok.lower()
                    if lw in _WORD_TO_NUM:
                        return int(_WORD_TO_NUM[lw])
            # fallback: last number on concise line
            mnum = re.findall(r"\b\d+\b", ln)
            if mnum:
                try:
                    return int(mnum[-1])
                except Exception:
                    pass

    # Strategy 2: Sentence patterns anywhere
    sentence_patterns = [
        r"there\s+are\s+(\w+)\s+\w+\s+in\s+the\s+image",
        r"there\s+are\s+(\w+)\s+\w+",
        r"i\s+(?:can\s+)?see\s+(\w+)\s+\w+",
        r"contains\s+(\w+)\s+\w+",
        r"shows\s+(\w+)\s+\w+",
    ]
    tl = t.lower()
    for pat in sentence_patterns:
        m = re.findall(pat, tl)
        if m:
            tok = m[-1]
            if tok.isdigit():
                return int(tok)
            if tok in _WORD_TO_NUM:
                return int(_WORD_TO_NUM[tok])

    # Strategy 3: Any number digit fallback
    m = re.findall(r"\b\d+\b", t)
    if m:
        try:
            return int(m[-1])
        except Exception:
            pass

    # Strategy 4: Word numbers anywhere
    for word in tl.split():
        w = re.sub(r"[^\w]", "", word)
        if w in _WORD_TO_NUM:
            return int(_WORD_TO_NUM[w])
    return None


def countbench_process_results(doc: Dict, results: List[str]):
    pred = results[0] if results else ""
    gt = doc.get("answer")
    try:
        gt_int = int(gt)
    except Exception:
        gt_int = None
    val = _extract_final_number(pred)
    score = 1.0 if (val is not None and gt_int is not None and int(val) == int(gt_int)) else 0.0
    return {"exact_match": score}
