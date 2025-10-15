"""
Datology ChartQA customizations

Deviations from upstream lmms-eval:
- Prompt: we keep our historical format with a leading image token and explicit
  instruction + trailing "Answer:" line.
- Scoring: extract concise final answer, add pragmatic normalizations (trailing
  periods/currency symbols/decimal commas), align list/percent/text to GT when
  equivalent, then apply relaxed correctness (5% tolerance, list order-insensitive,
  normalized text). This is slightly more permissive on edge cases than vanilla.

The original upstream relaxed correctness is preserved in structure; older
Datology wrapper (datology_utils) is now inlined here for a single entry point.
"""

# lmms-eval original (deprecated): doc_to_visual kept
def chartqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


# Datology: reproduce our previous prompt formatting using lmms-eval abstractions.
# We still accept lmms_eval_specific_kwargs (pre_prompt/post_prompt) so YAML can
# control the exact strings. Defaults should be set in the YAML to match our
# prior style ("<image>\n" and "Answer the question ...\nAnswer:").
def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc.get("question", "")
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{question}{post_prompt}"


# Datology: port our relaxed correctness logic and final-answer extraction
# directly into lmms-eval scoring. This now inlines previously separate
# datology_utils normalizations so a single entry point (this file) contains
# the full scoring behavior.

import re


def _extract_final_answer(text: str) -> str:
    if not isinstance(text, str):
        return text
    t = text.strip()
    # Prefer the last occurrence of 'Answer:' (case-insensitive)
    m_iter = list(re.finditer(r"(?im)^\s*answer\s*:\s*(.+)$", t))
    if m_iter:
        ans = m_iter[-1].group(1).strip()
        return ans.splitlines()[0].strip()
    # Fallback: last non-empty line if reasonably short (<= 10 tokens)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if len(ln.split()) <= 10:
            return ln
    return text


def _normalize_answer_text(answer: str) -> str:
    if not answer:
        return ""
    s = answer.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# --- Datology additional normalizations (migrated from datology_utils) ---

def _strip_trailing_periods(s: str) -> str:
    return s.rstrip().rstrip('.')


def _remove_currency_symbols(s: str) -> str:
    return re.sub(r"^[\$€£]\s*", "", s.strip())


def _convert_decimal_commas(s: str) -> str:
    """Convert decimal commas to dots in patterns like 6,6 or 10,5.
    Avoid touching thousand separators like 24,688.3.
    """
    def repl(m):
        left, right = m.group(1), m.group(2)
        if len(right) == 3:
            return m.group(0)
        return f"{left}.{right}"

    return re.sub(r"(\d),(\d)\b", repl, s)


def _strip_commas(s: str) -> str:
    return s.replace(",", "")


def _parse_float_maybe(s: str):
    try:
        return float(s)
    except Exception:
        return None


def _normalize_ws_lower_no_punct(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _align_bracketed_list_to_gt(pred: str, gt: str) -> str:
    ps = pred.strip(); gs = gt.strip()
    if not (len(ps) >= 2 and ps[0] == "[" and ps[-1] == "]" and len(gs) >= 2 and gs[0] == "[" and gs[-1] == "]"):
        return pred
    def norm_items(s: str) -> set[str]:
        inner = s[1:-1]
        items = [it.strip() for it in inner.split(",") if it.strip()]
        out = []
        for it in items:
            out.append(_normalize_ws_lower_no_punct(it))
        return set(out)
    if norm_items(ps) == norm_items(gs):
        return gs
    return pred


def _align_unbracketed_list_to_gt(pred: str, gt: str) -> str:
    """If GT is a bracketed list and pred is an unbracketed comma/space list,
    compare normalized item sets; if equal, return GT string.
    """
    gs = gt.strip()
    ps = pred.strip()
    if not (len(gs) >= 2 and gs[0] == "[" and gs[-1] == "]"):
        return pred
    # Extract GT items
    inner = gs[1:-1]
    gt_items = [it.strip() for it in inner.split(",") if it.strip()]
    gt_norm = set(_normalize_ws_lower_no_punct(x) for x in gt_items)
    # Extract pred items (split on commas)
    if "," in ps:
        p_items = [it.strip().rstrip('.') for it in ps.split(",") if it.strip()]
    else:
        # split on multiple spaces if no commas
        p_items = [x for x in re.split(r"\s{2,}", ps) if x]
        if len(p_items) <= 1:
            return pred
    p_norm = set(_normalize_ws_lower_no_punct(x) for x in p_items)
    if p_norm == gt_norm:
        return gs
    return pred


def _harmonize_percent_ratio(pred: str, gt: str) -> str:
    # Convert pred units to match GT units when possible
    p = pred.strip(); g = gt.strip()
    has_pct_gt = "%" in g
    has_pct_pred = "%" in p
    # Strip commas for numeric parsing
    pnum_s = _strip_commas(p).rstrip("%")
    gnum_s = _strip_commas(g).rstrip("%")
    if has_pct_gt:
        if not has_pct_pred:
            val = _parse_float_maybe(pnum_s)
            if val is not None and 0 <= val <= 1.5:
                s = ("%f" % (val * 100.0)).rstrip("0").rstrip(".")
                return f"{s}%"
            return f"{pnum_s}%"
        return p
    else:
        if has_pct_pred:
            try:
                pval = float(_strip_commas(pred).rstrip("%"))
            except Exception:
                return p
            gval = _parse_float_maybe(gnum_s)
            # consider raw, ratio, and absolute variants to match magnitude
            cand_raw = pval
            cand_ratio = pval / 100.0
            cand_raw_abs = abs(pval)
            cand_ratio_abs = abs(pval) / 100.0
            if gval is not None:
                cands = [cand_raw, cand_ratio, cand_raw_abs, cand_ratio_abs]
                best = min(cands, key=lambda x: abs(x - gval))
                return ("%f" % best).rstrip("0").rstrip(".")
            return ("%f" % cand_raw_abs).rstrip("0").rstrip(".")
        return p


def _align_text_to_gt_if_equivalent(pred: str, gt: str) -> str:
    # For non-numeric, non-list short text answers, if normalized forms match,
    # return GT string so exact string equality passes.
    p = pred.strip(); g = gt.strip()
    if any(ch.isdigit() for ch in p) or any(ch.isdigit() for ch in g):
        return pred
    if (len(p) >= 2 and p[0] == "[" and p[-1] == "]") or (len(g) >= 2 and g[0] == "[" and g[-1] == "]"):
        return pred
    if _normalize_ws_lower_no_punct(p) == _normalize_ws_lower_no_punct(g):
        return g
    return pred


def _relaxed_correctness_datology(prediction: str, target: str, max_relative_change: float = 0.05) -> bool:
    # Numeric pathway with percent + thousands normalization
    def strip_commas(s: str) -> str:
        return s.replace(",", "") if isinstance(s, str) else s

    def to_float_and_flags(text: str):
        try:
            t = (text or "").strip()
            t = t.replace("$", "")  # remove currency symbol if any
            # convert decimal commas (6,6 -> 6.6) for single-digit comma single-digit
            t = re.sub(r"(\d),(\d)\b", lambda m: f"{m.group(1)}.{m.group(2)}" if len(m.group(2)) != 3 else m.group(0), t)
            t_no_commas = strip_commas(t)
            is_percent = t_no_commas.endswith("%")
            core = t_no_commas.rstrip("%") if is_percent else t_no_commas
            val = float(core)
            return val, is_percent
        except Exception:
            return None, False

    pv, p_is_pct = to_float_and_flags(prediction)
    gv, g_is_pct = to_float_and_flags(target)

    if pv is not None and gv is not None:
        pv_eff = pv
        gv_eff = gv
        if p_is_pct != g_is_pct:
            # align ratio/percent when plausible
            if (not p_is_pct) and gv >= 2 and 0 <= pv <= 1.5:
                pv_eff = pv * 100.0
            if (not g_is_pct) and pv >= 2 and 0 <= gv <= 1.5:
                gv_eff = gv * 100.0
        # 5% tolerance or near-zero epsilon
        if gv_eff != 0:
            rel = abs(pv_eff - gv_eff) / abs(gv_eff)
            if rel <= max_relative_change:
                return True
            if abs(gv_eff) < 1e-6 and abs(pv_eff - gv_eff) <= 1e-6:
                return True
        else:
            if abs(pv_eff - gv_eff) <= 1e-6:
                return True

    # List (order-insensitive) if both bracketed
    def parse_list(ans: str):
        if not isinstance(ans, str):
            return []
        s = ans.strip()
        if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
            inner = s[1:-1]
            items = [it.strip() for it in inner.split(",") if it.strip()]
            norm_items = []
            for it in items:
                itn = _normalize_answer_text(it)
                if itn:
                    norm_items.append(itn)
            return sorted(norm_items)
        return []

    p_list = parse_list(prediction)
    g_list = parse_list(target)
    if p_list and g_list:
        return p_list == g_list

    # Non-numeric text equality with normalization
    return _normalize_answer_text(prediction) == _normalize_answer_text(target)


def chartqa_process_results(doc, results):
    # Datology implementation: extract final answer and apply extra normalizations
    # (migrated from datology_utils), then apply relaxed correctness.
    pred_raw = results[0] if results else ""
    pred = _extract_final_answer(pred_raw if isinstance(pred_raw, str) else str(pred_raw))
    gt = doc.get("answer", "")

    # Additional pre-normalizations formerly in datology_utils
    pred = _strip_trailing_periods(pred)
    pred = _convert_decimal_commas(pred)
    pred = _remove_currency_symbols(pred)
    pred = _align_unbracketed_list_to_gt(pred, str(gt))
    pred = _align_bracketed_list_to_gt(pred, str(gt))
    pred = _harmonize_percent_ratio(pred, str(gt))
    pred = _align_text_to_gt_if_equivalent(pred, str(gt))

    score = 1.0 if _relaxed_correctness_datology(pred, gt) else 0.0
    type_ = doc.get("type", "")
    return_dict = {"relaxed_overall": score}
    if type_ == "human_test":
        return_dict["relaxed_human_split"] = score
    else:
        return_dict["relaxed_augmented_split"] = score
    return return_dict

"""
Deprecated scoring (pre-migration):

The previous version of this module implemented a simpler chartqa_process_results
that extracted the final answer and applied relaxed correctness without the
additional pre-normalizations (strip trailing periods, decimal commas, currency
symbols, and list/text alignment). That implementation is now deprecated in favor
of the unified logic above, which inlines the formerly separate datology_utils
behavior for improved robustness while keeping the same scoring philosophy.
"""
