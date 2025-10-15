"""
Datology Pixmo Points Eval task

Prompt and scoring aligned with our datvlmeval implementation:
- Prompt: "<image>\n{question}" (no explicit Answer: suffix).
- Scoring: parse predicted (x, y) coordinates from response, assign predictions to
  ground-truth points, and compute precision/recall/F1 using segmentation masks.

Implementation details:
- Coordinates can be in [0,1] or [0,100] ranges; we normalize to [0,100].
- Assignment: try SciPy's linear_sum_assignment; if unavailable, fall back to a
  greedy nearest-neighbor assignment.
- Per-sample metrics returned:
  - precision, recall, f1
  - no_target_correct: 1.0 when GT has no target and model says so; else 0.0
  - parsing_failure: 1.0 when GT has target but no points parsed; else 0.0

These keys aggregate cleanly via mean in lmms-eval.
"""

import re
from typing import Dict, List, Tuple
from PIL import Image
import os
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover
    linear_sum_assignment = None  # type: ignore


def _open_image_with_fallbacks(path_str: str):
    """Try opening an image path with fallbacks for mismatched dataset roots.

    Some local JSONLs were authored with a prefix like
    /fsx/data/common/vlm-evaluation/datasets/download/... while images actually
    live under /fsx/data/common/vlm-evaluation/download/.... This helper tries a
    few safe rewrites before giving up.
    """
    candidates = [path_str]

    # If a custom datasets root is provided (e.g., HF cache mount), prefer it
    # by stitching the path segment after '/vlm-evaluation/'.
    env_root = os.environ.get("VLM_EVAL_DATA_ROOT_DIR")
    if env_root and "/vlm-evaluation/" in path_str:
        try:
            tail = path_str.split("/vlm-evaluation/", 1)[1]
            candidates.append(str(Path(env_root) / tail))
        except Exception:
            pass

    # Replace '/datasets/download/' with '/download/'
    if "/vlm-evaluation/datasets/download/" in path_str:
        candidates.append(path_str.replace(
            "/vlm-evaluation/datasets/download/", "/vlm-evaluation/download/"
        ))

    # Replace '/vlm-evaluation/datasets/' with '/vlm-evaluation/'
    if "/vlm-evaluation/datasets/" in path_str:
        candidates.append(path_str.replace(
            "/vlm-evaluation/datasets/", "/vlm-evaluation/"
        ))

    for cand in candidates:
        try:
            if os.path.exists(cand):
                return Image.open(cand).convert("RGB")
        except Exception:
            continue
    # Final attempt: if string is a valid Path pointing to a file
    try:
        p = Path(path_str)
        if p.exists():
            return Image.open(p).convert("RGB")
    except Exception:
        pass
    return None


def pixmo_points_eval_doc_to_visual(doc):
    """Return image as PIL. Supports either in-memory image or img_path string."""
    img = doc.get("image")
    if img is not None and hasattr(img, "convert"):
        return [img.convert("RGB")]
    path = doc.get("img_path") or doc.get("image_path")
    if isinstance(path, str):
        img_obj = _open_image_with_fallbacks(path)
        if img_obj is not None:
            return [img_obj]
        return []
    return []


def pixmo_points_eval_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre = (lmms_eval_specific_kwargs or {}).get("pre_prompt", "")
    post = (lmms_eval_specific_kwargs or {}).get("post_prompt", "")
    return f"{pre}{doc['question']}{post}"


def _parse_coordinates(text: str) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    if not isinstance(text, str):
        return coords
    s = text.strip()

    def _add(x: float, y: float):
        # normalize to [0, 100]
        if 0 <= x <= 1 and 0 <= y <= 1:
            coords.append((x * 100.0, y * 100.0))
        elif 0 <= x <= 100 and 0 <= y <= 100:
            coords.append((x, y))

    # (x, y)
    for m in re.findall(r"\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)", s):
        try:
            _add(float(m[0]), float(m[1]))
        except Exception:
            pass
    if coords:
        return coords
    # [x, y]
    for m in re.findall(r"\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]", s):
        try:
            _add(float(m[0]), float(m[1]))
        except Exception:
            pass
    if coords:
        return coords
    # x, y
    for m in re.findall(r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)", s):
        try:
            _add(float(m[0]), float(m[1]))
        except Exception:
            pass
    return coords


def _normalize_gt_points(gt_points: List) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for pt in (gt_points or []):
        if isinstance(pt, dict) and "x" in pt and "y" in pt:
            try:
                out.append((float(pt["x"]), float(pt["y"])))
            except Exception:
                pass
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                out.append((float(pt[0]), float(pt[1])))
            except Exception:
                pass
    return out


def _distance_matrix(pred: List[Tuple[float, float]], gt: List[Tuple[float, float]]):
    if not pred or not gt:
        return np.zeros((0, 0), dtype=float)
    a = np.array(pred, dtype=float)
    b = np.array(gt, dtype=float)
    d = np.zeros((len(a), len(b)), dtype=float)
    for i, pa in enumerate(a):
        for j, pb in enumerate(b):
            d[i, j] = float(np.sqrt(np.sum((pa - pb) ** 2)))
    return d


def _assign(pred: List[Tuple[float, float]], gt: List[Tuple[float, float]]):
    if not pred or not gt:
        return []
    D = _distance_matrix(pred, gt)
    if D.size == 0:
        return []
    if linear_sum_assignment is not None:
        pi, gi = linear_sum_assignment(D)
        return list(zip(pi, gi))
    # Greedy fallback
    taken_g = set()
    pairs = []
    for i, row in enumerate(D):
        j = int(np.argmin(row))
        if j not in taken_g:
            pairs.append((i, j))
            taken_g.add(j)
    return pairs


def _point_in_mask(point: Tuple[float, float], mask) -> bool:
    if not mask or not mask[0]:
        return False
    h = len(mask)
    w = len(mask[0])
    xm = int((point[0] / 100.0) * w)
    ym = int((point[1] / 100.0) * h)
    if ym < 0 or ym >= h or xm < 0 or xm >= w:
        return False
    return bool(mask[ym][xm])


def pixmo_points_eval_process_results(doc: Dict, results: List[str]):
    pred_raw = results[0] if results else ""
    gt_points = _normalize_gt_points(doc.get("points", []))
    masks = doc.get("masks", [])
    has_target = bool(doc.get("has_target", bool(gt_points)))

    # Parse predictions
    pred_points = _parse_coordinates(pred_raw)

    precision = 0.0
    recall = 0.0
    f1 = 0.0
    no_target_correct = 0.0
    parsing_failure = 0.0

    if not has_target:
        # treat explicit negatives as correct
        neg_phrases = [
            "no", "none", "not present", "not in", "not visible", "not found", "cannot find", "can't find"
        ]
        tl = (pred_raw or "").lower()
        if any(p in tl for p in neg_phrases):
            no_target_correct = 1.0
    elif gt_points and pred_points:
        pairs = _assign(pred_points, gt_points)
        tp = 0
        for pi, gi in pairs:
            if pi < len(pred_points) and gi < len(masks):
                if _point_in_mask(pred_points[pi], masks[gi]):
                    tp += 1
        precision = float(tp) / float(len(pred_points)) if pred_points else 0.0
        recall = float(tp) / float(len(gt_points)) if gt_points else 0.0
        if precision + recall > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
    else:
        # parsing failure case when target exists but no coords predicted
        if has_target and gt_points and not pred_points:
            parsing_failure = 1.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "no_target_correct": no_target_correct,
        "parsing_failure": parsing_failure,
        # For audit
        "predicted_points": pred_points,
    }
