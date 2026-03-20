from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from .common import EvalNode, taxonomy_paths
from .llm_client import EvalLLMClient


def _normalize_term(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _compress_path_names(names: List[str]) -> List[str]:
    out: List[str] = []
    for name in names:
        if not name:
            continue
        if out and _normalize_term(out[-1]) == _normalize_term(name):
            continue
        out.append(name)
    return out


def _extract_score_1_to_5(text: str) -> float:
    lowered = (text or "").lower()
    for pat in [
        r"<score[:=\s]*([1-5])>",
        r"\bscore\s*[:=]\s*([1-5])\b",
        r"\b([1-5])\s*/\s*5\b",
        r"\b([1-5])\b",
    ]:
        m = re.search(pat, lowered)
        if m:
            return float(int(m.group(1)))
    return float("nan")


def compute_path_granularity(
    nodes: Dict[str, EvalNode], root_id: str, root_topic: str, llm: EvalLLMClient
) -> Tuple[float, List[Dict]]:
    paths = taxonomy_paths(nodes, root_id)
    if not paths:
        return float("nan"), []

    rows: List[Dict] = []
    valid_scores = []
    for path in tqdm(paths, desc="Path Granularity"):
        names = _compress_path_names([root_topic] + [nodes[nid].name for nid in path if nid in nodes and nid != root_id])
        if len(names) < 2:
            continue
        path_str = " > ".join(names)
        prompt = (
            f"You are evaluating the granularity of a taxonomy path in the domain '{root_topic}'.\n\n"
            f"The taxonomy path is: '{path_str}'.\n\n"
            "Score whether the path shows a clear progression from broader categories to more specific categories. "
            "A high score means each child is a meaningfully more specific subcategory of its parent, without abrupt "
            "jumps, redundancy, or inconsistent levels of abstraction. A low score means the path is too flat, too "
            "coarse, too uneven, or otherwise poorly structured.\n\n"
            "Use a 1-5 scale where 1 = very poor granularity, 2 = weak, 3 = mixed/acceptable, "
            "4 = good, and 5 = excellent granularity. "
            "Return a short rationale and an explicit machine-readable score in the form '<score: X>'."
        )
        out = (llm.chat(prompt) or "").lower()
        score = _extract_score_1_to_5(out)
        rows.append({"path": path_str, "score": score, "reasoning": out})
        if not math.isnan(score):
            valid_scores.append(score)
    metric = float(np.mean(valid_scores)) if valid_scores else float("nan")
    return metric, rows
