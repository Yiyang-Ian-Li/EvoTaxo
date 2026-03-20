from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from .common import EvalNode, taxonomy_levels
from .llm_client import EvalLLMClient


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


def compute_sibling_separability(
    nodes: Dict[str, EvalNode], root_id: str, root_topic: str, llm: EvalLLMClient
) -> Tuple[float, List[Dict]]:
    levels = taxonomy_levels(nodes, root_id, include_root=True, root_topic=root_topic)
    if not levels:
        return float("nan"), []

    rows: List[Dict] = []
    valid_scores = []
    for level in tqdm(levels, desc="Sibling Separability"):
        parent = level["parent_name"]
        siblings = level["siblings"]
        if len(siblings) <= 1:
            score = float("nan")
            reasoning = "not_applicable_single_child"
        else:
            prompt = (
                f"You are evaluating the separability of sibling categories in a taxonomy for the domain '{root_topic}'.\n\n"
                f"The parent topic is: {parent}.\n\n"
                f"The sibling categories are: '{', '.join(siblings)}'\n\n"
                "Score how clearly these sibling categories are distinguishable from one another. "
                "A high score means each sibling has a clear, non-overlapping scope and could be reliably "
                "told apart from the others. A low score means the siblings are confusing, redundant, or "
                "strongly overlapping. Use a 1-5 scale where 1 = very poor separability, 2 = weak, "
                "3 = mixed/acceptable, 4 = good, and 5 = excellent separability. "
                "Return a short rationale and an explicit machine-readable score in the form '<score: X>'."
            )
            reasoning = (llm.chat(prompt) or "").lower()
            score = _extract_score_1_to_5(reasoning)

        rows.append(
            {
                "parent_id": level["parent_id"],
                "parent_name": parent,
                "siblings": " | ".join(siblings),
                "num_siblings": len(siblings),
                "score": score,
                "reasoning": reasoning,
            }
        )
        if not math.isnan(score):
            valid_scores.append(score)

    metric = float(np.mean(valid_scores)) if valid_scores else float("nan")
    return metric, rows
