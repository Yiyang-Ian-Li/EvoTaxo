from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from .common import EvalNode, taxonomy_levels
from .llm_client import EvalLLMClient


def compute_sibling_coherence(
    nodes: Dict[str, EvalNode], root_id: str, root_topic: str, llm: EvalLLMClient
) -> Tuple[float, List[Dict]]:
    levels = taxonomy_levels(nodes, root_id)
    if not levels:
        return float("nan"), []

    rows: List[Dict] = []
    valid_scores = []
    for level in tqdm(levels, desc="Sibling Coherence"):
        parent = level["parent_name"]
        siblings = level["siblings"]
        if len(siblings) <= 1:
            score = 0.0
            reasoning = "<no_sibling_coherence> (single child set)"
        else:
            prompt = (
                f"You are determining the coherence of a set of {root_topic} subtopics of the parent topic {parent}.\n\n"
                f"The parent topic is: {parent}.\n\n"
                f"The set of siblings, which are child subtopics of the parent, are: '{', '.join(siblings)}'\n\n"
                f"Evaluate the overall coherence of the sibling set based on their collective specificity and granularity "
                f"relative to {parent}. Use the following scoring criteria:\n\n"
                "Score=<no_sibling_coherence>: The set is highly inconsistent or incoherent (only one subtopic), "
                "with most topics significantly misaligned in specificity relative to the parent.\n"
                "Score=<weak_sibling_coherence>: The set shows considerable inconsistency, with several topics deviating "
                "noticeably from the expected level of specificity.\n"
                "Score=<reasonable_sibling_coherence>: The set is generally coherent, with only minor inconsistencies "
                "in specificity among the topics.\n"
                "Score=<strong_sibling_coherence>: The set is fully coherent, with all topics properly matching the "
                "expected level of specificity and granularity for the parent.\n"
                "Output options: '<no_sibling_coherence>', '<weak_sibling_coherence>', "
                "'<reasonable_sibling_coherence>', or '<strong_sibling_coherence>'. "
                "Do some simple rationalization before giving the output if possible."
            )
            reasoning = (llm.chat(prompt) or "").lower()
            if "<no_sibling_coherence>" in reasoning:
                score = 0.0
            elif "<weak_sibling_coherence>" in reasoning:
                score = 1.0 / 3.0
            elif "<reasonable_sibling_coherence>" in reasoning:
                score = 2.0 / 3.0
            elif "<strong_sibling_coherence>" in reasoning:
                score = 1.0
            else:
                score = float("nan")

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
