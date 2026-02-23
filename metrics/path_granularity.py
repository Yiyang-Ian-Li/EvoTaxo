from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from .common import EvalNode, taxonomy_paths
from .llm_client import EvalLLMClient


def compute_path_granularity(
    nodes: Dict[str, EvalNode], root_id: str, root_topic: str, llm: EvalLLMClient
) -> Tuple[float, List[Dict]]:
    paths = taxonomy_paths(nodes, root_id)
    if not paths:
        return float("nan"), []

    rows: List[Dict] = []
    valid_scores = []
    for path in tqdm(paths, desc="Path Granularity"):
        names = [nodes[nid].name for nid in path if nid in nodes and nid != root_id]
        if len(names) < 2:
            continue
        path_str = " > ".join(names)
        prompt = (
            "Scientific concepts are naturally organized in multi-dimensional taxonomic structures, "
            "with more specific concepts being the children of a broader research topic.\n\n"
            f"Given the root topic: '{root_topic}', decide whether this path from the scientific concept "
            f"taxonomy has good granularity: '{path_str}'. Check whether each child node is a more specific "
            "subaspect of its parent node.\n\n"
            "Output options: '<good granularity>' or '<bad granularity>'. "
            "Do some simple rationalization before giving the output if possible."
        )
        out = (llm.chat(prompt) or "").lower()
        if "<good granularity>" in out:
            score = 1.0
        elif "<bad granularity>" in out:
            score = 0.0
        else:
            score = float("nan")
        rows.append({"path": path_str, "score": score, "reasoning": out})
        if not math.isnan(score):
            valid_scores.append(score)
    metric = float(np.mean(valid_scores)) if valid_scores else float("nan")
    return metric, rows
