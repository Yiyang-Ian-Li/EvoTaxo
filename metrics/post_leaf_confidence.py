from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import pipeline

from .common import EvalNode, active_children


def leaf_node_names(nodes: Dict[str, EvalNode], root_id: str) -> List[str]:
    children = active_children(nodes)
    names: List[str] = []
    for node_id, node in nodes.items():
        if node_id == root_id or node.status != "active":
            continue
        if not children.get(node_id):
            names.append(node.name)
    return sorted({name for name in names if name})


def _entropy(probs: Sequence[float]) -> float:
    arr = np.asarray(probs, dtype=np.float64)
    arr = np.clip(arr, 1e-12, 1.0)
    return float(-(arr * np.log(arr)).sum())


def _normalized_entropy(probs: Sequence[float]) -> float:
    if not probs or len(probs) < 2:
        return float("nan")
    arr = np.asarray([float(p) for p in probs], dtype=np.float64)
    total = float(arr.sum())
    if total <= 0.0:
        return float("nan")
    arr = arr / total
    max_entropy = math.log(len(arr))
    if max_entropy <= 0.0:
        return float("nan")
    return _entropy(arr) / max_entropy


def compute_post_leaf_confidence(
    posts: Sequence[str],
    nodes: Dict[str, EvalNode],
    root_id: str,
    model_name: str,
    device: int,
    batch_size: int,
) -> Dict[str, float]:
    leaf_names = leaf_node_names(nodes, root_id)
    candidate_labels = list(leaf_names)
    if "others" not in {label.lower() for label in candidate_labels}:
        candidate_labels.append("others")

    valid_posts = [str(post).strip() for post in posts if str(post).strip()]
    if not valid_posts or len(candidate_labels) < 2:
        return {
            "mean_entropy": float("nan"),
            "std_entropy": float("nan"),
            "mean_margin_top1_top2": float("nan"),
            "std_margin_top1_top2": float("nan"),
            "others_ratio": float("nan"),
            "num_posts_predicted_others": 0,
            "num_posts": len(valid_posts),
            "num_leaf_labels": len(leaf_names),
            "entropy_normalization": "log_num_candidate_labels",
        }

    model_kwargs = {}
    if device >= 0:
        model_kwargs["dtype"] = torch.float16
    clf = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
        model_kwargs=model_kwargs,
    )

    entropies: List[float] = []
    margins: List[float] = []
    num_posts_predicted_others = 0
    for start in tqdm(range(0, len(valid_posts), batch_size), desc="Post Leaf Confidence"):
        batch = valid_posts[start : start + batch_size]
        outputs = clf(batch, candidate_labels, multi_label=False, batch_size=batch_size)
        if isinstance(outputs, dict):
            outputs = [outputs]
        for out in outputs:
            labels = [str(label) for label in out.get("labels", [])]
            scores = [float(score) for score in out.get("scores", [])]
            if len(scores) < 2:
                continue
            if labels and labels[0].strip().lower() == "others":
                num_posts_predicted_others += 1
            non_other_scores = [
                score
                for label, score in zip(labels, scores)
                if label.strip().lower() != "others"
            ]
            if len(non_other_scores) < 2:
                continue
            entropies.append(_normalized_entropy(non_other_scores))
            top2 = sorted(non_other_scores, reverse=True)[:2]
            margins.append(float(top2[0] - top2[1]))

    return {
        "mean_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "std_entropy": float(np.std(entropies)) if entropies else float("nan"),
        "mean_margin_top1_top2": float(np.mean(margins)) if margins else float("nan"),
        "std_margin_top1_top2": float(np.std(margins)) if margins else float("nan"),
        "others_ratio": (
            float(num_posts_predicted_others / len(valid_posts))
            if valid_posts
            else float("nan")
        ),
        "num_posts_predicted_others": int(num_posts_predicted_others),
        "num_posts": len(valid_posts),
        "num_leaf_labels": len(leaf_names),
        "entropy_normalization": "log_num_candidate_labels",
    }
