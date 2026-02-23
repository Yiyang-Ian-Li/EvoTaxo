from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import pipeline

from .common import EvalNode, geometric_mean, hearst_hypotheses, paths_to_root, semantic_text


def _find_label_score(rows: List[Dict], needle: str) -> float:
    for r in rows:
        lbl = str(r.get("label", "")).lower()
        if needle in lbl:
            return float(r.get("score", 0.0))
    raise ValueError(f"NLI output missing label containing '{needle}'")


def compute_nliv(
    nodes: Dict[str, EvalNode],
    root_id: str,
    model_name: str,
    device: int,
    batch_size: int,
    max_length: int,
    include_root_edges: bool,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    node_ids = [nid for nid, n in nodes.items() if nid != root_id and n.status == "active"]
    edges = []
    for nid in node_ids:
        parent = nodes[nid].parent_id
        if parent is None or parent not in nodes:
            continue
        if not include_root_edges and parent == root_id:
            continue
        edges.append((parent, nid))

    if not edges:
        return {"nliv_s": float("nan"), "nliv_w": float("nan"), "num_edges": 0, "num_paths": 0}, [], []

    model_kwargs = {}
    if device >= 0:
        model_kwargs["dtype"] = torch.float16
    clf = pipeline(
        "text-classification",
        model=model_name,
        device=device,
        top_k=None,
        model_kwargs=model_kwargs,
    )

    pair_inputs: List[Dict[str, str]] = []
    pair_meta: List[Tuple[str, str, str]] = []
    for parent_id, child_id in edges:
        premise = semantic_text(nodes[child_id])
        for hyp in hearst_hypotheses(nodes[parent_id].name, nodes[child_id].name):
            pair_inputs.append({"text": premise, "text_pair": hyp})
            pair_meta.append((parent_id, child_id, hyp))

    raw_scores = []
    for s in tqdm(range(0, len(pair_inputs), batch_size), desc="NLI edge scoring"):
        batch = pair_inputs[s : s + batch_size]
        out = clf(batch, truncation=True, max_length=max_length)
        raw_scores.extend(out)

    edge_buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    edge_rows: List[Dict] = []
    for idx, score_rows in enumerate(raw_scores):
        if isinstance(score_rows, dict):
            score_rows = [score_rows]
        parent_id, child_id, hyp = pair_meta[idx]
        entails = _find_label_score(score_rows, "entail")
        contradicts = _find_label_score(score_rows, "contrad")
        key = (parent_id, child_id)
        if key not in edge_buckets:
            edge_buckets[key] = {"entails": [], "wprob": []}
        edge_buckets[key]["entails"].append(entails)
        edge_buckets[key]["wprob"].append(1.0 - contradicts)
        edge_rows.append(
            {
                "parent_id": parent_id,
                "parent_name": nodes[parent_id].name,
                "child_id": child_id,
                "child_name": nodes[child_id].name,
                "hypothesis": hyp,
                "entails_prob": entails,
                "non_contrad_prob": 1.0 - contradicts,
            }
        )

    edge_probs: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for key, vals in edge_buckets.items():
        edge_probs[key] = (float(np.mean(vals["entails"])), float(np.mean(vals["wprob"])))

    paths = paths_to_root(nodes, root_id)
    path_rows: List[Dict] = []
    path_s = []
    path_w = []
    for nid in node_ids:
        path = paths.get(nid)
        if not path:
            continue
        s_probs = []
        w_probs = []
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge not in edge_probs:
                continue
            ps, pw = edge_probs[edge]
            s_probs.append(ps)
            w_probs.append(pw)
        if not s_probs:
            continue
        s_path = geometric_mean(s_probs)
        w_path = geometric_mean(w_probs)
        path_s.append(s_path)
        path_w.append(w_path)
        path_rows.append(
            {
                "target_node_id": nid,
                "target_name": nodes[nid].name,
                "depth": len(path) - 1,
                "path_nliv_s": s_path,
                "path_nliv_w": w_path,
            }
        )

    summary = {
        "nliv_s": float(np.mean(path_s)) if path_s else float("nan"),
        "nliv_w": float(np.mean(path_w)) if path_w else float("nan"),
        "num_edges": len(edges),
        "num_paths": len(path_rows),
    }
    return summary, edge_rows, path_rows
