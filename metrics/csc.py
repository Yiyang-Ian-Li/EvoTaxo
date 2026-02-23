from __future__ import annotations

from typing import Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from .common import EvalNode, cosine_pairwise, kendall_tau_b, paths_to_root, semantic_text, wu_palmer


def compute_csc(nodes: Dict[str, EvalNode], root_id: str, model_name: str, batch_size: int) -> Dict:
    kept = [n for n in nodes.values() if n.node_id != root_id and n.status == "active"]
    if len(kept) < 3:
        return {"score": float("nan"), "num_nodes": len(kept), "num_pairs": 0}

    id_to_idx = {n.node_id: i for i, n in enumerate(kept)}
    embedder = SentenceTransformer(model_name)
    vecs = embedder.encode([semantic_text(n) for n in kept], batch_size=batch_size, show_progress_bar=False)
    sem = cosine_pairwise(np.asarray(vecs, dtype=np.float64))

    paths = paths_to_root(nodes, root_id)
    tax = np.zeros_like(sem)
    for a in kept:
        for b in kept:
            ia = id_to_idx[a.node_id]
            ib = id_to_idx[b.node_id]
            tax[ia, ib] = wu_palmer(paths[a.node_id], paths[b.node_id])

    sem_pairs = []
    tax_pairs = []
    n = len(kept)
    for i in range(n - 1):
        for j in range(i + 1, n):
            sem_pairs.append(sem[i, j])
            tax_pairs.append(tax[i, j])
    sem_arr = np.array(sem_pairs, dtype=np.float64)
    tax_arr = np.array(tax_pairs, dtype=np.float64)
    return {
        "score": float(kendall_tau_b(sem_arr, tax_arr)),
        "num_nodes": len(kept),
        "num_pairs": int(len(sem_pairs)),
    }
