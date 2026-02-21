from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None


def group_key(p: Dict[str, Any]) -> Tuple[str, str]:
    return (str(p.get("action_type", "")), str(p.get("objective_node_id", "")))


def semantic_text(p: Dict[str, Any]) -> str:
    sem = p.get("semantic_payload", {})
    if not isinstance(sem, dict):
        return ""
    parts: List[str] = []
    if p.get("action_type") == "add_path":
        parts.append(str(p.get("objective_node_id", "")))
        for item in sem.get("nodes", []) if isinstance(sem.get("nodes", []), list) else []:
            if not isinstance(item, dict):
                continue
            parts.append(str(item.get("name", "")))
            parts.append(str(item.get("level", "")))
            cmb = item.get("cmb", {})
            if isinstance(cmb, dict):
                parts.extend(
                    [
                        str(cmb.get("definition", "")),
                        " ".join([str(x) for x in cmb.get("include_terms", [])]),
                        " ".join([str(x) for x in cmb.get("exclude_terms", [])]),
                        " ".join([str(x) for x in cmb.get("examples", [])]),
                    ]
                )
        return " ".join([x for x in parts if x.strip()])

    if "child_name" in sem:
        parts.append(str(sem.get("child_name", "")))
    if "child_level" in sem:
        parts.append(str(sem.get("child_level", "")))
    cc = sem.get("child_cmb", {})
    if isinstance(cc, dict):
        parts.extend(
            [
                str(cc.get("definition", "")),
                " ".join([str(x) for x in cc.get("include_terms", [])]),
                " ".join([str(x) for x in cc.get("exclude_terms", [])]),
                " ".join([str(x) for x in cc.get("examples", [])]),
            ]
        )
    nc = sem.get("new_cmb", {})
    if isinstance(nc, dict):
        parts.extend(
            [
                str(nc.get("definition", "")),
                " ".join([str(x) for x in nc.get("include_terms", [])]),
                " ".join([str(x) for x in nc.get("exclude_terms", [])]),
                " ".join([str(x) for x in nc.get("examples", [])]),
            ]
        )
    return " ".join([p for p in parts if p.strip()])


def _norm(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return v / denom


def _cosine_distance_matrix(emb: np.ndarray) -> np.ndarray:
    e = _norm(emb.astype(np.float64, copy=False))
    sim = np.clip(e @ e.T, -1.0, 1.0)
    return (1.0 - sim).astype(np.float64, copy=False)


def _time_distance_matrix(ts_seconds: np.ndarray) -> np.ndarray:
    if len(ts_seconds) <= 1:
        return np.zeros((len(ts_seconds), len(ts_seconds)), dtype=np.float64)
    tmin = float(np.min(ts_seconds))
    tmax = float(np.max(ts_seconds))
    span = max(tmax - tmin, 1.0)
    dif = np.abs(ts_seconds.reshape(-1, 1) - ts_seconds.reshape(1, -1))
    return (dif / span).astype(np.float64, copy=False)


def _run_hdbscan(dist_matrix: np.ndarray, min_cluster_size: int) -> np.ndarray:
    dist_matrix = dist_matrix.astype(np.float64, copy=False)
    n = dist_matrix.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n < max(2, min_cluster_size):
        return np.full(n, -1, dtype=int)

    if hdbscan is None:
        return np.full(n, -1, dtype=int)

    clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=min_cluster_size)
    return clusterer.fit_predict(dist_matrix)


def _cluster_quality(dist_matrix: np.ndarray, labels: np.ndarray, timestamps: np.ndarray) -> Dict[int, Dict[str, float]]:
    quality: Dict[int, Dict[str, float]] = {}
    for cid in sorted(set(labels.tolist())):
        if cid < 0:
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) < 2:
            quality[int(cid)] = {"cohesion": 1.0, "stability": 1.0, "time_compactness": 1.0}
            continue

        intra = dist_matrix[np.ix_(idx, idx)]
        cohesion = float(1.0 - np.mean(intra))
        cohesion = max(0.0, min(1.0, cohesion))

        t = timestamps[idx]
        t_span = float(np.max(t) - np.min(t))
        global_span = max(float(np.max(timestamps) - np.min(timestamps)), 1.0)
        time_compactness = max(0.0, min(1.0, 1.0 - (t_span / global_span)))

        # MVP stability proxy.
        stability = cohesion
        quality[int(cid)] = {
            "cohesion": round(cohesion, 4),
            "stability": round(stability, 4),
            "time_compactness": round(time_compactness, 4),
        }
    return quality


def cluster_group(
    proposals: List[Dict[str, Any]],
    embeddings: np.ndarray,
    min_cluster_size: int,
    w_sem: float,
    w_time: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not proposals:
        return [], []

    timestamps = np.array([float(x.get("timestamp_epoch", 0.0)) for x in proposals], dtype=float)
    sem_dist = _cosine_distance_matrix(embeddings)
    time_dist = _time_distance_matrix(timestamps)
    temp_dist = (w_sem * sem_dist) + (w_time * time_dist)

    sem_labels = _run_hdbscan(sem_dist, min_cluster_size=min_cluster_size)
    tmp_labels = _run_hdbscan(temp_dist, min_cluster_size=min_cluster_size)

    sem_q = _cluster_quality(sem_dist, sem_labels, timestamps)
    tmp_q = _cluster_quality(temp_dist, tmp_labels, timestamps)

    sem_clusters = []
    tmp_clusters = []

    for mode, labels, qmap, store in (
        ("semantic", sem_labels, sem_q, sem_clusters),
        ("temporal", tmp_labels, tmp_q, tmp_clusters),
    ):
        for cid in sorted(set(labels.tolist())):
            if cid < 0:
                continue
            idx = np.where(labels == cid)[0]
            if len(idx) == 0:
                continue
            first = proposals[int(idx[0])]
            store.append(
                {
                    "cluster_id": f"{mode}:{first.get('action_type')}:{first.get('objective_node_id')}:{cid}",
                    "window_id": first.get("window_id"),
                    "cluster_mode": mode,
                    "action_type": first.get("action_type"),
                    "objective_node_id": first.get("objective_node_id"),
                    "proposal_ids": [proposals[int(i)]["proposal_id"] for i in idx],
                    "size": int(len(idx)),
                    "quality": qmap.get(int(cid), {"cohesion": 0.0, "stability": 0.0, "time_compactness": 0.0}),
                }
            )

    return sem_clusters, tmp_clusters
