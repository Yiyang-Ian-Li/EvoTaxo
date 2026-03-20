from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None


def group_key(p: Dict[str, Any]) -> Tuple[str, str]:
    return (str(p.get("action_type", "")), str(p.get("objective_node_id", "")))


def semantic_text(p: Dict[str, Any]) -> str:
    explanation = str(p.get("action_explanation", "")).strip()
    summary = str(p.get("post_summary", "")).strip()
    return " ".join([x for x in [explanation, summary] if x])


def _norm(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return v / denom


def _cosine_distance_matrix(emb: np.ndarray) -> np.ndarray:
    e = _norm(emb.astype(np.float64, copy=False))
    sim = np.clip(e @ e.T, -1.0, 1.0)
    return (1.0 - sim).astype(np.float64, copy=False)


def _cosine_distance_to_centroid(emb: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if len(idx) == 0:
        return np.array([], dtype=np.float64)
    members = emb[idx].astype(np.float64, copy=False)
    member_norms = np.linalg.norm(members, axis=1, keepdims=True)
    member_norms[member_norms == 0] = 1.0
    members = members / member_norms
    centroid = np.mean(members, axis=0, keepdims=True)
    centroid_norm = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid_norm[centroid_norm == 0] = 1.0
    centroid = centroid / centroid_norm
    sim = np.clip((members @ centroid.T).reshape(-1), -1.0, 1.0)
    return (1.0 - sim).astype(np.float64, copy=False)


def _time_distance_matrix(ts_seconds: np.ndarray) -> np.ndarray:
    if len(ts_seconds) <= 1:
        return np.zeros((len(ts_seconds), len(ts_seconds)), dtype=np.float64)
    tmin = float(np.min(ts_seconds))
    tmax = float(np.max(ts_seconds))
    span = max(tmax - tmin, 1.0)
    dif = np.abs(ts_seconds.reshape(-1, 1) - ts_seconds.reshape(1, -1))
    return (dif / span).astype(np.float64, copy=False)


def _run_hdbscan(
    dist_matrix: np.ndarray,
    min_cluster_size: int,
) -> Tuple[np.ndarray, Dict[int, float]]:
    dist_matrix = dist_matrix.astype(np.float64, copy=False)
    n = dist_matrix.shape[0]
    if n == 0:
        return np.array([], dtype=int), {}
    if n < max(2, min_cluster_size):
        return np.full(n, -1, dtype=int), {}

    if hdbscan is None:
        return np.full(n, -1, dtype=int), {}

    clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(dist_matrix)
    persistence = {}
    raw_persistence = getattr(clusterer, "cluster_persistence_", None)
    if raw_persistence is None:
        raw_persistence = []
    for cid, value in enumerate(raw_persistence):
        persistence[int(cid)] = round(float(value), 4)
    return labels, persistence


def _cluster_quality(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    persistence_map: Dict[int, float],
) -> Dict[int, Dict[str, float]]:
    quality: Dict[int, Dict[str, float]] = {}
    for cid in sorted(set(labels.tolist())):
        if cid < 0:
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) < 2:
            quality[int(cid)] = {
                "cohesion": 1.0,
                "stability": persistence_map.get(int(cid), 1.0),
                "time_compactness": 1.0,
            }
            continue

        intra = dist_matrix[np.ix_(idx, idx)]
        cohesion = float(1.0 - np.mean(intra))
        cohesion = max(0.0, min(1.0, cohesion))

        t = timestamps[idx]
        t_span = float(np.max(t) - np.min(t))
        global_span = max(float(np.max(timestamps) - np.min(timestamps)), 1.0)
        time_compactness = max(0.0, min(1.0, 1.0 - (t_span / global_span)))

        stability = float(persistence_map.get(int(cid), 0.0))
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

    sem_labels, sem_persistence = _run_hdbscan(sem_dist, min_cluster_size=min_cluster_size)
    tmp_labels, tmp_persistence = _run_hdbscan(temp_dist, min_cluster_size=min_cluster_size)

    sem_q = _cluster_quality(sem_dist, sem_labels, timestamps, sem_persistence)
    tmp_q = _cluster_quality(temp_dist, tmp_labels, timestamps, tmp_persistence)

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
            centroid_dist = _cosine_distance_to_centroid(embeddings, idx)
            centroid_order = [int(x) for x in np.argsort(centroid_dist)]
            centroid_proposal_ids = [proposals[int(idx[i])]["proposal_id"] for i in centroid_order]
            store.append(
                {
                    "cluster_id": f"{mode}:{first.get('action_type')}:{first.get('objective_node_id')}:{cid}",
                    "window_id": first.get("window_id"),
                    "cluster_mode": mode,
                    "action_type": first.get("action_type"),
                    "objective_node_id": first.get("objective_node_id"),
                    "proposal_ids": [proposals[int(i)]["proposal_id"] for i in idx],
                    "centroid_proposal_ids": centroid_proposal_ids,
                    "size": int(len(idx)),
                    "quality": qmap.get(int(cid), {"cohesion": 0.0, "stability": 0.0, "time_compactness": 0.0}),
                }
            )

    return sem_clusters, tmp_clusters
