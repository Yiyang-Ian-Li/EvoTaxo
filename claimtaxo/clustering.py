from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap

from .config import ClusteringConfig


@dataclass
class ClusterResult:
    labels: np.ndarray
    n_clusters: int
    silhouette: float


def reduce_embeddings(embeddings: np.ndarray, cfg: ClusteringConfig) -> np.ndarray:
    n_samples = len(embeddings)
    if n_samples <= 2:
        return embeddings

    # UMAP spectral init can fail on tiny subsets (k >= N). Fall back to raw space.
    max_components = n_samples - 2
    if max_components < 1:
        return embeddings
    n_components = min(cfg.umap_n_components, max_components)
    n_neighbors = min(cfg.umap_n_neighbors, n_samples - 1)
    if n_components < 1 or n_neighbors < 2:
        return embeddings

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=cfg.umap_min_dist,
        n_components=n_components,
        random_state=cfg.random_state,
    )
    try:
        return reducer.fit_transform(embeddings)
    except Exception:
        return embeddings


def kmeans_best_k(embeddings: np.ndarray, cfg: ClusteringConfig) -> ClusterResult:
    best = None
    for k in range(cfg.k_min, cfg.k_max + 1, cfg.k_step):
        if k <= 1 or k >= len(embeddings):
            continue
        kmeans = KMeans(n_clusters=k, random_state=cfg.random_state, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
        try:
            score = silhouette_score(embeddings, labels)
        except ValueError:
            continue
        if best is None or score > best.silhouette:
            best = ClusterResult(labels=labels, n_clusters=k, silhouette=score)
    if best is None:
        labels = np.zeros(len(embeddings), dtype=int)
        best = ClusterResult(labels=labels, n_clusters=1, silhouette=0.0)
    return best
