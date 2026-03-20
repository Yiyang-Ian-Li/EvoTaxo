from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def map_posts_to_subtopics(
    post_vecs: np.ndarray,
    subtopic_vecs: np.ndarray,
    subtopic_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[Optional[str]]]:
    if subtopic_vecs.shape[0] > 0 and post_vecs.shape[0] > 0:
        sims = cosine_similarity(post_vecs, subtopic_vecs)
        best_idx_arr = np.argmax(sims, axis=1).astype(int)
        best_sim_arr = np.max(sims, axis=1).astype(float)
    else:
        best_idx_arr = np.full(post_vecs.shape[0], -1, dtype=int)
        best_sim_arr = np.zeros(post_vecs.shape[0], dtype=float)

    best_node_ids: List[Optional[str]] = []
    for idx in best_idx_arr:
        i = int(idx)
        if 0 <= i < len(subtopic_ids):
            best_node_ids.append(subtopic_ids[i])
        else:
            best_node_ids.append(None)
    return best_idx_arr, best_sim_arr, best_node_ids
