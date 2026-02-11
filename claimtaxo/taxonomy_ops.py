from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from .config import LifecycleConfig, MergeConfig, SplitConfig
from .embeddings import Embedder
from .taxonomy import Taxonomy


def slice_to_index(slice_id: str) -> int:
    try:
        year_text, quarter_text = slice_id.split("Q")
        year = int(year_text)
        quarter = int(quarter_text)
        return year * 4 + quarter
    except Exception:
        return 0


def _node_text(taxonomy: Taxonomy, node_id: str) -> str:
    node = taxonomy.nodes[node_id]
    cmb = node.cmb
    parts = [
        node.name,
        cmb.canonical_definition,
        " ".join(cmb.include_terms),
    ]
    return " ".join(p for p in parts if p)


def _pick_merge_winner(
    left_id: str,
    right_id: str,
    support_totals: Dict[str, int],
    taxonomy: Taxonomy,
) -> Tuple[str, str]:
    left_support = support_totals.get(left_id, 0)
    right_support = support_totals.get(right_id, 0)
    if left_support != right_support:
        return (left_id, right_id) if left_support > right_support else (right_id, left_id)

    left_node = taxonomy.nodes[left_id]
    right_node = taxonomy.nodes[right_id]
    left_seen = left_node.first_seen_slice or "9999Q4"
    right_seen = right_node.first_seen_slice or "9999Q4"
    if left_seen != right_seen:
        return (left_id, right_id) if left_seen < right_seen else (right_id, left_id)
    return (left_id, right_id) if left_id < right_id else (right_id, left_id)


def run_merge_pass(
    taxonomy: Taxonomy,
    embedder: Embedder,
    merge_cfg: MergeConfig,
    slice_id: str,
    support_totals: Dict[str, int],
    support_in_slice: Dict[str, int],
) -> Dict[str, Any]:
    if not merge_cfg.enabled:
        return {"merged_pairs": [], "operation_rows": []}

    by_parent: Dict[str, List[str]] = defaultdict(list)
    for node in taxonomy.nodes.values():
        if node.level != "argument":
            continue
        if node.status == "deprecated":
            continue
        if node.parent_id is None:
            continue
        by_parent[node.parent_id].append(node.node_id)

    merged_pairs: List[Tuple[str, str]] = []
    operation_rows: List[Dict[str, Any]] = []
    merges_done = 0

    for parent_id, node_ids in by_parent.items():
        active_ids = [nid for nid in node_ids if taxonomy.nodes[nid].status != "deprecated"]
        if len(active_ids) < 2:
            continue

        node_vectors = embedder.encode([_node_text(taxonomy, nid) for nid in active_ids])
        sim_matrix = cosine_similarity(node_vectors, node_vectors)
        pairs: List[Tuple[float, str, str]] = []
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                sim = float(sim_matrix[i, j])
                if sim >= merge_cfg.similarity_threshold:
                    pairs.append((sim, active_ids[i], active_ids[j]))
        pairs.sort(key=lambda x: x[0], reverse=True)

        blocked = set()
        for sim, left_id, right_id in pairs:
            if merges_done >= merge_cfg.max_merges_per_slice:
                break
            if left_id in blocked or right_id in blocked:
                continue
            winner_id, loser_id = _pick_merge_winner(left_id, right_id, support_totals, taxonomy)
            winner_support = support_totals.get(winner_id, 0)
            loser_support = support_totals.get(loser_id, 0)
            if loser_support < merge_cfg.min_support:
                pass
            elif winner_support == 0:
                continue
            elif (loser_support / winner_support) > merge_cfg.support_ratio:
                continue

            taxonomy.nodes[loser_id].status = "deprecated"
            taxonomy.redirect(
                source_id=loser_id,
                target_id=winner_id,
                reason="auto-merge",
                op_type="merge",
                slice_id=slice_id,
                score=sim,
                details={
                    "parent_id": parent_id,
                    "winner_support_total": winner_support,
                    "loser_support_total": loser_support,
                    "winner_support_slice": support_in_slice.get(winner_id, 0),
                    "loser_support_slice": support_in_slice.get(loser_id, 0),
                },
            )
            merged_pairs.append((loser_id, winner_id))
            blocked.add(winner_id)
            blocked.add(loser_id)
            merges_done += 1
            operation_rows.append(
                {
                    "slice_id": slice_id,
                    "op_type": "merge",
                    "source_id": loser_id,
                    "target_id": winner_id,
                    "score": sim,
                    "parent_id": parent_id,
                }
            )

        if merges_done >= merge_cfg.max_merges_per_slice:
            break

    return {"merged_pairs": merged_pairs, "operation_rows": operation_rows}


def _cluster_name(texts: List[str], fallback: str) -> Tuple[str, str, List[str]]:
    if not texts:
        return fallback, fallback, []
    vectorizer = TfidfVectorizer(max_features=30, stop_words="english")
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    scores = np.asarray(X.mean(axis=0)).ravel()
    top_idx = scores.argsort()[::-1][:5]
    include_terms = [terms[i] for i in top_idx]
    if not include_terms:
        return fallback, fallback, []
    name = f"{fallback} - {include_terms[0]}"
    definition = f"Sub-argument under '{fallback}' focused on: {', '.join(include_terms[:3])}."
    return name, definition, include_terms


def run_split_pass(
    taxonomy: Taxonomy,
    embedder: Embedder,
    split_cfg: SplitConfig,
    slice_id: str,
    posts_by_node: Dict[str, List[str]],
) -> Dict[str, Any]:
    if not split_cfg.enabled:
        return {"split_nodes": [], "operation_rows": []}

    split_nodes: List[Tuple[str, List[str]]] = []
    operation_rows: List[Dict[str, Any]] = []

    for node_id, texts in posts_by_node.items():
        node = taxonomy.nodes.get(node_id)
        if node is None or node.level != "argument" or node.status == "deprecated":
            continue
        if len(texts) < split_cfg.min_support:
            continue

        emb = embedder.encode(texts)
        best = None
        for k in range(split_cfg.min_clusters, split_cfg.max_clusters + 1):
            if k >= len(texts):
                continue
            labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(emb)
            try:
                score = float(silhouette_score(emb, labels))
            except ValueError:
                continue
            counts = np.bincount(labels)
            if counts.min() < split_cfg.min_cluster_size:
                continue
            if best is None or score > best[0]:
                best = (score, labels, k)

        if best is None or best[0] < split_cfg.silhouette_threshold:
            continue

        _, labels, n_clusters = best
        child_ids: List[str] = []
        for cluster_id in range(n_clusters):
            cluster_idx = np.where(labels == cluster_id)[0]
            cluster_texts = [texts[i] for i in cluster_idx]
            new_name, new_def, include_terms = _cluster_name(cluster_texts, node.name)
            new_node_id = taxonomy.add_node(new_name, "argument", node.parent_id)
            taxonomy.set_cmb(
                new_node_id,
                canonical_definition=new_def,
                include_terms=include_terms,
                representative_examples=cluster_texts[:3],
                exclude_terms=[],
            )
            taxonomy.mark_seen(new_node_id, slice_id)
            taxonomy.nodes[new_node_id].status = "candidate"
            child_ids.append(new_node_id)

        counts = [(child_ids[i], int(np.sum(labels == i))) for i in range(n_clusters)]
        counts.sort(key=lambda x: x[1], reverse=True)
        dominant_child = counts[0][0]

        node.status = "deprecated"
        taxonomy.redirect(
            source_id=node_id,
            target_id=dominant_child,
            reason="auto-split",
            op_type="split",
            slice_id=slice_id,
            score=float(best[0]),
            details={"child_ids": child_ids, "cluster_sizes": [c for _, c in counts]},
        )
        split_nodes.append((node_id, child_ids))
        operation_rows.append(
            {
                "slice_id": slice_id,
                "op_type": "split",
                "source_id": node_id,
                "target_id": dominant_child,
                "score": float(best[0]),
                "created_children": child_ids,
            }
        )

    return {"split_nodes": split_nodes, "operation_rows": operation_rows}


def update_lifecycle(
    taxonomy: Taxonomy,
    lifecycle_cfg: LifecycleConfig,
    slice_id: str,
    support_totals: Dict[str, int],
) -> Dict[str, Any]:
    promoted: List[str] = []
    deprecated: List[str] = []
    current_idx = slice_to_index(slice_id)

    for node in taxonomy.nodes.values():
        if node.level != "argument":
            continue
        if node.status == "deprecated":
            continue

        if node.status == "candidate":
            seen_span = 0
            if node.first_seen_slice and node.last_seen_slice:
                seen_span = slice_to_index(node.last_seen_slice) - slice_to_index(node.first_seen_slice) + 1
            if (
                seen_span >= lifecycle_cfg.promote_min_slices
                or support_totals.get(node.node_id, 0) >= lifecycle_cfg.promote_min_support
            ):
                node.status = "active"
                promoted.append(node.node_id)

        if node.last_seen_slice:
            stale_for = current_idx - slice_to_index(node.last_seen_slice)
            if stale_for >= lifecycle_cfg.stale_after_slices:
                node.status = "deprecated"
                deprecated.append(node.node_id)

    operation_rows: List[Dict[str, Any]] = []
    for nid in promoted:
        operation_rows.append(
            {
                "slice_id": slice_id,
                "op_type": "promote",
                "source_id": nid,
                "target_id": nid,
                "score": None,
            }
        )
    for nid in deprecated:
        operation_rows.append(
            {
                "slice_id": slice_id,
                "op_type": "deprecate",
                "source_id": nid,
                "target_id": nid,
                "score": None,
            }
        )
    return {"promoted": promoted, "deprecated": deprecated, "operation_rows": operation_rows}
