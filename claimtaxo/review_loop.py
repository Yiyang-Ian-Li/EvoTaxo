from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from apply_ops import apply_refined_actions
from cluster import cluster_group, group_key, semantic_text
from config import PipelineConfig
from embeddings import Embedder
from io_sinks import RunSinks
from llm import LLMClient
from llm_ops import propose_post_actions, review_action_cluster
from mapping import map_posts_to_claims
from taxonomy import Taxonomy
from utils import now_ts


def is_high_quality(cluster_row: Dict[str, Any], cfg: PipelineConfig) -> bool:
    q = cluster_row.get("quality", {})
    if int(cluster_row.get("size", 0)) < cfg.min_cluster_size_review:
        return False
    if float(q.get("cohesion", 0.0)) < cfg.min_cohesion:
        return False
    if cluster_row.get("cluster_mode") == "temporal" and float(q.get("time_compactness", 0.0)) < cfg.min_time_compactness:
        return False
    return True


def _window_taxonomy_context(taxonomy: Taxonomy) -> Dict[str, Any]:
    return {
        "root_id": taxonomy.root_id,
        "node_count": len(taxonomy.nodes),
        "nodes": [
            {
                "node_id": n.node_id,
                "name": n.name,
                "level": n.level,
                "parent_id": n.parent_id,
                "children": n.children,
                "cmb": {
                    "definition": n.cmb.definition,
                    "include_terms": n.cmb.include_terms,
                    "exclude_terms": n.cmb.exclude_terms,
                    "examples": n.cmb.examples,
                },
            }
            for n in sorted(taxonomy.nodes.values(), key=lambda x: (x.level, x.name, x.node_id))
        ],
    }


@dataclass
class WindowLoopResult:
    action_proposals: List[Dict[str, Any]]
    pending_ids: set[str]
    node_post_links: List[Dict[str, Any]]
    bursts: List[Dict[str, Any]]
    windows: List[str]


def process_windows(
    cfg: PipelineConfig,
    df: pd.DataFrame,
    taxonomy: Taxonomy,
    llm: LLMClient,
    embedder: Embedder,
    sinks: RunSinks,
    logger: Any,
) -> WindowLoopResult:
    action_proposals: List[Dict[str, Any]] = []
    node_post_links: List[Dict[str, Any]] = []
    bursts: List[Dict[str, Any]] = []
    pending_ids: set[str] = set()

    windows = list(df["window_id"].drop_duplicates())
    logger.info("Processing windows=%d", len(windows))

    for window_id, wdf in df.groupby("window_id", sort=True):
        logger.info("Window %s: posts=%d", window_id, len(wdf))
        sinks.event_log.append(
            {
                "ts": now_ts(),
                "event": "window_start",
                "window_id": window_id,
                "post_count": int(len(wdf)),
                "pending_before": int(len(pending_ids)),
                "taxonomy_nodes": len(taxonomy.nodes),
            }
        )

        claim_ids = taxonomy.claim_node_ids()
        claim_texts = [taxonomy.node_text(x) for x in claim_ids]
        claim_vecs = embedder.encode(claim_texts) if claim_texts else np.zeros((0, 1))
        taxonomy_ctx_for_window = _window_taxonomy_context(taxonomy) if llm.available() else None

        ordered_wdf = wdf.sort_values(cfg.timestamp_col).reset_index(drop=True)
        post_vecs = embedder.encode(ordered_wdf["_text"].tolist()) if len(ordered_wdf) else np.zeros((0, 1))
        best_idx_arr, best_sim_arr, best_node_ids = map_posts_to_claims(post_vecs, claim_vecs, claim_ids)

        mapped_direct = 0
        unmapped = 0
        new_props = 0

        pbar = tqdm(ordered_wdf.iterrows(), total=len(ordered_wdf), desc=f"window {window_id}", leave=False)
        for row_idx, row in pbar:
            post_id = str(row[cfg.id_col])
            text = row["_text"]
            ts = row[cfg.timestamp_col]
            ts_iso = ts.isoformat()
            best_idx = int(best_idx_arr[row_idx]) if row_idx < len(best_idx_arr) else -1
            best_sim = float(best_sim_arr[row_idx]) if row_idx < len(best_sim_arr) else 0.0
            best_node_id = best_node_ids[row_idx] if row_idx < len(best_node_ids) else None

            if best_idx >= 0 and best_sim >= cfg.high_sim_threshold:
                mapped_direct += 1
                sinks.assignment.append(
                    {
                        "post_id": post_id,
                        "timestamp": ts_iso,
                        "window_id": window_id,
                        "node_id_at_time": best_node_id,
                        "canonical_node_id": best_node_id,
                        "similarity": round(best_sim, 6),
                        "mapping_mode": "direct_high_sim",
                    }
                )
                node_post_links.append(
                    {"post_id": post_id, "node_id": best_node_id, "timestamp": ts_iso, "window_id": window_id, "source": "direct_high_sim"}
                )
                pbar.set_postfix(mapped=mapped_direct, unmapped=unmapped, pending=len(pending_ids))
                sinks.event_log.append(
                    {
                        "ts": now_ts(),
                        "event": "post_mapped_direct",
                        "window_id": window_id,
                        "post_id": post_id,
                        "best_node_id": best_node_id,
                        "best_similarity": round(best_sim, 6),
                    }
                )
                continue

            unmapped += 1
            actions = propose_post_actions(
                llm=llm,
                taxonomy=taxonomy,
                root_topic=cfg.root_topic,
                post_text=text,
                post_id=post_id,
                window_id=window_id,
                best_candidate_node_id=best_node_id,
                best_similarity=best_sim,
                max_parse_attempts=cfg.llm.max_parse_attempts,
                trace=sinks.llm_trace,
                taxonomy_ctx=taxonomy_ctx_for_window,
            )
            sinks.event_log.append(
                {
                    "ts": now_ts(),
                    "event": "post_unmapped_llm_actions",
                    "window_id": window_id,
                    "post_id": post_id,
                    "best_node_id": best_node_id,
                    "best_similarity": round(best_sim, 6),
                    "action_count": len(actions),
                    "action_types": [str(a.get("action_type", "")) for a in actions],
                }
            )

            for a in actions:
                pid = str(uuid.uuid4())
                record = {
                    "proposal_id": pid,
                    "post_id": post_id,
                    "post_text": text,
                    "timestamp": ts_iso,
                    "timestamp_epoch": float(row["timestamp_epoch"]),
                    "window_id": window_id,
                    "action_type": a["action_type"],
                    "objective_node_id": a.get("objective_node_id"),
                    "objective": a.get("objective", ""),
                    "semantic_payload": a.get("semantic_payload", {}),
                    "confidence": a.get("confidence", 0.0),
                    "reasoning_short": a.get("reasoning_short", ""),
                    "status": "pending",
                    "cluster_ids": {"semantic": None, "temporal": None},
                }
                action_proposals.append(record)
                pending_ids.add(pid)
                new_props += 1
                sinks.event_log.append(
                    {
                        "ts": now_ts(),
                        "event": "proposal_added",
                        "window_id": window_id,
                        "proposal_id": pid,
                        "post_id": post_id,
                        "action_type": record["action_type"],
                        "objective_node_id": record["objective_node_id"],
                    }
                )

            sinks.assignment.append(
                {
                    "post_id": post_id,
                    "timestamp": ts_iso,
                    "window_id": window_id,
                    "node_id_at_time": None,
                    "canonical_node_id": None,
                    "similarity": round(best_sim, 6),
                    "mapping_mode": "unmapped",
                }
            )
            pbar.set_postfix(mapped=mapped_direct, unmapped=unmapped, pending=len(pending_ids))
        pbar.close()

        logger.info(
            "Window %s mapping complete: mapped_direct=%d unmapped=%d new_proposals=%d pending=%d",
            window_id,
            mapped_direct,
            unmapped,
            new_props,
            len(pending_ids),
        )

        pending_records = [p for p in action_proposals if p["proposal_id"] in pending_ids]
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for p in pending_records:
            if p.get("action_type") not in {"add_child", "add_path", "update_cmb"}:
                continue
            groups[group_key(p)].append(p)

        sem_clusters: List[Dict[str, Any]] = []
        tmp_clusters: List[Dict[str, Any]] = []
        for _, props in groups.items():
            if len(props) < 2:
                continue
            texts = [semantic_text(x) for x in props]
            emb = embedder.encode(texts)
            srows, trows = cluster_group(
                proposals=props,
                embeddings=emb,
                min_cluster_size=cfg.min_cluster_size_hdbscan,
                w_sem=cfg.temporal_w_sem,
                w_time=cfg.temporal_w_time,
            )
            sem_clusters.extend(srows)
            tmp_clusters.extend(trows)
            sinks.event_log.append(
                {
                    "ts": now_ts(),
                    "event": "cluster_group_built",
                    "window_id": window_id,
                    "group_key": {"action_type": props[0].get("action_type"), "objective_node_id": props[0].get("objective_node_id")},
                    "group_size": len(props),
                    "semantic_clusters": len(srows),
                    "temporal_clusters": len(trows),
                }
            )

        logger.info(
            "Window %s clustering: semantic_clusters=%d temporal_clusters=%d clusterable_groups=%d",
            window_id,
            len(sem_clusters),
            len(tmp_clusters),
            len(groups),
        )

        for c in sem_clusters:
            c["is_high_quality"] = is_high_quality(c, cfg)
            sinks.semantic_clusters.append(c)
        for c in tmp_clusters:
            c["is_high_quality"] = is_high_quality(c, cfg)
            sinks.temporal_clusters.append(c)

        proposal_map = {p["proposal_id"]: p for p in action_proposals}
        approved_this_window: set[str] = set()

        for clusters in (sem_clusters, tmp_clusters):
            for c in clusters:
                if not is_high_quality(c, cfg):
                    continue
                pids = [pid for pid in c["proposal_ids"] if pid in pending_ids]
                if not pids:
                    continue
                c_payload = dict(c)
                c_payload["proposal_ids"] = pids
                records = [proposal_map[x] for x in pids]

                review = review_action_cluster(
                    llm=llm,
                    taxonomy=taxonomy,
                    root_topic=cfg.root_topic,
                    window_id=window_id,
                    cluster_record=c_payload,
                    proposal_records=records,
                    max_parse_attempts=cfg.llm.max_parse_attempts,
                    max_review_examples=cfg.review_max_examples,
                    max_review_post_chars=cfg.review_max_post_chars,
                    trace=sinks.llm_trace,
                )

                review_row = {
                    "ts": now_ts(),
                    "window_id": window_id,
                    "cluster_id": c["cluster_id"],
                    "cluster_mode": c["cluster_mode"],
                    "decision": review["decision"],
                    "reason": review.get("reason", ""),
                    "refined_actions": review.get("refined_actions", []),
                    "proposal_ids": pids,
                }
                sinks.cluster_reviews.append(review_row)
                sinks.event_log.append(
                    {
                        "ts": now_ts(),
                        "event": "cluster_review",
                        "window_id": window_id,
                        "cluster_id": c["cluster_id"],
                        "cluster_mode": c["cluster_mode"],
                        "decision": review["decision"],
                        "proposal_count": len(pids),
                        "refined_action_count": len(review.get("refined_actions", [])),
                    }
                )

                if c["cluster_mode"] == "temporal":
                    burst_row = {
                        "window_id": window_id,
                        "cluster_id": c["cluster_id"],
                        "action_type": c["action_type"],
                        "objective_node_id": c["objective_node_id"],
                        "size": c["size"],
                        "quality": c["quality"],
                        "decision": review["decision"],
                        "refined_action_count": len(review.get("refined_actions", [])),
                        "refined_actions": review.get("refined_actions", []),
                    }
                    bursts.append(burst_row)
                    sinks.bursts.append(burst_row)

                if review["decision"] != "approve":
                    continue

                apply_refined_actions(
                    taxonomy=taxonomy,
                    refined_actions=review.get("refined_actions", []),
                    cluster_proposals=records,
                    window_id=window_id,
                    taxonomy_ops=sinks.taxonomy_ops,
                    assignment_rows=sinks.assignment,
                    node_post_links=node_post_links,
                    logger=logger,
                    event_log=sinks.event_log,
                )
                for pid in pids:
                    approved_this_window.add(pid)

        logger.info(
            "Window %s review/apply: approved_proposals=%d pending_after=%d",
            window_id,
            len(approved_this_window),
            len(pending_ids) - len(approved_this_window),
        )

        for pid in approved_this_window:
            pending_ids.discard(pid)
            proposal_map[pid]["status"] = "applied"

        sinks.window_summary.append(
            {
                "window_id": window_id,
                "posts": int(len(wdf)),
                "mapped_direct": mapped_direct,
                "unmapped": unmapped,
                "new_proposals": new_props,
                "pending_backlog": int(len(pending_ids)),
                "semantic_clusters": len(sem_clusters),
                "temporal_clusters": len(tmp_clusters),
                "approved_proposals": len(approved_this_window),
                "taxonomy_nodes": len(taxonomy.nodes),
            }
        )
        logger.info("Window %s done: taxonomy_nodes=%d pending_backlog=%d", window_id, len(taxonomy.nodes), len(pending_ids))
        sinks.event_log.append(
            {
                "ts": now_ts(),
                "event": "window_end",
                "window_id": window_id,
                "taxonomy_nodes": len(taxonomy.nodes),
                "pending_backlog": int(len(pending_ids)),
                "approved_proposals": len(approved_this_window),
            }
        )

    return WindowLoopResult(
        action_proposals=action_proposals,
        pending_ids=pending_ids,
        node_post_links=node_post_links,
        bursts=bursts,
        windows=windows,
    )
