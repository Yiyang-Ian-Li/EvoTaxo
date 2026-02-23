from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from apply_ops import apply_refined_actions
from cluster import cluster_group, group_key, semantic_text
from config import PipelineConfig
from embeddings import Embedder
from io_sinks import RunSinks
from llm import LLMClient
from propose_llm import propose_post_actions
from review_llm import repair_final_action_candidate, review_action_cluster, review_final_action_pool
from action_schema import validate_refined_action_executable
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


def _proposal_taxonomy_context(taxonomy: Taxonomy) -> Dict[str, Any]:
    return {
        "root_id": taxonomy.root_id,
        "root_name": taxonomy.nodes[taxonomy.root_id].name if taxonomy.root_id in taxonomy.nodes else "ROOT",
        "root_level": taxonomy.nodes[taxonomy.root_id].level if taxonomy.root_id in taxonomy.nodes else "root",
        "node_count": len(taxonomy.nodes),
        "nodes": [
            {
                "node_id": n.node_id,
                "name": n.name,
                "level": n.level,
                "parent_id": n.parent_id,
                "children": n.children,
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
    review_interval = max(1, int(cfg.review_batch_every_n_posts))
    logger.info("Processing posts=%d with review interval every %d new posts", len(df), review_interval)
    ordered_df = df.sort_values(cfg.timestamp_col).reset_index(drop=True)
    post_vecs = embedder.encode(ordered_df["_text"].tolist()) if len(ordered_df) else np.zeros((0, 1))
    proposal_map: Dict[str, Dict[str, Any]] = {}
    batch_counter = 0
    mapped_direct_total = 0
    unmapped_total = 0
    new_props_total = 0

    claim_ids: List[str] = []
    claim_vecs = np.zeros((0, 1))
    taxonomy_ctx_cached: Optional[Dict[str, Any]] = _proposal_taxonomy_context(taxonomy) if llm.available() else None
    taxonomy_dirty = True

    def _refresh_claim_cache() -> None:
        nonlocal claim_ids, claim_vecs, taxonomy_ctx_cached, taxonomy_dirty
        claim_ids = taxonomy.claim_node_ids()
        if claim_ids:
            claim_texts = [taxonomy.node_text(x) for x in claim_ids]
            claim_vecs = embedder.encode(claim_texts)
        else:
            claim_vecs = np.zeros((0, 1))
        taxonomy_ctx_cached = _proposal_taxonomy_context(taxonomy) if llm.available() else None
        taxonomy_dirty = False

    def _proposal_to_action(p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action_type": p.get("action_type"),
            "objective_node_id": p.get("objective_node_id"),
            "objective": p.get("objective", ""),
            "semantic_payload": {},
            "confidence": p.get("confidence", 0.0),
            "reasoning_short": p.get("reasoning_short", ""),
        }

    def _run_review_batch(batch_id: str, trigger_reason: str, last_window_id: str, posts_since_last_batch: int) -> None:
        nonlocal taxonomy_dirty
        pending_records = [p for p in action_proposals if p["proposal_id"] in pending_ids]
        if not pending_records:
            return

        sinks.event_log.append(
            {
                "ts": now_ts(),
                "event": "batch_start",
                "batch_id": batch_id,
                "trigger_reason": trigger_reason,
                "window_id": last_window_id,
                "pending_before": int(len(pending_ids)),
                "posts_since_last_batch": int(posts_since_last_batch),
                "taxonomy_nodes": len(taxonomy.nodes),
            }
        )

        # Directly apply simple no-structure actions; keep structural actions for clustering/review.
        direct_records = [p for p in pending_records if p.get("action_type") in {"set_node", "skip_post"}]
        applied_direct_ids: set[str] = set()
        for rec in direct_records:
            action = _proposal_to_action(rec)
            apply_refined_actions(
                taxonomy=taxonomy,
                refined_actions=[action],
                cluster_proposals=[rec],
                window_id=str(rec.get("window_id", last_window_id)),
                taxonomy_ops=sinks.taxonomy_ops,
                assignment_rows=sinks.assignment,
                node_post_links=node_post_links,
                logger=logger,
                event_log=sinks.event_log,
            )
            applied_direct_ids.add(str(rec["proposal_id"]))
            proposal_map[str(rec["proposal_id"])]["status"] = "applied"
        for pid in applied_direct_ids:
            pending_ids.discard(pid)

        structural_records = [p for p in pending_records if p["proposal_id"] in pending_ids and p.get("action_type") in {"add_child", "add_path", "update_cmb"}]
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for p in structural_records:
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
                    "batch_id": batch_id,
                    "window_id": last_window_id,
                    "group_key": {
                        "action_type": props[0].get("action_type"),
                        "objective_node_id": props[0].get("objective_node_id"),
                    },
                    "group_size": len(props),
                    "semantic_clusters": len(srows),
                    "temporal_clusters": len(trows),
                }
            )

        for c in sem_clusters:
            c["is_high_quality"] = is_high_quality(c, cfg)
            sinks.semantic_clusters.append(c)
        for c in tmp_clusters:
            c["is_high_quality"] = is_high_quality(c, cfg)
            sinks.temporal_clusters.append(c)

        approved_candidates: List[Dict[str, Any]] = []
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
                    window_id=str(c_payload.get("window_id", last_window_id)),
                    cluster_record=c_payload,
                    proposal_records=records,
                    max_parse_attempts=cfg.llm.max_parse_attempts,
                    max_review_examples=cfg.review_max_examples,
                    max_review_post_chars=cfg.review_max_post_chars,
                    trace=sinks.llm_trace,
                )
                # If approved but empty actions, ask LLM once more to regenerate.
                if review["decision"] == "approve" and not review.get("refined_actions"):
                    regen = review_action_cluster(
                        llm=llm,
                        taxonomy=taxonomy,
                        root_topic=cfg.root_topic,
                        window_id=str(c_payload.get("window_id", last_window_id)),
                        cluster_record=c_payload,
                        proposal_records=records,
                        max_parse_attempts=cfg.llm.max_parse_attempts,
                        max_review_examples=cfg.review_max_examples,
                        max_review_post_chars=cfg.review_max_post_chars,
                        trace=sinks.llm_trace,
                    )
                    sinks.event_log.append(
                        {
                            "ts": now_ts(),
                            "event": "cluster_review_regen",
                            "batch_id": batch_id,
                            "window_id": last_window_id,
                            "cluster_id": c["cluster_id"],
                            "cluster_mode": c["cluster_mode"],
                            "regen_decision": regen["decision"],
                            "regen_refined_action_count": len(regen.get("refined_actions", [])),
                        }
                    )
                    if regen["decision"] == "approve" and regen.get("refined_actions"):
                        review = regen
                sinks.cluster_reviews.append(
                    {
                        "ts": now_ts(),
                        "window_id": str(c.get("window_id", last_window_id)),
                        "cluster_id": c["cluster_id"],
                        "cluster_mode": c["cluster_mode"],
                        "decision": review["decision"],
                        "reason": review.get("reason", ""),
                        "refined_actions": review.get("refined_actions", []),
                        "proposal_ids": pids,
                    }
                )
                sinks.event_log.append(
                    {
                        "ts": now_ts(),
                        "event": "cluster_review",
                        "batch_id": batch_id,
                        "window_id": last_window_id,
                        "cluster_id": c["cluster_id"],
                        "cluster_mode": c["cluster_mode"],
                        "decision": review["decision"],
                        "proposal_count": len(pids),
                        "refined_action_count": len(review.get("refined_actions", [])),
                    }
                )
                if c["cluster_mode"] == "temporal":
                    burst_row = {
                        "window_id": str(c.get("window_id", last_window_id)),
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
                if review["decision"] == "approve" and review.get("refined_actions"):
                    approved_candidates.append(
                        {
                            "cluster_id": c["cluster_id"],
                            "cluster_mode": c["cluster_mode"],
                            "action_type": c.get("action_type"),
                            "objective_node_id": c.get("objective_node_id"),
                            "window_id": str(c.get("window_id", last_window_id)),
                            "quality": c.get("quality", {}),
                            "proposal_ids": pids,
                            "records": records,
                            "refined_actions": list(review.get("refined_actions", [])),
                        }
                    )

        selected = review_final_action_pool(
            llm=llm,
            taxonomy=taxonomy,
            root_topic=cfg.root_topic,
            batch_id=batch_id,
            candidates=approved_candidates,
            max_parse_attempts=cfg.llm.max_parse_attempts,
            trace=sinks.llm_trace,
        )
        consumed: set[str] = set()
        applied_ids: set[str] = set(applied_direct_ids)
        for item in selected:
            idx = int(item["candidate_index"])
            cand = approved_candidates[idx]
            pids = [pid for pid in cand["proposal_ids"] if pid in pending_ids]
            if not pids:
                continue
            if any(pid in consumed for pid in pids):
                sinks.event_log.append(
                    {
                        "ts": now_ts(),
                        "event": "final_arbitration_conflict_skip",
                        "batch_id": batch_id,
                        "cluster_id": cand["cluster_id"],
                        "proposal_ids": pids,
                    }
                )
                continue
            refined_actions = item.get("refined_actions", cand["refined_actions"]) or []
            if not refined_actions:
                refined_actions = repair_final_action_candidate(
                    llm=llm,
                    taxonomy=taxonomy,
                    root_topic=cfg.root_topic,
                    batch_id=batch_id,
                    candidate=cand,
                    invalid_reason="missing_refined_actions",
                    max_parse_attempts=cfg.llm.max_parse_attempts,
                    trace=sinks.llm_trace,
                )

            valid_actions: List[Dict[str, Any]] = []
            invalid_reasons: List[str] = []
            for ra in refined_actions:
                ok, reason = validate_refined_action_executable(ra, taxonomy)
                if ok:
                    valid_actions.append(ra)
                else:
                    invalid_reasons.append(reason)

            # If all invalid, ask LLM once more to regenerate a valid action set.
            if not valid_actions:
                repaired_actions = repair_final_action_candidate(
                    llm=llm,
                    taxonomy=taxonomy,
                    root_topic=cfg.root_topic,
                    batch_id=batch_id,
                    candidate=cand,
                    invalid_reason=invalid_reasons[0] if invalid_reasons else "all_invalid_refined_actions",
                    max_parse_attempts=cfg.llm.max_parse_attempts,
                    trace=sinks.llm_trace,
                )
                for ra in repaired_actions:
                    ok, reason = validate_refined_action_executable(ra, taxonomy)
                    if ok:
                        valid_actions.append(ra)
                    else:
                        invalid_reasons.append(reason)

            if not valid_actions:
                sinks.event_log.append(
                    {
                        "ts": now_ts(),
                        "event": "final_action_invalid_deferred",
                        "batch_id": batch_id,
                        "cluster_id": cand["cluster_id"],
                        "proposal_ids": pids,
                        "invalid_reason": invalid_reasons[0] if invalid_reasons else "missing_refined_actions",
                    }
                )
                for pid in pids:
                    proposal_map[pid]["status"] = "pending"
                continue

            if invalid_reasons:
                sinks.event_log.append(
                    {
                        "ts": now_ts(),
                        "event": "final_action_partially_invalid_dropped",
                        "batch_id": batch_id,
                        "cluster_id": cand["cluster_id"],
                        "proposal_ids": pids,
                        "invalid_reason": invalid_reasons[0],
                        "valid_action_count": len(valid_actions),
                    }
                )

            apply_refined_actions(
                taxonomy=taxonomy,
                refined_actions=valid_actions,
                cluster_proposals=[proposal_map[pid] for pid in pids],
                window_id=str(cand.get("window_id", last_window_id)),
                taxonomy_ops=sinks.taxonomy_ops,
                assignment_rows=sinks.assignment,
                node_post_links=node_post_links,
                logger=logger,
                event_log=sinks.event_log,
            )
            for pid in pids:
                consumed.add(pid)
                applied_ids.add(pid)
                proposal_map[pid]["status"] = "applied"

        for pid in applied_ids:
            pending_ids.discard(pid)
        if applied_ids:
            taxonomy_dirty = True

        sinks.window_summary.append(
            {
                "window_id": last_window_id,
                "batch_id": batch_id,
                "trigger_reason": trigger_reason,
                "last_window_id": last_window_id,
                "posts_since_last_batch": int(posts_since_last_batch),
                "pending_backlog": int(len(pending_ids)),
                "direct_applied": len(applied_direct_ids),
                "semantic_clusters": len(sem_clusters),
                "temporal_clusters": len(tmp_clusters),
                "approved_candidates": len(approved_candidates),
                "applied_proposals": len(applied_ids),
                "taxonomy_nodes": len(taxonomy.nodes),
            }
        )
        sinks.event_log.append(
            {
                "ts": now_ts(),
                "event": "batch_end",
                "batch_id": batch_id,
                "window_id": last_window_id,
                "pending_after": int(len(pending_ids)),
                "applied_proposals": int(len(applied_ids)),
                "taxonomy_nodes": len(taxonomy.nodes),
            }
        )

    if taxonomy_dirty:
        _refresh_claim_cache()

    pbar = tqdm(ordered_df.iterrows(), total=len(ordered_df), desc="posts", leave=False)
    posts_since_last_batch = 0
    last_window_id = windows[0] if windows else "INIT"
    for row_idx, row in pbar:
        if taxonomy_dirty:
            _refresh_claim_cache()
        post_id = str(row[cfg.id_col])
        text = row["_text"]
        ts = row[cfg.timestamp_col]
        ts_iso = ts.isoformat()
        window_id = str(row["window_id"])
        last_window_id = window_id
        posts_since_last_batch += 1

        if len(claim_ids) and claim_vecs.shape[0] > 0:
            vec = post_vecs[row_idx : row_idx + 1]
            sims = (vec @ claim_vecs.T) / (
                (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
                * (np.linalg.norm(claim_vecs, axis=1, keepdims=True).T + 1e-12)
            )
            best_idx = int(np.argmax(sims, axis=1)[0])
            best_sim = float(np.max(sims, axis=1)[0])
            best_node_id = claim_ids[best_idx] if 0 <= best_idx < len(claim_ids) else None
        else:
            best_idx = -1
            best_sim = 0.0
            best_node_id = None

        if best_idx >= 0 and best_sim >= cfg.high_sim_threshold:
            mapped_direct_total += 1
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
            node_post_links.append({"post_id": post_id, "node_id": best_node_id, "timestamp": ts_iso, "window_id": window_id, "source": "direct_high_sim"})
            pbar.set_postfix(mapped=mapped_direct_total, unmapped=unmapped_total, pending=len(pending_ids))
            continue

        unmapped_total += 1
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
            taxonomy_ctx=taxonomy_ctx_cached,
        )

        has_applied_set_node = False
        for a in actions:
            pid = str(uuid.uuid4())
            record = {
                "proposal_id": pid,
                "post_id": post_id,
                "timestamp": ts_iso,
                "timestamp_epoch": float(row["timestamp_epoch"]),
                "window_id": window_id,
                "action_type": a["action_type"],
                "objective_node_id": a.get("objective_node_id"),
                "objective": a.get("objective", ""),
                "action_explanation": a.get("action_explanation", ""),
                "post_summary": a.get("post_summary", ""),
                "confidence": a.get("confidence", 0.0),
                "reasoning_short": a.get("reasoning_short", ""),
                "status": "pending",
                "cluster_ids": {"semantic": None, "temporal": None},
            }
            action_proposals.append(record)
            proposal_map[pid] = record
            new_props_total += 1
            if record["action_type"] == "set_node":
                objective_node_id = record.get("objective_node_id")
                set_node_valid = objective_node_id in taxonomy.nodes if objective_node_id is not None else False
                apply_refined_actions(
                    taxonomy=taxonomy,
                    refined_actions=[_proposal_to_action(record)],
                    cluster_proposals=[record],
                    window_id=window_id,
                    taxonomy_ops=sinks.taxonomy_ops,
                    assignment_rows=sinks.assignment,
                    node_post_links=node_post_links,
                    logger=logger,
                    event_log=sinks.event_log,
                )
                record["status"] = "applied" if set_node_valid else "rejected"
                if record["status"] == "applied":
                    has_applied_set_node = True
            else:
                pending_ids.add(pid)

        if not has_applied_set_node:
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
        pbar.set_postfix(mapped=mapped_direct_total, unmapped=unmapped_total, pending=len(pending_ids))

        if posts_since_last_batch >= review_interval:
            batch_counter += 1
            batch_id = f"batch_{batch_counter:04d}"
            _run_review_batch(
                batch_id=batch_id,
                trigger_reason="post_interval",
                last_window_id=last_window_id,
                posts_since_last_batch=posts_since_last_batch,
            )
            posts_since_last_batch = 0
    pbar.close()

    if pending_ids:
        batch_counter += 1
        batch_id = f"batch_{batch_counter:04d}"
        _run_review_batch(
            batch_id=batch_id,
            trigger_reason="eof_flush",
            last_window_id=last_window_id,
            posts_since_last_batch=posts_since_last_batch,
        )

    logger.info(
        "Done processing posts. mapped_direct=%d unmapped=%d proposals=%d pending=%d batches=%d",
        mapped_direct_total,
        unmapped_total,
        new_props_total,
        len(pending_ids),
        batch_counter,
    )

    return WindowLoopResult(
        action_proposals=action_proposals,
        pending_ids=pending_ids,
        node_post_links=node_post_links,
        bursts=bursts,
        windows=windows,
    )
