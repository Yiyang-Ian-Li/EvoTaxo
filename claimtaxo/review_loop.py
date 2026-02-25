from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from action_schema import validate_refined_action_executable
from apply_ops import apply_refined_actions
from cluster import cluster_group, group_key, semantic_text
from config import PipelineConfig
from embeddings import Embedder
from io_sinks import RunSinks
from llm import LLMClient
from propose_llm import propose_post_actions
from review_llm import repair_final_action_candidate, review_action_cluster, review_final_action_pool
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


def _node_path_names(taxonomy: Taxonomy, node_id: Optional[str]) -> List[str]:
    if not node_id or node_id not in taxonomy.nodes:
        return []
    out: List[str] = []
    cur: Optional[str] = node_id
    while cur is not None and cur in taxonomy.nodes:
        n = taxonomy.nodes[cur]
        out.append(n.name)
        cur = n.parent_id
    out.reverse()
    return out


def _format_refined_actions_readable(taxonomy: Taxonomy, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        action_type = str(a.get("action_type", "")).strip()
        target = _node_path_names(taxonomy, a.get("objective_node_id"))
        sem = a.get("semantic_payload", {}) if isinstance(a.get("semantic_payload", {}), dict) else {}
        row: Dict[str, Any] = {"action_type": action_type, "target_path": target}
        if action_type == "add_child":
            row["child_name"] = str(sem.get("child_name", "")).strip()
            row["child_level"] = str(sem.get("child_level", "")).strip()
        elif action_type == "add_path":
            nodes = sem.get("nodes", []) if isinstance(sem.get("nodes", []), list) else []
            row["path_nodes"] = [
                {"name": str(x.get("name", "")).strip(), "level": str(x.get("level", "")).strip()}
                for x in nodes
                if isinstance(x, dict)
            ]
        elif action_type == "update_cmb":
            new_cmb = sem.get("new_cmb", {}) if isinstance(sem.get("new_cmb", {}), dict) else {}
            row["new_definition"] = str(new_cmb.get("definition", "")).strip()
        out.append(row)
    return out


def _taxonomy_nested_snapshot(taxonomy: Taxonomy) -> Dict[str, Any]:
    root = taxonomy.nodes[taxonomy.root_id]
    topics: List[Dict[str, Any]] = []
    topic_ids = [cid for cid in root.children if cid in taxonomy.nodes and taxonomy.nodes[cid].level == "topic"]
    for tid in sorted(topic_ids, key=lambda x: (taxonomy.nodes[x].name.lower(), x)):
        t = taxonomy.nodes[tid]
        topic_row: Dict[str, Any] = {
            "name": t.name,
            "definition": t.cmb.definition,
            "subtopics": [],
            "claims": [],
        }
        subtopic_ids = [cid for cid in t.children if cid in taxonomy.nodes and taxonomy.nodes[cid].level == "subtopic"]
        for sid in sorted(subtopic_ids, key=lambda x: (taxonomy.nodes[x].name.lower(), x)):
            s = taxonomy.nodes[sid]
            claim_ids = [cid for cid in s.children if cid in taxonomy.nodes and taxonomy.nodes[cid].level == "claim"]
            topic_row["subtopics"].append(
                {
                    "name": s.name,
                    "definition": s.cmb.definition,
                    "claims": [
                        {"name": taxonomy.nodes[cid].name, "definition": taxonomy.nodes[cid].cmb.definition}
                        for cid in sorted(claim_ids, key=lambda x: (taxonomy.nodes[x].name.lower(), x))
                    ],
                }
            )
        direct_claim_ids = [cid for cid in t.children if cid in taxonomy.nodes and taxonomy.nodes[cid].level == "claim"]
        topic_row["claims"] = [
            {"name": taxonomy.nodes[cid].name, "definition": taxonomy.nodes[cid].cmb.definition}
            for cid in sorted(direct_claim_ids, key=lambda x: (taxonomy.nodes[x].name.lower(), x))
        ]
        topics.append(topic_row)
    return {"root": {"name": root.name, "topics": topics}}


@dataclass
class WindowLoopResult:
    action_proposals: List[Dict[str, Any]]
    pending_ids: set[str]
    node_post_links: List[Dict[str, Any]]
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
    set_node_total = 0
    skip_post_total = 0
    new_props_total = 0

    claim_ids: List[str] = []
    claim_vecs = np.zeros((0, 1))
    taxonomy_ctx_cached: Optional[Dict[str, Any]] = _proposal_taxonomy_context(taxonomy) if llm.available() else None
    taxonomy_dirty = True

    def _append_proposal_log(record: Dict[str, Any]) -> None:
        row = dict(record)
        row["ts"] = now_ts()
        sinks.action_proposals.append(row)

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
            "semantic_payload": {},
        }

    def _run_review_batch(batch_id: str, last_window_id: str) -> None:
        nonlocal taxonomy_dirty
        pending_records = [p for p in action_proposals if p["proposal_id"] in pending_ids]
        if not pending_records:
            return

        direct_records = [p for p in pending_records if p.get("action_type") in {"set_node", "skip_post"}]
        applied_direct_ids: set[str] = set()
        for rec in direct_records:
            action = _proposal_to_action(rec)
            apply_refined_actions(
                taxonomy=taxonomy,
                refined_actions=[action],
                cluster_proposals=[rec],
                window_id=str(rec.get("window_id", last_window_id)),
                assignment_rows=sinks.assignment,
                node_post_links=node_post_links,
                logger=logger,
                taxonomy_updates=None,
            )
            applied_direct_ids.add(str(rec["proposal_id"]))
            proposal_map[str(rec["proposal_id"])]["status"] = "applied"
        for pid in applied_direct_ids:
            pending_ids.discard(pid)

        structural_records = [
            p
            for p in pending_records
            if p["proposal_id"] in pending_ids and p.get("action_type") in {"add_child", "add_path", "update_cmb"}
        ]
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

        approved_candidates: List[Dict[str, Any]] = []
        for clusters in (sem_clusters, tmp_clusters):
            for c in clusters:
                c["is_high_quality"] = is_high_quality(c, cfg)
                pids_for_examples = list(c.get("proposal_ids", []))
                examples = []
                for pid in pids_for_examples[: min(3, len(pids_for_examples))]:
                    rec = proposal_map.get(pid, {})
                    examples.append(
                        {
                            "proposal_id": pid,
                            "post_id": rec.get("post_id"),
                            "action_type": rec.get("action_type"),
                            "action_explanation": str(rec.get("action_explanation", ""))[:240],
                            "post_summary": str(rec.get("post_summary", ""))[:240],
                        }
                    )
                overview_debug_row = {
                    "ts": now_ts(),
                    "batch_id": batch_id,
                    "window_id": str(c.get("window_id", last_window_id)),
                    "cluster_id": c["cluster_id"],
                    "cluster_mode": c["cluster_mode"],
                    "action_type": c.get("action_type"),
                    "objective_node_id": c.get("objective_node_id"),
                    "size": int(c.get("size", 0)),
                    "quality": c.get("quality", {}),
                    "is_high_quality": bool(c.get("is_high_quality", False)),
                    "proposal_ids": pids_for_examples,
                    "centroid_proposal_ids": list(c.get("centroid_proposal_ids", [])),
                    "examples": examples,
                }
                sinks.clusters_overview.append(
                    {
                        "ts": overview_debug_row["ts"],
                        "batch_id": batch_id,
                        "window_id": str(c.get("window_id", last_window_id)),
                        "cluster_mode": c["cluster_mode"],
                        "action_type": c.get("action_type"),
                        "objective_node_path": _node_path_names(taxonomy, c.get("objective_node_id")),
                        "size": int(c.get("size", 0)),
                        "quality": c.get("quality", {}),
                        "is_high_quality": bool(c.get("is_high_quality", False)),
                        "proposal_count": len(pids_for_examples),
                        "samples": [
                            {
                                "post_id": x.get("post_id"),
                                "action_type": x.get("action_type"),
                                "action_explanation": x.get("action_explanation", ""),
                                "post_summary": x.get("post_summary", ""),
                            }
                            for x in examples
                        ],
                    }
                )

                if not c["is_high_quality"]:
                    continue
                pids = [pid for pid in pids_for_examples if pid in pending_ids]
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
                    max_review_post_words=cfg.max_post_words,
                    model_override=cfg.llm.later_stage_model,
                )
                while review["decision"] == "approve" and not review.get("refined_actions"):
                    review = review_action_cluster(
                        llm=llm,
                        taxonomy=taxonomy,
                        root_topic=cfg.root_topic,
                        window_id=str(c_payload.get("window_id", last_window_id)),
                        cluster_record=c_payload,
                        proposal_records=records,
                        max_parse_attempts=cfg.llm.max_parse_attempts,
                        max_review_examples=cfg.review_max_examples,
                        max_review_post_words=cfg.max_post_words,
                        model_override=cfg.llm.later_stage_model,
                    )

                decision_debug_row = {
                    "ts": now_ts(),
                    "stage": "cluster_review",
                    "batch_id": batch_id,
                    "window_id": str(c.get("window_id", last_window_id)),
                    "cluster_id": c["cluster_id"],
                    "cluster_mode": c["cluster_mode"],
                    "decision": review["decision"],
                    "reason": review.get("reason", ""),
                    "proposal_ids": pids,
                    "review_refined_actions": review.get("refined_actions", []),
                }
                sinks.cluster_decisions.append(
                    {
                        "ts": decision_debug_row["ts"],
                        "stage": "cluster_review",
                        "batch_id": batch_id,
                        "window_id": str(c.get("window_id", last_window_id)),
                        "cluster_mode": c["cluster_mode"],
                        "action_type": c.get("action_type"),
                        "objective_node_path": _node_path_names(taxonomy, c.get("objective_node_id")),
                        "decision": review["decision"],
                        "proposal_count": len(pids),
                        "review_refined_actions": _format_refined_actions_readable(taxonomy, review.get("refined_actions", [])),
                        "reason": review.get("reason", ""),
                    }
                )
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
            model_override=cfg.llm.later_stage_model,
        )

        consumed: set[str] = set()
        applied_ids: set[str] = set(applied_direct_ids)
        resolved_cluster_ids: set[str] = set()
        for item in selected:
            idx = int(item["candidate_index"])
            cand = approved_candidates[idx]
            final_justification = str(item.get("justification", "")).strip()
            pids = [pid for pid in cand["proposal_ids"] if pid in pending_ids]
            if not pids:
                continue
            if any(pid in consumed for pid in pids):
                decision_debug_row = {
                    "ts": now_ts(),
                    "stage": "final_arbitration",
                    "batch_id": batch_id,
                    "window_id": str(cand.get("window_id", last_window_id)),
                    "cluster_id": cand["cluster_id"],
                    "cluster_mode": cand["cluster_mode"],
                    "decision": "defer",
                    "final_status": "conflict_skip",
                    "reason": "proposal_ids_already_consumed",
                    "proposal_ids": pids,
                    "final_actions": [],
                }
                sinks.cluster_decisions.append(
                    {
                        "ts": decision_debug_row["ts"],
                        "stage": "final_arbitration",
                        "batch_id": batch_id,
                        "window_id": str(cand.get("window_id", last_window_id)),
                        "cluster_mode": cand["cluster_mode"],
                        "action_type": cand.get("action_type"),
                        "objective_node_path": _node_path_names(taxonomy, cand.get("objective_node_id")),
                        "decision": "defer",
                        "final_status": "conflict_skip",
                        "proposal_count": len(pids),
                        "final_actions": [],
                        "reason": "proposal_ids_already_consumed",
                        "justification": final_justification,
                    }
                )
                resolved_cluster_ids.add(cand["cluster_id"])
                continue

            refined_actions = item.get("refined_actions", cand["refined_actions"]) or []
            valid_actions: List[Dict[str, Any]] = []
            invalid_reasons: List[str] = []
            while not valid_actions:
                if not refined_actions:
                    refined_actions = repair_final_action_candidate(
                        llm=llm,
                        taxonomy=taxonomy,
                        root_topic=cfg.root_topic,
                        batch_id=batch_id,
                        candidate=cand,
                        invalid_reason=invalid_reasons[0] if invalid_reasons else "missing_refined_actions",
                        max_parse_attempts=cfg.llm.max_parse_attempts,
                        model_override=cfg.llm.later_stage_model,
                    )

                invalid_reasons = []
                valid_actions = []
                for ra in refined_actions:
                    ok, reason = validate_refined_action_executable(ra, taxonomy)
                    if ok:
                        valid_actions.append(ra)
                    else:
                        invalid_reasons.append(reason)

                if valid_actions:
                    break

                refined_actions = repair_final_action_candidate(
                    llm=llm,
                    taxonomy=taxonomy,
                    root_topic=cfg.root_topic,
                    batch_id=batch_id,
                    candidate=cand,
                    invalid_reason=invalid_reasons[0] if invalid_reasons else "all_invalid_refined_actions",
                    max_parse_attempts=cfg.llm.max_parse_attempts,
                    model_override=cfg.llm.later_stage_model,
                )

            if not valid_actions:
                for pid in pids:
                    proposal_map[pid]["status"] = "pending"
                decision_debug_row = {
                    "ts": now_ts(),
                    "stage": "final_arbitration",
                    "batch_id": batch_id,
                    "window_id": str(cand.get("window_id", last_window_id)),
                    "cluster_id": cand["cluster_id"],
                    "cluster_mode": cand["cluster_mode"],
                    "decision": "defer",
                    "final_status": "invalid_deferred",
                    "reason": invalid_reasons[0] if invalid_reasons else "missing_refined_actions",
                    "proposal_ids": pids,
                    "final_actions": [],
                }
                sinks.cluster_decisions.append(
                    {
                        "ts": decision_debug_row["ts"],
                        "stage": "final_arbitration",
                        "batch_id": batch_id,
                        "window_id": str(cand.get("window_id", last_window_id)),
                        "cluster_mode": cand["cluster_mode"],
                        "action_type": cand.get("action_type"),
                        "objective_node_path": _node_path_names(taxonomy, cand.get("objective_node_id")),
                        "decision": "defer",
                        "final_status": "invalid_deferred",
                        "proposal_count": len(pids),
                        "final_actions": [],
                        "reason": invalid_reasons[0] if invalid_reasons else "missing_refined_actions",
                        "justification": final_justification,
                    }
                )
                resolved_cluster_ids.add(cand["cluster_id"])
                continue

            apply_refined_actions(
                taxonomy=taxonomy,
                refined_actions=valid_actions,
                cluster_proposals=[proposal_map[pid] for pid in pids],
                window_id=str(cand.get("window_id", last_window_id)),
                assignment_rows=sinks.assignment,
                node_post_links=node_post_links,
                logger=logger,
                taxonomy_updates=None,
            )
            for pid in pids:
                consumed.add(pid)
                applied_ids.add(pid)
                proposal_map[pid]["status"] = "applied"
            decision_debug_row = {
                "ts": now_ts(),
                "stage": "final_arbitration",
                "batch_id": batch_id,
                "window_id": str(cand.get("window_id", last_window_id)),
                "cluster_id": cand["cluster_id"],
                "cluster_mode": cand["cluster_mode"],
                "decision": "approve",
                "final_status": "applied",
                "proposal_ids": pids,
                "final_actions": valid_actions,
            }
            sinks.cluster_decisions.append(
                {
                    "ts": decision_debug_row["ts"],
                    "stage": "final_arbitration",
                    "batch_id": batch_id,
                    "window_id": str(cand.get("window_id", last_window_id)),
                    "cluster_mode": cand["cluster_mode"],
                    "action_type": cand.get("action_type"),
                    "objective_node_path": _node_path_names(taxonomy, cand.get("objective_node_id")),
                    "decision": "approve",
                    "final_status": "applied",
                    "proposal_count": len(pids),
                    "final_actions": _format_refined_actions_readable(taxonomy, valid_actions),
                    "reason": "",
                    "justification": final_justification,
                }
            )
            resolved_cluster_ids.add(cand["cluster_id"])

        for cand in approved_candidates:
            cid = str(cand.get("cluster_id", ""))
            if cid in resolved_cluster_ids:
                continue
            pids = [pid for pid in cand["proposal_ids"] if pid in pending_ids]
            if not pids:
                continue
            decision_debug_row = {
                "ts": now_ts(),
                "stage": "final_arbitration",
                "batch_id": batch_id,
                "window_id": str(cand.get("window_id", last_window_id)),
                "cluster_id": cid,
                "cluster_mode": cand.get("cluster_mode"),
                "decision": "defer",
                "final_status": "not_selected_in_pool",
                "reason": "not_selected_by_final_pool",
                "proposal_ids": pids,
                "final_actions": [],
            }
            sinks.cluster_decisions.append(
                {
                    "ts": decision_debug_row["ts"],
                    "stage": "final_arbitration",
                    "batch_id": batch_id,
                    "window_id": str(cand.get("window_id", last_window_id)),
                    "cluster_mode": cand.get("cluster_mode"),
                    "action_type": cand.get("action_type"),
                    "objective_node_path": _node_path_names(taxonomy, cand.get("objective_node_id")),
                    "decision": "defer",
                    "final_status": "not_selected_in_pool",
                    "proposal_count": len(pids),
                    "final_actions": [],
                    "reason": "not_selected_by_final_pool",
                    "justification": "",
                }
            )

        for pid in applied_ids:
            pending_ids.discard(pid)
        if applied_ids:
            taxonomy_dirty = True
        sinks.taxonomy_after_clustering.append(
            {
                "ts": now_ts(),
                "batch_id": batch_id,
                "window_id": last_window_id,
                "taxonomy": _taxonomy_nested_snapshot(taxonomy),
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
            node_post_links.append(
                {
                    "post_id": post_id,
                    "node_id": best_node_id,
                    "timestamp": ts_iso,
                    "window_id": window_id,
                    "source": "direct_high_sim",
                }
            )
            pbar.set_postfix(
                mapped=mapped_direct_total,
                pending=len(pending_ids),
                set_node=set_node_total,
                skip_post=skip_post_total,
            )
            continue

        unmapped_total += 1
        actions = propose_post_actions(
            llm=llm,
            taxonomy=taxonomy,
            root_topic=cfg.root_topic,
            post_text=text,
            post_id=post_id,
            window_id=window_id,
            max_parse_attempts=cfg.llm.max_parse_attempts,
            taxonomy_ctx=taxonomy_ctx_cached,
        )

        has_applied_set_node = False
        for a in actions:
            action_type = str(a.get("action_type", ""))
            if action_type == "set_node":
                set_node_total += 1
            elif action_type == "skip_post":
                skip_post_total += 1

            pid = str(uuid.uuid4())
            record = {
                "proposal_id": pid,
                "post_id": post_id,
                "timestamp": ts_iso,
                "timestamp_epoch": float(row["timestamp_epoch"]),
                "window_id": window_id,
                "post_title": str(row.get(cfg.title_col, "")).strip() if cfg.title_col else "",
                "post_text": str(row.get(cfg.text_col, "")).strip(),
                "action_type": a["action_type"],
                "objective_node_id": a.get("objective_node_id"),
                "action_explanation": a.get("action_explanation", ""),
                "post_summary": a.get("post_summary", ""),
                "status": "pending",
                "cluster_ids": {"semantic": None, "temporal": None},
            }
            action_proposals.append(record)
            proposal_map[pid] = record
            new_props_total += 1
            _append_proposal_log(record)

            if record["action_type"] == "set_node":
                objective_node_id = record.get("objective_node_id")
                set_node_valid = objective_node_id in taxonomy.nodes and objective_node_id != taxonomy.root_id if objective_node_id is not None else False
                apply_refined_actions(
                    taxonomy=taxonomy,
                    refined_actions=[_proposal_to_action(record)],
                    cluster_proposals=[record],
                    window_id=window_id,
                    assignment_rows=sinks.assignment,
                    node_post_links=node_post_links,
                    logger=logger,
                    taxonomy_updates=None,
                    log_set_node=False,
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
        pbar.set_postfix(
            mapped=mapped_direct_total,
            pending=len(pending_ids),
            set_node=set_node_total,
            skip_post=skip_post_total,
        )

        if posts_since_last_batch >= review_interval:
            batch_counter += 1
            batch_id = f"batch_{batch_counter:04d}"
            _run_review_batch(batch_id=batch_id, last_window_id=last_window_id)
            posts_since_last_batch = 0
    pbar.close()

    if pending_ids:
        batch_counter += 1
        batch_id = f"batch_{batch_counter:04d}"
        _run_review_batch(batch_id=batch_id, last_window_id=last_window_id)

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
        windows=windows,
    )
