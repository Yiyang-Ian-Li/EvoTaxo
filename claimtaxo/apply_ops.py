from __future__ import annotations

import logging
from typing import Any, Dict, List

from taxonomy import Taxonomy
from utils import now_ts


VALID_LEVELS = {"topic", "subtopic", "claim"}


def apply_refined_actions(
    taxonomy: Taxonomy,
    refined_actions: List[Dict[str, Any]],
    cluster_proposals: List[Dict[str, Any]],
    window_id: str,
    taxonomy_ops: Any,
    assignment_rows: Any,
    node_post_links: List[Dict[str, Any]],
    logger: logging.Logger,
    event_log: Any,
) -> None:
    proposal_post_ids = [str(x["post_id"]) for x in cluster_proposals]
    proposal_ts = {str(x["post_id"]): x.get("timestamp") for x in cluster_proposals}

    for action in refined_actions:
        action_type = action.get("action_type")
        objective_node_id = action.get("objective_node_id")
        sem = action.get("semantic_payload", {})
        op_base = {
            "ts": now_ts(),
            "window_id": window_id,
            "action_type": action_type,
            "objective_node_id": objective_node_id,
            "post_ids": proposal_post_ids,
            "semantic_payload": sem,
        }
        if action_type != "skip_post":
            logger.info("Applying refined action type=%s objective=%s posts=%d", action_type, objective_node_id, len(proposal_post_ids))

        if action_type == "set_node":
            if objective_node_id not in taxonomy.nodes:
                taxonomy_ops.append({**op_base, "op_type": "set_node", "op_result": "invalid_objective"})
                event_log.append({**op_base, "event": "apply_refined_action_skipped", "reason": "invalid_objective"})
                continue
            for pid in proposal_post_ids:
                ts = proposal_ts.get(pid)
                assignment_rows.append(
                    {
                        "post_id": pid,
                        "timestamp": ts,
                        "window_id": window_id,
                        "node_id_at_time": objective_node_id,
                        "canonical_node_id": objective_node_id,
                        "similarity": None,
                        "mapping_mode": "post_apply_refine",
                    }
                )
                node_post_links.append(
                    {"post_id": pid, "node_id": objective_node_id, "timestamp": ts, "window_id": window_id, "source": "set_node_refined"}
                )
            taxonomy_ops.append(
                {**op_base, "op_type": "set_node", "target_id": objective_node_id, "post_count": len(proposal_post_ids), "op_result": "applied"}
            )
            event_log.append({**op_base, "event": "apply_set_node", "target_id": objective_node_id, "post_count": len(proposal_post_ids)})

        elif action_type == "add_child":
            if objective_node_id not in taxonomy.nodes:
                taxonomy_ops.append({**op_base, "op_type": "add_child", "op_result": "invalid_objective"})
                event_log.append({**op_base, "event": "apply_refined_action_skipped", "reason": "invalid_objective"})
                continue
            child_name = str(sem.get("child_name", "")).strip()
            child_level = str(sem.get("child_level", "claim")).strip().lower()
            child_cmb = sem.get("child_cmb", {})
            if not child_name or child_level not in VALID_LEVELS:
                taxonomy_ops.append({**op_base, "op_type": "add_child", "op_result": "invalid_semantic"})
                event_log.append({**op_base, "event": "apply_refined_action_skipped", "reason": "invalid_semantic"})
                continue

            child_id = taxonomy.add_node(parent_id=objective_node_id, name=child_name, level=child_level, cmb=child_cmb, window_id=window_id)
            for pid in proposal_post_ids:
                ts = proposal_ts.get(pid)
                assignment_rows.append(
                    {
                        "post_id": pid,
                        "timestamp": ts,
                        "window_id": window_id,
                        "node_id_at_time": child_id,
                        "canonical_node_id": child_id,
                        "similarity": None,
                        "mapping_mode": "post_apply_refine_add_child",
                    }
                )
                node_post_links.append(
                    {"post_id": pid, "node_id": child_id, "timestamp": ts, "window_id": window_id, "source": "add_child_refined"}
                )
            taxonomy_ops.append(
                {
                    **op_base,
                    "op_type": "add_child",
                    "source_id": objective_node_id,
                    "target_id": child_id,
                    "name": child_name,
                    "child_level": child_level,
                    "post_count": len(proposal_post_ids),
                    "op_result": "applied",
                }
            )
            event_log.append({**op_base, "event": "apply_add_child", "target_id": child_id, "name": child_name})

        elif action_type == "add_path":
            if objective_node_id not in taxonomy.nodes:
                taxonomy_ops.append({**op_base, "op_type": "add_path", "op_result": "invalid_objective"})
                event_log.append({**op_base, "event": "apply_refined_action_skipped", "reason": "invalid_objective"})
                continue
            anchor_level = taxonomy.nodes[objective_node_id].level
            if anchor_level not in {"root", "topic"}:
                taxonomy_ops.append({**op_base, "op_type": "add_path", "op_result": "invalid_anchor_level"})
                event_log.append({**op_base, "event": "apply_refined_action_skipped", "reason": "invalid_anchor_level"})
                continue
            nodes = sem.get("nodes", [])
            if not isinstance(nodes, list) or len(nodes) not in {2, 3}:
                taxonomy_ops.append({**op_base, "op_type": "add_path", "op_result": "invalid_path_length"})
                event_log.append({**op_base, "event": "apply_refined_action_skipped", "reason": "invalid_path_length"})
                continue
            if not all(isinstance(x, dict) for x in nodes):
                taxonomy_ops.append({**op_base, "op_type": "add_path", "op_result": "invalid_path_nodes"})
                event_log.append({**op_base, "event": "apply_refined_action_skipped", "reason": "invalid_path_nodes"})
                continue
            levels = [str(x.get("level", "")).strip().lower() for x in nodes]
            allowed_levels = (
                (["topic", "subtopic"], anchor_level == "root"),
                (["topic", "subtopic", "claim"], anchor_level == "root"),
                (["subtopic", "claim"], anchor_level == "topic"),
            )
            if not any(levels == lv and ok for lv, ok in allowed_levels):
                taxonomy_ops.append({**op_base, "op_type": "add_path", "op_result": "invalid_path_shape"})
                event_log.append({**op_base, "event": "apply_refined_action_skipped", "reason": "invalid_path_shape"})
                continue
            names = [str(x.get("name", "")).strip() for x in nodes]
            if any(not x for x in names):
                taxonomy_ops.append({**op_base, "op_type": "add_path", "op_result": "invalid_path_names"})
                event_log.append({**op_base, "event": "apply_refined_action_skipped", "reason": "invalid_path_names"})
                continue

            created_node_ids: List[str] = []
            parent = objective_node_id
            for item in nodes:
                next_id = taxonomy.add_node(
                    parent_id=parent,
                    name=str(item.get("name", "")).strip(),
                    level=str(item.get("level", "claim")).strip().lower(),
                    cmb=item.get("cmb", {}),
                    window_id=window_id,
                )
                created_node_ids.append(next_id)
                parent = next_id
            final_id = created_node_ids[-1]
            for pid in proposal_post_ids:
                ts = proposal_ts.get(pid)
                assignment_rows.append(
                    {
                        "post_id": pid,
                        "timestamp": ts,
                        "window_id": window_id,
                        "node_id_at_time": final_id,
                        "canonical_node_id": final_id,
                        "similarity": None,
                        "mapping_mode": "post_apply_refine_add_path",
                    }
                )
                node_post_links.append(
                    {"post_id": pid, "node_id": final_id, "timestamp": ts, "window_id": window_id, "source": "add_path_refined"}
                )
            taxonomy_ops.append(
                {
                    **op_base,
                    "op_type": "add_path",
                    "anchor_id": objective_node_id,
                    "created_node_ids": created_node_ids,
                    "final_node_id": final_id,
                    "path_levels": levels,
                    "path_names": names,
                    "post_count": len(proposal_post_ids),
                    "op_result": "applied",
                }
            )
            event_log.append({**op_base, "event": "apply_add_path", "created_node_ids": created_node_ids, "final_node_id": final_id})

        elif action_type == "update_cmb":
            if objective_node_id not in taxonomy.nodes:
                taxonomy_ops.append({**op_base, "op_type": "update_cmb", "op_result": "invalid_objective"})
                event_log.append({**op_base, "event": "apply_refined_action_skipped", "reason": "invalid_objective"})
                continue
            taxonomy.set_cmb(objective_node_id, sem.get("new_cmb", {}), window_id=window_id)
            taxonomy_ops.append({**op_base, "op_type": "update_cmb", "target_id": objective_node_id, "op_result": "applied"})
            event_log.append({**op_base, "event": "apply_update_cmb", "target_id": objective_node_id})

        elif action_type == "skip_post":
            taxonomy_ops.append({**op_base, "op_type": "skip_post", "post_count": len(proposal_post_ids), "op_result": "applied"})
            event_log.append({**op_base, "event": "apply_skip_post", "post_count": len(proposal_post_ids)})
        else:
            taxonomy_ops.append({**op_base, "op_type": "unknown", "op_result": "invalid_action_type"})
            event_log.append({**op_base, "event": "apply_refined_action_skipped", "reason": "invalid_action_type"})
