from __future__ import annotations

import logging
from typing import Any, Dict, List

from .taxonomy import Taxonomy
from .utils import now_ts


VALID_LEVELS = {"topic", "subtopic"}


def apply_refined_actions(
    taxonomy: Taxonomy,
    refined_actions: List[Dict[str, Any]],
    cluster_proposals: List[Dict[str, Any]],
    window_id: str,
    assignment_rows: Any,
    node_post_links: List[Dict[str, Any]],
    logger: logging.Logger,
    taxonomy_updates: Any = None,
) -> None:
    proposal_post_ids = [str(x["post_id"]) for x in cluster_proposals]
    proposal_ts = {str(x["post_id"]): x.get("timestamp") for x in cluster_proposals}

    for action in refined_actions:
        action_type = action.get("action_type")
        objective_node_id = action.get("objective_node_id")
        sem = action.get("semantic_payload", {})
        if action_type != "skip_post":
            logger.info("Applying refined action type=%s objective=%s posts=%d", action_type, objective_node_id, len(proposal_post_ids))

        if action_type == "add_child":
            if objective_node_id not in taxonomy.nodes:
                continue
            child_name = str(sem.get("child_name", "")).strip()
            child_level = str(sem.get("child_level", "subtopic")).strip().lower()
            child_cmb = sem.get("child_cmb", {})
            if not child_name or child_level not in VALID_LEVELS:
                continue
            parent_level = taxonomy.nodes[objective_node_id].level
            if (parent_level == "root" and child_level != "topic") or (parent_level == "topic" and child_level != "subtopic"):
                continue
            if parent_level not in {"root", "topic"}:
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
            if taxonomy_updates is not None:
                taxonomy_updates.append(
                    {
                        "ts": now_ts(),
                        "window_id": window_id,
                        "trigger": "apply_add_child",
                        "action_type": action_type,
                        "objective_node_id": objective_node_id,
                        "post_ids": proposal_post_ids,
                        "taxonomy_nodes": taxonomy.to_rows(),
                    }
                )

        elif action_type == "add_path":
            if objective_node_id not in taxonomy.nodes:
                continue
            anchor_level = taxonomy.nodes[objective_node_id].level
            if anchor_level != "root":
                continue
            nodes = sem.get("nodes", [])
            if not isinstance(nodes, list) or len(nodes) != 2:
                continue
            if not all(isinstance(x, dict) for x in nodes):
                continue
            levels = [str(x.get("level", "")).strip().lower() for x in nodes]
            if levels != ["topic", "subtopic"]:
                continue
            names = [str(x.get("name", "")).strip() for x in nodes]
            if any(not x for x in names):
                continue

            created_node_ids: List[str] = []
            parent = objective_node_id
            for item in nodes:
                next_id = taxonomy.add_node(
                    parent_id=parent,
                    name=str(item.get("name", "")).strip(),
                    level=str(item.get("level", "subtopic")).strip().lower(),
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
            if taxonomy_updates is not None:
                taxonomy_updates.append(
                    {
                        "ts": now_ts(),
                        "window_id": window_id,
                        "trigger": "apply_add_path",
                        "action_type": action_type,
                        "objective_node_id": objective_node_id,
                        "post_ids": proposal_post_ids,
                        "taxonomy_nodes": taxonomy.to_rows(),
                    }
                )

        elif action_type == "update_cmb":
            if objective_node_id not in taxonomy.nodes:
                continue
            taxonomy.set_cmb(objective_node_id, sem.get("new_cmb", {}), window_id=window_id)
            if taxonomy_updates is not None:
                taxonomy_updates.append(
                    {
                        "ts": now_ts(),
                        "window_id": window_id,
                        "trigger": "apply_update_cmb",
                        "action_type": action_type,
                        "objective_node_id": objective_node_id,
                        "post_ids": proposal_post_ids,
                        "taxonomy_nodes": taxonomy.to_rows(),
                    }
                )

        elif action_type == "skip_post":
            continue
