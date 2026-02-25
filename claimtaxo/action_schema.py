from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from taxonomy import Taxonomy


ALLOWED_ACTIONS = {"set_node", "add_child", "add_path", "update_cmb", "skip_post"}
ALLOWED_LEVELS = {"topic", "subtopic", "claim"}


def normalize_cmb(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    return {
        "definition": str(value.get("definition", "")).strip(),
        "include_terms": [str(x).strip() for x in value.get("include_terms", []) if str(x).strip()][:20],
        "exclude_terms": [str(x).strip() for x in value.get("exclude_terms", []) if str(x).strip()][:20],
        "examples": [str(x).strip() for x in value.get("examples", []) if str(x).strip()][:10],
    }


def normalize_refined_action(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    action_type = str(raw.get("action_type", "")).strip()
    if action_type not in ALLOWED_ACTIONS:
        return None

    objective_node_id = raw.get("objective_node_id", None)
    if objective_node_id is not None:
        objective_node_id = str(objective_node_id).strip() or None
    if action_type != "skip_post" and not objective_node_id:
        return None

    normalized = {
        "action_type": action_type,
        "objective_node_id": objective_node_id,
        "semantic_payload": {},
    }

    sem = raw.get("semantic_payload", {})
    if action_type == "add_child":
        if not isinstance(sem, dict):
            sem = {}
        level = str(sem.get("child_level", "claim")).strip().lower()
        if level not in ALLOWED_LEVELS:
            return None
        child_name = str(sem.get("child_name", "")).strip()
        if not child_name:
            return None
        normalized["semantic_payload"] = {
            "child_name": child_name,
            "child_level": level,
            "child_cmb": normalize_cmb(sem.get("child_cmb", {})),
        }
    elif action_type == "add_path":
        if not isinstance(sem, dict):
            sem = {}
        nodes = sem.get("nodes", [])
        out_nodes = []
        if isinstance(nodes, list):
            for item in nodes:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                level = str(item.get("level", "")).strip().lower()
                if not name or level not in {"topic", "subtopic", "claim"}:
                    continue
                out_nodes.append(
                    {
                        "name": name,
                        "level": level,
                        "cmb": normalize_cmb(item.get("cmb", {})),
                    }
                )
        levels = [x["level"] for x in out_nodes]
        if levels not in (["topic", "subtopic"], ["subtopic", "claim"], ["topic", "subtopic", "claim"]):
            return None
        normalized["semantic_payload"] = {"nodes": out_nodes}
    elif action_type == "update_cmb":
        if not isinstance(sem, dict):
            sem = {}
        normalized["semantic_payload"] = {"new_cmb": normalize_cmb(sem.get("new_cmb", {}))}
    return normalized


def normalize_proposal_action(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    action_type = str(raw.get("action_type", "")).strip()
    if action_type not in ALLOWED_ACTIONS:
        return None

    objective_node_id = raw.get("objective_node_id")
    if objective_node_id is not None:
        objective_node_id = str(objective_node_id).strip() or None
    if action_type != "skip_post" and not objective_node_id:
        return None

    action_explanation = str(raw.get("action_explanation", "")).strip()
    post_summary = str(raw.get("post_summary", "")).strip()
    if action_type != "skip_post" and not (action_explanation or post_summary):
        return None

    return {
        "action_type": action_type,
        "objective_node_id": objective_node_id,
        "action_explanation": action_explanation,
        "post_summary": post_summary,
    }


def validate_refined_action_executable(
    action: Dict[str, Any],
    taxonomy: Taxonomy,
) -> Tuple[bool, str]:
    if not isinstance(action, dict):
        return False, "action_not_dict"
    action_type = str(action.get("action_type", "")).strip()
    if action_type not in ALLOWED_ACTIONS:
        return False, "invalid_action_type"

    objective_node_id = action.get("objective_node_id")
    if objective_node_id is not None:
        objective_node_id = str(objective_node_id).strip() or None
    sem = action.get("semantic_payload", {})
    if not isinstance(sem, dict):
        sem = {}

    if action_type == "skip_post":
        return True, ""

    if not objective_node_id:
        return False, "missing_objective_node_id"
    if objective_node_id not in taxonomy.nodes:
        return False, "objective_node_not_found"

    if action_type == "set_node":
        if objective_node_id == taxonomy.root_id:
            return False, "set_node_on_root_not_allowed"
        return True, ""

    if action_type == "update_cmb":
        if objective_node_id == taxonomy.root_id:
            return False, "update_cmb_on_root_not_allowed"
        return True, ""

    if action_type == "add_child":
        child_name = str(sem.get("child_name", "")).strip()
        child_level = str(sem.get("child_level", "claim")).strip().lower()
        if not child_name:
            return False, "missing_child_name"
        if child_level not in ALLOWED_LEVELS:
            return False, "invalid_child_level"
        return True, ""

    if action_type == "add_path":
        anchor_level = taxonomy.nodes[objective_node_id].level
        if anchor_level not in {"root", "topic"}:
            return False, "invalid_anchor_level"
        nodes = sem.get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) not in {2, 3}:
            return False, "invalid_path_length"
        if not all(isinstance(x, dict) for x in nodes):
            return False, "invalid_path_nodes"
        levels = [str(x.get("level", "")).strip().lower() for x in nodes]
        allowed_levels = (
            (["topic", "subtopic"], anchor_level == "root"),
            (["topic", "subtopic", "claim"], anchor_level == "root"),
            (["subtopic", "claim"], anchor_level == "topic"),
        )
        if not any(levels == lv and ok for lv, ok in allowed_levels):
            return False, "invalid_path_shape"
        names = [str(x.get("name", "")).strip() for x in nodes]
        if any(not x for x in names):
            return False, "invalid_path_names"
        return True, ""

    return False, "unsupported_action_type"


# Backward-compatible aliases used by older imports.
_normalize_cmb = normalize_cmb
_normalize_refined_action = normalize_refined_action
_normalize_proposal_action = normalize_proposal_action
