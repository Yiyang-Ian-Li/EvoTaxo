from __future__ import annotations

from typing import Any, Dict, List, Optional

from .action_schema import normalize_refined_action
from .llm import LLMClient
from .llm_json import ask_json_with_retries
from .prompts import (
    build_final_review_prompt,
    build_initial_taxonomy_prompt,
    build_repair_prompt,
    build_review_cluster_prompt,
    taxonomy_context,
)
from .taxonomy import Taxonomy


ALLOWED_REVIEW_ACTIONS = {"add_child", "add_path", "update_cmb"}
ALLOWED_BOOTSTRAP_ACTIONS = {"add_child", "add_path"}


def generate_initial_taxonomy_actions(
    llm: LLMClient,
    taxonomy: Taxonomy,
    root_topic: str,
    max_parse_attempts: int,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    if not llm.available():
        return {"refined_actions": [], "reason": "llm_unavailable"}

    prompt = build_initial_taxonomy_prompt(
        root_topic=root_topic,
        taxonomy_ctx=taxonomy_context(taxonomy, max_nodes=None),
    )
    system = "You are a taxonomy bootstrap planner. Return valid JSON only."
    payload = ask_json_with_retries(
        llm,
        prompt,
        system,
        max_parse_attempts=max_parse_attempts,
        model_override=model_override,
    )
    if not payload:
        return {"refined_actions": [], "reason": "parse_fail"}

    refined_actions = []
    for item in payload.get("refined_actions", []):
        norm = normalize_refined_action(item)
        if norm is not None and str(norm.get("action_type", "")) in ALLOWED_BOOTSTRAP_ACTIONS:
            refined_actions.append(norm)

    return {
        "refined_actions": refined_actions,
        "reason": str(payload.get("reason", "")).strip(),
    }


def _sample_proposals_for_review(
    proposal_records: List[Dict[str, Any]],
    max_examples: int,
    max_post_words: int,
    centroid_proposal_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if not proposal_records:
        return []
    k = max(1, min(int(max_examples), len(proposal_records)))

    records = list(proposal_records)
    by_pid = {str(r.get("proposal_id", "")): r for r in records}
    picked: List[Dict[str, Any]] = []
    seen = set()

    if isinstance(centroid_proposal_ids, list) and centroid_proposal_ids:
        for pid_raw in centroid_proposal_ids:
            pid = str(pid_raw)
            rec = by_pid.get(pid)
            if rec is None or pid in seen:
                continue
            seen.add(pid)
            picked.append(rec)
            if len(picked) >= k:
                break

    records.sort(
        key=lambda x: (
            float(x.get("timestamp_epoch", 0.0)),
            str(x.get("proposal_id", "")),
        )
    )
    if len(picked) < k:
        for rec in records:
            pid = str(rec.get("proposal_id", ""))
            if pid in seen:
                continue
            seen.add(pid)
            picked.append(rec)
            if len(picked) >= k:
                break

    out = []
    for r in picked:
        post_title = str(r.get("post_title", ""))
        post_text = str(r.get("post_text", ""))
        if max_post_words > 0:
            post_title = " ".join(post_title.split()[:max_post_words])
            post_text = " ".join(post_text.split()[:max_post_words])
        out.append(
            {
                "action_type": r.get("action_type"),
                "objective_node_id": r.get("objective_node_id"),
                "action_explanation": r.get("action_explanation", ""),
                "post_title": post_title,
                "post_text": post_text,
            }
        )
    return out


def review_action_cluster(
    llm: LLMClient,
    taxonomy: Taxonomy,
    root_topic: str,
    window_id: str,
    cluster_record: Dict[str, Any],
    proposal_records: List[Dict[str, Any]],
    max_parse_attempts: int,
    max_review_examples: int,
    max_review_post_words: int,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    sampled = _sample_proposals_for_review(
        proposal_records,
        max_examples=max_review_examples,
        max_post_words=max_review_post_words,
        centroid_proposal_ids=cluster_record.get("centroid_proposal_ids"),
    )
    cluster_brief = {
        "cluster_mode": cluster_record.get("cluster_mode"),
        "action_type": cluster_record.get("action_type"),
        "objective_node_id": cluster_record.get("objective_node_id"),
        "size": cluster_record.get("size"),
        "quality": cluster_record.get("quality", {}),
    }
    prompt = build_review_cluster_prompt(
        root_topic=root_topic,
        cluster_brief=cluster_brief,
        sampled=sampled,
        taxonomy_ctx=taxonomy_context(taxonomy, max_nodes=None),
        total_cluster_count=len(proposal_records),
    )
    system = "You are a taxonomy reviewer. Return valid JSON only."
    payload = ask_json_with_retries(
        llm,
        prompt,
        system,
        max_parse_attempts=max_parse_attempts,
        model_override=model_override,
    )
    if not payload:
        return {"decision": "defer", "refined_actions": [], "reason": "parse_fail"}

    decision = str(payload.get("decision", "defer")).strip().lower()
    if decision not in {"approve", "defer"}:
        decision = "defer"

    refined_actions = []
    for item in payload.get("refined_actions", []):
        norm = normalize_refined_action(item)
        if norm is not None and str(norm.get("action_type", "")) in ALLOWED_REVIEW_ACTIONS:
            refined_actions.append(norm)

    return {
        "decision": decision,
        "refined_actions": refined_actions,
        "reason": str(payload.get("reason", "")).strip(),
    }


def review_final_action_pool(
    llm: LLMClient,
    taxonomy: Taxonomy,
    root_topic: str,
    batch_id: str,
    candidates: List[Dict[str, Any]],
    max_parse_attempts: int,
    model_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    if not llm.available():
        return [
            {"candidate_index": int(i), "refined_actions": list(c.get("refined_actions", [])), "justification": ""}
            for i, c in enumerate(candidates)
        ]

    compact = []
    for i, c in enumerate(candidates):
        compact.append(
            {
                "candidate_index": i,
                "refined_actions": c.get("refined_actions", []),
            }
        )

    prompt = build_final_review_prompt(
        root_topic=root_topic,
        batch_id=batch_id,
        compact_candidates=compact,
        taxonomy_ctx=taxonomy_context(taxonomy, max_nodes=None),
    )
    system = "You are a taxonomy editor resolving overlaps. Return valid JSON only."
    payload = ask_json_with_retries(
        llm,
        prompt,
        system,
        max_parse_attempts=max_parse_attempts,
        model_override=model_override,
    )
    if not payload:
        return [
            {"candidate_index": int(i), "refined_actions": list(c.get("refined_actions", [])), "justification": ""}
            for i, c in enumerate(candidates)
        ]

    selected: List[Dict[str, Any]] = []
    for item in payload.get("selected", []):
        if not isinstance(item, dict):
            continue
        idx_raw = item.get("candidate_index")
        if isinstance(idx_raw, bool):
            continue
        try:
            idx = int(idx_raw)
        except Exception:
            continue
        if idx < 0 or idx >= len(candidates):
            continue
        refined = []
        for a in item.get("refined_actions", []):
            norm = normalize_refined_action(a)
            if norm is not None and str(norm.get("action_type", "")) in ALLOWED_REVIEW_ACTIONS:
                refined.append(norm)
        if not refined:
            refined = list(candidates[idx].get("refined_actions", []))
        selected.append(
            {
                "candidate_index": idx,
                "refined_actions": refined,
                "justification": str(item.get("justification", "")).strip(),
            }
        )

    if not selected:
        return [
            {"candidate_index": int(i), "refined_actions": list(c.get("refined_actions", [])), "justification": ""}
            for i, c in enumerate(candidates)
        ]
    return selected


def repair_final_action_candidate(
    llm: LLMClient,
    taxonomy: Taxonomy,
    root_topic: str,
    batch_id: str,
    candidate: Dict[str, Any],
    invalid_reason: str,
    max_parse_attempts: int,
    model_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not llm.available():
        return []

    records = candidate.get("records", []) if isinstance(candidate.get("records", []), list) else []
    examples = []
    for r in records[:3]:
        if not isinstance(r, dict):
            continue
        examples.append(
            {
                "action_explanation": str(r.get("action_explanation", ""))[:240],
                "post_summary": str(r.get("post_summary", ""))[:240],
            }
        )
    cand_compact = {
        "cluster_id": candidate.get("cluster_id"),
        "cluster_mode": candidate.get("cluster_mode"),
        "action_type": candidate.get("action_type"),
        "objective_node_id": candidate.get("objective_node_id"),
        "proposal_count": len(candidate.get("proposal_ids", [])),
        "quality": candidate.get("quality", {}),
        "current_refined_action": (candidate.get("refined_actions", []) or [None])[0],
        "examples": examples,
    }
    prompt = build_repair_prompt(
        root_topic=root_topic,
        batch_id=batch_id,
        invalid_reason=invalid_reason,
        candidate_compact=cand_compact,
        taxonomy_ctx=taxonomy_context(taxonomy, max_nodes=None),
    )
    system = "You are fixing an invalid taxonomy action. Return valid JSON only."
    payload = ask_json_with_retries(
        llm,
        prompt,
        system,
        max_parse_attempts=max_parse_attempts,
        model_override=model_override,
    )
    if not payload:
        return []

    # Prefer plural key; keep singular for backward compatibility.
    raw_actions = payload.get("refined_actions")
    if not isinstance(raw_actions, list):
        one = payload.get("refined_action")
        raw_actions = [one] if one is not None else []

    out: List[Dict[str, Any]] = []
    for raw in raw_actions:
        norm = normalize_refined_action(raw)
        if norm is not None and str(norm.get("action_type", "")) in ALLOWED_REVIEW_ACTIONS:
            out.append(norm)
    return out
