from __future__ import annotations

from typing import Any, Dict, List, Optional

from action_schema import normalize_refined_action
from llm import LLMClient
from llm_json import ask_json_with_retries
from prompts import (
    build_final_review_prompt,
    build_repair_prompt,
    build_review_cluster_prompt,
    taxonomy_context,
)
from taxonomy import Taxonomy


def _sample_proposals_for_review(
    proposal_records: List[Dict[str, Any]],
    max_examples: int,
    max_post_chars: int,
) -> List[Dict[str, Any]]:
    if not proposal_records:
        return []
    k = max(1, min(int(max_examples), len(proposal_records)))

    records = list(proposal_records)
    records.sort(key=lambda x: float(x.get("timestamp_epoch", 0.0)))
    time_picks = []
    if len(records) <= k:
        time_picks = records
    else:
        idxs = []
        for i in range(k):
            idx = int(round(i * (len(records) - 1) / max(1, k - 1)))
            if idx not in idxs:
                idxs.append(idx)
        time_picks = [records[i] for i in idxs][:k]

    ranked = sorted(
        records,
        key=lambda x: (
            -float(x.get("confidence", 0.0)),
            str(x.get("proposal_id", "")),
        ),
    )
    top_k = ranked[: max(1, k // 2)]

    seen = set()
    picked = []
    for rec in top_k + time_picks:
        pid = str(rec.get("proposal_id", ""))
        if pid in seen:
            continue
        seen.add(pid)
        picked.append(rec)
        if len(picked) >= k:
            break

    out = []
    for r in picked:
        txt = str(r.get("post_summary", ""))
        if max_post_chars > 0:
            txt = txt[:max_post_chars]
        out.append(
            {
                "proposal_id": r.get("proposal_id"),
                "post_id": r.get("post_id"),
                "window_id": r.get("window_id"),
                "timestamp": r.get("timestamp"),
                "action_type": r.get("action_type"),
                "objective_node_id": r.get("objective_node_id"),
                "action_explanation": r.get("action_explanation", ""),
                "confidence": r.get("confidence", 0.0),
                "reasoning_short": r.get("reasoning_short", ""),
                "post_summary": txt,
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
    max_review_post_chars: int,
    trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    sampled = _sample_proposals_for_review(
        proposal_records,
        max_examples=max_review_examples,
        max_post_chars=max_review_post_chars,
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
        window_id=window_id,
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
        trace=trace,
        trace_meta={
            "call": "review_action_cluster",
            "window_id": window_id,
            "cluster_id": str(cluster_record.get("cluster_id", "")),
            "proposal_total": len(proposal_records),
            "proposal_sampled": len(sampled),
        },
    )
    if not payload:
        return {"decision": "defer", "refined_actions": [], "reason": "parse_fail"}

    decision = str(payload.get("decision", "defer")).strip().lower()
    if decision not in {"approve", "defer"}:
        decision = "defer"

    refined_actions = []
    for item in payload.get("refined_actions", []):
        norm = normalize_refined_action(item)
        if norm is not None:
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
    trace: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    if not llm.available():
        return [
            {"candidate_index": int(i), "refined_actions": list(c.get("refined_actions", []))}
            for i, c in enumerate(candidates)
        ]

    compact = []
    for i, c in enumerate(candidates):
        records = c.get("records", []) if isinstance(c.get("records", []), list) else []
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
        compact.append(
            {
                "candidate_index": i,
                "cluster_id": c.get("cluster_id"),
                "cluster_mode": c.get("cluster_mode"),
                "action_type": c.get("action_type"),
                "objective_node_id": c.get("objective_node_id"),
                "proposal_count": len(c.get("proposal_ids", [])),
                "quality": c.get("quality", {}),
                "refined_actions": c.get("refined_actions", []),
                "examples": examples,
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
        trace=trace,
        trace_meta={
            "call": "review_final_action_pool",
            "batch_id": batch_id,
            "candidate_count": len(candidates),
        },
    )
    if not payload:
        return [
            {"candidate_index": int(i), "refined_actions": list(c.get("refined_actions", []))}
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
            if norm is not None:
                refined.append(norm)
        if not refined:
            refined = list(candidates[idx].get("refined_actions", []))
        selected.append({"candidate_index": idx, "refined_actions": refined})

    if not selected:
        return [
            {"candidate_index": int(i), "refined_actions": list(c.get("refined_actions", []))}
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
    trace: Optional[List[Dict[str, Any]]] = None,
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
        trace=trace,
        trace_meta={
            "call": "repair_final_action_candidate",
            "batch_id": batch_id,
            "cluster_id": str(candidate.get("cluster_id", "")),
            "invalid_reason": invalid_reason,
        },
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
        if norm is not None:
            out.append(norm)
    return out
