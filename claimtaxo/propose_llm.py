from __future__ import annotations

from typing import Any, Dict, List, Optional

from action_schema import normalize_proposal_action
from llm import LLMClient
from llm_json import ask_json_with_retries
from prompts import build_propose_post_prompt, taxonomy_context
from taxonomy import Taxonomy


def propose_post_actions(
    llm: LLMClient,
    taxonomy: Taxonomy,
    root_topic: str,
    post_text: str,
    post_id: str,
    window_id: str,
    best_candidate_node_id: Optional[str],
    best_similarity: float,
    max_parse_attempts: int,
    trace: Optional[List[Dict[str, Any]]] = None,
    taxonomy_ctx: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not llm.available():
        return [
            {
                "action_type": "skip_post",
                "objective_node_id": None,
                "objective": "llm_unavailable",
                "action_explanation": "LLM unavailable; skipped.",
                "post_summary": "",
                "confidence": 0.0,
                "reasoning_short": "LLM unavailable",
            }
        ]

    ctx = taxonomy_ctx if isinstance(taxonomy_ctx, dict) else taxonomy_context(taxonomy)
    prompt = build_propose_post_prompt(
        root_topic=root_topic,
        post_id=post_id,
        taxonomy_ctx=ctx,
        post_text=post_text,
    )
    system = "You help maintain a coherent taxonomy. Return valid JSON only."
    payload = ask_json_with_retries(
        llm,
        prompt,
        system,
        max_parse_attempts=max_parse_attempts,
        trace=trace,
        trace_meta={"call": "propose_post_actions", "window_id": window_id, "post_id": post_id},
    )
    if not payload:
        return [
            {
                "action_type": "skip_post",
                "objective_node_id": None,
                "objective": "parse_fail",
                "action_explanation": "Skipped because proposal JSON parsing failed.",
                "post_summary": "",
                "confidence": 0.0,
                "reasoning_short": "LLM output parse failed",
            }
        ]

    out: List[Dict[str, Any]] = []
    for item in payload.get("actions", []):
        norm = normalize_proposal_action(item)
        if norm is not None:
            out.append(norm)
    if not out:
        out.append(
            {
                "action_type": "skip_post",
                "objective_node_id": None,
                "objective": "no_valid_action",
                "action_explanation": "Skipped because no valid action parsed.",
                "post_summary": "",
                "confidence": 0.0,
                "reasoning_short": "No valid action parsed",
            }
        )
    return out
