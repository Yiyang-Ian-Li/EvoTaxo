from __future__ import annotations

from typing import Any, Dict, List, Optional

from action_schema import normalize_proposal_action
from llm import LLMClient
from llm_json import ask_json_with_retries
from prompts import build_propose_post_prompt, taxonomy_context
from taxonomy import Taxonomy


def _parse_proposal_actions(payload: Dict[str, Any], taxonomy: Taxonomy) -> tuple[List[Dict[str, Any]], bool]:
    out: List[Dict[str, Any]] = []
    used_root_forbidden_action = False
    for item in payload.get("actions", []):
        norm = normalize_proposal_action(item)
        if norm is None:
            continue
        action_type = str(norm.get("action_type", ""))
        objective_node_id = str(norm.get("objective_node_id"))
        if action_type in {"set_node", "update_cmb"} and objective_node_id == str(taxonomy.root_id):
            used_root_forbidden_action = True
            continue
        out.append(norm)
    return out, used_root_forbidden_action


def propose_post_actions(
    llm: LLMClient,
    taxonomy: Taxonomy,
    root_topic: str,
    post_text: str,
    post_id: str,
    window_id: str,
    max_parse_attempts: int,
    taxonomy_ctx: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not llm.available():
        return [
            {
                "action_type": "skip_post",
                "objective_node_id": None,
                "action_explanation": "LLM unavailable; skipped.",
                "post_summary": "",
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
    )
    if not payload:
        return [
            {
                "action_type": "skip_post",
                "objective_node_id": None,
                "action_explanation": "Skipped because proposal JSON parsing failed.",
                "post_summary": "",
            }
        ]

    out, used_root_forbidden_action = _parse_proposal_actions(payload, taxonomy)
    if used_root_forbidden_action:
        retry_prompt = (
            prompt
            + "\n\nYour previous output used set_node/update_cmb on the root node, which is invalid."
            + " Re-answer with valid actions only. Never use set_node or update_cmb on root."
        )
        retry_payload = ask_json_with_retries(
            llm,
            retry_prompt,
            system,
            max_parse_attempts=max_parse_attempts,
        )
        if retry_payload:
            out, _ = _parse_proposal_actions(retry_payload, taxonomy)

    if not out:
        out.append(
            {
                "action_type": "skip_post",
                "objective_node_id": None,
                "action_explanation": "Skipped because no valid action parsed.",
                "post_summary": "",
            }
        )
    return out
