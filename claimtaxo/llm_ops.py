from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from llm import LLMClient
from taxonomy import Taxonomy
from utils import parse_json_object


ALLOWED_ACTIONS = {"set_node", "add_child", "add_path", "update_cmb", "skip_post"}
ALLOWED_LEVELS = {"topic", "subtopic", "claim"}


def taxonomy_context(taxonomy: Taxonomy, max_nodes: Optional[int] = 300) -> Dict[str, Any]:
    nodes = []
    for n in taxonomy.nodes.values():
        nodes.append(
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
        )
    nodes.sort(key=lambda x: (x["level"], x["name"], x["node_id"]))
    if max_nodes is None:
        node_rows = nodes
    else:
        node_rows = nodes[: max(0, int(max_nodes))]
    return {
        "root_id": taxonomy.root_id,
        "node_count": len(taxonomy.nodes),
        "nodes": node_rows,
    }


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
        txt = str(r.get("post_text", ""))
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
                "semantic_payload": r.get("semantic_payload", {}),
                "confidence": r.get("confidence", 0.0),
                "reasoning_short": r.get("reasoning_short", ""),
                "post_text": txt,
            }
        )
    return out


def ask_json_with_retries(
    llm: LLMClient,
    prompt: str,
    system_prompt: str,
    max_parse_attempts: int,
    trace: Optional[List[Dict[str, Any]]] = None,
    trace_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    attempts = max(1, max_parse_attempts)
    for i in range(attempts):
        p = prompt
        if i > 0:
            p = prompt + "\n\nPrevious response was invalid JSON. Return one valid JSON object only."
        resp = llm.chat(p, response_format={"type": "json_object"}, system_prompt=system_prompt)
        if trace is not None and llm.cfg.trace_mode != "off":
            resp_len = len(resp or "")
            prompt_len = len(p)
            trace_row = {
                "kind": "llm_json_attempt",
                "trace_meta": trace_meta or {},
                "attempt": i + 1,
                "max_attempts": attempts,
                "prompt_len": prompt_len,
                "response_len": resp_len,
            }
            if llm.cfg.trace_mode == "full":
                trace_row["prompt"] = p
                trace_row["response_text"] = resp
            elif llm.cfg.trace_mode == "compact":
                max_chars = max(0, int(llm.cfg.trace_max_chars))
                trace_row["prompt_head"] = p[:max_chars]
                trace_row["response_head"] = (resp or "")[:max_chars]
            trace.append(
                trace_row
            )
        if not resp:
            continue
        obj = parse_json_object(resp)
        if isinstance(obj, dict):
            if trace is not None and llm.cfg.trace_mode != "off":
                trace.append(
                    {
                        "kind": "llm_json_success",
                        "trace_meta": trace_meta or {},
                        "attempt": i + 1,
                        "keys": sorted([str(k) for k in obj.keys()]),
                    }
                )
            return obj
    if trace is not None and llm.cfg.trace_mode != "off":
        trace.append(
            {
                "kind": "llm_json_failed",
                "trace_meta": trace_meta or {},
                "max_attempts": attempts,
            }
        )
    return None


def _normalize_cmb(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    return {
        "definition": str(value.get("definition", "")).strip(),
        "include_terms": [str(x).strip() for x in value.get("include_terms", []) if str(x).strip()][:20],
        "exclude_terms": [str(x).strip() for x in value.get("exclude_terms", []) if str(x).strip()][:20],
        "examples": [str(x).strip() for x in value.get("examples", []) if str(x).strip()][:10],
    }


def bootstrap_taxonomy_with_llm(
    llm: LLMClient,
    sample_posts: List[Dict[str, Any]],
    root_topic: str,
    max_parse_attempts: int,
    trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    prompt = (
        "Task: Build an initial taxonomy and concept memory bank (CMB) from sampled posts.\\n"
        f"Intended root topic: {json.dumps(root_topic, ensure_ascii=False)}\\n"
        "Node level definitions (must follow):\\n"
        "- topic: broad recurring theme under root topic.\\n"
        "- subtopic: narrower bucket inside one topic.\\n"
        "- claim: atomic, post-level mappable assertion/opinion/event statement.\\n"
        "Granularity rules:\\n"
        "- Do not make claims too broad (bad: 'policy debate').\\n"
        "- Do not make claims too specific to a single post phrasing.\\n"
        "- Prefer 2-4 topics, optional subtopics, and enough claim nodes for mapping coverage.\\n"
        "JSON output only with keys: root_name, nodes.\\n"
        "nodes[] object keys: temp_id, name, level(topic|subtopic|claim), parent_temp_id, cmb.\\n"
        "cmb keys: definition, include_terms[], exclude_terms[], examples[].\\n"
        "Example node: "
        '{"temp_id":"c1","name":"Insurance Denial for X","level":"claim","parent_temp_id":"s2","cmb":{"definition":"...","include_terms":["deny","reject"],"exclude_terms":["off-topic"],"examples":["..."]}}\\n'
        f"Sample posts:\\n{json.dumps(sample_posts, ensure_ascii=False)}"
    )
    system = "You are a precise taxonomy architect. Output strict JSON only."
    payload = ask_json_with_retries(
        llm,
        prompt,
        system,
        max_parse_attempts=max_parse_attempts,
        trace=trace,
        trace_meta={"call": "bootstrap_taxonomy"},
    )
    if not payload:
        return {"root_name": "ROOT", "nodes": []}

    out_nodes = []
    for item in payload.get("nodes", []):
        if not isinstance(item, dict):
            continue
        level = str(item.get("level", "claim")).strip().lower()
        if level not in ALLOWED_LEVELS:
            level = "claim"
        out_nodes.append(
            {
                "temp_id": str(item.get("temp_id", "")).strip(),
                "name": str(item.get("name", "")).strip() or "unnamed",
                "level": level,
                "parent_temp_id": str(item.get("parent_temp_id", "")).strip() or None,
                "cmb": _normalize_cmb(item.get("cmb", {})),
            }
        )

    return {"root_name": str(payload.get("root_name", "ROOT")).strip() or "ROOT", "nodes": out_nodes}


def _normalize_action(raw: Any) -> Optional[Dict[str, Any]]:
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
        "objective": str(raw.get("objective", "")).strip(),
        "semantic_payload": {},
        "confidence": float(raw.get("confidence", 0.0)) if str(raw.get("confidence", "")).strip() else 0.0,
        "reasoning_short": str(raw.get("reasoning_short", "")).strip(),
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
            "child_cmb": _normalize_cmb(sem.get("child_cmb", {})),
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
                if not name or level not in {"subtopic", "claim"}:
                    continue
                out_nodes.append(
                    {
                        "name": name,
                        "level": level,
                        "cmb": _normalize_cmb(item.get("cmb", {})),
                    }
                )
        levels = [x["level"] for x in out_nodes]
        if levels not in (["subtopic"], ["subtopic", "claim"]):
            return None
        normalized["semantic_payload"] = {"nodes": out_nodes}
    elif action_type == "update_cmb":
        if not isinstance(sem, dict):
            sem = {}
        normalized["semantic_payload"] = {"new_cmb": _normalize_cmb(sem.get("new_cmb", {}))}
    return normalized


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
        return [{"action_type": "skip_post", "objective_node_id": None, "objective": "llm_unavailable", "semantic_payload": {}, "confidence": 0.0, "reasoning_short": "LLM unavailable"}]
    ctx = taxonomy_ctx if isinstance(taxonomy_ctx, dict) else taxonomy_context(taxonomy)
    prompt = (
        "You review one post and propose taxonomy actions.\\n"
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\\n"
        'Return JSON: {"actions": [...]}\\n'
        "Allowed action_type: set_node, add_child, add_path, update_cmb, skip_post\\n"
        "Level definitions:\\n"
        "- topic: broad recurring theme\\n"
        "- subtopic: narrower bucket under a topic\\n"
        "- claim: atomic assertion directly mappable from one post\\n"
        "Rules:\\n"
        "- set_node: classify to existing node; semantic_payload must be {}\\n"
        "- add_child: semantic_payload must include child_name, child_level, child_cmb\\n"
        "- add_path: objective_node_id must be an existing topic node; semantic_payload.nodes must be either:\\n"
        "  * [ {level:'subtopic',...} ]  meaning topic->subtopic\\n"
        "  * [ {level:'subtopic',...}, {level:'claim',...} ] meaning topic->subtopic->claim\\n"
        "- update_cmb: semantic_payload must include new_cmb\\n"
        "- skip_post: for meaningless/noise post\\n"
        "- objective_node_id should be explicit except skip_post\\n"
        "- If post fits existing node, prefer set_node over adding new nodes.\\n"
        "- If off-topic from root topic, use skip_post.\\n"
        "- You may emit multiple actions if needed.\\n"
        "Examples:\\n"
        '- Existing claim fit: {"action_type":"set_node","objective_node_id":"<existing_claim_id>","semantic_payload":{}}\\n'
        '- New subtopic under existing topic: {"action_type":"add_path","objective_node_id":"<existing_topic_id>","semantic_payload":{"nodes":[{"name":"X","level":"subtopic","cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}]}}\\n'
        '- New claim under existing subtopic: {"action_type":"add_child","objective_node_id":"<existing_subtopic_id>","semantic_payload":{"child_name":"Y","child_level":"claim","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}\\n'
        '- Noise/off-topic: {"action_type":"skip_post","objective_node_id":null,"semantic_payload":{}}\\n'
        f"Window: {window_id}\\n"
        f"Post ID: {post_id}\\n"
        f"Best existing claim candidate: {best_candidate_node_id or 'none'}\\n"
        f"Best similarity: {best_similarity:.4f}\\n"
        f"Taxonomy context:\\n{json.dumps(ctx, ensure_ascii=False)}\\n"
        f"Post text: {json.dumps(post_text, ensure_ascii=False)}"
    )
    system = "You are a strict taxonomy maintainer. Output valid JSON only."
    payload = ask_json_with_retries(
        llm,
        prompt,
        system,
        max_parse_attempts=max_parse_attempts,
        trace=trace,
        trace_meta={"call": "propose_post_actions", "window_id": window_id, "post_id": post_id},
    )
    if not payload:
        return [{"action_type": "skip_post", "objective_node_id": None, "objective": "parse_fail", "semantic_payload": {}, "confidence": 0.0, "reasoning_short": "LLM output parse failed"}]

    out: List[Dict[str, Any]] = []
    for item in payload.get("actions", []):
        norm = _normalize_action(item)
        if norm is not None:
            out.append(norm)
    if not out:
        out.append({"action_type": "skip_post", "objective_node_id": None, "objective": "no_valid_action", "semantic_payload": {}, "confidence": 0.0, "reasoning_short": "No valid action parsed"})
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
    prompt = (
        "Review this action cluster and decide whether to apply it now.\\n"
        "Return strict JSON with keys: decision, refined_actions, reason\\n"
        "decision in {approve,reject,defer}.\\n"
        "refined_actions use same action schema as proposals.\\n"
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\\n"
        "Policy:\\n"
        "- Reject duplicated or weakly-supported creation actions.\\n"
        "- Prefer set_node when existing nodes already cover the semantics.\\n"
        "- add_path must preserve allowed shapes: topic->subtopic or topic->subtopic->claim.\\n"
        "- Return only high-confidence refined actions.\\n"
        "Example output: "
        '{"decision":"approve","refined_actions":[{"action_type":"add_child","objective_node_id":"...","semantic_payload":{"child_name":"...","child_level":"claim","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}],"reason":"..."}\\n'
        f"Window: {window_id}\\n"
        f"Cluster:\\n{json.dumps(cluster_record, ensure_ascii=False)}\\n"
        f"Proposal sample for review (sampled_count={len(sampled)} total_cluster_count={len(proposal_records)}):\\n"
        f"{json.dumps(sampled, ensure_ascii=False)}\\n"
        f"Taxonomy context (full):\\n{json.dumps(taxonomy_context(taxonomy, max_nodes=None), ensure_ascii=False)}"
    )
    system = "You are a rigorous taxonomy reviewer. Only approve high-confidence coherent actions."
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
    if decision not in {"approve", "reject", "defer"}:
        decision = "defer"

    refined_actions = []
    for item in payload.get("refined_actions", []):
        norm = _normalize_action(item)
        if norm is not None:
            refined_actions.append(norm)

    return {
        "decision": decision,
        "refined_actions": refined_actions,
        "reason": str(payload.get("reason", "")).strip(),
    }
