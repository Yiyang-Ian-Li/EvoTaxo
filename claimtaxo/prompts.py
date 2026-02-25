from __future__ import annotations

import json
from typing import Any, Dict, Optional

from taxonomy import Taxonomy


QUALITY_RUBRIC = (
    "Taxonomy quality rubric:\\n"
    "1) Parent-child validity: child must be a true subtype of parent.\\n"
    "2) Sibling non-overlap: avoid duplicate/near-duplicate siblings.\\n"
    "3) Granularity parity: siblings should have similar specificity.\\n"
    "4) Naming clarity: prefer clear, descriptive names (avoid vague single-word labels).\\n"
    "5) Scope balance: avoid catch-all buckets and overly narrow/post-specific nodes.\\n"
    "6) Claim quality: claims should be clear propositions, not just labels or slogans.\\n"
)


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
        "root_name": taxonomy.nodes[taxonomy.root_id].name if taxonomy.root_id in taxonomy.nodes else "ROOT",
        "root_level": taxonomy.nodes[taxonomy.root_id].level if taxonomy.root_id in taxonomy.nodes else "root",
        "node_count": len(taxonomy.nodes),
        "nodes": node_rows,
    }


def build_propose_post_prompt(
    root_topic: str,
    post_id: str,
    taxonomy_ctx: Dict[str, Any],
    post_text: str,
) -> str:
    return (
        "You are the post-proposal stage of taxonomy maintenance.\\n"
        "Responsibility:\\n"
        "- Review one post and propose candidate taxonomy action(s).\\n"
        "- Prefer reuse of existing nodes when they already fit.\\n"
        "- Use skip_post only when the post is truly off-topic/noise.\\n"
        "\\n"
        "Context:\\n"
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\\n"
        f"Post ID: {post_id}\\n"
        f"Taxonomy context:\\n{json.dumps(taxonomy_ctx, ensure_ascii=False)}\\n"
        f"Post text: {json.dumps(post_text, ensure_ascii=False)}\\n"
        "\\n"
        "Decision rubric and action-quality rules:\\n"
        "- objective_node_id must be an existing node except skip_post.\\n"
        "- Root node has level='root'.\\n"
        "- Allowed action_type: set_node, add_child, add_path, update_cmb, skip_post.\\n"
        "- Prefer set_node when existing non-root node already fits and the post does not introduce new semantic content or fine-grained details.\\n"
        "- Prefer update_cmb when node structure is correct but definition/include/exclude/examples should be refined.\\n"
        "- set_node should not be used on root node.\\n"
        "- update_cmb should not be used on root node.\\n"
        "- add_child is for one new child under objective_node_id.\\n"
        "- add_path shapes only: root->topic->subtopic, root->topic->subtopic->claim, topic->subtopic->claim.\\n"
        "- For add_path, objective_node_id is the existing anchor node; do not repeat the anchor inside semantic_payload.nodes.\\n"
        f"{QUALITY_RUBRIC}\\n"
        "\\n"
        "Output contract:\\n"
        'Return strict JSON: {"actions": [...]}\\n'
        "Each action must include: action_type, objective_node_id (null only for skip_post), action_explanation, post_summary.\\n"
        "Reference Patterns:\\n"
        '- Existing claim fit: {"action_type":"set_node","objective_node_id":"<existing_claim_id>","action_explanation":"Map to claim X.","post_summary":"..."}\\n'
        '- Add topic under root: {"action_type":"add_child","objective_node_id":"<root_topic_id>","action_explanation":"Add topic X under root.","post_summary":"..."}\\n'
        '- Add subtopic under topic: {"action_type":"add_child","objective_node_id":"<existing_topic_id>","action_explanation":"Add subtopic X under topic Y.","post_summary":"..."}\\n'
        '- Add claim under subtopic: {"action_type":"add_child","objective_node_id":"<existing_subtopic_id>","action_explanation":"Add claim X under subtopic Y.","post_summary":"..."}\\n'
        '- Add root->topic->subtopic: {"action_type":"add_path","objective_node_id":"<root_node_id>","action_explanation":"Add Topic A with Subtopic B under root.","post_summary":"..."}\\n'
        '- Add root->topic->subtopic->claim: {"action_type":"add_path","objective_node_id":"<root_node_id>","action_explanation":"Add Topic A, Subtopic B, and Claim C under root.","post_summary":"..."}\\n'
        '- Add topic->subtopic->claim: {"action_type":"add_path","objective_node_id":"<existing_topic_id>","action_explanation":"Add Subtopic A with Claim B under Topic C.","post_summary":"..."}\\n'
        '- Update existing node wording: {"action_type":"update_cmb","objective_node_id":"<existing_node_id>","action_explanation":"Refine node wording and scope.","post_summary":"..."}\\n'
        '- Off-topic/noise: {"action_type":"skip_post","objective_node_id":null,"action_explanation":"Skip this post.","post_summary":"..."}\\n'
    )


def build_review_cluster_prompt(
    root_topic: str,
    cluster_brief: Dict[str, Any],
    sampled: Any,
    taxonomy_ctx: Dict[str, Any],
    total_cluster_count: int,
) -> str:
    return (
        "You are the cluster-review stage of taxonomy maintenance.\\n"
        "Responsibility:\\n"
        "- Decide whether this cluster should be promoted as a candidate to final-review arbitration.\\n"
        "- If promoted, provide refined action(s) that represent the dominant intent of this cluster.\\n"
        "- If not coherent enough, defer.\\n"
        "Semantics:\\n"
        "- approve: coherent cluster with a clear shared intent that can be summarized safely.\\n"
        "- defer: noisy/mixed/conflicting cluster without a reliable single direction.\\n"
        "- approve does NOT apply actions immediately; final apply is decided later in final arbitration.\\n"
        "- Sampled proposals are drafts from a weaker upstream model; evaluate intent, not wording, and refine freely.\\n"
        "\\n"
        "Context:\\n"
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\\n"
        "Root node has level='root'.\\n"
        f"Current taxonomy context (full):\\n{json.dumps(taxonomy_ctx, ensure_ascii=False)}\\n"
        f"Cluster summary:\\n{json.dumps(cluster_brief, ensure_ascii=False)}\\n"
        f"Proposal sample for review (sampled_count={len(sampled)} total_cluster_count={total_cluster_count}):\\n"
        f"{json.dumps(sampled, ensure_ascii=False)}\\n"
        "\\n"
        "Decision rubric and action-quality rules:\\n"
        f"{QUALITY_RUBRIC}\\n"
        "- Prefer conservative edits and avoid overlapping duplicates.\\n"
        "- Prefer update_cmb when existing node structure is right but wording/scope needs refinement.\\n"
        "- add_path shapes: root->topic->subtopic OR root->topic->subtopic->claim OR topic->subtopic->claim.\\n"
        "- For add_path, objective_node_id is the existing anchor node; do not repeat the anchor inside semantic_payload.nodes.\\n"
        "- For add_child, the intended new child must include child_name, child_level, and child_cmb.\\n"
        "- For add_path, every intended new node in semantic_payload.nodes must include name, level, and cmb.\\n"
        "- Approve only when a single coherent intent is visible across the cluster.\\n"
        "- Defer when proposals are conflicting, inconsistent, or too noisy.\\n"
        "- If approved, return only refined actions that represent the cluster intent.\\n"
        "\\n"
        "Output contract:\\n"
        "Return strict JSON with keys: decision, refined_actions, reason\\n"
        "decision in {approve,defer}.\\n"
        "refined_actions must be strict executable actions using schema:\\n"
        '- {"action_type":"add_child|add_path|update_cmb","objective_node_id":"...","semantic_payload":{...}}\\n'
        "- Keep reason concise (<= 2 sentences).\\n"
        "Reference patterns:\\n"
        '- add_child: {"decision":"approve","refined_actions":[{"action_type":"add_child","objective_node_id":"<existing_parent_id>","semantic_payload":{"child_name":"...","child_level":"claim","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}],"reason":"..."}\\n'
        '- add_path: {"decision":"approve","refined_actions":[{"action_type":"add_path","objective_node_id":"<anchor_id>","semantic_payload":{"nodes":[{"name":"...","level":"topic","cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}},{"name":"...","level":"subtopic","cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}]}}],"reason":"..."}\\n'
        '- update_cmb: {"decision":"approve","refined_actions":[{"action_type":"update_cmb","objective_node_id":"<existing_node_id>","semantic_payload":{"new_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}],"reason":"..."}\\n'
        "- add_path shape patterns (anchor -> nodes):\\n"
        '- root anchor: {"action_type":"add_path","objective_node_id":"<root_node_id>","semantic_payload":{"nodes":[{"level":"topic",...},{"level":"subtopic",...}]}}\\n'
        '- root anchor: {"action_type":"add_path","objective_node_id":"<root_node_id>","semantic_payload":{"nodes":[{"level":"topic",...},{"level":"subtopic",...},{"level":"claim",...}]}}\\n'
        '- topic anchor: {"action_type":"add_path","objective_node_id":"<existing_topic_id>","semantic_payload":{"nodes":[{"level":"subtopic",...},{"level":"claim",...}]}}\\n'
    )


def build_final_review_prompt(
    root_topic: str,
    batch_id: str,
    compact_candidates: Any,
    taxonomy_ctx: Dict[str, Any],
) -> str:
    return (
        "You are the final-review arbitration stage of taxonomy maintenance.\\n"
        "Responsibility:\\n"
        "- Review all cluster-approved candidates and decide which candidate actions to apply now.\\n"
        "- Resolve overlaps/conflicts and return a coherent high-quality final set.\\n"
        "- Some approved candidates may still be deferred by not selecting them.\\n"
        "- Treat provided candidate refined_actions as trusted inputs; focus on conflict resolution and final refinement.\\n"
        "\\n"
        "Context:\\n"
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\\n"
        f"Batch: {batch_id}\\n"
        f"Candidates:\\n{json.dumps(compact_candidates, ensure_ascii=False)}\\n"
        f"Taxonomy context:\\n{json.dumps(taxonomy_ctx, ensure_ascii=False)}\\n"
        "\\n"
        "Decision rubric and action-quality rules:\\n"
        f"{QUALITY_RUBRIC}\\n"
        "- Root node has level='root'.\\n"
        "- Select a coherent subset of candidates and refine actions when needed.\\n"
        "- Prefer conservative edits and existing-node reuse.\\n"
        "- Keep final actions non-overlapping.\\n"
        "- If candidates are near-duplicates, keep one.\\n"
        "- For add_path, objective_node_id is the existing anchor node; do not repeat the anchor inside semantic_payload.nodes.\\n"
        "- For add_child, the intended new child must include child_name, child_level, and child_cmb.\\n"
        "- For add_path, every intended new node in semantic_payload.nodes must include name, level, and cmb.\\n"
        "\\n"
        "Output contract:\\n"
        "Return strict JSON with key: selected.\\n"
        'Format: {"selected":[{"candidate_index":0,"refined_actions":[...],"justification":"..."}, ...]}\\n'
        "Refined action schema (only these three action types):\\n"
        '- add_child: {"action_type":"add_child","objective_node_id":"<existing_parent_id>","semantic_payload":{"child_name":"...","child_level":"topic|subtopic|claim","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}\\n'
        '- update_cmb: {"action_type":"update_cmb","objective_node_id":"<existing_node_id>","semantic_payload":{"new_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}\\n'
        '- add_path: {"action_type":"add_path","objective_node_id":"<anchor_id>","semantic_payload":{"nodes":[...]}}\\n'
        "Reference patterns:\\n"
        '- {"candidate_index":0,"refined_actions":[{"action_type":"add_child","objective_node_id":"...","semantic_payload":{"child_name":"...","child_level":"claim","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}]}\\n'
        '- {"candidate_index":1,"refined_actions":[{"action_type":"add_path","objective_node_id":"...","semantic_payload":{"nodes":[{"name":"...","level":"subtopic","cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}},{"name":"...","level":"claim","cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}]}}]}\\n'
        '- {"candidate_index":2,"refined_actions":[{"action_type":"update_cmb","objective_node_id":"...","semantic_payload":{"new_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}]}\\n'
        "- add_path shape patterns (anchor -> nodes):\\n"
        '- root anchor: {"action_type":"add_path","objective_node_id":"<root_node_id>","semantic_payload":{"nodes":[{"level":"topic",...},{"level":"subtopic",...}]}}\\n'
        '- root anchor: {"action_type":"add_path","objective_node_id":"<root_node_id>","semantic_payload":{"nodes":[{"level":"topic",...},{"level":"subtopic",...},{"level":"claim",...}]}}\\n'
        '- topic anchor: {"action_type":"add_path","objective_node_id":"<existing_topic_id>","semantic_payload":{"nodes":[{"level":"subtopic",...},{"level":"claim",...}]}}\\n'
        "- candidate_index must refer to provided candidates.\\n"
        "- refined_actions must follow allowed schema and contain one or more actions.\\n"
        "- justification must explain why this candidate was selected/refined (<= 2 sentences).\\n"
    )


def build_repair_prompt(
    root_topic: str,
    batch_id: str,
    invalid_reason: str,
    candidate_compact: Dict[str, Any],
    taxonomy_ctx: Dict[str, Any],
) -> str:
    return (
        "You are the repair stage for one final-review candidate.\\n"
        "Responsibility:\\n"
        "- Repair invalid/empty refined action(s) for this candidate only.\\n"
        "- Do not re-judge cluster coherence; only return executable corrected action(s).\\n"
        "\\n"
        "Context:\\n"
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\\n"
        f"Batch: {batch_id}\\n"
        f"Invalid reason: {invalid_reason}\\n"
        f"Candidate:\\n{json.dumps(candidate_compact, ensure_ascii=False)}\\n"
        f"Taxonomy context:\\n{json.dumps(taxonomy_ctx, ensure_ascii=False)}\\n"
        "\\n"
        "Decision rubric and action-quality rules:\\n"
        f"{QUALITY_RUBRIC}\\n"
        "- objective_node_id must reference an existing node (except skip_post).\\n"
        "- add_path shapes must be: root->topic->subtopic OR root->topic->subtopic->claim OR topic->subtopic->claim.\\n"
        "- For add_child, the intended new child must include child_name, child_level, and child_cmb.\\n"
        "- For add_path, every intended new node in semantic_payload.nodes must include name, level, and cmb.\\n"
        "\\n"
        "Output contract:\\n"
        "Return strict JSON with key: refined_actions\\n"
        'Format: {"refined_actions":[{...}, ...]}\\n'
        "Action schema options (only these three action types):\\n"
        '- add_child: {"action_type":"add_child","objective_node_id":"<existing_parent_id>","semantic_payload":{"child_name":"...","child_level":"topic|subtopic|claim","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}\\n'
        '- update_cmb: {"action_type":"update_cmb","objective_node_id":"<existing_node_id>","semantic_payload":{"new_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}\\n'
        '- add_path: {"action_type":"add_path","objective_node_id":"<anchor_id>","semantic_payload":{"nodes":[...]}}\\n'
        "- Return one or more actions.\\n"
    )
