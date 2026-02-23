from __future__ import annotations

import json
from typing import Any, Dict, Optional

from taxonomy import Taxonomy


QUALITY_RUBRIC = (
    "Taxonomy quality rubric:\\n"
    "1) Parent-child validity: child must be a true subtype of parent.\\n"
    "2) Sibling non-overlap: avoid duplicate/near-duplicate siblings.\\n"
    "3) Granularity parity: siblings should have similar specificity.\\n"
    "4) Conservative edits: reuse existing nodes when they already fit.\\n"
    "5) Avoid catch-all inflation: do not create vague general buckets.\\n"
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
        "Review one post and propose taxonomy actions.\\n"
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\\n"
        'Return JSON: {"actions": [...]}\\n'
        "Each action must include: action_type, objective_node_id (null only for skip_post), action_explanation, post_summary.\\n"
        "Allowed action_type: set_node, add_child, add_path, update_cmb, skip_post.\\n"
        "Guidance:\\n"
        "- objective_node_id must be an existing node except skip_post.\\n"
        "- Root node has level='root'.\\n"
        "- Prefer set_node when existing taxonomy already fits.\\n"
        "- add_child is for one new child under objective_node_id.\\n"
        "- add_path shapes only: root->topic->subtopic, root->topic->subtopic->claim, topic->subtopic->claim.\\n"
        "- Use skip_post only for clearly off-topic/noise content.\\n"
        "Examples:\\n"
        '- Existing claim fit: {"action_type":"set_node","objective_node_id":"<existing_claim_id>","action_explanation":"Map to this claim.","post_summary":"..."}\\n'
        '- Add subtopic under topic: {"action_type":"add_child","objective_node_id":"<existing_topic_id>","action_explanation":"Add subtopic X under this topic.","post_summary":"..."}\\n'
        '- Add claim under subtopic: {"action_type":"add_child","objective_node_id":"<existing_subtopic_id>","action_explanation":"Add claim Y under this subtopic.","post_summary":"..."}\\n'
        '- Add root->topic->subtopic: {"action_type":"add_path","objective_node_id":"<root_node_id>","action_explanation":"Add Topic A with Subtopic B.","post_summary":"..."}\\n'
        '- Add root->topic->subtopic->claim: {"action_type":"add_path","objective_node_id":"<root_node_id>","action_explanation":"Add Topic A, Subtopic B, and Claim C.","post_summary":"..."}\\n'
        '- Add topic->subtopic->claim: {"action_type":"add_path","objective_node_id":"<existing_topic_id>","action_explanation":"Add Subtopic B with Claim C.","post_summary":"..."}\\n'
        '- Off-topic/noise: {"action_type":"skip_post","objective_node_id":null,"action_explanation":"Skip this post.","post_summary":"..."}\\n'
        f"Post ID: {post_id}\\n"
        f"Taxonomy context:\\n{json.dumps(taxonomy_ctx, ensure_ascii=False)}\\n"
        f"Post text: {json.dumps(post_text, ensure_ascii=False)}"
    )


def build_review_cluster_prompt(
    root_topic: str,
    window_id: str,
    cluster_brief: Dict[str, Any],
    sampled: Any,
    taxonomy_ctx: Dict[str, Any],
    total_cluster_count: int,
) -> str:
    return (
        "Review this action cluster and decide whether to apply it now.\\n"
        "Return strict JSON with keys: decision, refined_actions, reason\\n"
        "decision in {approve,defer}.\\n"
        "refined_actions must be strict executable actions using schema:\\n"
        '- {"action_type":"set_node|add_child|add_path|update_cmb|skip_post","objective_node_id":"...|null","semantic_payload":{...}}\\n'
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\\n"
        "Root node has level='root'.\\n"
        f"{QUALITY_RUBRIC}\\n"
        "Policy:\\n"
        "- Prefer conservative edits and avoid overlapping duplicates.\\n"
        "- Prefer set_node if existing nodes already cover cluster semantics.\\n"
        "- add_path shapes: root->topic->subtopic OR root->topic->subtopic->claim OR topic->subtopic->claim.\\n"
        "- Return only high-confidence refined actions.\\n"
        "- Keep reason concise (<= 2 sentences).\\n"
        "Example output: "
        '{"decision":"approve","refined_actions":[{"action_type":"add_child","objective_node_id":"...","semantic_payload":{"child_name":"...","child_level":"claim","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}],"reason":"..."}\\n'
        f"Window: {window_id}\\n"
        f"Cluster summary:\\n{json.dumps(cluster_brief, ensure_ascii=False)}\\n"
        f"Proposal sample for review (sampled_count={len(sampled)} total_cluster_count={total_cluster_count}):\\n"
        f"{json.dumps(sampled, ensure_ascii=False)}\\n"
        f"Taxonomy context (full):\\n{json.dumps(taxonomy_ctx, ensure_ascii=False)}"
    )


def build_final_review_prompt(
    root_topic: str,
    batch_id: str,
    compact_candidates: Any,
    taxonomy_ctx: Dict[str, Any],
) -> str:
    return (
        "Review all cluster-approved actions and produce conflict-resolved final refined actions.\\n"
        "Return strict JSON with key: selected.\\n"
        'Format: {"selected":[{"candidate_index":0,"refined_actions":[...]}, ...]}\\n'
        "Refined action schema:\\n"
        '- set_node: {"action_type":"set_node","objective_node_id":"<existing_node_id>","semantic_payload":{}}\\n'
        '- skip_post: {"action_type":"skip_post","objective_node_id":null,"semantic_payload":{}}\\n'
        '- add_child: {"action_type":"add_child","objective_node_id":"<existing_parent_id>","semantic_payload":{"child_name":"...","child_level":"topic|subtopic|claim","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}\\n'
        '- update_cmb: {"action_type":"update_cmb","objective_node_id":"<existing_node_id>","semantic_payload":{"new_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}\\n'
        '- add_path: {"action_type":"add_path","objective_node_id":"<anchor_id>","semantic_payload":{"nodes":[...]}}\\n'
        f"{QUALITY_RUBRIC}\\n"
        "- Root node has level='root'.\\n"
        "Task:\\n"
        "- Select a coherent subset of candidates and refine actions when needed.\\n"
        "- Resolve overlaps/conflicts between candidates.\\n"
        "- Prefer conservative edits and existing-node reuse.\\n"
        "Rules:\\n"
        "- candidate_index must refer to provided candidates.\\n"
        "- refined_actions must follow allowed schema and contain one or more actions.\\n"
        "- Keep final actions non-overlapping.\\n"
        "- If candidates are near-duplicates, keep one.\\n"
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\\n"
        f"Batch: {batch_id}\\n"
        f"Candidates:\\n{json.dumps(compact_candidates, ensure_ascii=False)}\\n"
        f"Taxonomy context:\\n{json.dumps(taxonomy_ctx, ensure_ascii=False)}"
    )


def build_repair_prompt(
    root_topic: str,
    batch_id: str,
    invalid_reason: str,
    candidate_compact: Dict[str, Any],
    taxonomy_ctx: Dict[str, Any],
) -> str:
    return (
        "The previous refined actions are invalid or empty and cannot be applied. Return corrected refined actions.\\n"
        "Return strict JSON with key: refined_actions\\n"
        'Format: {"refined_actions":[{...}, ...]}\\n'
        "Action schema options:\\n"
        '- set_node: {"action_type":"set_node","objective_node_id":"<existing_node_id>","semantic_payload":{}}\\n'
        '- skip_post: {"action_type":"skip_post","objective_node_id":null,"semantic_payload":{}}\\n'
        '- add_child: {"action_type":"add_child","objective_node_id":"<existing_parent_id>","semantic_payload":{"child_name":"...","child_level":"topic|subtopic|claim","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}\\n'
        '- update_cmb: {"action_type":"update_cmb","objective_node_id":"<existing_node_id>","semantic_payload":{"new_cmb":{"definition":"...","include_terms":[],"exclude_terms":[],"examples":[]}}}\\n'
        '- add_path: {"action_type":"add_path","objective_node_id":"<anchor_id>","semantic_payload":{"nodes":[...]}}\\n'
        f"{QUALITY_RUBRIC}\\n"
        "Rules:\\n"
        "- objective_node_id must reference an existing node (except skip_post).\\n"
        "- add_path shapes must be: root->topic->subtopic OR root->topic->subtopic->claim OR topic->subtopic->claim.\\n"
        "- Return one or more actions.\\n"
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\\n"
        f"Batch: {batch_id}\\n"
        f"Invalid reason: {invalid_reason}\\n"
        f"Candidate:\\n{json.dumps(candidate_compact, ensure_ascii=False)}\\n"
        f"Taxonomy context:\\n{json.dumps(taxonomy_ctx, ensure_ascii=False)}"
    )
