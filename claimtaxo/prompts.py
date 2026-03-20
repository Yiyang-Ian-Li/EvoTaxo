from __future__ import annotations

import json
from typing import Any, Dict, Optional

from taxonomy import Taxonomy


QUALITY_RUBRIC = (
    "Taxonomy quality rubric:\n"
    "1) Parent-child validity: child must be a true subtype of parent.\n"
    "2) Sibling non-overlap: avoid duplicate/near-duplicate siblings.\n"
    "3) Granularity parity: siblings should have similar specificity.\n"
    "4) Naming clarity: prefer clear, descriptive names (avoid vague single-word labels).\n"
    "5) Scope balance: avoid catch-all buckets and overly narrow/post-specific nodes.\n"
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
        "You are the item-proposal stage of taxonomy maintenance.\n"
        "Responsibility:\n"
        "- Review one input item and propose exactly one candidate taxonomy action.\n"
        "- The item may be a plain submission or a structured example containing sections like [POST_TEXT] and [TOP_LEVEL_COMMENT].\n"
        "- Use all provided sections as context, but focus on the claim-bearing content when deciding the taxonomy action.\n"
        "- Use skip_post only when the item is too noisy, too vague, or otherwise lacks taxonomy-worthy content.\n"
        "\n"
        "Context:\n"
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\n"
        f"Item ID: {post_id}\n"
        f"Taxonomy context:\n{json.dumps(taxonomy_ctx, ensure_ascii=False)}\n"
        f"Item text: {json.dumps(post_text, ensure_ascii=False)}\n"
        "\n"
        "Decision rubric and action-quality rules:\n"
        "- Allowed action_type: add_child, add_path, update_cmb, set_node, skip_post.\n"
        "- objective_node_id must be an existing node except skip_post.\n"
        "- add_child: add exactly one new child under objective_node_id; the child must be clearly narrower than its parent.\n"
        "- add_path: use only for the shape root->topic->subtopic; do not repeat the anchor inside nodes.\n"
        "- update_cmb: use when the node structure is right but its wording/scope should be refined; do not use on root.\n"
        "- set_node: use when an existing topic or subtopic already precisely fits this item and the item should be assigned there directly.\n"
        "- skip_post: use only for items that are too noisy, too vague, or otherwise not useful for taxonomy updates or assignment.\n"
        "- New topic/subtopic names must be specific and discriminative, not broad umbrella areas.\n"
        "- Do not create generic buckets such as opioid use, dependence, recovery, treatment issues, or drug discussion.\n"
        "- Do not use set_node when the item introduces new information or more fine-grained details.\n"
        "- If a proposed child is almost as broad as its parent, do not add it; prefer reuse or update_cmb instead.\n"
        "- Return exactly one action, choosing the single best next step for this item.\n"
        f"{QUALITY_RUBRIC}\n"
        "\n"
        "Output contract:\n"
        "Only return one strict JSON object.\n"
        "The object must include: action_type, objective_node_id (null only for skip_post), action_explanation, post_summary.\n"
        "Reference Patterns:\n"
        '- Add topic under root: {"action_type":"add_child","objective_node_id":"<root_topic_id>","action_explanation":"Add topic X under root.","post_summary":"..."}\n'
        '- Add subtopic under topic: {"action_type":"add_child","objective_node_id":"<existing_topic_id>","action_explanation":"Add subtopic X under topic Y.","post_summary":"..."}\n'
        '- Add root->topic->subtopic: {"action_type":"add_path","objective_node_id":"<root_node_id>","action_explanation":"Add Topic A with Subtopic B under root.","post_summary":"..."}\n'
        '- Update existing node wording: {"action_type":"update_cmb","objective_node_id":"<existing_node_id>","action_explanation":"Refine node wording and scope.","post_summary":"..."}\n'
        '- Assign directly to existing existing node: {"action_type":"set_node","objective_node_id":"<existing_topic_or_subtopic_id>","action_explanation":"Assign this item to existing node X.","post_summary":"..."}\n'
        '- noise: {"action_type":"skip_post","objective_node_id":null,"action_explanation":"Skip this item.","post_summary":"..."}\n'
    )


def build_initial_taxonomy_prompt(
    root_topic: str,
    taxonomy_ctx: Dict[str, Any],
) -> str:
    return (
        "You are the initial-taxonomy bootstrap stage of taxonomy maintenance.\n"
        "Responsibility:\n"
        "- Propose a small, high-precision starting taxonomy under the root topic before post processing begins.\n"
        "- Prefer a sparse, defensible structure over broad coverage.\n"
        "- This taxonomy is for organizing user-generated social media (reddit) texts, not for building a general encyclopedia of the domain.\n"
        "- Return executable refined actions only; these will be validated and applied directly.\n"
        "\n"
        "Context:\n"
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\n"
        f"Current taxonomy context:\n{json.dumps(taxonomy_ctx, ensure_ascii=False)}\n"
        "\n"
        "Decision rubric and action-quality rules:\n"
        "- Allowed action_type: add_child, add_path. Do not use update_cmb or skip_post.\n"
        "- objective_node_id must reference an existing node in the provided taxonomy context.\n"
        "- add_child: add exactly one child under the objective node.\n"
        "- add_path: use only for the shape root->topic->subtopic; do not repeat the anchor inside nodes.\n"
        "- Prefer 3-8 high-confidence topic areas total.\n"
        "- Only add subtopics when they are clearly justified and materially narrower than the topic.\n"
        "- Prefer categories that are observable and useful for assigning real social media posts.\n"
        "- New topic/subtopic names must be specific and discriminative, not broad umbrella areas.\n"
        "- Do not create generic buckets such as opioid use, dependence, recovery, treatment issues, or drug discussion.\n"
        "- Avoid near-duplicate siblings and avoid over-fragmenting the root into many weak categories.\n"
        f"{QUALITY_RUBRIC}\n"
        "\n"
        "Output contract:\n"
        'Return strict JSON: {"refined_actions":[...],"reason":"..."}\n'
        "Each refined action must follow the executable schema below.\n"
        "Reference patterns:\n"
        '- add_child topic: {"action_type":"add_child","objective_node_id":"<root_node_id>","semantic_payload":{"child_name":"...","child_level":"topic","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}}}\n'
        '- add_child subtopic: {"action_type":"add_child","objective_node_id":"<topic_node_id>","semantic_payload":{"child_name":"...","child_level":"subtopic","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}}}\n'
        '- add_path: {"action_type":"add_path","objective_node_id":"<root_node_id>","semantic_payload":{"nodes":[{"name":"...","level":"topic","cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}},{"name":"...","level":"subtopic","cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}}]}}\n'
        "- Keep reason concise (<= 2 sentences).\n"
    )


def build_review_cluster_prompt(
    root_topic: str,
    cluster_brief: Dict[str, Any],
    sampled: Any,
    taxonomy_ctx: Dict[str, Any],
    total_cluster_count: int,
) -> str:
    return (
        "You are the cluster-review stage of taxonomy maintenance.\n"
        "Responsibility:\n"
        "- Decide whether this cluster should be promoted as a candidate to final-review arbitration.\n"
        "- If promoted, provide refined action(s) that represent the dominant intent of this cluster.\n"
        "- If not coherent enough, defer.\n"
        "Semantics:\n"
        "- approve: coherent cluster with a clear shared intent that can be summarized safely.\n"
        "- defer: noisy/mixed/conflicting cluster without a reliable single direction.\n"
        "- approve does NOT apply actions immediately; final apply is decided later in final arbitration.\n"
        "- Sampled proposals are drafts from a weaker upstream model; evaluate intent, not wording, and refine freely.\n"
        "\n"
        "Context:\n"
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\n"
        f"Current taxonomy context (full):\n{json.dumps(taxonomy_ctx, ensure_ascii=False)}\n"
        f"Cluster summary:\n{json.dumps(cluster_brief, ensure_ascii=False)}\n"
        f"Proposal sample for review (sampled_count={len(sampled)} total_cluster_count={total_cluster_count}):\n"
        f"{json.dumps(sampled, ensure_ascii=False)}\n"
        "\n"
        "Decision rubric and action-quality rules:\n"
        "- add_child: use for one clearly justified new child that is narrower than its parent.\n"
        "- add_path: use only for the shape root->topic->subtopic; do not repeat the anchor inside nodes.\n"
        "- update_cmb: use when the node exists but its wording/scope should be refined.\n"
        "- New topic/subtopic names must be specific and discriminative, not broad umbrella areas.\n"
        "- Do not approve generic buckets such as opioid use, dependence, recovery, treatment issues, or drug discussion.\n"
        "- If a proposed child is almost as broad as its parent, reject or rewrite it; prefer reuse or update_cmb instead.\n"
        "- Approve only when a single coherent intent is visible across the cluster.\n"
        "- Defer when proposals are conflicting, inconsistent, or too noisy.\n"
        "- If approved, return only refined actions that represent the cluster intent.\n"
        f"{QUALITY_RUBRIC}\n"
        "\n"
        "Output contract:\n"
        "Return strict JSON with keys: decision, refined_actions, reason\n"
        "decision in {approve,defer}.\n"
        "refined_actions must be strict executable actions using schema:\n"
        '- {"action_type":"add_child|add_path|update_cmb","objective_node_id":"...","semantic_payload":{...}}\n'
        "- Keep reason concise (<= 2 sentences).\n"
        "Reference patterns:\n"
        '- add_child: {"decision":"approve","refined_actions":[{"action_type":"add_child","objective_node_id":"<existing_parent_id>","semantic_payload":{"child_name":"...","child_level":"subtopic","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}}}],"reason":"..."}\n'
        '- add_path: {"decision":"approve","refined_actions":[{"action_type":"add_path","objective_node_id":"<anchor_id>","semantic_payload":{"nodes":[{"name":"...","level":"topic","cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}},{"name":"...","level":"subtopic","cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}}]}}],"reason":"..."}\n'
        '- update_cmb: {"decision":"approve","refined_actions":[{"action_type":"update_cmb","objective_node_id":"<existing_node_id>","semantic_payload":{"new_cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}}}],"reason":"..."}\n'
        "- add_path shape pattern (anchor -> nodes):\n"
        '- root anchor: {"action_type":"add_path","objective_node_id":"<root_node_id>","semantic_payload":{"nodes":[{"level":"topic",...},{"level":"subtopic",...}]}}\n'
    )


def build_final_review_prompt(
    root_topic: str,
    batch_id: str,
    compact_candidates: Any,
    taxonomy_ctx: Dict[str, Any],
) -> str:
    return (
        "You are the final-review arbitration stage of taxonomy maintenance.\n"
        "Responsibility:\n"
        "- Review all cluster-approved candidates and decide which candidate actions to apply now.\n"
        "- Resolve overlaps/conflicts and return a coherent high-quality final set.\n"
        "- Some approved candidates may still be deferred by not selecting them.\n"
        "- Treat provided candidate refined_actions as trusted inputs; focus on conflict resolution and final refinement.\n"
        "\n"
        "Context:\n"
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\n"
        f"Batch: {batch_id}\n"
        f"Candidates:\n{json.dumps(compact_candidates, ensure_ascii=False)}\n"
        f"Taxonomy context:\n{json.dumps(taxonomy_ctx, ensure_ascii=False)}\n"
        "\n"
        "Decision rubric and action-quality rules:\n"
        "- add_child: keep only if the new child is clearly narrower than its parent and not redundant.\n"
        "- add_path: keep only for the shape root->topic->subtopic; do not repeat the anchor inside nodes.\n"
        "- update_cmb: prefer when structure is right but wording/scope needs refinement.\n"
        "- Topic/subtopic names must be specific and discriminative, not broad umbrella areas.\n"
        "- Do not keep generic buckets such as opioid use, dependence, recovery, treatment issues, or drug discussion.\n"
        "- If a candidate node is almost as broad as its parent, prefer reuse or update_cmb instead of adding it.\n"
        "- Keep final actions non-overlapping; if candidates are near-duplicates, keep one.\n"
        f"{QUALITY_RUBRIC}\n"
        "\n"
        "Output contract:\n"
        "Return strict JSON with key: selected.\n"
        'Format: {"selected":[{"candidate_index":0,"refined_actions":[...],"justification":"..."}, ...]}\n'
        "Refined action schema (only these three action types):\n"
        '- add_child: {"action_type":"add_child","objective_node_id":"<existing_parent_id>","semantic_payload":{"child_name":"...","child_level":"topic|subtopic","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}}}\n'
        '- update_cmb: {"action_type":"update_cmb","objective_node_id":"<existing_node_id>","semantic_payload":{"new_cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}}}\n'
        '- add_path: {"action_type":"add_path","objective_node_id":"<anchor_id>","semantic_payload":{"nodes":[...]}}\n'
        "Reference patterns:\n"
        '- {"candidate_index":0,"refined_actions":[{"action_type":"add_child","objective_node_id":"...","semantic_payload":{"child_name":"...","child_level":"subtopic","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}}}]}\n'
        '- {"candidate_index":1,"refined_actions":[{"action_type":"add_path","objective_node_id":"...","semantic_payload":{"nodes":[{"name":"...","level":"topic","cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}},{"name":"...","level":"subtopic","cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}}]}}]}\n'
        '- {"candidate_index":2,"refined_actions":[{"action_type":"update_cmb","objective_node_id":"...","semantic_payload":{"new_cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}}}]}\n'
        "- add_path shape pattern (anchor -> nodes):\n"
        '- root anchor: {"action_type":"add_path","objective_node_id":"<root_node_id>","semantic_payload":{"nodes":[{"level":"topic",...},{"level":"subtopic",...}]}}\n'
        "- candidate_index must refer to provided candidates.\n"
        "- refined_actions must follow allowed schema and contain one or more actions.\n"
        "- justification must explain why this candidate was selected/refined (<= 2 sentences).\n"
    )


def build_repair_prompt(
    root_topic: str,
    batch_id: str,
    invalid_reason: str,
    candidate_compact: Dict[str, Any],
    taxonomy_ctx: Dict[str, Any],
) -> str:
    return (
        "You are the repair stage for one final-review candidate.\n"
        "Responsibility:\n"
        "- Repair invalid/empty refined action(s) for this candidate only.\n"
        "- Do not re-judge cluster coherence; only return executable corrected action(s).\n"
        "\n"
        "Context:\n"
        f"Root topic: {json.dumps(root_topic, ensure_ascii=False)}\n"
        f"Batch: {batch_id}\n"
        f"Invalid reason: {invalid_reason}\n"
        f"Candidate:\n{json.dumps(candidate_compact, ensure_ascii=False)}\n"
        f"Taxonomy context:\n{json.dumps(taxonomy_ctx, ensure_ascii=False)}\n"
        "\n"
        "Decision rubric and action-quality rules:\n"
        "- objective_node_id must reference an existing node.\n"
        "- add_child: return one valid child with child_name, child_level, and child_cmb.\n"
        "- add_path: use only the shape root->topic->subtopic.\n"
        "- update_cmb: return only a valid new_cmb payload for an existing non-root node.\n"
        "\n"
        "Output contract:\n"
        "Return strict JSON with key: refined_actions\n"
        'Format: {"refined_actions":[{...}, ...]}\n'
        "Action schema options (only these three action types):\n"
        '- add_child: {"action_type":"add_child","objective_node_id":"<existing_parent_id>","semantic_payload":{"child_name":"...","child_level":"topic|subtopic","child_cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}}}\n'
        '- update_cmb: {"action_type":"update_cmb","objective_node_id":"<existing_node_id>","semantic_payload":{"new_cmb":{"definition":"...","include_terms":[],"exclude_terms":[]}}}\n'
        '- add_path: {"action_type":"add_path","objective_node_id":"<anchor_id>","semantic_payload":{"nodes":[...]}}\n'
        "- Return one or more actions.\n"
    )
