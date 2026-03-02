# ClaimTaxo V2 Specification

## 1) Goal
Build a time-evolving claim taxonomy where:
- taxonomy nodes are grounded in posts,
- unmapped posts generate structured action proposals (not immediate edits),
- actions are clustered (semantic and temporal-aware) per window,
- LLM reviews clusters and approves batched taxonomy updates,
- unused actions remain in backlog for future windows.

---

## 2) Core Pipeline

1. Root-only initialization
- Start with only the root node.
- Root node name is set from configured `root_topic`.

2. Chronological processing
- Sort posts by timestamp and process sequentially.
- For each post:
  - map to existing claim node only if similarity >= `high_sim_threshold`.
  - otherwise call LLM to propose zero or more structured actions.
  - if post is meaningless/noise, LLM can emit `skip_post`.
- Store proposals in persistent action backlog; do not mutate taxonomy here.

3. End-of-window clustering
- Window can be `month`, `quarter`, or `year`.
- For each `(action_type, objective_node_id)` group:
  - run semantic HDBSCAN (no time).
  - run temporal-aware HDBSCAN on distance:
    - `d_total = w_sem * d_sem + w_time * d_time_norm`
- Keep both cluster views; identify high-quality clusters for review.

4. LLM cluster review and apply
- Send each high-quality cluster with supporting posts/proposals to LLM.
- LLM outputs decision plus refined action(s): `approve`, `reject`, or `defer`.
- Approved refined actions are converted into concrete taxonomy operations and applied.
- Remove approved proposals from backlog; keep rejected/deferred/backlog residue.

4. Finalization and projection
- After all windows, produce final taxonomy.
- Build per-window taxonomy views from node-post grounding by timestamps (not operation replay).
- Produce burst timeline from temporal clusters.

---

## 3) Data Model

## 3.1 Taxonomy Node
```json
{
  "node_id": "uuid",
  "name": "string",
  "level": "topic|subtopic|claim",
  "parent_id": "uuid|null",
  "children": ["uuid"],
  "status": "active|candidate|frozen",
  "cmb": {
    "definition": "string",
    "include_terms": ["string"],
    "exclude_terms": ["string"]
  },
  "created_at_window": "2024-03",
  "updated_at_window": "2024-04"
}
```

## 3.2 Post Assignment Log
```json
{
  "post_id": "string",
  "timestamp": "iso8601",
  "window_id": "2024-03",
  "node_id_at_time": "uuid|null",
  "canonical_node_id": "uuid|null",
  "similarity": 0.87,
  "mapping_mode": "direct_high_sim|unmapped|post_apply_remap"
}
```

## 3.3 Action Proposal (atomic backlog unit)
```json
{
  "proposal_id": "uuid",
  "post_id": "string",
  "timestamp": "iso8601",
  "window_id": "2024-03",
  "action_type": "set_node|add_child|add_path|update_cmb|skip_post",
  "objective_node_id": "uuid|null",
  "action_explanation": "string",
  "post_summary": "string",
  "status": "pending|clustered|approved|rejected|deferred|applied",
  "cluster_ids": {
    "semantic": "string|null",
    "temporal": "string|null"
  }
}
```

`semantic_payload` contract by action type:
- `set_node`: `{}` (no semantic payload)
- `skip_post`: `{}` (no semantic payload)
- `add_child`:
```json
{
  "child_name": "string",
  "child_level": "topic|subtopic|claim",
  "child_cmb": {
    "definition": "string",
    "include_terms": ["string"],
    "exclude_terms": ["string"]
  }
}
```
- `update_cmb`:
```json
{
  "new_cmb": {
    "definition": "string",
    "include_terms": ["string"],
    "exclude_terms": ["string"]
  }
}
```
- `add_path`:
```json
{
  "nodes": [
    {"name": "New Subtopic", "level": "subtopic", "cmb": {"definition": "...", "include_terms": [], "exclude_terms": []}},
    {"name": "New Claim (optional)", "level": "claim", "cmb": {"definition": "...", "include_terms": [], "exclude_terms": []}}
  ]
}
```
`objective_node_id` is the anchor topic node and must already exist.
Allowed shapes only:
- `topic -> subtopic`
- `topic -> subtopic -> claim`

## 3.4 Cluster Record
```json
{
  "cluster_id": "string",
  "window_id": "2024-03",
  "cluster_mode": "semantic|temporal",
  "action_type": "set_node|add_child|add_path|update_cmb|skip_post",
  "objective_node_id": "uuid|null",
  "proposal_ids": ["uuid"],
  "size": 14,
  "quality": {
    "cohesion": 0.71,
    "stability": 0.66,
    "time_compactness": 0.84
  },
  "is_high_quality": true
}
```

## 3.5 Review Decision
```json
{
  "review_id": "uuid",
  "window_id": "2024-03",
  "cluster_id": "string",
  "decision": "approve|reject|defer",
  "refined_actions": [
    {
      "action_type": "set_node|add_child|add_path|update_cmb|skip_post",
      "objective_node_id": "uuid|null",
      "semantic_payload": {}
    }
  ],
  "approved_operations": [
    {
      "op_type": "set_node|add_child|add_path|update_cmb|skip_post",
      "payload": {}
    }
  ],
  "decision_reason": "string"
}
```

---

## 4) Action Types (V2 minimal set)

Keep action space narrow first to reduce noisy proposals:
- `set_node`: classify one post to an existing node.
- `add_child`: add a child node with full CMB.
- `add_path`: add `topic->subtopic` or `topic->subtopic->claim` from existing topic anchor.
- `update_cmb`: replace/update CMB for objective node.
- `skip_post`: explicit no-op for meaningless post.

Rule: every action must include:
- `action_type`
- `objective_node_id` (or `null` for `skip_post`)
- `objective`

Semantic payload rule:
- only `add_child`, `add_path`, and `update_cmb` carry semantics.
- `set_node` and `skip_post` use empty semantic payload.

---

## 5) Clustering Design

## 5.1 Semantic representation
- Build proposal embedding from:
  - normalized action signature string:
    - `[semantic payload text only]`
- Cluster proposals only inside same `(action_type, objective_node_id)` bucket.
- Use same embedding model as mapping for simplicity in v1 of v2.

## 5.2 Two cluster passes
- Pass A: semantic HDBSCAN on embedding vectors.
- Pass B: temporal-aware HDBSCAN on pairwise distance:
  - `d_total = w_sem * cosine_dist + w_time * |t_i - t_j|/window_span`
  - recommended initial: `w_sem=0.8`, `w_time=0.2`.

## 5.3 High-quality cluster filter
Cluster is reviewable if:
- `size >= min_cluster_size_review`,
- `cohesion >= min_cohesion`,
- and for temporal mode: `time_compactness >= min_time_compactness`.

---

## 6) LLM Review Input/Output Contract

Review input per cluster includes:
- taxonomy snapshot summary around target node,
- clustered proposal summaries,
- representative posts from cluster,
- conflict checks are handled by final LLM review; apply stage does not deduplicate.

LLM output must be strict JSON:
```json
{
  "decision": "approve|reject|defer",
  "refined_actions": [],
  "approved_operations": [],
  "reason": "string"
}
```

`refined_actions` are the key output of review. `approved_operations` are deterministic, validated transformations from refined actions.

---

## 7) Required Artifacts

- `taxonomy_nodes_final.json`
- `taxonomy_ops_log.jsonl`
- `post_assignments.csv`
- `action_proposals.jsonl` (full backlog state transitions)
- `action_clusters_semantic.jsonl`
- `action_clusters_temporal.jsonl`
- `cluster_reviews.jsonl`
- `window_summary.jsonl`
- `bursts.jsonl` (temporal clusters accepted/rejected/deferred)

---

## 8) Module Layout (proposed)

Under `claimtaxo/`:
- `config.py`: thresholds + window/clustering settings.
- `models.py`: dataclasses/pydantic models above.
- `mapping.py`: claim-node mapping and assignment logging.
- `proposer.py`: per-post LLM action proposal generation.
- `backlog.py`: persistence and state transitions for proposals.
- `cluster_semantic.py`: semantic clustering.
- `cluster_temporal.py`: temporal-aware clustering.
- `review.py`: LLM cluster review, JSON validation.
- `apply_ops.py`: deterministic taxonomy mutation engine.
- `projection.py`: build per-window taxonomy views from node-post timestamps.
- `pipeline.py`: orchestration.

---

## 9) Milestone Plan

## Milestone 1 (build now)
- Implement models + file schemas.
- Implement root-only initialization + chronological mapping.
- Implement proposal generation + backlog persistence.
- No clustering/apply yet.
- Deliverables:
  - `action_proposals.jsonl`
  - `post_assignments.csv`

## Milestone 2
- Add semantic + temporal clustering modules.
- Add cluster quality scoring + review candidate selection.

## Milestone 3
- Add LLM review contract + operation apply engine.
- Add projection and burst analytics outputs.

---

## 10) Immediate Open Decisions

1. Mapping strictness:
- exact value of `high_sim_threshold` (default `0.9` in current implementation).

2. Backlog aging:
- should stale proposals auto-expire after `K` windows?

3. Window unit default:
- month (more burst-sensitive) vs quarter (more stable) vs year (more coarse-grained).
