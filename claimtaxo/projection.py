from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from taxonomy import Taxonomy


def build_window_taxonomy_views(
    taxonomy: Taxonomy,
    node_post_links: List[Dict[str, Any]],
    windows: List[str],
) -> List[Dict[str, Any]]:
    node_first_idx: Dict[str, int] = {}
    window_to_idx = {w: i for i, w in enumerate(windows)}

    for link in node_post_links:
        nid = link.get("node_id")
        w = link.get("window_id")
        if nid not in taxonomy.nodes or w not in window_to_idx:
            continue
        idx = window_to_idx[w]
        if nid not in node_first_idx or idx < node_first_idx[nid]:
            node_first_idx[nid] = idx

    out = []
    counts_per_window = defaultdict(lambda: defaultdict(int))
    for link in node_post_links:
        w = link.get("window_id")
        nid = link.get("node_id")
        if w in window_to_idx and nid in taxonomy.nodes:
            counts_per_window[w][nid] += 1

    for w in windows:
        wi = window_to_idx[w]
        visible = set([taxonomy.root_id])
        for nid, fi in node_first_idx.items():
            if fi <= wi:
                cur = nid
                while cur is not None and cur in taxonomy.nodes:
                    visible.add(cur)
                    cur = taxonomy.nodes[cur].parent_id

        rows = []
        for nid in visible:
            n = taxonomy.nodes[nid]
            rows.append(
                {
                    "node_id": nid,
                    "name": n.name,
                    "level": n.level,
                    "parent_id": n.parent_id,
                    "post_count_in_window": counts_per_window[w].get(nid, 0),
                }
            )
        out.append({"window_id": w, "nodes": rows})

    return out


def build_final_burst_summary(
    bursts: List[Dict[str, Any]],
    taxonomy: Taxonomy,
) -> Dict[str, Any]:
    approved = [b for b in bursts if str(b.get("decision", "")).lower() == "approve"]
    approved.sort(key=lambda x: (str(x.get("window_id", "")), str(x.get("cluster_id", ""))))

    timeline = []
    affected_nodes: Dict[str, Dict[str, Any]] = {}
    affected_actions: Dict[str, int] = defaultdict(int)

    for b in approved:
        node_id = b.get("objective_node_id")
        action_type = str(b.get("action_type", ""))
        affected_actions[action_type] += 1

        node_name = None
        node_level = None
        if node_id in taxonomy.nodes:
            node_name = taxonomy.nodes[node_id].name
            node_level = taxonomy.nodes[node_id].level
            if node_id not in affected_nodes:
                affected_nodes[node_id] = {
                    "node_id": node_id,
                    "name": node_name,
                    "level": node_level,
                    "approved_temporal_clusters": 0,
                    "windows": set(),
                    "actions": defaultdict(int),
                }
            affected_nodes[node_id]["approved_temporal_clusters"] += 1
            affected_nodes[node_id]["windows"].add(str(b.get("window_id", "")))
            affected_nodes[node_id]["actions"][action_type] += 1

        timeline.append(
            {
                "window_id": b.get("window_id"),
                "cluster_id": b.get("cluster_id"),
                "action_type": action_type,
                "objective_node_id": node_id,
                "objective_node_name": node_name,
                "objective_node_level": node_level,
                "size": int(b.get("size", 0)),
                "quality": b.get("quality", {}),
                "refined_actions": b.get("refined_actions", []),
                "refined_action_count": int(b.get("refined_action_count", 0)),
            }
        )

    nodes_out = []
    for _, v in affected_nodes.items():
        nodes_out.append(
            {
                "node_id": v["node_id"],
                "name": v["name"],
                "level": v["level"],
                "approved_temporal_clusters": int(v["approved_temporal_clusters"]),
                "windows": sorted([w for w in v["windows"] if w]),
                "actions": dict(v["actions"]),
            }
        )
    nodes_out.sort(key=lambda x: (-x["approved_temporal_clusters"], x["name"] or "", x["node_id"]))

    return {
        "approved_temporal_cluster_count": len(approved),
        "affected_action_counts": dict(sorted(affected_actions.items())),
        "affected_nodes": nodes_out,
        "timeline": timeline,
    }
