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


def build_final_node_post_counts(
    taxonomy: Taxonomy,
    node_post_links: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    node_to_posts: Dict[str, set[str]] = defaultdict(set)
    for row in node_post_links:
        nid = str(row.get("node_id", "")).strip()
        pid = str(row.get("post_id", "")).strip()
        if nid in taxonomy.nodes and pid:
            node_to_posts[nid].add(pid)

    out: List[Dict[str, Any]] = []
    for nid, node in taxonomy.nodes.items():
        out.append(
            {
                "node_id": nid,
                "name": node.name,
                "level": node.level,
                "parent_id": node.parent_id,
                "post_count": int(len(node_to_posts.get(nid, set()))),
            }
        )
    out.sort(key=lambda x: (-x["post_count"], x["level"], x["name"], x["node_id"]))
    return out
