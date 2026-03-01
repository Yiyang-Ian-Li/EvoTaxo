from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class EvalNode:
    node_id: str
    name: str
    parent_id: str | None
    definition: str
    status: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(mode: str, device_id: int) -> int:
    if mode == "cpu":
        return -1
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return device_id
    return device_id if torch.cuda.is_available() else -1


def load_nodes(path: str) -> Tuple[Dict[str, EvalNode], str]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    nodes: Dict[str, EvalNode] = {}
    root_id = None
    for r in rows:
        cmb = r.get("cmb", {}) or {}
        node = EvalNode(
            node_id=str(r.get("node_id")),
            name=str(r.get("name", "")).strip(),
            parent_id=r.get("parent_id"),
            definition=str(cmb.get("definition", "")).strip(),
            status=str(r.get("status", "active")),
        )
        nodes[node.node_id] = node
        if node.parent_id is None and node.name.upper() == "ROOT":
            root_id = node.node_id
    if root_id is None:
        raise ValueError("Could not find ROOT node in taxonomy.")
    return nodes, root_id


def semantic_text(node: EvalNode, text_source: str = "auto") -> str:
    mode = (text_source or "auto").strip().lower()
    if mode == "name":
        return node.name
    return node.definition if node.definition else node.name


def paths_to_root(nodes: Dict[str, EvalNode], root_id: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for nid in nodes:
        cur = nid
        path = []
        seen = set()
        while cur is not None:
            if cur in seen:
                raise ValueError("Cycle detected in taxonomy.")
            seen.add(cur)
            path.append(cur)
            parent = nodes[cur].parent_id
            cur = parent if parent in nodes else None
        path = list(reversed(path))
        if path and path[0] == root_id:
            out[nid] = path
    return out


def lca_depth(path_a: Sequence[str], path_b: Sequence[str]) -> int:
    depth = 0
    for x, y in zip(path_a, path_b):
        if x != y:
            break
        depth += 1
    return depth


def wu_palmer(path_a: Sequence[str], path_b: Sequence[str]) -> float:
    lca = lca_depth(path_a, path_b)
    if not path_a or not path_b:
        return 0.0
    return (2.0 * lca) / (len(path_a) + len(path_b))


def cosine_pairwise(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    x = vecs / norms
    return x @ x.T


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    tie_x = 0
    tie_y = 0
    for i in range(n - 1):
        xi = x[i]
        yi = y[i]
        for j in range(i + 1, n):
            dx = xi - x[j]
            dy = yi - y[j]
            if dx == 0 and dy == 0:
                tie_x += 1
                tie_y += 1
            elif dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) // 2
    denom = math.sqrt((n0 - tie_x) * (n0 - tie_y))
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def article(term: str) -> str:
    t = (term or "").strip().lower()
    if not t:
        return "a"
    return "an" if t[0] in {"a", "e", "i", "o", "u"} else "a"


def pluralize(term: str) -> str:
    t = (term or "").strip()
    if not t:
        return "things"
    lower = t.lower()
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        return t[:-1] + "ies"
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return t + "es"
    return t + "s"


def hearst_hypotheses(parent: str, child: str) -> List[str]:
    ap = article(parent)
    ac = article(child)
    pp = pluralize(parent)
    return [
        f"{ac} {child} is a type of {parent}.",
        f"{ac} {child} is an example of {parent}.",
        f"{ac} {child} is {ap} {parent}.",
        f"{ac} {child} is a kind of {parent}.",
        f"{ap} {parent} such as {ac} {child}.",
        f"such {pp} as {child}.",
        f"{ac} {child} or other {pp}.",
        f"{ac} {child} and other {pp}.",
        f"{pp}, including {child}.",
        f"{pp}, especially {child}.",
    ]


def geometric_mean(values: Iterable[float]) -> float:
    vals = [max(float(v), 1e-12) for v in values]
    if not vals:
        return float("nan")
    return float(np.exp(np.mean(np.log(np.array(vals, dtype=np.float64)))))


def active_children(nodes: Dict[str, EvalNode]) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = {nid: [] for nid in nodes}
    for nid, n in nodes.items():
        if n.parent_id in children and n.status == "active":
            children[n.parent_id].append(nid)
    return children


def taxonomy_paths(nodes: Dict[str, EvalNode], root_id: str) -> List[List[str]]:
    children = active_children(nodes)
    paths: List[List[str]] = []

    def dfs(cur: str, path: List[str]) -> None:
        kids = [k for k in children.get(cur, []) if k in nodes and nodes[k].status == "active"]
        if not kids:
            paths.append(path.copy())
            return
        for k in kids:
            path.append(k)
            dfs(k, path)
            path.pop()

    dfs(root_id, [root_id])
    return paths


def taxonomy_levels(nodes: Dict[str, EvalNode], root_id: str) -> List[Dict]:
    children = active_children(nodes)
    levels = []
    for parent_id, child_ids in children.items():
        if parent_id == root_id or not child_ids:
            continue
        levels.append(
            {
                "parent_id": parent_id,
                "parent_name": nodes[parent_id].name,
                "siblings": [nodes[c].name for c in child_ids if c in nodes],
                "child_ids": [c for c in child_ids if c in nodes],
            }
        )
    return levels
