from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid


@dataclass
class CMB:
    definition: str = ""
    include_terms: List[str] = field(default_factory=list)
    exclude_terms: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


@dataclass
class Node:
    node_id: str
    name: str
    level: str
    parent_id: Optional[str]
    children: List[str] = field(default_factory=list)
    cmb: CMB = field(default_factory=CMB)
    status: str = "active"
    created_at_window: Optional[str] = None
    updated_at_window: Optional[str] = None


class Taxonomy:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.root_id = self._create_node("ROOT", "topic", None)

    def _create_node(self, name: str, level: str, parent_id: Optional[str]) -> str:
        node_id = str(uuid.uuid4())
        self.nodes[node_id] = Node(node_id=node_id, name=name, level=level, parent_id=parent_id)
        if parent_id:
            self.nodes[parent_id].children.append(node_id)
        return node_id

    def add_node(self, parent_id: str, name: str, level: str, cmb: Dict, window_id: str) -> str:
        if parent_id not in self.nodes:
            parent_id = self.root_id
        node_id = self._create_node(name=name, level=level, parent_id=parent_id)
        self.set_cmb(node_id, cmb, window_id)
        self.nodes[node_id].created_at_window = window_id
        self.nodes[node_id].updated_at_window = window_id
        return node_id

    def set_cmb(self, node_id: str, cmb: Dict, window_id: str) -> None:
        node = self.nodes[node_id]
        node.cmb = CMB(
            definition=str(cmb.get("definition", "")),
            include_terms=[str(x) for x in cmb.get("include_terms", []) if str(x).strip()],
            exclude_terms=[str(x) for x in cmb.get("exclude_terms", []) if str(x).strip()],
            examples=[str(x) for x in cmb.get("examples", []) if str(x).strip()],
        )
        node.updated_at_window = window_id

    def claim_node_ids(self) -> List[str]:
        return [n.node_id for n in self.nodes.values() if n.level == "claim"]

    def node_text(self, node_id: str) -> str:
        n = self.nodes[node_id]
        parts = [n.name, n.cmb.definition, " ".join(n.cmb.include_terms)]
        return " ".join(p for p in parts if p)

    def find_child_by_name(self, parent_id: str, name: str) -> Optional[str]:
        wanted = name.strip().lower()
        for cid in self.nodes[parent_id].children:
            if self.nodes[cid].name.strip().lower() == wanted:
                return cid
        return None

    def to_rows(self) -> List[Dict]:
        rows = []
        for n in self.nodes.values():
            rows.append(
                {
                    "node_id": n.node_id,
                    "name": n.name,
                    "level": n.level,
                    "parent_id": n.parent_id,
                    "children": n.children,
                    "status": n.status,
                    "cmb": {
                        "definition": n.cmb.definition,
                        "include_terms": n.cmb.include_terms,
                        "exclude_terms": n.cmb.exclude_terms,
                        "examples": n.cmb.examples,
                    },
                    "created_at_window": n.created_at_window,
                    "updated_at_window": n.updated_at_window,
                }
            )
        return rows
