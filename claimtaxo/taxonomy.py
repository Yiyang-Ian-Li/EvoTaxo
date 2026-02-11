from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class ConceptMemoryBank:
    canonical_definition: str = ""
    representative_examples: List[str] = field(default_factory=list)
    include_terms: List[str] = field(default_factory=list)
    exclude_terms: List[str] = field(default_factory=list)


@dataclass
class TaxonomyNode:
    node_id: str
    name: str
    level: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    cmb: ConceptMemoryBank = field(default_factory=ConceptMemoryBank)
    status: str = "active"  # active | candidate | deprecated
    first_seen_slice: Optional[str] = None
    last_seen_slice: Optional[str] = None


@dataclass
class Redirect:
    source_id: str
    target_id: str
    reason: str
    op_type: str = "manual"
    slice_id: Optional[str] = None
    score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class Taxonomy:
    def __init__(self):
        self.nodes: Dict[str, TaxonomyNode] = {}
        self.redirects: List[Redirect] = []
        self.root_id = self._create_node("ROOT", "root", None)

    def _create_node(self, name: str, level: str, parent_id: Optional[str]) -> str:
        node_id = str(uuid.uuid4())
        self.nodes[node_id] = TaxonomyNode(node_id=node_id, name=name, level=level, parent_id=parent_id)
        if parent_id:
            self.nodes[parent_id].children.append(node_id)
        return node_id

    def add_node(self, name: str, level: str, parent_id: Optional[str]) -> str:
        return self._create_node(name, level, parent_id)

    def set_cmb(
        self,
        node_id: str,
        canonical_definition: str,
        include_terms: List[str],
        representative_examples: List[str],
        exclude_terms: Optional[List[str]] = None,
    ) -> None:
        node = self.nodes[node_id]
        node.cmb = ConceptMemoryBank(
            canonical_definition=canonical_definition,
            representative_examples=representative_examples,
            include_terms=include_terms,
            exclude_terms=exclude_terms or [],
        )

    def mark_seen(self, node_id: str, slice_id: str) -> None:
        node = self.nodes[node_id]
        if node.first_seen_slice is None:
            node.first_seen_slice = slice_id
        node.last_seen_slice = slice_id

    def promote_if_needed(self, node_id: str, min_slices: int = 2) -> None:
        node = self.nodes[node_id]
        if node.first_seen_slice and node.last_seen_slice:
            if node.first_seen_slice != node.last_seen_slice:
                node.status = "active"

    def redirect(
        self,
        source_id: str,
        target_id: str,
        reason: str,
        op_type: str = "manual",
        slice_id: Optional[str] = None,
        score: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.redirects.append(
            Redirect(
                source_id=source_id,
                target_id=target_id,
                reason=reason,
                op_type=op_type,
                slice_id=slice_id,
                score=score,
                details=details or {},
            )
        )

    def canonicalize(self, node_id: str) -> str:
        current = node_id
        visited = set()
        while True:
            if current in visited:
                break
            visited.add(current)
            next_id = None
            for r in self.redirects:
                if r.source_id == current:
                    next_id = r.target_id
                    break
            if next_id is None:
                break
            current = next_id
        return current
