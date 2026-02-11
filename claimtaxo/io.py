from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List

import pandas as pd

from .taxonomy import Taxonomy
from .utils import ensure_dir, write_json, write_jsonl


def save_taxonomy(taxonomy: Taxonomy, output_dir: str) -> None:
    ensure_dir(output_dir)
    nodes = []
    for node in taxonomy.nodes.values():
        nodes.append(
            {
                "node_id": node.node_id,
                "name": node.name,
                "level": node.level,
                "parent_id": node.parent_id,
                "children": node.children,
                "status": node.status,
                "first_seen_slice": node.first_seen_slice,
                "last_seen_slice": node.last_seen_slice,
                "cmb": {
                    "canonical_definition": node.cmb.canonical_definition,
                    "representative_examples": node.cmb.representative_examples,
                    "boundaries": {
                        "include": node.cmb.include_terms,
                        "exclude": node.cmb.exclude_terms,
                    },
                    "definition": node.cmb.canonical_definition,
                    "keywords": node.cmb.include_terms,
                    "examples": node.cmb.representative_examples,
                },
            }
        )
    write_json(os.path.join(output_dir, "taxonomy_nodes.json"), nodes)

    redirects = [r.__dict__ for r in taxonomy.redirects]
    write_json(os.path.join(output_dir, "redirects.json"), redirects)


def save_assignments(assignments: List[Dict[str, Any]], output_dir: str) -> None:
    ensure_dir(output_dir)
    df = pd.DataFrame(assignments)
    df.to_csv(os.path.join(output_dir, "assignments.csv"), index=False)


def save_slice_summary(rows: List[Dict[str, Any]], output_dir: str) -> None:
    ensure_dir(output_dir)
    write_jsonl(os.path.join(output_dir, "slice_summary.jsonl"), rows)


def save_taxonomy_ops(rows: List[Dict[str, Any]], output_dir: str) -> None:
    ensure_dir(output_dir)
    write_jsonl(os.path.join(output_dir, "taxonomy_ops.jsonl"), rows)


def save_stance(rows: List[Dict[str, Any]], output_dir: str) -> None:
    ensure_dir(output_dir)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "stance_estimates.csv"), index=False)
