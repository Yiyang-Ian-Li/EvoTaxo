from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict

from .utils import JsonlSink


class AssignmentSink:
    def __init__(self, path: str):
        self.path = path
        self.count = 0
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "post_id",
                    "timestamp",
                    "window_id",
                    "node_id_at_time",
                    "canonical_node_id",
                    "similarity",
                    "mapping_mode",
                ],
            )
            w.writeheader()

    def append(self, row: Dict) -> None:
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "post_id",
                    "timestamp",
                    "window_id",
                    "node_id_at_time",
                    "canonical_node_id",
                    "similarity",
                    "mapping_mode",
                ],
            )
            w.writerow(row)
        self.count += 1


class PrettyJsonAppendSink:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w", encoding="utf-8").close()
        self.count = 0

    def append(self, row: Dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, indent=2))
            f.write("\n")
        self.count += 1


@dataclass
class RunSinks:
    assignment: AssignmentSink
    action_proposals: JsonlSink
    clusters_overview: JsonlSink
    cluster_decisions: JsonlSink
    taxonomy_after_clustering: PrettyJsonAppendSink


def create_run_sinks(output_dir: str) -> RunSinks:
    return RunSinks(
        assignment=AssignmentSink(os.path.join(output_dir, "post_assignments.csv")),
        action_proposals=JsonlSink(os.path.join(output_dir, "action_proposals.jsonl")),
        clusters_overview=JsonlSink(os.path.join(output_dir, "clusters_overview.jsonl")),
        cluster_decisions=JsonlSink(os.path.join(output_dir, "cluster_decisions.jsonl")),
        taxonomy_after_clustering=PrettyJsonAppendSink(os.path.join(output_dir, "taxonomy_after_clustering.jsonl")),
    )
