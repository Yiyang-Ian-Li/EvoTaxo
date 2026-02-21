from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict

from utils import JsonlSink


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


@dataclass
class RunSinks:
    assignment: AssignmentSink
    event_log: JsonlSink
    llm_trace: JsonlSink
    taxonomy_ops: JsonlSink
    cluster_reviews: JsonlSink
    semantic_clusters: JsonlSink
    temporal_clusters: JsonlSink
    window_summary: JsonlSink
    bursts: JsonlSink


def create_run_sinks(output_dir: str) -> RunSinks:
    return RunSinks(
        assignment=AssignmentSink(os.path.join(output_dir, "post_assignments.csv")),
        event_log=JsonlSink(os.path.join(output_dir, "event_log.jsonl")),
        llm_trace=JsonlSink(os.path.join(output_dir, "llm_trace.jsonl")),
        taxonomy_ops=JsonlSink(os.path.join(output_dir, "taxonomy_ops_log.jsonl")),
        cluster_reviews=JsonlSink(os.path.join(output_dir, "cluster_reviews.jsonl")),
        semantic_clusters=JsonlSink(os.path.join(output_dir, "action_clusters_semantic.jsonl")),
        temporal_clusters=JsonlSink(os.path.join(output_dir, "action_clusters_temporal.jsonl")),
        window_summary=JsonlSink(os.path.join(output_dir, "window_summary.jsonl")),
        bursts=JsonlSink(os.path.join(output_dir, "bursts.jsonl")),
    )
