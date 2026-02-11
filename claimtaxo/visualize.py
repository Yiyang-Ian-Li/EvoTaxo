from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .utils import ensure_dir


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_outputs(input_dir: str) -> Dict[str, Any]:
    required = {
        "taxonomy_nodes": os.path.join(input_dir, "taxonomy_nodes.json"),
        "assignments": os.path.join(input_dir, "assignments.csv"),
        "slice_summary": os.path.join(input_dir, "slice_summary.jsonl"),
        "redirects": os.path.join(input_dir, "redirects.json"),
        "taxonomy_ops": os.path.join(input_dir, "taxonomy_ops.jsonl"),
    }
    for name, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required output file: {name} at {path}")

    return {
        "taxonomy_nodes": _read_json(required["taxonomy_nodes"]),
        "assignments": pd.read_csv(required["assignments"]),
        "slice_summary": pd.DataFrame(_read_jsonl(required["slice_summary"])),
        "redirects": pd.DataFrame(_read_json(required["redirects"])),
        "taxonomy_ops": pd.DataFrame(_read_jsonl(required["taxonomy_ops"])),
    }


def _write_html(fig: go.Figure, path: str) -> None:
    # Self-contained HTML avoids blank plots when CDN is blocked/offline.
    fig.write_html(path, include_plotlyjs=True)


def build_slice_coverage_plot(slice_summary: pd.DataFrame) -> go.Figure:
    data = slice_summary.sort_values("slice_id").copy()
    fig = px.line(
        data,
        x="slice_id",
        y="coverage",
        markers=True,
        title="Slice Coverage Over Time",
        labels={"slice_id": "Time Slice", "coverage": "Coverage"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def build_node_activity_heatmap(assignments: pd.DataFrame, taxonomy_nodes: List[Dict[str, Any]]) -> go.Figure:
    node_name = {n["node_id"]: n["name"] for n in taxonomy_nodes}
    df = assignments.dropna(subset=["canonical_node_id"]).copy()
    counts = (
        df.groupby(["slice_id", "canonical_node_id"], as_index=False)["post_id"]
        .count()
        .rename(columns={"post_id": "count"})
    )
    counts["node_label"] = counts["canonical_node_id"].map(node_name).fillna(counts["canonical_node_id"])
    top_nodes = (
        counts.groupby("canonical_node_id")["count"].sum().sort_values(ascending=False).head(30).index.tolist()
    )
    counts = counts[counts["canonical_node_id"].isin(top_nodes)].copy()
    pivot = counts.pivot_table(index="node_label", columns="slice_id", values="count", fill_value=0)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Blues",
            colorbar={"title": "Post Count"},
        )
    )
    fig.update_layout(
        title="Top Canonical Argument Node Activity Across Time Slices",
        xaxis_title="Time Slice",
        yaxis_title="Canonical Node",
    )
    return fig


def build_taxonomy_sunburst(taxonomy_nodes: List[Dict[str, Any]]) -> go.Figure:
    df = pd.DataFrame(taxonomy_nodes)
    node_ids = set(df["node_id"].tolist())
    # Keep hierarchy unambiguous by using stable ids. Names are only display labels.
    df["id"] = df["node_id"]
    df["parent"] = df["parent_id"].where(df["parent_id"].isin(node_ids), "")
    df["label"] = df["name"]
    color_map = {"active": "#2E8B57", "candidate": "#F4A261", "deprecated": "#A0A0A0"}
    marker_colors = [color_map.get(status, "#4C78A8") for status in df["status"].tolist()]
    fig = go.Figure(
        go.Sunburst(
            ids=df["id"],
            labels=df["label"],
            parents=df["parent"],
            marker={"colors": marker_colors},
            customdata=df[["status", "level"]],
            hovertemplate=(
                "label=%{label}<br>"
                "status=%{customdata[0]}<br>"
                "level=%{customdata[1]}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(title="Final Taxonomy Structure (Status Colored)")
    return fig


def build_redirect_sankey(redirects: pd.DataFrame, taxonomy_nodes: List[Dict[str, Any]]) -> go.Figure:
    if redirects.empty:
        fig = go.Figure()
        fig.update_layout(title="Redirect Graph: no redirects found")
        return fig

    id_to_name = {n["node_id"]: n["name"] for n in taxonomy_nodes}
    redirects = redirects.copy()
    redirects["source_label"] = redirects["source_id"].map(id_to_name).fillna(redirects["source_id"])
    redirects["target_label"] = redirects["target_id"].map(id_to_name).fillna(redirects["target_id"])

    labels = sorted(set(redirects["source_label"].tolist() + redirects["target_label"].tolist()))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    source = redirects["source_label"].map(label_to_idx).tolist()
    target = redirects["target_label"].map(label_to_idx).tolist()
    value = [1] * len(redirects)
    hover = [
        f"reason={r} | type={t} | slice={s}"
        for r, t, s in zip(
            redirects.get("reason", pd.Series([""] * len(redirects))),
            redirects.get("op_type", pd.Series([""] * len(redirects))),
            redirects.get("slice_id", pd.Series([""] * len(redirects))),
        )
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                node={"label": labels, "pad": 15, "thickness": 12},
                link={"source": source, "target": target, "value": value, "customdata": hover, "hovertemplate": "%{customdata}<extra></extra>"},
            )
        ]
    )
    fig.update_layout(title="Redirect / Merge / Split Flows")
    return fig


def build_ops_timeline(taxonomy_ops: pd.DataFrame) -> go.Figure:
    if taxonomy_ops.empty:
        fig = go.Figure()
        fig.update_layout(title="Taxonomy Operations Timeline: no operations found")
        return fig

    data = taxonomy_ops.groupby(["slice_id", "op_type"], as_index=False).size().rename(columns={"size": "count"})
    fig = px.bar(
        data,
        x="slice_id",
        y="count",
        color="op_type",
        barmode="stack",
        title="Taxonomy Operations by Time Slice",
        labels={"slice_id": "Time Slice", "count": "Operation Count", "op_type": "Operation"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def run_visualization(input_dir: str, output_dir: str) -> None:
    ensure_dir(output_dir)
    data = _load_outputs(input_dir)

    coverage_fig = build_slice_coverage_plot(data["slice_summary"])
    activity_fig = build_node_activity_heatmap(data["assignments"], data["taxonomy_nodes"])
    taxonomy_fig = build_taxonomy_sunburst(data["taxonomy_nodes"])
    redirect_fig = build_redirect_sankey(data["redirects"], data["taxonomy_nodes"])
    ops_fig = build_ops_timeline(data["taxonomy_ops"])

    _write_html(coverage_fig, os.path.join(output_dir, "slice_coverage.html"))
    _write_html(activity_fig, os.path.join(output_dir, "node_activity_heatmap.html"))
    _write_html(taxonomy_fig, os.path.join(output_dir, "taxonomy_sunburst.html"))
    _write_html(redirect_fig, os.path.join(output_dir, "redirect_sankey.html"))
    _write_html(ops_fig, os.path.join(output_dir, "taxonomy_ops_timeline.html"))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ClaimTaxo post-run visualization generator")
    p.add_argument("--input-dir", default="outputs", help="Pipeline output directory")
    p.add_argument("--output-dir", default="outputs/viz", help="Visualization output directory")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_visualization(args.input_dir, args.output_dir)
    print(f"Visualization HTML files generated in: {args.output_dir}")


if __name__ == "__main__":
    main()
