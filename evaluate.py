from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import pandas as pd

from metrics.common import ensure_dir, load_nodes, resolve_device, write_csv, write_json
from metrics.llm_client import EvalLLMClient, EvalLLMConfig
from metrics.nliv import compute_nliv
from metrics.path_granularity import compute_path_granularity
from metrics.post_leaf_confidence import compute_post_leaf_confidence
from metrics.sibling_coherence import compute_sibling_coherence
from metrics.sibling_separability import compute_sibling_separability


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate taxonomy with NLIV, post-leaf confidence, path granularity, sibling coherence, and sibling separability."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="EvoTaxo run directory containing taxonomy_nodes_final.json")
    src.add_argument("--taxonomy-json", help="Path to taxonomy_nodes_final.json")
    p.add_argument("--input-csv", help="Optional input CSV for post-level metrics; auto-resolved from run-dir when possible")
    p.add_argument("--output-dir", default="metrics", help="Directory for metric outputs")
    p.add_argument("--root-topic", default="taxonomy")
    p.add_argument("--embedding-model", default="all-mpnet-base-v2")
    p.add_argument("--nli-model", default="facebook/bart-large-mnli")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--include-root-edges", action="store_true", default=True)
    p.add_argument("--exclude-root-edges", action="store_false", dest="include_root_edges")
    p.add_argument("--node-text-source", choices=["auto", "name"], default="auto")
    p.add_argument("--id-col", default="id")
    p.add_argument("--text-col", default="text")
    p.add_argument("--title-col", default="title")
    p.add_argument("--kind-col", default="kind")
    p.add_argument("--kind-value", default="submissions")
    p.add_argument("--kind-values", nargs="+", default=None)
    p.add_argument("--timestamp-col", default="created_dt")
    p.add_argument("--min-year", type=int, default=2014)
    p.add_argument("--max-post-words", type=int, default=500)

    p.add_argument("--llm-model", default="gpt-4o-mini")
    p.add_argument("--llm-api-url", default=None)
    p.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--llm-timeout-s", type=int, default=60)
    p.add_argument("--llm-max-retries", type=int, default=2)
    return p


def resolve_taxonomy_path(args: argparse.Namespace) -> str:
    if args.run_dir:
        return os.path.join(args.run_dir, "taxonomy_nodes_final.json")
    return args.taxonomy_json


def resolve_input_csv(args: argparse.Namespace) -> Optional[str]:
    if args.input_csv:
        return args.input_csv
    if not args.run_dir:
        return None
    cfg_path = os.path.join(args.run_dir, "config.json")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    input_path = str(cfg.get("input_path", "")).strip()
    if not input_path:
        return None
    if os.path.isabs(input_path):
        return input_path
    run_relative = os.path.abspath(os.path.join(args.run_dir, input_path))
    if os.path.exists(run_relative):
        return run_relative
    return os.path.abspath(input_path)


def load_posts_for_eval(args: argparse.Namespace) -> List[str]:
    input_csv = resolve_input_csv(args)
    if not input_csv or not os.path.exists(input_csv):
        return []
    df = pd.read_csv(input_csv)
    if args.kind_col in df.columns:
        kind_values = list(args.kind_values) if args.kind_values else [args.kind_value]
        df = df[df[args.kind_col].isin(kind_values)].copy()
    if args.timestamp_col in df.columns:
        df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col], errors="coerce")
        df = df.dropna(subset=[args.timestamp_col]).copy()
        df = df[df[args.timestamp_col].dt.year >= args.min_year].copy()

    if args.title_col not in df.columns:
        df[args.title_col] = ""
    if args.text_col not in df.columns:
        raise ValueError(f"text column '{args.text_col}' not found in {input_csv}")

    titles = df[args.title_col].fillna("").astype(str).str.strip()
    bodies = df[args.text_col].fillna("").astype(str).str.strip()
    posts = []
    for title, body in zip(titles.tolist(), bodies.tolist()):
        if title and body:
            text = f"{title}\n\n{body}"
        else:
            text = title or body
        if args.max_post_words > 0:
            text = " ".join(text.split()[: args.max_post_words])
        text = text.strip()
        if text:
            posts.append(text)
    return posts


def write_metric_file(path: str, metric_name: str, value: float, meta: Dict) -> None:
    write_json(path, {"metric": metric_name, "value": value, "meta": meta})


def main() -> None:
    load_dotenv(override=False)
    args = build_parser().parse_args()
    taxonomy_json = resolve_taxonomy_path(args)
    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)

    nodes, root_id = load_nodes(taxonomy_json)
    device = resolve_device(args.device, args.device_id)
    posts = load_posts_for_eval(args)

    nliv, edge_rows, path_rows = compute_nliv(
        nodes=nodes,
        root_id=root_id,
        model_name=args.nli_model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        include_root_edges=args.include_root_edges,
        text_source=args.node_text_source,
    )
    post_leaf_conf = compute_post_leaf_confidence(
        posts=posts,
        nodes=nodes,
        root_id=root_id,
        model_name=args.nli_model,
        device=device,
        batch_size=args.batch_size,
    )

    llm_cfg = EvalLLMConfig(
        provider="openai",
        api_key_env=args.llm_api_key_env,
        api_url=args.llm_api_url,
        model=args.llm_model,
        timeout_s=args.llm_timeout_s,
        max_retries=args.llm_max_retries,
        temperature=0.0,
    )
    llm = EvalLLMClient(llm_cfg)
    if llm.available():
        path_granularity, path_granularity_rows = compute_path_granularity(nodes, root_id, args.root_topic, llm)
        sibling_coherence, sibling_coherence_rows = compute_sibling_coherence(nodes, root_id, args.root_topic, llm)
        sibling_separability, sibling_separability_rows = compute_sibling_separability(nodes, root_id, args.root_topic, llm)
    else:
        path_granularity = float("nan")
        sibling_coherence = float("nan")
        sibling_separability = float("nan")
        path_granularity_rows = []
        sibling_coherence_rows = []
        sibling_separability_rows = []

    base_meta = {
        "taxonomy_json": os.path.abspath(taxonomy_json),
        "embedding_model": args.embedding_model,
        "nli_model": args.nli_model,
        "root_topic": args.root_topic,
        "llm_provider": "openai",
        "llm_model": args.llm_model,
        "node_text_source": args.node_text_source,
        "include_root_edges": args.include_root_edges,
        "details": {
            "nliv_num_edges": nliv["num_edges"],
            "nliv_num_paths": nliv["num_paths"],
            "post_leaf_num_posts": post_leaf_conf["num_posts"],
            "post_leaf_num_leaf_labels": post_leaf_conf["num_leaf_labels"],
            "post_leaf_entropy_normalization": post_leaf_conf.get("entropy_normalization"),
            "post_leaf_num_posts_predicted_others": post_leaf_conf.get("num_posts_predicted_others"),
            "llm_available": llm.available(),
        },
    }

    metrics_summary = {
        "nliv_s": nliv["nliv_s"],
        "nliv_w": nliv["nliv_w"],
        "post_leaf_mean_entropy": post_leaf_conf["mean_entropy"],
        "post_leaf_mean_margin_top1_top2": post_leaf_conf["mean_margin_top1_top2"],
        "post_leaf_others_ratio": post_leaf_conf["others_ratio"],
        "path_granularity": path_granularity,
        "sibling_coherence": sibling_coherence,
        "sibling_separability": sibling_separability,
        "meta": base_meta,
    }

    write_json(os.path.join(output_dir, "taxonomy_eval_metrics.json"), metrics_summary)
    write_metric_file(os.path.join(output_dir, "nliv_s.json"), "NLIV-S", nliv["nliv_s"], base_meta)
    write_metric_file(os.path.join(output_dir, "nliv_w.json"), "NLIV-W", nliv["nliv_w"], base_meta)
    write_metric_file(
        os.path.join(output_dir, "post_leaf_mean_entropy.json"),
        "Post Leaf Mean Normalized Entropy",
        post_leaf_conf["mean_entropy"],
        base_meta,
    )
    write_metric_file(
        os.path.join(output_dir, "post_leaf_mean_margin_top1_top2.json"),
        "Post Leaf Mean Margin Top1-Top2",
        post_leaf_conf["mean_margin_top1_top2"],
        base_meta,
    )
    write_metric_file(
        os.path.join(output_dir, "post_leaf_others_ratio.json"),
        "Post Leaf Others Ratio",
        post_leaf_conf["others_ratio"],
        base_meta,
    )
    write_metric_file(os.path.join(output_dir, "path_granularity.json"), "Path Granularity", path_granularity, base_meta)
    write_metric_file(os.path.join(output_dir, "sibling_coherence.json"), "Sibling Coherence", sibling_coherence, base_meta)
    write_metric_file(
        os.path.join(output_dir, "sibling_separability.json"),
        "Sibling Separability",
        sibling_separability,
        base_meta,
    )

    write_csv(os.path.join(output_dir, "taxonomy_eval_edge_scores.csv"), edge_rows)
    write_csv(os.path.join(output_dir, "taxonomy_eval_path_scores.csv"), path_rows)
    write_csv(os.path.join(output_dir, "path_granularity_details.csv"), path_granularity_rows)
    write_csv(os.path.join(output_dir, "sibling_coherence_details.csv"), sibling_coherence_rows)
    write_csv(os.path.join(output_dir, "sibling_separability_details.csv"), sibling_separability_rows)

    print("Evaluation complete.")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
