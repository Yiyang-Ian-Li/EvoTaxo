from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Tuple

from dotenv import load_dotenv

from metrics.common import ensure_dir, load_nodes, resolve_device, write_csv, write_json
from metrics.csc import compute_csc
from metrics.llm_client import EvalLLMClient, EvalLLMConfig
from metrics.nliv import compute_nliv
from metrics.path_granularity import compute_path_granularity
from metrics.sibling_coherence import compute_sibling_coherence


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate taxonomy with CSC, NLIV, Path Granularity, Sibling Coherence.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="ClaimTaxo run directory containing taxonomy_nodes_final.json")
    src.add_argument("--taxonomy-json", help="Path to taxonomy_nodes_final.json")
    p.add_argument("--output-dir", default="metrics", help="Directory for metric outputs")
    p.add_argument("--root-topic", default="taxonomy")
    p.add_argument("--embedding-model", default="all-mpnet-base-v2")
    p.add_argument("--nli-model", default="facebook/bart-large-mnli")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--include-root-edges", action="store_true")

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

    csc = compute_csc(nodes, root_id, args.embedding_model, args.batch_size)
    nliv, edge_rows, path_rows = compute_nliv(
        nodes=nodes,
        root_id=root_id,
        model_name=args.nli_model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        include_root_edges=args.include_root_edges,
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
    else:
        path_granularity = float("nan")
        sibling_coherence = float("nan")
        path_granularity_rows = []
        sibling_coherence_rows = []

    csc_x_nliv_s = (
        float(csc["score"] * nliv["nliv_s"])
        if not (math.isnan(csc["score"]) or math.isnan(nliv["nliv_s"]))
        else float("nan")
    )

    base_meta = {
        "taxonomy_json": os.path.abspath(taxonomy_json),
        "embedding_model": args.embedding_model,
        "nli_model": args.nli_model,
        "root_topic": args.root_topic,
        "llm_provider": "openai",
        "llm_model": args.llm_model,
        "include_root_edges": args.include_root_edges,
        "details": {
            "csc_num_nodes": csc["num_nodes"],
            "csc_num_pairs": csc["num_pairs"],
            "nliv_num_edges": nliv["num_edges"],
            "nliv_num_paths": nliv["num_paths"],
            "llm_available": llm.available(),
        },
    }

    metrics_summary = {
        "csc": csc["score"],
        "nliv_s": nliv["nliv_s"],
        "nliv_w": nliv["nliv_w"],
        "path_granularity": path_granularity,
        "sibling_coherence": sibling_coherence,
        "csc_x_nliv_s": csc_x_nliv_s,
        "meta": base_meta,
    }

    write_json(os.path.join(output_dir, "taxonomy_eval_metrics.json"), metrics_summary)
    write_metric_file(os.path.join(output_dir, "csc.json"), "CSC", csc["score"], base_meta)
    write_metric_file(os.path.join(output_dir, "nliv_s.json"), "NLIV-S", nliv["nliv_s"], base_meta)
    write_metric_file(os.path.join(output_dir, "nliv_w.json"), "NLIV-W", nliv["nliv_w"], base_meta)
    write_metric_file(os.path.join(output_dir, "path_granularity.json"), "Path Granularity", path_granularity, base_meta)
    write_metric_file(os.path.join(output_dir, "sibling_coherence.json"), "Sibling Coherence", sibling_coherence, base_meta)
    write_metric_file(os.path.join(output_dir, "csc_x_nliv_s.json"), "CSCxNLIV-S", csc_x_nliv_s, base_meta)

    write_csv(os.path.join(output_dir, "taxonomy_eval_edge_scores.csv"), edge_rows)
    write_csv(os.path.join(output_dir, "taxonomy_eval_path_scores.csv"), path_rows)
    write_csv(os.path.join(output_dir, "path_granularity_details.csv"), path_granularity_rows)
    write_csv(os.path.join(output_dir, "sibling_coherence_details.csv"), sibling_coherence_rows)

    print("Evaluation complete.")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
