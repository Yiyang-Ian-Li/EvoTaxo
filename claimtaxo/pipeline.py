from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from datetime import datetime

from dotenv import load_dotenv

from config import DEFAULT_CONFIG, PipelineConfig
from data import load_data
from embeddings import Embedder
from io_sinks import create_run_sinks
from llm import LLMClient
from projection import build_final_node_post_counts, build_window_taxonomy_views
from review_loop import process_windows
from taxonomy import Taxonomy
from utils import ensure_dir, now_ts, write_json, write_jsonl


def setup_logger(output_dir: str) -> logging.Logger:
    logger = logging.getLogger("claimtaxo")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(output_dir, "run.log"), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def resolve_output_dir(raw_output: str) -> str:
    out = (raw_output or "").strip() or "results"
    norm = os.path.normpath(out)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(norm, stamp)


def run_pipeline(cfg: PipelineConfig) -> None:
    ensure_dir(cfg.output_dir)
    logger = setup_logger(cfg.output_dir)
    logger.info("Starting ClaimTaxo pipeline")
    logger.info("Input=%s Output=%s RootTopic=%s", cfg.input_path, cfg.output_dir, cfg.root_topic)

    df = load_data(cfg)
    logger.info("Loaded posts=%d (filtered kind=%s)", len(df), cfg.kind_value)

    llm = LLMClient(cfg.llm)
    embedder = Embedder(cfg.embedding)
    taxonomy = Taxonomy()
    # Root-only initialization.
    if len(df):
        first_window = str(df["window_id"].iloc[0])
    else:
        first_window = "INIT"
    taxonomy.nodes[taxonomy.root_id].name = cfg.root_topic
    taxonomy.nodes[taxonomy.root_id].created_at_window = first_window
    taxonomy.nodes[taxonomy.root_id].updated_at_window = first_window
    sinks = create_run_sinks(cfg.output_dir)
    sinks.taxonomy_updates.append(
        {
            "ts": now_ts(),
            "window_id": first_window,
            "trigger": "root_init",
            "action_type": "init",
            "objective_node_id": taxonomy.root_id,
            "post_ids": [],
            "taxonomy_nodes": taxonomy.to_rows(),
        }
    )

    logger.info(
        "LLM enabled=%s available=%s provider=%s model=%s",
        cfg.llm.enabled,
        llm.available(),
        cfg.llm.provider,
        cfg.llm.model,
    )

    logger.info(
        "Root-only init: taxonomy root_id=%s root_name=%s window=%s",
        taxonomy.root_id,
        taxonomy.nodes[taxonomy.root_id].name,
        first_window,
    )

    loop = process_windows(
        cfg=cfg,
        df=df,
        taxonomy=taxonomy,
        llm=llm,
        embedder=embedder,
        sinks=sinks,
        logger=logger,
    )

    taxonomy_views = build_window_taxonomy_views(taxonomy, loop.node_post_links, loop.windows)
    final_node_post_counts = build_final_node_post_counts(taxonomy, loop.node_post_links)

    write_json(os.path.join(cfg.output_dir, "taxonomy_nodes_final.json"), taxonomy.to_rows())
    write_json(os.path.join(cfg.output_dir, "taxonomy_node_post_counts_final.json"), final_node_post_counts)
    write_jsonl(os.path.join(cfg.output_dir, "taxonomy_by_window.jsonl"), taxonomy_views)

    write_json(
        os.path.join(cfg.output_dir, "run_meta.json"),
        {
            "generated_at": now_ts(),
            "config": dataclasses.asdict(cfg),
            "counts": {
                "nodes": len(taxonomy.nodes),
                "assignments": sinks.assignment.count,
                "proposals": len(loop.action_proposals),
                "pending": len(loop.pending_ids),
                "cluster_decisions": sinks.cluster_decisions.count,
            },
        },
    )
    logger.info(
        "Completed. nodes=%d assignments=%d proposals=%d pending=%d cluster_decisions=%d",
        len(taxonomy.nodes),
        sinks.assignment.count,
        len(loop.action_proposals),
        len(loop.pending_ids),
        sinks.cluster_decisions.count,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ClaimTaxo MVP")
    p.add_argument("--input", default=DEFAULT_CONFIG.input_path)
    p.add_argument("--output", default=DEFAULT_CONFIG.output_dir)
    p.add_argument("--high-sim", type=float, default=DEFAULT_CONFIG.high_sim_threshold)
    p.add_argument("--min-year", type=int, default=DEFAULT_CONFIG.min_year)
    p.add_argument("--llm-provider", default=DEFAULT_CONFIG.llm.provider, choices=["custom", "openai", "openrouter"])
    p.add_argument("--llm-model", default=DEFAULT_CONFIG.llm.model)
    p.add_argument("--llm-later-stage-model", default=DEFAULT_CONFIG.llm.later_stage_model)
    p.add_argument("--llm-api-url", default=None)
    p.add_argument("--llm-api-key-env", default=None)
    p.add_argument("--llm-timeout-s", type=int, default=DEFAULT_CONFIG.llm.timeout_s)
    p.add_argument("--llm-max-retries", type=int, default=DEFAULT_CONFIG.llm.max_retries)
    p.add_argument("--llm-max-parse-attempts", type=int, default=DEFAULT_CONFIG.llm.max_parse_attempts)
    p.add_argument("--review-max-examples", type=int, default=DEFAULT_CONFIG.review_max_examples)
    p.add_argument("--review-batch-every-n-posts", type=int, default=DEFAULT_CONFIG.review_batch_every_n_posts)
    p.add_argument("--disable-llm", action="store_true")
    p.add_argument("--window", default=DEFAULT_CONFIG.window_unit, choices=["month", "quarter", "year"])
    p.add_argument("--root-topic", default=DEFAULT_CONFIG.root_topic)
    return p


def main() -> None:
    # Auto-load local .env into process environment (does not override existing env vars).
    load_dotenv(override=False)
    args = build_parser().parse_args()
    cfg = DEFAULT_CONFIG
    cfg.input_path = args.input
    cfg.output_dir = resolve_output_dir(args.output)
    cfg.high_sim_threshold = args.high_sim
    cfg.min_year = args.min_year
    cfg.llm.provider = args.llm_provider
    cfg.llm.model = args.llm_model
    cfg.llm.later_stage_model = args.llm_later_stage_model
    if args.llm_api_url:
        cfg.llm.api_url = args.llm_api_url
    if args.llm_api_key_env:
        cfg.llm.api_key_env = args.llm_api_key_env
    cfg.llm.timeout_s = args.llm_timeout_s
    cfg.llm.max_retries = args.llm_max_retries
    cfg.llm.max_parse_attempts = args.llm_max_parse_attempts
    cfg.review_max_examples = args.review_max_examples
    cfg.review_batch_every_n_posts = args.review_batch_every_n_posts
    cfg.window_unit = args.window
    cfg.root_topic = args.root_topic
    if args.disable_llm:
        cfg.llm.enabled = False

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
