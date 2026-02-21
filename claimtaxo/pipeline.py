from __future__ import annotations

import argparse
import dataclasses
import logging
import os

from bootstrap import bootstrap_taxonomy
from config import DEFAULT_CONFIG, PipelineConfig
from data import diversity_sample, load_data
from embeddings import Embedder
from io_sinks import create_run_sinks
from llm import LLMClient
from projection import build_final_burst_summary, build_window_taxonomy_views
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
    sinks = create_run_sinks(cfg.output_dir)

    logger.info(
        "LLM enabled=%s available=%s provider=%s model=%s",
        cfg.llm.enabled,
        llm.available(),
        cfg.llm.provider,
        cfg.llm.model,
    )

    logger.info("Bootstrap: diversity sample n=%d", cfg.bootstrap_sample_size)
    bootstrap_df = diversity_sample(df, embedder, cfg.bootstrap_sample_size)
    bootstrap_taxonomy(
        taxonomy=taxonomy,
        llm=llm,
        sample_df=bootstrap_df,
        cfg=cfg,
        logger=logger,
        event_log=sinks.event_log,
        llm_trace=sinks.llm_trace,
    )
    write_json(os.path.join(cfg.output_dir, "taxonomy_nodes_bootstrap.json"), taxonomy.to_rows())
    logger.info("Bootstrap complete. taxonomy_nodes=%d claim_nodes=%d", len(taxonomy.nodes), len(taxonomy.claim_node_ids()))

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
    final_burst_summary = build_final_burst_summary(loop.bursts, taxonomy)

    write_json(os.path.join(cfg.output_dir, "taxonomy_nodes_final.json"), taxonomy.to_rows())
    write_jsonl(os.path.join(cfg.output_dir, "action_proposals.jsonl"), loop.action_proposals)
    write_json(os.path.join(cfg.output_dir, "bursts_final_summary.json"), final_burst_summary)
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
                "cluster_reviews": sinks.cluster_reviews.count,
            },
        },
    )
    logger.info(
        "Completed. nodes=%d assignments=%d proposals=%d pending=%d reviews=%d",
        len(taxonomy.nodes),
        sinks.assignment.count,
        len(loop.action_proposals),
        len(loop.pending_ids),
        sinks.cluster_reviews.count,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ClaimTaxo MVP")
    p.add_argument("--input", default=DEFAULT_CONFIG.input_path)
    p.add_argument("--output", default=DEFAULT_CONFIG.output_dir)
    p.add_argument("--high-sim", type=float, default=DEFAULT_CONFIG.high_sim_threshold)
    p.add_argument("--bootstrap-n", type=int, default=DEFAULT_CONFIG.bootstrap_sample_size)
    p.add_argument("--llm-provider", default=DEFAULT_CONFIG.llm.provider, choices=["custom", "openai"])
    p.add_argument("--llm-model", default=DEFAULT_CONFIG.llm.model)
    p.add_argument("--llm-api-url", default=DEFAULT_CONFIG.llm.api_url)
    p.add_argument("--llm-api-key-env", default=DEFAULT_CONFIG.llm.api_key_env)
    p.add_argument("--llm-timeout-s", type=int, default=DEFAULT_CONFIG.llm.timeout_s)
    p.add_argument("--llm-max-retries", type=int, default=DEFAULT_CONFIG.llm.max_retries)
    p.add_argument("--llm-max-parse-attempts", type=int, default=DEFAULT_CONFIG.llm.max_parse_attempts)
    p.add_argument("--llm-trace-mode", default=DEFAULT_CONFIG.llm.trace_mode, choices=["off", "compact", "full"])
    p.add_argument("--llm-trace-max-chars", type=int, default=DEFAULT_CONFIG.llm.trace_max_chars)
    p.add_argument("--review-max-examples", type=int, default=DEFAULT_CONFIG.review_max_examples)
    p.add_argument("--review-max-post-chars", type=int, default=DEFAULT_CONFIG.review_max_post_chars)
    p.add_argument("--disable-llm", action="store_true")
    p.add_argument("--window", default=DEFAULT_CONFIG.window_unit, choices=["quarter"])
    p.add_argument("--root-topic", default=DEFAULT_CONFIG.root_topic)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = DEFAULT_CONFIG
    cfg.input_path = args.input
    cfg.output_dir = args.output
    cfg.high_sim_threshold = args.high_sim
    cfg.bootstrap_sample_size = args.bootstrap_n
    cfg.llm.provider = args.llm_provider
    cfg.llm.model = args.llm_model
    cfg.llm.api_url = args.llm_api_url
    cfg.llm.api_key_env = args.llm_api_key_env
    cfg.llm.timeout_s = args.llm_timeout_s
    cfg.llm.max_retries = args.llm_max_retries
    cfg.llm.max_parse_attempts = args.llm_max_parse_attempts
    cfg.llm.trace_mode = args.llm_trace_mode
    cfg.llm.trace_max_chars = args.llm_trace_max_chars
    cfg.review_max_examples = args.review_max_examples
    cfg.review_max_post_chars = args.review_max_post_chars
    cfg.window_unit = args.window
    cfg.root_topic = args.root_topic
    if args.disable_llm:
        cfg.llm.enabled = False

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
