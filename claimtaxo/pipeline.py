from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from .clustering import kmeans_best_k, reduce_embeddings
from .config import DEFAULT_CONFIG, PipelineConfig
from .embeddings import Embedder
from .io import save_assignments, save_slice_summary, save_stance, save_taxonomy, save_taxonomy_ops
from .llm import LLMClient
from .stance import estimate_stance_distribution
from .taxonomy import Taxonomy
from .taxonomy_ops import run_merge_pass, run_split_pass, update_lifecycle
from .utils import ensure_dir, safe_text
from .visualize import run_visualization


LEVEL_TOPIC = "topic"
LEVEL_CLAIM = "claim"
LEVEL_ARGUMENT = "argument"


def setup_logger(cfg: PipelineConfig) -> logging.Logger:
    logger = logging.getLogger("claimtaxo")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if cfg.logging.log_to_file:
        file_path = os.path.join(cfg.output_dir, cfg.logging.file_name)
        file_handler = logging.FileHandler(file_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def load_data(cfg: PipelineConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_path)
    df = df[df[cfg.kind_col] == cfg.kind_value].copy()
    df[cfg.slicing.timestamp_col] = pd.to_datetime(df[cfg.slicing.timestamp_col], errors="coerce")
    df = df.dropna(subset=[cfg.slicing.timestamp_col])

    # Note: current dataset's `text` already contains title. If that changes in the future,
    # use `use_title_if_missing` or combine title + text explicitly.
    def build_text(row: pd.Series) -> str:
        text = safe_text(row.get(cfg.text_col, ""))
        if cfg.use_title_if_missing and (not text.strip()):
            title = safe_text(row.get(cfg.title_col, ""))
            return title
        return text

    df["_text"] = df.apply(build_text, axis=1)
    df = df[df["_text"].str.strip().astype(bool)].copy()
    return df


def add_slice_id(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    if cfg.slicing.time_unit != "quarter":
        raise ValueError("Only quarter slicing is supported in this pipeline.")
    df["slice_id"] = df[cfg.slicing.timestamp_col].dt.to_period("Q").astype(str)
    return df


def summarize_cluster_heuristic(texts: List[str]) -> Tuple[str, str, List[str], List[str], List[str]]:
    vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    scores = np.asarray(X.mean(axis=0)).ravel()
    top_idx = scores.argsort()[::-1][:5]
    keywords = [terms[i] for i in top_idx]
    name = keywords[0] if keywords else "cluster"
    definition = " ".join(keywords)
    examples = texts[:3]
    return name, definition, keywords, examples, []


def _parse_json_object(text: str) -> Dict[str, Any]:
    raw = text.strip()
    if not raw:
        raise json.JSONDecodeError("empty", raw, 0)

    # Common model pattern: fenced JSON block
    fenced = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced)
    fenced = fenced.strip()

    candidates = [raw, fenced]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Last resort: extract first JSON object substring
    start = fenced.find("{")
    end = fenced.rfind("}")
    if start >= 0 and end > start:
        return json.loads(fenced[start : end + 1])
    raise json.JSONDecodeError("no json object found", raw, 0)


def summarize_cluster_llm(llm: LLMClient, texts: List[str]) -> Tuple[str, str, List[str], List[str], List[str]]:
    logger = logging.getLogger("claimtaxo")
    sample = texts[:5]
    prompt = (
        "Summarize the cluster of posts. Return JSON with keys: "
        "name (short phrase), definition (1-2 sentences), keywords (list), "
        "examples (list), exclude_terms (list).\n\n"
        f"Posts: {json.dumps(sample, ensure_ascii=False)}"
    )
    t0 = time.perf_counter()
    resp = llm.chat(prompt, response_format={"type": "json_object"})
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if not resp:
        logger.info("LLM summarize fallback=heuristic reason=empty_response elapsed_ms=%.1f n_texts=%d", elapsed_ms, len(texts))
        return summarize_cluster_heuristic(texts)
    try:
        data = _parse_json_object(resp)
        name = str(data.get("name", "cluster"))
        definition = str(data.get("definition", ""))
        keywords = list(data.get("keywords", []))
        examples = list(data.get("examples", []))
        exclude_terms = list(data.get("exclude_terms", []))
        if not keywords:
            keywords = [name]
        if not examples:
            examples = texts[:3]
        logger.info("LLM summarize ok elapsed_ms=%.1f n_texts=%d name=%s", elapsed_ms, len(texts), name[:80])
        return name, definition, keywords, examples, exclude_terms
    except json.JSONDecodeError:
        logger.info(
            "LLM summarize fallback=heuristic reason=json_parse_error elapsed_ms=%.1f n_texts=%d",
            elapsed_ms,
            len(texts),
        )
        return summarize_cluster_heuristic(texts)


def summarize_cluster(llm: LLMClient, texts: List[str]) -> Tuple[str, str, List[str], List[str], List[str]]:
    if llm.available():
        return summarize_cluster_llm(llm, texts)
    return summarize_cluster_heuristic(texts)


def build_initial_taxonomy(
    taxonomy: Taxonomy,
    llm: LLMClient,
    embedder: Embedder,
    cfg: PipelineConfig,
    texts: List[str],
) -> None:
    logger = logging.getLogger("claimtaxo")
    t0 = time.perf_counter()
    embeddings = embedder.encode(texts)
    logger.info("Initial taxonomy: embedding_done n_texts=%d elapsed_s=%.2f", len(texts), time.perf_counter() - t0)
    t1 = time.perf_counter()
    reduced = reduce_embeddings(embeddings, cfg.clustering)
    topic_clusters = kmeans_best_k(reduced, cfg.clustering)
    logger.info(
        "Initial taxonomy: topic_clustering_done n_topics=%d elapsed_s=%.2f",
        topic_clusters.n_clusters,
        time.perf_counter() - t1,
    )

    for topic_id in range(topic_clusters.n_clusters):
        topic_t0 = time.perf_counter()
        idx = np.where(topic_clusters.labels == topic_id)[0]
        topic_texts = [texts[i] for i in idx]
        topic_name, definition, include_terms, examples, exclude_terms = summarize_cluster(llm, topic_texts)
        topic_node_id = taxonomy.add_node(topic_name, LEVEL_TOPIC, taxonomy.root_id)
        taxonomy.set_cmb(
            topic_node_id,
            canonical_definition=definition,
            include_terms=include_terms,
            representative_examples=examples,
            exclude_terms=exclude_terms,
        )

        # Claim level
        topic_emb = embeddings[idx]
        claim_t0 = time.perf_counter()
        claim_clusters = kmeans_best_k(reduce_embeddings(topic_emb, cfg.clustering), cfg.clustering)
        logger.info(
            "Initial taxonomy: topic_index=%d topic_size=%d claim_clusters=%d cluster_elapsed_s=%.2f",
            topic_id,
            len(topic_texts),
            claim_clusters.n_clusters,
            time.perf_counter() - claim_t0,
        )
        for claim_id in range(claim_clusters.n_clusters):
            cidx = idx[np.where(claim_clusters.labels == claim_id)[0]]
            claim_texts = [texts[i] for i in cidx]
            claim_name, cdef, ckw, cex, cexcl = summarize_cluster(llm, claim_texts)
            claim_node_id = taxonomy.add_node(claim_name, LEVEL_CLAIM, topic_node_id)
            taxonomy.set_cmb(
                claim_node_id,
                canonical_definition=cdef,
                include_terms=ckw,
                representative_examples=cex,
                exclude_terms=cexcl,
            )

            # Argument level
            claim_emb = embeddings[cidx]
            arg_t0 = time.perf_counter()
            arg_clusters = kmeans_best_k(reduce_embeddings(claim_emb, cfg.clustering), cfg.clustering)
            logger.info(
                "Initial taxonomy: topic_index=%d claim_index=%d claim_size=%d arg_clusters=%d cluster_elapsed_s=%.2f",
                topic_id,
                claim_id,
                len(claim_texts),
                arg_clusters.n_clusters,
                time.perf_counter() - arg_t0,
            )
            for arg_id in range(arg_clusters.n_clusters):
                aidx = cidx[np.where(arg_clusters.labels == arg_id)[0]]
                arg_texts = [texts[i] for i in aidx]
                arg_name, adef, akw, aex, aexcl = summarize_cluster(llm, arg_texts)
                arg_node_id = taxonomy.add_node(arg_name, LEVEL_ARGUMENT, claim_node_id)
                taxonomy.set_cmb(
                    arg_node_id,
                    canonical_definition=adef,
                    include_terms=akw,
                    representative_examples=aex,
                    exclude_terms=aexcl,
                )
        logger.info(
            "Initial taxonomy: topic_index=%d finished topic_elapsed_s=%.2f nodes_so_far=%d",
            topic_id,
            time.perf_counter() - topic_t0,
            len(taxonomy.nodes),
        )


def node_text(taxonomy: Taxonomy, node_id: str) -> str:
    node = taxonomy.nodes[node_id]
    cmb = node.cmb
    parts = [node.name, cmb.canonical_definition, " ".join(cmb.include_terms)]
    return " ".join(p for p in parts if p)


def compute_argument_embeddings(taxonomy: Taxonomy, embedder: Embedder) -> Tuple[List[str], np.ndarray]:
    arg_ids = [n.node_id for n in taxonomy.nodes.values() if n.level == LEVEL_ARGUMENT]
    texts = [node_text(taxonomy, nid) for nid in arg_ids]
    if not texts:
        return [], np.zeros((0, 1))
    emb = embedder.encode(texts)
    return arg_ids, emb


def map_posts_to_arguments(
    post_emb: np.ndarray,
    arg_ids: List[str],
    arg_emb: np.ndarray,
    min_similarity: float,
) -> Tuple[List[int], List[float]]:
    if arg_emb.shape[0] == 0:
        return [-1] * len(post_emb), [0.0] * len(post_emb)
    sims = cosine_similarity(post_emb, arg_emb)
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    mapped = []
    scores = []
    for i in range(len(post_emb)):
        if best_sim[i] >= min_similarity:
            mapped.append(best_idx[i])
            scores.append(float(best_sim[i]))
        else:
            mapped.append(-1)
            scores.append(float(best_sim[i]))
    return mapped, scores


def expand_unmapped(
    taxonomy: Taxonomy,
    llm: LLMClient,
    embedder: Embedder,
    cfg: PipelineConfig,
    slice_id: str,
    texts: List[str],
    embeddings: np.ndarray,
) -> List[str]:
    reduced = reduce_embeddings(embeddings, cfg.clustering)
    clusters = kmeans_best_k(reduced, cfg.clustering)

    new_arg_ids: List[str] = []
    topic_nodes = [n.node_id for n in taxonomy.nodes.values() if n.level == LEVEL_TOPIC]
    topic_texts = [node_text(taxonomy, nid) for nid in topic_nodes]
    topic_emb = embedder.encode(topic_texts) if topic_texts else np.zeros((0, embeddings.shape[1]))

    for cluster_id in range(clusters.n_clusters):
        idx = np.where(clusters.labels == cluster_id)[0]
        cluster_texts = [texts[i] for i in idx]
        name, definition, include_terms, examples, exclude_terms = summarize_cluster(llm, cluster_texts)

        parent_topic_id = taxonomy.root_id
        if topic_emb.shape[0] > 0:
            centroid = embeddings[idx].mean(axis=0, keepdims=True)
            sim = cosine_similarity(centroid, topic_emb).ravel()
            best = sim.argmax()
            if sim[best] >= 0.4:
                parent_topic_id = topic_nodes[best]
        if parent_topic_id == taxonomy.root_id:
            topic_node_id = taxonomy.add_node(name, LEVEL_TOPIC, taxonomy.root_id)
            taxonomy.set_cmb(
                topic_node_id,
                canonical_definition=definition,
                include_terms=include_terms,
                representative_examples=examples,
                exclude_terms=exclude_terms,
            )
        else:
            topic_node_id = parent_topic_id

        claim_node_id = taxonomy.add_node(name, LEVEL_CLAIM, topic_node_id)
        taxonomy.set_cmb(
            claim_node_id,
            canonical_definition=definition,
            include_terms=include_terms,
            representative_examples=examples,
            exclude_terms=exclude_terms,
        )

        arg_node_id = taxonomy.add_node(name, LEVEL_ARGUMENT, claim_node_id)
        taxonomy.set_cmb(
            arg_node_id,
            canonical_definition=definition,
            include_terms=include_terms,
            representative_examples=examples,
            exclude_terms=exclude_terms,
        )
        taxonomy.nodes[arg_node_id].status = "candidate"
        taxonomy.mark_seen(arg_node_id, slice_id)
        new_arg_ids.append(arg_node_id)

    return new_arg_ids


def run_pipeline(cfg: PipelineConfig = DEFAULT_CONFIG) -> None:
    ensure_dir(cfg.output_dir)
    logger = setup_logger(cfg)
    logger.info("Starting ClaimTaxo pipeline")
    logger.info("Input=%s Output=%s", cfg.input_path, cfg.output_dir)

    df = load_data(cfg)
    df = add_slice_id(df, cfg)
    logger.info("Loaded %d posts after filtering", len(df))

    llm = LLMClient(cfg.llm)
    embedder = Embedder(cfg.embedding)
    taxonomy = Taxonomy()
    logger.info("LLM enabled=%s available=%s", cfg.llm.enabled, llm.available())

    # Initial taxonomy from full corpus
    texts = df["_text"].tolist()
    logger.info("Building initial taxonomy from %d texts", len(texts))
    build_initial_taxonomy(taxonomy, llm, embedder, cfg, texts)
    logger.info("Initial taxonomy built with %d nodes", len(taxonomy.nodes))

    assignments: List[Dict[str, Any]] = []
    slice_summary: List[Dict[str, Any]] = []
    stance_rows: List[Dict[str, Any]] = []
    op_rows: List[Dict[str, Any]] = []
    support_totals: Dict[str, int] = {}

    for slice_id, sdf in df.sort_values(cfg.slicing.timestamp_col).groupby("slice_id"):
        slice_texts = sdf["_text"].tolist()
        post_ids = sdf[cfg.id_col].tolist()
        logger.info("Slice %s: processing %d posts", slice_id, len(slice_texts))
        post_emb = embedder.encode(slice_texts)

        arg_ids, arg_emb = compute_argument_embeddings(taxonomy, embedder)
        mapped_idx, mapped_scores = map_posts_to_arguments(
            post_emb,
            arg_ids,
            arg_emb,
            cfg.mapping.min_similarity,
        )

        mapped_mask = np.array([i >= 0 for i in mapped_idx])
        total = len(slice_texts)
        mapped = int(mapped_mask.sum())
        coverage = mapped / total if total else 0.0
        logger.info(
            "Slice %s: initial mapping coverage %.4f (%d/%d)",
            slice_id,
            coverage,
            mapped,
            total,
        )

        consecutive_small_gain = 0
        rounds = 0
        while rounds < cfg.unmapped.max_rounds:
            unmapped_idx = np.where(~mapped_mask)[0]
            if len(unmapped_idx) < cfg.unmapped.min_unmapped:
                logger.info(
                    "Slice %s: stop expansion, unmapped=%d below min_unmapped=%d",
                    slice_id,
                    len(unmapped_idx),
                    cfg.unmapped.min_unmapped,
                )
                break

            new_arg_ids = expand_unmapped(
                taxonomy,
                llm,
                embedder,
                cfg,
                slice_id,
                [slice_texts[i] for i in unmapped_idx],
                post_emb[unmapped_idx],
            )
            if not new_arg_ids:
                logger.info("Slice %s: stop expansion, no new nodes generated", slice_id)
                break

            arg_ids, arg_emb = compute_argument_embeddings(taxonomy, embedder)
            mapped_idx_new, mapped_scores_new = map_posts_to_arguments(
                post_emb,
                arg_ids,
                arg_emb,
                cfg.mapping.min_similarity,
            )
            mapped_mask_new = np.array([i >= 0 for i in mapped_idx_new])
            mapped_new = int(mapped_mask_new.sum())
            coverage_new = mapped_new / total if total else 0.0
            delta = coverage_new - coverage

            if delta < cfg.unmapped.epsilon:
                consecutive_small_gain += 1
            else:
                consecutive_small_gain = 0

            mapped_mask = mapped_mask_new
            mapped_idx = mapped_idx_new
            mapped_scores = mapped_scores_new
            coverage = coverage_new

            rounds += 1
            logger.info(
                "Slice %s: expansion round %d new_args=%d coverage=%.4f delta=%.4f",
                slice_id,
                rounds,
                len(new_arg_ids),
                coverage,
                delta,
            )
            if consecutive_small_gain >= cfg.unmapped.consecutive_rounds:
                logger.info(
                    "Slice %s: stop expansion, coverage gain below epsilon for %d rounds",
                    slice_id,
                    cfg.unmapped.consecutive_rounds,
                )
                break

        # Prepare per-slice supports before taxonomy updates
        arg_ids, arg_emb = compute_argument_embeddings(taxonomy, embedder)
        posts_by_node: Dict[str, List[str]] = {}
        support_in_slice: Dict[str, int] = {}
        node_ids_at_time: List[str | None] = []
        for i, post_id in enumerate(post_ids):
            node_id = None
            if mapped_idx[i] >= 0:
                node_id = arg_ids[mapped_idx[i]]
            node_ids_at_time.append(node_id)
            if node_id:
                taxonomy.mark_seen(node_id, slice_id)
                posts_by_node.setdefault(node_id, []).append(slice_texts[i])
                support_in_slice[node_id] = support_in_slice.get(node_id, 0) + 1
                support_totals[node_id] = support_totals.get(node_id, 0) + 1

        merge_result = run_merge_pass(
            taxonomy=taxonomy,
            embedder=embedder,
            merge_cfg=cfg.merge,
            slice_id=slice_id,
            support_totals=support_totals,
            support_in_slice=support_in_slice,
        )
        split_result = run_split_pass(
            taxonomy=taxonomy,
            embedder=embedder,
            split_cfg=cfg.split,
            slice_id=slice_id,
            posts_by_node=posts_by_node,
        )
        lifecycle_result = update_lifecycle(
            taxonomy=taxonomy,
            lifecycle_cfg=cfg.lifecycle,
            slice_id=slice_id,
            support_totals=support_totals,
        )
        op_rows.extend(merge_result["operation_rows"])
        op_rows.extend(split_result["operation_rows"])
        op_rows.extend(lifecycle_result["operation_rows"])
        logger.info(
            (
                "Slice %s ops: merges=%d splits=%d promotions=%d deprecations=%d "
                "active_arguments=%d redirects=%d"
            ),
            slice_id,
            len(merge_result["merged_pairs"]),
            len(split_result["split_nodes"]),
            len(lifecycle_result["promoted"]),
            len(lifecycle_result["deprecated"]),
            sum(1 for n in taxonomy.nodes.values() if n.level == LEVEL_ARGUMENT and n.status == "active"),
            len(taxonomy.redirects),
        )

        # Log assignments after taxonomy updates so canonical IDs include new redirects.
        for i, post_id in enumerate(post_ids):
            node_id = node_ids_at_time[i]
            assignments.append(
                {
                    "post_id": post_id,
                    "slice_id": slice_id,
                    "node_id_at_time": node_id,
                    "canonical_node_id": taxonomy.canonicalize(node_id) if node_id else None,
                    "similarity": mapped_scores[i],
                }
            )

        slice_summary.append(
            {
                "slice_id": slice_id,
                "total_posts": total,
                "mapped_posts": int(mapped_mask.sum()),
                "coverage": coverage,
                "expansion_rounds": rounds,
                "merge_count": len(merge_result["merged_pairs"]),
                "split_count": len(split_result["split_nodes"]),
                "promoted_count": len(lifecycle_result["promoted"]),
                "deprecated_count": len(lifecycle_result["deprecated"]),
                "redirect_total": len(taxonomy.redirects),
            }
        )

        # Stance estimation
        if cfg.stance.enabled:
            for node_id, texts in posts_by_node.items():
                dist = estimate_stance_distribution(llm, node_id, texts, cfg.stance.sample_per_node)
                stance_rows.append(
                    {
                        "slice_id": slice_id,
                        "node_id": node_id,
                        **dist,
                    }
                )
            logger.info("Slice %s stance rows added: %d", slice_id, len(posts_by_node))

    save_taxonomy(taxonomy, cfg.output_dir)
    save_assignments(assignments, cfg.output_dir)
    save_slice_summary(slice_summary, cfg.output_dir)
    save_taxonomy_ops(op_rows, cfg.output_dir)
    if cfg.stance.enabled:
        save_stance(stance_rows, cfg.output_dir)
    logger.info(
        "Completed pipeline. nodes=%d redirects=%d assignments=%d slices=%d ops=%d",
        len(taxonomy.nodes),
        len(taxonomy.redirects),
        len(assignments),
        len(slice_summary),
        len(op_rows),
    )

    if cfg.visualization.auto_generate:
        viz_dir = os.path.join(cfg.output_dir, cfg.visualization.output_subdir)
        try:
            logger.info("Generating visualizations: input_dir=%s output_dir=%s", cfg.output_dir, viz_dir)
            run_visualization(cfg.output_dir, viz_dir)
            logger.info("Visualization generation complete: %s", viz_dir)
        except Exception as exc:
            logger.exception("Visualization generation failed: %s", exc)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ClaimTaxo pipeline")
    p.add_argument("--input", default=DEFAULT_CONFIG.input_path)
    p.add_argument("--output", default=DEFAULT_CONFIG.output_dir)
    p.add_argument("--min-sim", type=float, default=DEFAULT_CONFIG.mapping.min_similarity)
    p.add_argument("--epsilon", type=float, default=DEFAULT_CONFIG.unmapped.epsilon)
    p.add_argument("--rounds", type=int, default=DEFAULT_CONFIG.unmapped.max_rounds)
    p.add_argument("--disable-llm", action="store_true")
    p.add_argument("--disable-stance", action="store_true")
    p.add_argument("--disable-merge", action="store_true")
    p.add_argument("--enable-split", action="store_true")
    p.add_argument("--no-file-log", action="store_true")
    p.add_argument("--no-auto-viz", action="store_true")
    p.add_argument("--viz-dir", default=DEFAULT_CONFIG.visualization.output_subdir)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = DEFAULT_CONFIG
    cfg.input_path = args.input
    cfg.output_dir = args.output
    cfg.mapping.min_similarity = args.min_sim
    cfg.unmapped.epsilon = args.epsilon
    cfg.unmapped.max_rounds = args.rounds
    if args.disable_llm:
        cfg.llm.enabled = False
    if args.disable_stance:
        cfg.stance.enabled = False
    if args.disable_merge:
        cfg.merge.enabled = False
    if args.enable_split:
        cfg.split.enabled = True
    if args.no_file_log:
        cfg.logging.log_to_file = False
    if args.no_auto_viz:
        cfg.visualization.auto_generate = False
    cfg.visualization.output_subdir = args.viz_dir
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
