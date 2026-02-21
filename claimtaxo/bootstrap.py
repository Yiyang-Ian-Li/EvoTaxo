from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from config import PipelineConfig
from llm import LLMClient
from llm_ops import bootstrap_taxonomy_with_llm
from taxonomy import Taxonomy
from utils import now_ts


def bootstrap_taxonomy(
    taxonomy: Taxonomy,
    llm: LLMClient,
    sample_df: pd.DataFrame,
    cfg: PipelineConfig,
    logger: logging.Logger,
    event_log: Any,
    llm_trace: Any,
) -> None:
    sample_posts = []
    for _, row in sample_df.iterrows():
        sample_posts.append(
            {
                "post_id": str(row[cfg.id_col]),
                "timestamp": str(row[cfg.timestamp_col]),
                "text": row["_text"],
            }
        )

    payload = bootstrap_taxonomy_with_llm(
        llm,
        sample_posts=sample_posts,
        root_topic=cfg.root_topic,
        max_parse_attempts=cfg.llm.max_parse_attempts,
        trace=llm_trace,
    )
    event_log.append(
        {
            "ts": now_ts(),
            "event": "bootstrap_llm_payload",
            "root_topic": cfg.root_topic,
            "sample_size": len(sample_posts),
            "node_candidates": len(payload.get("nodes", [])),
            "root_name": payload.get("root_name", "ROOT"),
        }
    )

    tmp_to_node: Dict[str, str] = {}
    window_id = sample_df["window_id"].min() if len(sample_df) else "INIT"
    nodes = payload.get("nodes", [])
    staged = sorted(nodes, key=lambda x: {"topic": 0, "subtopic": 1, "claim": 2}.get(x.get("level", "claim"), 3))

    for item in staged:
        tid = item.get("temp_id")
        parent_tid = item.get("parent_temp_id")
        parent_id = taxonomy.root_id if not parent_tid else tmp_to_node.get(parent_tid, taxonomy.root_id)
        node_id = taxonomy.add_node(
            parent_id=parent_id,
            name=item.get("name", "unnamed"),
            level=item.get("level", "claim"),
            cmb=item.get("cmb", {}),
            window_id=window_id,
        )
        logger.info(
            "Bootstrap node created node_id=%s level=%s parent_id=%s name=%s",
            node_id,
            item.get("level", "claim"),
            parent_id,
            item.get("name", "unnamed"),
        )
        event_log.append(
            {
                "ts": now_ts(),
                "event": "bootstrap_node_created",
                "window_id": window_id,
                "node_id": node_id,
                "parent_id": parent_id,
                "level": item.get("level", "claim"),
                "name": item.get("name", "unnamed"),
            }
        )
        if tid:
            tmp_to_node[tid] = node_id

    if not taxonomy.claim_node_ids():
        fallback_id = taxonomy.add_node(
            parent_id=taxonomy.root_id,
            name="misc_claim",
            level="claim",
            cmb={
                "definition": "Fallback claim node for unmatched posts",
                "include_terms": [],
                "exclude_terms": [],
                "examples": [],
            },
            window_id=window_id,
        )
        logger.info("Bootstrap fallback claim node created node_id=%s", fallback_id)
        event_log.append(
            {
                "ts": now_ts(),
                "event": "bootstrap_fallback_claim_created",
                "window_id": window_id,
                "node_id": fallback_id,
            }
        )
