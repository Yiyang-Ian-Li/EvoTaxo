from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmbeddingConfig:
    # SentenceTransformer model used for post/node semantic embeddings.
    model_name: str = "all-mpnet-base-v2"
    # Batch size for embedding inference.
    batch_size: int = 32


@dataclass
class LLMConfig:
    # Global on/off switch for all LLM calls.
    enabled: bool = True
    # LLM provider backend.
    provider: str = "custom"  # custom | openai
    # Environment variable name used to fetch API key.
    api_key_env: str = "OPENAI_API_KEY"
    # Chat completion endpoint URL.
    api_url: str = "https://openwebui.crc.nd.edu/api/v1/chat/completions"
    # Model name for selected provider.
    model: str = "gpt-oss:120b"
    # HTTP timeout per LLM request (seconds).
    timeout_s: int = 60
    # Sampling temperature for LLM generation.
    temperature: float = 0.0
    # Retries for transport/non-200 failures.
    max_retries: int = 2
    # Backoff sleep between request retries (seconds).
    retry_backoff_s: float = 1.0
    # Retries for malformed/non-parseable JSON outputs.
    max_parse_attempts: int = 4
    # Trace verbosity for llm_trace.jsonl.
    trace_mode: str = "compact"  # off | compact | full
    # Max stored chars for prompt/response in compact trace mode.
    trace_max_chars: int = 400


@dataclass
class PipelineConfig:
    # Input CSV path.
    input_path: str = "naloxone_mentions.csv"
    # Output directory for all artifacts.
    output_dir: str = "results_v2"
    # Column names and filter for selecting working rows.
    kind_col: str = "kind"
    kind_value: str = "submissions"
    id_col: str = "id"
    text_col: str = "text"
    title_col: Optional[str] = "title"
    timestamp_col: str = "created_dt"
    # Root topic string injected into LLM prompts.
    root_topic: str = "naloxone"
    # Drop rows before this year.
    min_year: int = 2020
    # Truncate post text to first N words.
    max_post_words: int = 300

    # Time window unit (MVP currently supports quarter only).
    window_unit: str = "quarter"  # fixed by user request
    # Diversity sample size for bootstrap taxonomy generation.
    bootstrap_sample_size: int = 50
    # Direct map threshold for post->claim cosine similarity.
    high_sim_threshold: float = 0.9

    # Cluster quality gates and HDBSCAN settings.
    min_cluster_size_review: int = 4
    min_cluster_size_hdbscan: int = 3
    min_cohesion: float = 0.55
    min_time_compactness: float = 0.6
    # Temporal clustering distance weights.
    temporal_w_sem: float = 0.8
    temporal_w_time: float = 0.2
    # Number of proposal samples shown to final-review LLM per cluster.
    review_max_examples: int = 12
    # Per-sample post text truncation for final-review prompt.
    review_max_post_chars: int = 400

    # Nested configs.
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


DEFAULT_CONFIG = PipelineConfig()
