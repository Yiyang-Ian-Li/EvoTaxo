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
    provider: str = "openrouter"  # openai | openrouter
    # Optional environment variable name override for API key.
    # If None, provider-specific defaults are used automatically.
    api_key_env: Optional[str] = None
    # Optional endpoint URL override for chat completions.
    # If None, provider-specific defaults are used automatically.
    api_url: Optional[str] = None
    # Model name for selected provider.
    model: str = "gpt-4o-mini"
    # Optional shared model override for review/final-review/repair stages.
    later_stage_model: Optional[str] = "gpt-5.1"
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


@dataclass
class PipelineConfig:
    # Input CSV path.
    input_path: str = "data/opiates_claimtaxo_input_5labels_ge0.75.csv"
    # Output directory for all artifacts.
    output_dir: str = "results"
    # Column names and filter for selecting working rows.
    kind_col: str = "kind"
    kind_value: str = "submissions"
    id_col: str = "id"
    text_col: str = "text"
    title_col: Optional[str] = "title"
    timestamp_col: str = "created_dt"
    # Root topic string injected into LLM prompts.
    root_topic: str = "opiates"
    # Drop rows before this year.
    min_year: int = 2014
    # Truncate post text to first N words.
    max_post_words: int = 500

    # Time window unit.
    window_unit: str = "year"  # month | quarter | year
    # Direct map threshold for post->claim cosine similarity.
    high_sim_threshold: float = 0.7

    # Cluster quality gates and HDBSCAN settings.
    min_cluster_size_review: int = 10
    min_cluster_size_hdbscan: int = 3
    min_cohesion: float = 0.5
    min_time_compactness: float = 0.2
    # Temporal clustering distance weights.
    temporal_w_sem: float = 0.5
    temporal_w_time: float = 0.5
    # Number of proposal samples shown to final-review LLM per cluster.
    review_max_examples: int = 10
    # Trigger a review/apply cycle every N newly processed posts.
    review_batch_every_n_posts: int = 500

    # Nested configs.
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


DEFAULT_CONFIG = PipelineConfig()
