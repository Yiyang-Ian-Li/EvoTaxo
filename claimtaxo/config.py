from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EmbeddingConfig:
    model_name: str = "all-mpnet-base-v2"
    batch_size: int = 32


@dataclass
class ClusteringConfig:
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_n_components: int = 5
    k_min: int = 2
    k_max: int = 12
    k_step: int = 1
    random_state: int = 42


@dataclass
class SliceConfig:
    time_unit: str = "quarter"  # only "quarter" supported for now
    timestamp_col: str = "created_dt"


@dataclass
class MappingConfig:
    min_similarity: float = 0.35


@dataclass
class UnmappedConfig:
    epsilon: float = 0.005
    consecutive_rounds: int = 2
    max_rounds: int = 5
    min_unmapped: int = 30


@dataclass
class LLMConfig:
    enabled: bool = True
    api_url: str = "https://openwebui.crc.nd.edu/api/v1/chat/completions"
    model: str = "gpt-oss:120b"
    timeout_s: int = 60


@dataclass
class StanceConfig:
    enabled: bool = True
    sample_per_node: int = 20


@dataclass
class MergeConfig:
    enabled: bool = True
    similarity_threshold: float = 0.85
    min_support: int = 8
    support_ratio: float = 0.5
    max_merges_per_slice: int = 20


@dataclass
class SplitConfig:
    enabled: bool = False
    min_support: int = 50
    min_clusters: int = 2
    max_clusters: int = 4
    min_cluster_size: int = 10
    silhouette_threshold: float = 0.2


@dataclass
class LifecycleConfig:
    promote_min_slices: int = 2
    promote_min_support: int = 20
    stale_after_slices: int = 3


@dataclass
class LoggingConfig:
    log_to_file: bool = True
    file_name: str = "run.log"


@dataclass
class VisualizationConfig:
    auto_generate: bool = True
    output_subdir: str = "viz"


@dataclass
class PipelineConfig:
    input_path: str = "naloxone_mentions.csv"
    output_dir: str = "outputs"
    text_col: str = "text"
    kind_col: str = "kind"
    kind_value: str = "submissions"
    id_col: str = "id"
    title_col: Optional[str] = "title"
    use_title_if_missing: bool = True
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    slicing: SliceConfig = field(default_factory=SliceConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    unmapped: UnmappedConfig = field(default_factory=UnmappedConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    stance: StanceConfig = field(default_factory=StanceConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


DEFAULT_CONFIG = PipelineConfig()
