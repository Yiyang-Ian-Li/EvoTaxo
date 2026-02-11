from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EmbeddingConfig


class Embedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return self.model.encode(list(texts), batch_size=self.cfg.batch_size, show_progress_bar=False)
