from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from config import PipelineConfig
from embeddings import Embedder
from utils import safe_text


def load_data(cfg: PipelineConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_path)
    df = df[df[cfg.kind_col] == cfg.kind_value].copy()
    df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col], errors="coerce")
    df = df.dropna(subset=[cfg.timestamp_col]).copy()
    df = df[df[cfg.timestamp_col].dt.year >= cfg.min_year].copy()

    def build_text(row: pd.Series) -> str:
        body = safe_text(row.get(cfg.text_col, "")).strip()
        if body:
            return body
        return safe_text(row.get(cfg.title_col, "")).strip()

    df["_text"] = df.apply(build_text, axis=1)
    if cfg.max_post_words > 0:
        df["_text"] = df["_text"].apply(lambda t: " ".join(str(t).split()[: cfg.max_post_words]))
    df = df[df["_text"].str.strip().astype(bool)].copy()

    if cfg.window_unit != "quarter":
        raise ValueError("Only quarter is currently supported in this MVP.")
    df["window_id"] = df[cfg.timestamp_col].dt.to_period("Q").astype(str)
    df["timestamp_epoch"] = df[cfg.timestamp_col].astype("int64") / 1e9
    return df.sort_values(cfg.timestamp_col).reset_index(drop=True)


def diversity_sample(df: pd.DataFrame, embedder: Embedder, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    texts = df["_text"].tolist()
    emb = embedder.encode(texts)
    kmeans = KMeans(n_clusters=n, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(emb)
    centers = kmeans.cluster_centers_

    picks = []
    for k in range(n):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        subset = emb[idx]
        sims = cosine_similarity(subset, centers[k : k + 1]).reshape(-1)
        picks.append(int(idx[int(np.argmax(sims))]))

    picks = sorted(set(picks))
    if len(picks) < n:
        extra = [i for i in range(len(df)) if i not in set(picks)]
        picks.extend(extra[: n - len(picks)])
    return df.iloc[picks[:n]].copy()
