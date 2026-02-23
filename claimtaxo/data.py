from __future__ import annotations

import pandas as pd

from config import PipelineConfig
from utils import safe_text


def _window_period_code(window_unit: str) -> str:
    unit = (window_unit or "").strip().lower()
    if unit == "month":
        return "M"
    if unit == "quarter":
        return "Q"
    if unit == "year":
        return "Y"
    raise ValueError(f"Unsupported window_unit '{window_unit}'. Use one of: month, quarter, year.")


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

    period_code = _window_period_code(cfg.window_unit)
    ts = df[cfg.timestamp_col]
    # Convert timezone-aware timestamps to naive to avoid pandas warning when using Period.
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    df["window_id"] = ts.dt.to_period(period_code).astype(str)
    df["timestamp_epoch"] = ts.astype("int64") / 1e9
    return df.sort_values(cfg.timestamp_col).reset_index(drop=True)
