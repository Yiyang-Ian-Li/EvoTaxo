#!/usr/bin/env python3
import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline


LABELS = [
    "Factual_Claim",
    "Evaluative_Opinion",
    "Causal_Claim",
    "Policy_Prescription",
    "Argumentative_Reasoning",
    "Personal_Experience",
    "Information_Sharing",
    "Advice_Seeking",
    "Emotional_Reaction",
    "Meme_or_Irony",
    "Mobilization_or_Coordination",
    "Other",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/opiates_text_filtered.csv")
    p.add_argument("--text-col", default="text")
    p.add_argument("--id-col", default="id")
    p.add_argument("--model", default="facebook/bart-large-mnli")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--truncation", action="store_true")
    p.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Tokenizer max_length when truncation is enabled.",
    )
    p.add_argument("--row-start", type=int, default=0)
    p.add_argument("--row-end", type=int, default=None)
    p.add_argument("--output-scores", default="data/zero_shot_bart_mnli_scores.csv")
    p.add_argument(
        "--output-distribution",
        default="data/zero_shot_bart_mnli_distribution.csv",
    )
    p.add_argument(
        "--output-histogram",
        default="data/zero_shot_bart_mnli_histogram.csv",
    )
    return p.parse_args()


def resolve_device(mode: str, device_id: int) -> int:
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return device_id
    if mode == "auto":
        return device_id if torch.cuda.is_available() else -1
    return -1


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device, args.device_id)

    usecols: List[str] = [args.text_col]
    if args.id_col:
        usecols = [args.id_col, args.text_col]
    df = pd.read_csv(args.input, usecols=usecols)
    df = df.iloc[args.row_start : args.row_end].reset_index(drop=True)
    texts = df[args.text_col].fillna("").astype(str).tolist()

    model_kwargs = {}
    if isinstance(device, int) and device >= 0:
        model_kwargs["dtype"] = torch.float16
    clf = pipeline(
        "zero-shot-classification",
        model=args.model,
        device=device,
        model_kwargs=model_kwargs,
    )

    n = len(texts)
    scores = np.zeros((n, len(LABELS)), dtype=np.float32)
    if n > 0:
        for s in tqdm(range(0, n, args.batch_size), desc="Scoring"):
            e = min(s + args.batch_size, n)
            out = clf(
                texts[s:e],
                candidate_labels=LABELS,
                multi_label=True,
                truncation=args.truncation,
                max_length=args.max_length,
            )
            if isinstance(out, dict):
                out = [out]
            for i, item in enumerate(out):
                m = dict(zip(item["labels"], item["scores"]))
                scores[s + i, :] = [m[label] for label in LABELS]

    os.makedirs(os.path.dirname(args.output_scores), exist_ok=True)

    base_cols = [args.id_col] if args.id_col and args.id_col in df.columns else []
    out_scores = pd.concat(
        [df[base_cols].reset_index(drop=True), pd.DataFrame(scores, columns=LABELS)],
        axis=1,
    )
    out_scores.to_csv(args.output_scores, index=False)

    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    dist_rows = []
    hist_rows = []
    bins = np.linspace(0.0, 1.0, 11)

    for j, label in enumerate(LABELS):
        col = scores[:, j]
        q = np.quantile(col, qs)
        dist_rows.append(
            {
                "label": label,
                "count": int(len(col)),
                "mean": float(col.mean()),
                "std": float(col.std()),
                "min": float(col.min()),
                "p01": float(q[0]),
                "p05": float(q[1]),
                "p10": float(q[2]),
                "p25": float(q[3]),
                "p50": float(q[4]),
                "p75": float(q[5]),
                "p90": float(q[6]),
                "p95": float(q[7]),
                "p99": float(q[8]),
                "max": float(col.max()),
                "share_ge_0_5": float((col >= 0.5).mean()),
                "share_ge_0_7": float((col >= 0.7).mean()),
                "share_ge_0_9": float((col >= 0.9).mean()),
            }
        )
        counts, edges = np.histogram(col, bins=bins)
        for i in range(len(counts)):
            hist_rows.append(
                {
                    "label": label,
                    "bin_left": float(edges[i]),
                    "bin_right": float(edges[i + 1]),
                    "count": int(counts[i]),
                    "share": float(counts[i] / len(col)),
                }
            )

    dist_df = pd.DataFrame(dist_rows).sort_values("mean", ascending=False)
    dist_df.to_csv(args.output_distribution, index=False)
    pd.DataFrame(hist_rows).to_csv(args.output_histogram, index=False)

    print("Done.")
    print(f"Rows scored: {n}")
    print(f"Scores: {args.output_scores}")
    print(f"Distribution: {args.output_distribution}")
    print(f"Histogram: {args.output_histogram}")
    print(
        dist_df[
            ["label", "mean", "p50", "p90", "share_ge_0_5", "share_ge_0_7", "share_ge_0_9"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
