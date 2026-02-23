#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List

import pandas as pd


DEFAULT_LABELS = [
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
    p = argparse.ArgumentParser(
        description="Filter scored dataset by label thresholds (multi-label or single-label)."
    )
    p.add_argument(
        "--scores",
        required=True,
        help="CSV containing per-label scores (from score_zero_shot_bart_mnli.py).",
    )
    p.add_argument(
        "--data",
        default=None,
        help="Optional original data CSV to merge by --id-col and keep original columns.",
    )
    p.add_argument("--id-col", default="id")
    p.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help="Label columns to use from --scores.",
    )
    p.add_argument(
        "--mode",
        choices=["multi", "single"],
        default="multi",
        help="multi: keep rows with any label >= threshold; single: keep based on top-1 label threshold.",
    )
    p.add_argument(
        "--global-threshold",
        type=float,
        default=0.5,
        help="Fallback threshold for labels not specified in --threshold-config.",
    )
    p.add_argument(
        "--threshold-config",
        default=None,
        help="Optional JSON file: {\"Label\": threshold, ...}",
    )
    p.add_argument(
        "--require-labels",
        nargs="*",
        default=[],
        help="Optional whitelist: only keep rows whose selected labels intersect this set.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output filtered CSV path.",
    )
    p.add_argument(
        "--output-summary",
        default=None,
        help="Optional JSON summary path.",
    )
    return p.parse_args()


def load_thresholds(path: str | None, labels: List[str], fallback: float) -> Dict[str, float]:
    thresholds = {l: float(fallback) for l in labels}
    if path is None:
        return thresholds
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for l in labels:
        if l in cfg:
            thresholds[l] = float(cfg[l])
    return thresholds


def main() -> None:
    args = parse_args()
    labels = args.labels
    thresholds = load_thresholds(args.threshold_config, labels, args.global_threshold)

    score_df = pd.read_csv(args.scores)
    missing = [l for l in labels if l not in score_df.columns]
    if missing:
        raise ValueError(f"Missing label columns in scores CSV: {missing}")
    if args.id_col not in score_df.columns and args.data is not None:
        raise ValueError(f"--id-col '{args.id_col}' must exist in scores CSV when --data is provided.")

    if args.mode == "multi":
        keep_mask = pd.Series(False, index=score_df.index)
        for l in labels:
            keep_mask = keep_mask | (score_df[l] >= thresholds[l])
        selected_labels = []
        for _, row in score_df[labels].iterrows():
            hits = [l for l in labels if row[l] >= thresholds[l]]
            selected_labels.append("|".join(hits))
    else:
        top_label = score_df[labels].idxmax(axis=1)
        top_score = score_df[labels].max(axis=1)
        per_row_thr = top_label.map(lambda l: thresholds[l])
        keep_mask = top_score >= per_row_thr
        selected_labels = top_label.astype(str).tolist()

    if args.require_labels:
        allow = set(args.require_labels)

        def has_allowed(lbls: str) -> bool:
            if not lbls:
                return False
            return any(part in allow for part in lbls.split("|"))

        allow_mask = pd.Series([has_allowed(x) for x in selected_labels], index=score_df.index)
        keep_mask = keep_mask & allow_mask

    annotated = score_df.copy()
    annotated["selected_labels"] = selected_labels

    kept_scores = annotated[keep_mask].copy()
    dropped_scores = annotated[~keep_mask].copy()

    if args.data is not None:
        base_df = pd.read_csv(args.data)
        if args.id_col not in base_df.columns:
            raise ValueError(f"--id-col '{args.id_col}' not found in --data CSV.")
        kept_df = base_df.merge(
            kept_scores[[args.id_col, "selected_labels"] + labels],
            on=args.id_col,
            how="inner",
        )
    else:
        kept_df = kept_scores

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    kept_df.to_csv(args.output, index=False)

    label_counts = (
        kept_scores["selected_labels"]
        .str.split("|")
        .explode()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .to_dict()
    )
    summary = {
        "mode": args.mode,
        "scores_path": args.scores,
        "data_path": args.data,
        "output_path": args.output,
        "n_input": int(len(score_df)),
        "n_kept": int(len(kept_scores)),
        "n_dropped": int(len(dropped_scores)),
        "keep_rate": float(len(kept_scores) / len(score_df)) if len(score_df) else 0.0,
        "thresholds": thresholds,
        "require_labels": args.require_labels,
        "kept_selected_label_counts": label_counts,
    }

    if args.output_summary:
        os.makedirs(os.path.dirname(args.output_summary), exist_ok=True)
        with open(args.output_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Input rows: {summary['n_input']}")
    print(f"Kept rows: {summary['n_kept']} ({summary['keep_rate']:.2%})")
    print(f"Dropped rows: {summary['n_dropped']}")
    print(f"Output: {args.output}")
    if args.output_summary:
        print(f"Summary: {args.output_summary}")
    print("Kept selected label counts:")
    for k, v in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
