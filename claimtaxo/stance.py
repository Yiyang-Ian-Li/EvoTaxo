from __future__ import annotations

from typing import Dict, List, Optional
import random

from .llm import LLMClient


STANCE_LABELS = ["pro", "anti", "neutral", "unclear"]


def classify_stance(llm: LLMClient, text: str) -> Optional[str]:
    if not llm.available():
        return None
    prompt = (
        "Classify the stance in the text toward the main claim or argument mentioned. "
        "Return one label from: pro, anti, neutral, unclear.\n\n"
        f"Text: {text}\n\n"
        "Label:"
    )
    resp = llm.chat(prompt)
    if not resp:
        return None
    resp = resp.strip().lower()
    for label in STANCE_LABELS:
        if label in resp:
            return label
    return None


def estimate_stance_distribution(
    llm: LLMClient,
    node_id: str,
    texts: List[str],
    sample_n: int,
) -> Dict[str, float]:
    if not texts:
        return {label: 0.0 for label in STANCE_LABELS}
    sample = texts if len(texts) <= sample_n else random.sample(texts, sample_n)
    counts = {label: 0 for label in STANCE_LABELS}
    total = 0
    for text in sample:
        label = classify_stance(llm, text)
        if label is None:
            continue
        counts[label] += 1
        total += 1
    if total == 0:
        return {label: 0.0 for label in STANCE_LABELS}
    return {label: counts[label] / total for label in STANCE_LABELS}
