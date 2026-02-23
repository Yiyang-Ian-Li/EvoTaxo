# ClaimTaxo

ClaimTaxo is a time-evolving taxonomy induction pipeline for social media posts.

## Usage

Run with defaults:

```bash
uv run python claimtaxo/pipeline.py \
  --input naloxone_mentions.csv \
  --output /tmp/claimtaxo_run
```

`claimtaxo/pipeline.py` auto-loads a local `.env` file (via `python-dotenv`) before reading API-key environment variables.

Run with LLM provider/model override:

```bash
uv run python claimtaxo/pipeline.py \
  --input naloxone_mentions.csv \
  --output /tmp/claimtaxo_run \
  --llm-provider custom \
  --llm-model gpt-oss:120b
```

Run with OpenRouter:

```bash
uv run python claimtaxo/pipeline.py \
  --input naloxone_mentions.csv \
  --output /tmp/claimtaxo_run \
  --llm-provider openrouter \
  --llm-model openai/gpt-4o-mini
```

Evaluate generated taxonomy (reference-free CSC / NLIV):

```bash
uv run python evaluate.py \
  --run-dir /tmp/claimtaxo_run \
  --device auto \
  --root-topic naloxone \
  --output-dir results
```

Outputs are written to `results/`:
- `taxonomy_eval_metrics.json`
- `taxonomy_eval_edge_scores.csv`
- `taxonomy_eval_path_scores.csv`
- `csc.json`
- `nliv_s.json`
- `nliv_w.json`
- `csc_x_nliv_s.json`
- `path_granularity.json`
- `sibling_coherence.json`

## Parameter Reference

CLI parameters currently exposed:

- `--input`: path to input CSV.
- `--output`: output directory for artifacts/logs.
- `--high-sim`: cosine threshold for direct post-to-claim mapping.
- `--min-year`: drop posts before this year.
- `--llm-provider`: `custom`, `openai`, or `openrouter`.
- `--llm-model`: model name sent to provider endpoint.
- `--llm-api-url`: optional chat completions URL override.
- `--llm-api-key-env`: optional env var name override for API key lookup.
- `--llm-timeout-s`: request timeout per LLM call (seconds).
- `--llm-max-retries`: request retries on transport/non-200 failures.
- `--llm-max-parse-attempts`: retries when LLM output JSON is invalid.
- `--llm-trace-mode`: `off`, `compact`, or `full` trace logging.
- `--llm-trace-max-chars`: max prompt/response chars stored in compact trace.
- `--review-max-examples`: sampled proposals shown to final-review LLM per cluster.
- `--review-max-post-chars`: max chars per sampled post text in review prompt.
- `--proposal-pool-trigger-size`: trigger a review/apply batch when pending proposal backlog reaches this size.
- `--disable-llm`: disable LLM and use fallback behavior.
- `--window`: time window unit (`month`, `quarter`, or `year`).
- `--root-topic`: root-topic string used for root node naming and proposal/review prompts.

Additional tunables not currently on CLI (edit in `claimtaxo/config.py`):

- `min_year`: drop posts before this year.
- `max_post_words`: truncate text length by word count.
- `min_cluster_size_review`, `min_cluster_size_hdbscan`: clustering/review gates.
- `min_cohesion`, `min_time_compactness`: quality filters for reviewable clusters.
- `temporal_w_sem`, `temporal_w_time`: weights for temporal-aware distance.
- `embedding.model_name`, `embedding.batch_size`: embedding backend settings.
