# ClaimTaxo

ClaimTaxo is a time-evolving taxonomy induction pipeline for social media posts.

## Usage

Run with defaults:

```bash
uv run python claimtaxo/pipeline.py \
  --input naloxone_mentions.csv \
  --output /tmp/claimtaxo_run
```

Run with LLM provider/model override:

```bash
uv run python claimtaxo/pipeline.py \
  --input naloxone_mentions.csv \
  --output /tmp/claimtaxo_run \
  --llm-provider custom \
  --llm-model gpt-oss:120b
```

## Parameter Reference

CLI parameters currently exposed:

- `--input`: path to input CSV.
- `--output`: output directory for artifacts/logs.
- `--high-sim`: cosine threshold for direct post-to-claim mapping.
- `--bootstrap-n`: diversity sample size used in bootstrap taxonomy stage.
- `--llm-provider`: `custom` or `openai`.
- `--llm-model`: model name sent to provider endpoint.
- `--llm-api-url`: chat completions URL.
- `--llm-api-key-env`: env var name for API key lookup.
- `--llm-timeout-s`: request timeout per LLM call (seconds).
- `--llm-max-retries`: request retries on transport/non-200 failures.
- `--llm-max-parse-attempts`: retries when LLM output JSON is invalid.
- `--llm-trace-mode`: `off`, `compact`, or `full` trace logging.
- `--llm-trace-max-chars`: max prompt/response chars stored in compact trace.
- `--review-max-examples`: sampled proposals shown to final-review LLM per cluster.
- `--review-max-post-chars`: max chars per sampled post text in review prompt.
- `--disable-llm`: disable LLM and use fallback behavior.
- `--window`: time window unit (`quarter` in current MVP).
- `--root-topic`: root-topic string injected into bootstrap/proposal/review prompts.

Additional tunables not currently on CLI (edit in `claimtaxo/config.py`):

- `min_year`: drop posts before this year.
- `max_post_words`: truncate text length by word count.
- `min_cluster_size_review`, `min_cluster_size_hdbscan`: clustering/review gates.
- `min_cohesion`, `min_time_compactness`: quality filters for reviewable clusters.
- `temporal_w_sem`, `temporal_w_time`: weights for temporal-aware distance.
- `embedding.model_name`, `embedding.batch_size`: embedding backend settings.
