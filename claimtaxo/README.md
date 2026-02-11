# ClaimTaxo (Pipeline)
A claim-centered, time-evolving taxonomy induction pipeline for social media posts.

This README documents the implementation under `claimtaxo/` and how to run it on
`naloxone_mentions.csv`.

## What It Does
The pipeline builds a multi-level taxonomy (topic → claim → argument), then processes posts in
chronological slices (quarterly). In each slice it:
1. Maps posts to existing argument nodes (via embedding similarity).
2. Iteratively expands the taxonomy using unmapped posts, until coverage gain saturates.
3. Logs post-to-node assignments and canonicalizes them via a redirect table.
4. Estimates stance distributions per argument node (optional).
5. Runs automated taxonomy maintenance (merge, optional split, lifecycle promotion/deprecation).

Key constraints in this implementation:
- Uses `created_dt` for time slicing.
- Uses only `submissions` rows (`kind == submissions`).
- Assumes `text` already contains `title`; if this changes, update the text builder logic in
  `claimtaxo/pipeline.py`.
- Claim filtering is currently skipped (future module).

## Repository Layout
- `claimtaxo/`
  - `pipeline.py`: end-to-end orchestration and CLI
  - `taxonomy.py`: node model, concept memory bank, redirect handling
  - `embeddings.py`: SentenceTransformer embedding
  - `clustering.py`: UMAP + KMeans with silhouette selection
  - `llm.py`: LLM client (OpenWebUI-style API)
  - `stance.py`: stance sampling + classification
  - `io.py`: outputs
  - `config.py`: hyperparameters and defaults

## Installation
This repo uses `uv` and `pyproject.toml` for dependencies.

```bash
uv sync
```

## Quick Start
```bash
# Optional: set API key for LLM summaries + stance classification
export OPENAI_API_KEY='your_api_key_here'

# Run the pipeline
python -m claimtaxo --input naloxone_mentions.csv --output outputs
```

Or:
```bash
bash run_claimtaxo.sh
```

## Outputs
All outputs are written under the output directory (default: `outputs/`).

- `taxonomy_nodes.json`: all taxonomy nodes (topic/claim/argument), with CMB fields
- `redirects.json`: redirect table (future merge/split hooks)
- `assignments.csv`: post assignments per slice
  - `post_id`, `slice_id`, `node_id_at_time`, `canonical_node_id`, `similarity`
- `slice_summary.jsonl`: coverage statistics per slice
- `taxonomy_ops.jsonl`: per-slice taxonomy update operations (`merge`, `split`, `promote`, `deprecate`)
- `run.log`: execution trace with per-slice coverage, expansion rounds, and op counts
- `stance_estimates.csv`: stance distribution per argument node per slice

### Concept Memory Bank Format
Each node stores a concept memory bank (CMB) entry with canonical definition, representative examples,
and boundaries.

Example:
```text
Node ID: C17
Name: Vaccine myocarditis
Canonical definition:
  "Claims about myocarditis caused by COVID vaccines"
Representative examples:
  - post1
  - post2
  - post3
Boundaries:
  include: heart inflammation, myocarditis
  exclude: general heart disease
```

## Hyperparameters
Hyperparameters live in `claimtaxo/config.py` and can be overridden via CLI.

### Embedding
- `embedding.model_name` (default: `all-mpnet-base-v2`)
  - SentenceTransformer model used for post and node embeddings.
- `embedding.batch_size` (default: `32`)
  - Batch size for embedding inference.

### Clustering (UMAP + KMeans)
- `clustering.umap_n_neighbors` (default: `30`)
- `clustering.umap_min_dist` (default: `0.1`)
- `clustering.umap_n_components` (default: `5`)
  - UMAP controls for dimensionality reduction prior to KMeans.
- `clustering.k_min`, `clustering.k_max`, `clustering.k_step` (default: `2..12`)
  - Candidate K values for KMeans. Best K chosen by silhouette score.
- `clustering.random_state` (default: `42`)
  - Random seed for UMAP and KMeans.

### Time Slicing
- `slicing.time_unit` (fixed: `quarter`)
  - Uses pandas `to_period("Q")` on `created_dt`.
- `slicing.timestamp_col` (default: `created_dt`)
  - Source timestamp column.

### Mapping
- `mapping.min_similarity` (default: `0.35`)
  - Minimum cosine similarity required to map a post to an existing argument node.
  - Higher → fewer mappings, more “unmapped” posts; lower → more aggressive mapping.
  - CLI: `--min-sim`.

### Unmapped Iteration / Expansion
- `unmapped.epsilon` (default: `0.005`)
  - Coverage gain saturation threshold. If coverage gain per iteration is below this
    value for `consecutive_rounds` iterations, stop expanding.
  - CLI: `--epsilon`.
- `unmapped.consecutive_rounds` (default: `2`)
  - Number of consecutive low-gain rounds required to stop.
- `unmapped.max_rounds` (default: `5`)
  - Hard cap on expansion rounds per slice.
  - CLI: `--rounds`.
- `unmapped.min_unmapped` (default: `30`)
  - If fewer unmapped posts than this, skip expansion for the slice.

### Taxonomy Update Operations
- `merge.enabled` (default: `True`)
  - Enables automated argument-node merge detection and redirect generation.
- `merge.similarity_threshold` (default: `0.85`)
  - Minimum argument-node similarity to consider merging.
- `split.enabled` (default: `False`)
  - Enables automated split detection for broad argument nodes.
- `lifecycle.promote_min_slices`, `lifecycle.promote_min_support`
  - Candidate nodes are promoted to active when persistence/support thresholds are met.
- `lifecycle.stale_after_slices`
  - Active/candidate nodes are deprecated if not seen for this many slices.

### LLM
- `llm.enabled` (default: `True`)
  - If disabled, cluster summaries use TF-IDF heuristics and stance is skipped.
  - CLI: `--disable-llm`.
- `llm.api_url` (default: `https://openwebui.crc.nd.edu/api/v1/chat/completions`)
- `llm.model` (default: `gpt-oss:120b`)
- `llm.timeout_s` (default: `60`)

### Stance Estimation
- `stance.enabled` (default: `True`)
  - Estimates stance distributions per argument node per slice.
  - CLI: `--disable-stance`.
- `stance.sample_per_node` (default: `20`)
  - Number of posts sampled per node for stance classification.

## Notes on Text Fields
The current dataset’s `text` column already includes the `title`. If you move to a dataset where
`text` does not include the title, update `build_text` in `claimtaxo/pipeline.py` to concatenate
`title` + `text` (or use the title when text is missing). The pipeline has a small guard for the
missing-text case (`use_title_if_missing`).

## CLI Options
```bash
python -m claimtaxo \
  --input naloxone_mentions.csv \
  --output outputs \
  --min-sim 0.35 \
  --epsilon 0.005 \
  --rounds 5 \
  --disable-merge \
  --enable-split \
  --no-auto-viz
```

## Visualize After Run
Generate interactive HTML visualizations from pipeline outputs:

```bash
python -m claimtaxo.visualize --input-dir outputs --output-dir outputs/viz
```

By default, `python -m claimtaxo ...` now auto-generates visualizations after pipeline completion into
`<output_dir>/viz`. You can disable with `--no-auto-viz` or change subdirectory with `--viz-dir`.

Generated files:
- `slice_coverage.html`: coverage trend by slice
- `node_activity_heatmap.html`: top canonical argument-node activity over time
- `taxonomy_sunburst.html`: final taxonomy structure colored by node status
- `redirect_sankey.html`: redirect flows from merge/split operations
- `taxonomy_ops_timeline.html`: operation counts (`merge`/`split`/`promote`/`deprecate`) by slice

## Known Limitations
- Claim filtering is not implemented yet.
- Split operations are conservative and disabled by default; tune thresholds before large runs.
- Cluster summary quality depends on LLM availability; heuristic summaries are coarse.
- Silhouette-based K selection can be unstable on small clusters.
