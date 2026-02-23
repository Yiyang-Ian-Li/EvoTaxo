# Some Notes When Running the Program

## About opiates data

```bash
(claimtaxo) (base) yli62@yeslab-10:~/Documents/ClaimTaxo$ uv run python data/3_filter_by_label_thresholds.py \
    --scores data/zero_shot_bart_mnli_scores.csv \
    --data data/opiates_text_filtered.csv \
    --id-col id \
    --labels Factual_Claim Evaluative_Opinion Causal_Claim Policy_Prescription Argumentative_Reasoning \
    --mode multi \
    --global-threshold 0.7 \
    --require-labels Factual_Claim Evaluative_Opinion Causal_Claim Policy_Prescription Argumentative_Reasoning \
    --output data/opiates_claimtaxo_input_5labels_ge0.7.csv \
    --output-dropped data/opiates_claimtaxo_dropped_5labels_ge0.7.csv \
    --output-summary data/opiates_claimtaxo_input_5labels_ge0.7_summary.json
Done.
Input rows: 158496
Kept rows: 17138 (10.81%)
Dropped rows: 141358
Output: data/opiates_claimtaxo_input_5labels_ge0.7.csv
Dropped output: data/opiates_claimtaxo_dropped_5labels_ge0.7.csv
Summary: data/opiates_claimtaxo_input_5labels_ge0.7_summary.json
Kept selected label counts:
  Causal_Claim: 9363
  Evaluative_Opinion: 8045
  Policy_Prescription: 6209
  Factual_Claim: 2555
  Argumentative_Reasoning: 868
```
