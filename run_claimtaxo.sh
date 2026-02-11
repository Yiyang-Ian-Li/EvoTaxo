#!/bin/bash

# Run ClaimTaxo pipeline on naloxone_mentions.csv
# Usage: bash run_claimtaxo.sh

# export OPENAI_API_KEY='your_api_key_here'

python -m claimtaxo \
  --input naloxone_mentions.csv \
  --output outputs \
  --min-sim 0.35 \
  --epsilon 0.005 \
  --rounds 5
