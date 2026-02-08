#!/bin/bash

# Run TaxoAdapt for naloxone Reddit data
# Usage: bash run_naloxone_taxonomy.sh

# Set API key (if not already in environment)
# export OPENAI_API_KEY='your_api_key_here'

echo "=========================================="
echo "Running TaxoAdapt for Naloxone Reddit Data"
echo "=========================================="

cd taxoadapt-copy

python main.py \
    --topic "naloxone discussion" \
    --dataset "naloxone_reddit" \
    --csv_path "../naloxone_mentions.csv" \
    --sample_size 50 \
    --llm "custom" \
    --max_depth 2 \
    --init_levels 1 \
    --max_density 5

echo ""
echo "=========================================="
echo "Done! Check taxoadapt-copy/datasets/naloxone_reddit/ for results"
echo "=========================================="
