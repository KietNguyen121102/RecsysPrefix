#!/bin/bash
# =============================================================================
# Goodreads RecSys Pipeline - Full Experiment Runner
# =============================================================================
# This script runs the complete pipeline for Goodreads book recommendations:
# 1. Generate item popularity groups
# 2. Train RecSys model and generate recommendations
# 3. Apply rank aggregation methods
# 4. Evaluate consensus rankings (Kendall Tau, Axiom Satisfaction, Diversity)
# 5. Gather and summarize results
# =============================================================================

set -e  # Exit on error

# Configuration
DATA_PATH="data/goodreads_interactions.csv"
GROUP_FILE="data/goodreads/item_groups.pkl"
MODEL_PATH="./model"
RECOMMENDATIONS_FILE="recommendations.csv"
OUTPUT_DIR="consensus_results"

# Dataset subsampling (set to limit original dataset size)
MAX_USERS=3000         # Max users from original dataset (set to "" for all)
MAX_ITEMS=3000       # Max items from original dataset (set to "" for all)
MIN_USER_RATINGS=5     # Min ratings per user to keep
MIN_BOOK_RATINGS=5     # Min ratings per book to keep

# Aggregation sampling
N_SAMPLES=10
USER_SAMPLE_SIZE=10
ITEM_SAMPLE_SIZE=20
SEED=42

echo "=============================================================="
echo "GOODREADS RECSYS PIPELINE"
echo "=============================================================="
echo "Data Path: $DATA_PATH"
echo "Max Users: ${MAX_USERS:-all}"
echo "Max Items: ${MAX_ITEMS:-all}"
echo "Min User Ratings: $MIN_USER_RATINGS"
echo "Min Book Ratings: $MIN_BOOK_RATINGS"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of Samples: $N_SAMPLES"
echo "Users per Sample: $USER_SAMPLE_SIZE"
echo "Items per Sample: $ITEM_SAMPLE_SIZE"
echo "=============================================================="

# Step 0: Generate item groups (if not exists)
if [ ! -f "$GROUP_FILE" ]; then
    echo ""
    echo "[Step 0] Generating item popularity groups..."
    python utils/generate_groups.py \
        --ratings "$DATA_PATH" \
        --output "$GROUP_FILE" \
        --n-bins 5
fi

# Step 1: Train model and generate recommendations
echo ""
echo "[Step 1] Training RecSys model and generating recommendations..."

# Build command with optional max-users and max-items
CMD="python 01_recsys_pipeline.py \
    --data-path $DATA_PATH \
    --model-out $MODEL_PATH \
    --output-file $RECOMMENDATIONS_FILE \
    --min-user-ratings $MIN_USER_RATINGS \
    --min-book-ratings $MIN_BOOK_RATINGS \
    --seed $SEED"

if [ -n "$MAX_USERS" ]; then
    CMD="$CMD --max-users $MAX_USERS"
fi

if [ -n "$MAX_ITEMS" ]; then
    CMD="$CMD --max-items $MAX_ITEMS"
fi

eval $CMD

# Step 2: Generate rank aggregations with sampling
echo ""
echo "[Step 2] Applying rank aggregation methods..."
python 02_generate_agg_with_sampling.py \
    --input "$RECOMMENDATIONS_FILE" \
    --outdir "$OUTPUT_DIR" \
    --group-file "$GROUP_FILE" \
    --user-sample-size $USER_SAMPLE_SIZE \
    --item-sample-size $ITEM_SAMPLE_SIZE \
    --n-samples $N_SAMPLES \
    --seed $SEED

# Step 3: Evaluate each sample
echo ""
echo "[Step 3] Evaluating consensus rankings..."

for i in $(seq 0 $((N_SAMPLES - 1))); do
    SAMPLE_DIR="$OUTPUT_DIR/sample_$i"
    PREF_FILE="$SAMPLE_DIR/sampled_rankings.pkl"
    
    echo ""
    echo "--- Processing Sample $i ---"
    
    # 3a. Kendall Tau
    echo "  Computing Kendall Tau..."
    python 03a_kendall_tau_calc.py \
        --pref "$PREF_FILE" \
        --agg "$SAMPLE_DIR" \
        --output "$SAMPLE_DIR/kendall_tau_results.csv" 2>/dev/null || true
    
    # 3b. Axiom Satisfaction
    echo "  Computing Axiom Satisfaction..."
    python 03b_axiom_satisfaction_calc.py \
        --pref "$PREF_FILE" \
        --agg "$SAMPLE_DIR" 2>/dev/null || true
    
    # 3c. Diversity
    echo "  Computing Diversity Metrics..."
    python 03c_calc_diversity.py \
        --pref "$PREF_FILE" \
        --agg "$SAMPLE_DIR" \
        --group-file "$GROUP_FILE" \
        --top-k 10 2>/dev/null || true
done

# Step 4: Gather results
echo ""
echo "[Step 4] Gathering and summarizing results..."
python 04_gather_results.py \
    --base-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR/summary"

echo ""
echo "=============================================================="
echo "PIPELINE COMPLETE"
echo "=============================================================="
echo "Results available in: $OUTPUT_DIR/summary/"
echo "=============================================================="

