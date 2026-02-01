#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# RUNNING AGGREGATION
# =============================================================================



# PYTHON="/data2/rsalgani/miniconda3/envs/prefix/bin/python"
# SCRIPT="/u/rsalgani/2024-2025/RecsysPrefix/02b_generate_agg_with_sampling_parallel.py"

# NSAMPLES=100
# USERS=10
# ITEMS=30
# JOBS=4   # adjust based on CPU + SCIP stability (try 4/8/16/24)

# echo "[Launch] Parallel aggregation"
# echo "PYTHON=${PYTHON}"
# echo "SCRIPT=${SCRIPT}"
# echo "NSAMPLES=${NSAMPLES} USERS=${USERS} ITEMS=${ITEMS} JOBS=${JOBS}"
# # mkdir -p "${OUTDIR}/logs"
# "${PYTHON}" -u "${SCRIPT}" \
#   --dataset "goodreads" \
#   --n-samples "${NSAMPLES}" \
#   --user-sample-size "${USERS}" \
#   --item-sample-size "${ITEMS}" \
#   --jobs "${JOBS}" 


# =============================================================================
# RUNNING AXIOM SATISFACTION CALCULATION
# =============================================================================

PYTHON="/data2/rsalgani/miniconda3/envs/prefix/bin/python"
SCRIPT="/u/rsalgani/2024-2025/RecsysPrefix/03b_axiom_satisfaction_calc_parallel.py"
BASE="/data2/rsalgani/Prefix/goodreads/agg_files" #"/data2/rsalgani/Prefix/ml-1m/agg_files"

MAX_JOBS=8 #was 8 
WORKERS=24
running=0

for k in $(seq 0 99); do
  echo "[Dispatch] sample_${k}"

  "${PYTHON}" -u "${SCRIPT}" \
    --dataset "goodreads" \
    --agg "${BASE}/sample_${k}" \
    --pref "${BASE}/sample_${k}/sampled_rankings.pkl" \
    --workers "${WORKERS}" \
    > "${BASE}/logs/sample_${k}.out" 2> "${BASE}/logs/sample_${k}.err" &

  running=$((running + 1))

  if [[ "${running}" -ge "${MAX_JOBS}" ]]; then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "All samples finished."
