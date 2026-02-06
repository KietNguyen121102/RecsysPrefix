#!/usr/bin/env bash
set -euo pipefail

DATASET="$2"
echo "Dataset: ${DATASET}"
PYTHON="/data2/rsalgani/miniconda3/envs/prefix/bin/python"
BASE="/data2/rsalgani/Prefix/backup_res/${DATASET}/agg_files" #"/data2/rsalgani/Prefix/ml-1m/agg_files"
mkdir -p "${BASE}"
# =============================================================================
# RUNNING AGGREGATION
# =============================================================================

SCRIPT="/u/rsalgani/2024-2025/RecsysPrefix/02b_generate_agg_with_sampling_parallel.py"

NSAMPLES=100
USERS=10
ITEMS=30
JOBS=4   # adjust based on CPU + SCIP stability (try 4/8/16/24)

echo "[Launch] Parallel aggregation"
echo "PYTHON=${PYTHON}"
echo "SCRIPT=${SCRIPT}"
echo "NSAMPLES=${NSAMPLES} USERS=${USERS} ITEMS=${ITEMS} JOBS=${JOBS}"

"${PYTHON}" -u "${SCRIPT}" \
  --dataset "${DATASET}" \
  --n-samples "${NSAMPLES}" \
  --user-sample-size "${USERS}" \
  --item-sample-size "${ITEMS}" \
  --jobs "${JOBS}" \
  --outdir "${BASE}"


# =============================================================================
# RUNNING AXIOM SATISFACTION CALCULATION
# =============================================================================

# PYTHON="/data2/rsalgani/miniconda3/envs/prefix/bin/python"
SCRIPT="/u/rsalgani/2024-2025/RecsysPrefix/03b_axiom_satisfaction_calc_parallel.py"


MAX_JOBS=8 #was 8 
WORKERS=24
running=0

mkdir -p "${BASE}/axiom_logs"

for k in $(seq 0 99); do
  echo "[Dispatch] sample_${k}"

  "${PYTHON}" -u "${SCRIPT}" \
    --dataset "${DATASET}" \
    --agg "${BASE}/sample_${k}" \
    --pref "${BASE}/sample_${k}/sampled_rankings.pkl" \
    --workers "${WORKERS}" \
    > "${BASE}/axiom_logs/sample_${k}.out" 2> "${BASE}/axiom_logs/sample_${k}.err" &

  running=$((running + 1))

  if [[ "${running}" -ge "${MAX_JOBS}" ]]; then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "All axiom satisfaction samples finished."

# =============================================================================
# RUNNING KENDALL TAU CALCULATION
# =============================================================================

# PYTHON="/data2/rsalgani/miniconda3/envs/prefix/bin/python"
SCRIPT="/u/rsalgani/2024-2025/RecsysPrefix/03a_kendall_tau_calc_parallel.py"
# BASE="/data2/rsalgani/Prefix/${DATASET}/agg_files" 

MAX_JOBS=8
running=0

mkdir -p "${BASE}/kt_logs"

for k in $(seq 0 99); do
  echo "[Dispatch] sample_${k}"

  "${PYTHON}" -u "${SCRIPT}" \
    --dataset "${DATASET}" \
    --agg "${BASE}/sample_${k}" \
    --pref "${BASE}/sample_${k}/sampled_rankings.pkl" \
    > "${BASE}/kt_logs/sample_${k}.out" 2> "${BASE}/kt_logs/sample_${k}.err" &

  running=$((running + 1))

  if [[ "${running}" -ge "${MAX_JOBS}" ]]; then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "All KT calculation samples finished."


# =============================================================================
# RUNNING DIVERSITY CALCULATION
# =============================================================================

# PYTHON="/data2/rsalgani/miniconda3/envs/prefix/bin/python"
SCRIPT="/u/rsalgani/2024-2025/RecsysPrefix/03c_calc_diversity.py"
# BASE="/data2/rsalgani/Prefix/${DATASET}/agg_files" 

MAX_JOBS=8
running=0

mkdir -p "${BASE}/div_logs"

for k in $(seq 0 99); do
  echo "[Dispatch] sample_${k}"

  "${PYTHON}" -u "${SCRIPT}" \
    --dataset "${DATASET}" \
    --agg "${BASE}/sample_${k}" \
    --pref "${BASE}/sample_${k}/sampled_rankings.pkl" \
    > "${BASE}/div_logs/sample_${k}.out" 2> "${BASE}/div_logs/sample_${k}.err" &

  running=$((running + 1))

  if [[ "${running}" -ge "${MAX_JOBS}" ]]; then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "All samples finished."


python3 /u/rsalgani/2024-2025/RecsysPrefix/04_gather_results.py \
  --dataset "${DATASET}"\
  --input_base "${BASE}"

echo "All done."
