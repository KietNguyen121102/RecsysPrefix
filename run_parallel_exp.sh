#!/usr/bin/env bash

#Run aggregation methods 
set -euo pipefail

PYTHON="/data2/rsalgani/miniconda3/envs/prefix/bin/python"
SCRIPT="/u/rsalgani/2024-2025/RecsysPrefix/02_generate_agg_with_sampling_parallel.py"

INPUT="/u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/full_recset.csv"
OUTDIR="/data2/rsalgani/Prefix/ml-1m/agg_files"
GROUP_FILE="/u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/item_groups.pkl"

NSAMPLES=100
USERS=10
ITEMS=30

# parallelism inside each sample
# JOBS=4   # adjust based on CPU + SCIP stability (try 4/8/16/24)

# echo "[Launch] Parallel aggregation"
# echo "PYTHON=${PYTHON}"
# echo "SCRIPT=${SCRIPT}"
# echo "OUTDIR=${OUTDIR}"
# echo "NSAMPLES=${NSAMPLES} USERS=${USERS} ITEMS=${ITEMS} JOBS=${JOBS}"
# mkdir -p "${OUTDIR}/logs"
# "${PYTHON}" -u "${SCRIPT}" \
#   --input "${INPUT}" \
#   --outdir "${OUTDIR}" \
#   --n-samples "${NSAMPLES}" \
#   --user-sample-size "${USERS}" \
#   --item-sample-size "${ITEMS}" \
#   --group-file "${GROUP_FILE}" \
#   --jobs "${JOBS}" 



# set -euo pipefail

PYTHON="/data2/rsalgani/miniconda3/envs/prefix/bin/python"
SCRIPT="/u/rsalgani/2024-2025/RecsysPrefix/03b_axiom_satisfaction_calc_parallel.py"
BASE="/data2/rsalgani/Prefix/ml-1m/agg_files"

MAX_JOBS=8 #was 8 
WORKERS=24

mkdir -p "${BASE}/logs"

running=0

for k in $(seq 0 99); do
  echo "[Dispatch] sample_${k}"

  "${PYTHON}" -u "${SCRIPT}" \
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
