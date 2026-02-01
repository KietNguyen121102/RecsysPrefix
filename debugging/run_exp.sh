
#!/usr/bin/env bash

# INPUT="/u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/full_recset.csv" #"/u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/full_recset.csv"
# OUTDIR="/data2/rsalgani/Prefix/ml-1m/agg_files" #"/data2/rsalgani/Prefix/ml-1m/agg_files"
# GROUP_FILE="/u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/item_groups.pkl" #"/u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/item_groups.pkl"

# PYTHON="/data2/rsalgani/miniconda3/envs/prefix/bin/python"
# SCRIPT="/u/rsalgani/2024-2025/RecsysPrefix/02b_generate_agg_with_sampling.py"

# NSAMPLES=1
# USERS=10
# ITEMS=30

# mkdir -p "${OUTDIR}/logs"

# echo "[Launch] aggregation"
# echo "PYTHON=${PYTHON}"
# echo "SCRIPT=${SCRIPT}"
# echo "OUTDIR=${OUTDIR}"

# #Run aggregation methods 
# "${PYTHON}" -u "${SCRIPT}" \
#   --input "${INPUT}" \
#   --outdir "${OUTDIR}" \
#   --n-samples "${NSAMPLES}" \
#   --user-sample-size "${USERS}" \
#   --item-sample-size "${ITEMS}" \
#   --group-file "${GROUP_FILE}"






# python -u /u/rsalgani/2024-2025/RecsysPrefix/02_generate_agg_with_sampling.py \
#     --input /u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/recommendations.csv \
#     --outdir /data2/rsalgani/Prefix/ml-1m/agg_files \
#     --n-samples 1 \
#     --user-sample-size 10 \
#     --item-sample-size 30 \
#     --group-file data/ml-1m/item_groups.pkl

##Run Axiom Satisfaction Calculation
# for k in $(seq 0 1); do
#     python -u /u/rsalgani/2024-2025/RecsysPrefix/03b_axiom_satisfaction_calc.py \
#         --agg /data2/rsalgani/Prefix/ml-1m/agg_files/sample_${k} \
#         --pref /data2/rsalgani/Prefix/ml-1m/agg_files/sample_${k}/sampled_rankings.pkl     
# done 

# ##Calculate diversity metrics for each sample
# for k in $(seq 0 99); do
#     python -u /u/rsalgani/2024-2025/RecsysPrefix/03c_calc_diversity.py \
#         --agg /data2/rsalgani/Prefix/ml-1m/agg_files/sample_${k} \
#         --pref /data2/rsalgani/Prefix/ml-1m/agg_files/sample_${k}/sampled_rankings.pkl      
# done 


DATASET="$2"
echo "Dataset: ${DATASET}"

PYTHON="/data2/rsalgani/miniconda3/envs/prefix/bin/python"
SCRIPT="/u/rsalgani/2024-2025/RecsysPrefix/03a_kendall_tau_calc.py"   
BASE="/data2/rsalgani/Prefix/${DATASET}/agg_files"                   # or /data2/rsalgani/Prefix/ml-1m/agg_files

MAX_JOBS=8
running=0

mkdir -p "${BASE}/kt_logs"

for k in $(seq 0 1); do
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
echo "All samples finished."
