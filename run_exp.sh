
#Run aggregation methods 
python -u /u/rsalgani/2024-2025/RecsysPrefix/02_generate_agg_with_sampling.py \
    --input /u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/recommendations.csv \
    --outdir /data2/rsalgani/Prefix/ml-1m/agg_files \
    --n-samples 100 \
    --user-sample-size 10 \
    --item-sample-size 30 \
    --group-file data/ml-1m/item_groups.pkl

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