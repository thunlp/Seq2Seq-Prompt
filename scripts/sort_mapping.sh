for seed in 13 21 42 87 100
do
    for mapping_id in {0..19}
    do
        # To save time, we fix these hyper-parameters
        bs=8
        lr=6e-5

        # Since we only use dev performance here, use --no_predict to skip testing
TAG=exp-mapping-$task \
TYPE=prompt \
TASK=$task \
BS=$bs \
LR=$lr \
SEED=$seed \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment.sh "--mapping_path auto_label_sequences/$task/16-$seed.txt --mapping_id $mapping_id --no_predict"
    done
done