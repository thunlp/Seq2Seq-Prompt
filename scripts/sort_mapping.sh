for mapping_id in {0..19}
do
        # To save time, we fix these hyper-parameters
        bs=8
        lr=6e-5

        # Since we only use dev performance here, use --no_predict to skip testing
sbatch -p rtx2080 --job-name exp-mapping-$task  --output=slurm/%j-$mapping_id.out -N 1 -n 1 --cpus-per-task=8 --mem=15G --gres=gpu:1  <<EOF
#!/bin/sh
TAG=exp-mapping-$task \
TYPE=prompt \
TASK=$task \
BS=$bs \
LR=$lr \
SEED=13 \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment.sh "--mapping_path auto_label_sequences/$task/16-13.txt --mapping_id $mapping_id --no_predict"
TAG=exp-mapping-$task \
TYPE=prompt \
TASK=$task \
BS=$bs \
LR=$lr \
SEED=21 \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment.sh "--mapping_path auto_label_sequences/$task/16-21.txt --mapping_id $mapping_id --no_predict"
TAG=exp-mapping-$task \
TYPE=prompt \
TASK=$task \
BS=$bs \
LR=$lr \
SEED=42 \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment.sh "--mapping_path auto_label_sequences/$task/16-42.txt --mapping_id $mapping_id --no_predict"
TAG=exp-mapping-$task \
TYPE=prompt \
TASK=$task \
BS=$bs \
LR=$lr \
SEED=87 \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment.sh "--mapping_path auto_label_sequences/$task/16-87.txt --mapping_id $mapping_id --no_predict"
TAG=exp-mapping-$task \
TYPE=prompt \
TASK=$task \
BS=$bs \
LR=$lr \
SEED=100 \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment.sh "--mapping_path auto_label_sequences/$task/16-100.txt --mapping_id $mapping_id --no_predict"
EOF
done