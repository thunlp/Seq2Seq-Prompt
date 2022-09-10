for seed in 13 21 42 87 100
do
    for bs in 2 4 8
    do
        for lr in 7e-5 1e-4 2e-4
        do
TAG=exp-fine-tuning-$task \
TYPE=fprompt \
TASK=$task \
BS=$bs \
LR=$lr \
SEED=$seed \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment_fine_tuning_k.sh
        done
    done
done