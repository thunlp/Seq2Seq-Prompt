for seed in 13 21 42 87 100
do
    for bs in 8
    do
        for lr in 8e-5
        do
TAG=exp-full-fine-tuning-$task \
TYPE=fprompt \
TASK=$task \
BS=$bs \
LR=$lr \
SEED=$seed \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment_fine_tuning.sh
        done
    done
done