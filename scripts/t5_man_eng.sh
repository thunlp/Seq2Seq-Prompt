for seed in 13 21 42 87 100
do
    for bs in 2 4 8
    do
        for lr in 2e-5 6e-5 9e-5
        do
TAG=exp-man-eng-$task \
TYPE=prompt \
TASK=$task \
BS=$bs \
LR=$lr \
SEED=$seed \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment_man_eng.sh
        done
    done
done