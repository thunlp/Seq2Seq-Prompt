for seed in 13 21 42 87 100
do
    for bs in 8
    do
        for lr in 8e-5
        do
sbatch -p rtx2080 --job-name exp-test-$task  --output=slurm/%j-$seed.out -N 1 -n 1 --cpus-per-task=8 --mem=15G --gres=gpu:1  <<EOF
#!/bin/sh
TAG=exp-test-$task \
TYPE=fprompt \
TASK=$task \
BS=$bs \
LR=$lr \
SEED=$seed \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment_fine_tuning.sh
EOF
        done
    done
done