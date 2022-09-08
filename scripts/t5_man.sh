for seed in 13 21 42 87 100
do
    for bs in 2 4 8
    do
sbatch -p rtx2080 --job-name exp-SPL-$task-man  --output=slurm/%j-man-$seed-$bs.out -N 1 -n 1 --cpus-per-task=8 --mem=15G --gres=gpu:1  <<EOF
#!/bin/sh
TAG=exp-SPL-$task-man \
TYPE=prompt \
TASK=$task \
BS=$bs \
LR=2e-5 \
SEED=$seed \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment_man.sh
TAG=exp-SPL-$task-man \
TYPE=prompt \
TASK=$task \
BS=$bs \
LR=6e-5 \
SEED=$seed \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment_man.sh
TAG=exp-SPL-$task-man \
TYPE=prompt \
TASK=$task \
BS=$bs \
LR=9e-5 \
SEED=$seed \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment_man.sh
EOF
    done
done