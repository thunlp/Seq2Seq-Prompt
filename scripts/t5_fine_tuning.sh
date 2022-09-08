for seed in 13 21 42 87 100
do
    for bs in 2 4 8
    do
sbatch -p rtx2080 --job-name exp-fine-tuning-$task  --output=slurm/%j-$seed-$bs.out -N 1 -n 1 --cpus-per-task=8 --mem=15G --gres=gpu:1  <<EOF
#!/bin/sh 
TAG=exp-fine-tuning-$task \
TYPE=fprompt \
TASK=$task \
BS=$bs \
LR=7e-5 \
SEED=$seed \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment_fine_tuning_k.sh
TAG=exp-fine-tuning-$task \
TYPE=fprompt \
TASK=$task \
BS=$bs \
LR=1e-4 \
SEED=$seed \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment_fine_tuning_k.sh
TAG=exp-fine-tuning-$task \
TYPE=fprompt \
TASK=$task \
BS=$bs \
LR=2e-4 \
SEED=$seed \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment_fine_tuning_k.sh
EOF
    done
done