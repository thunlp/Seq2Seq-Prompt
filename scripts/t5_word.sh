for seed in 13 21 42 87 100
do
    for bs in 2 4 8
    do
        for lr in 2e-5 6e-5 9e-5
        do
TAG=exp-autoword-$task \
TYPE=prompt \
TASK=$task \
BS=$bs \
LR=$lr \
SEED=$seed \
MODEL=google/t5-v1_1-base \
bash scripts/run_experiment.sh "--mapping_path auto_label_words/$task/16-$seed.sort.txt --mapping_id 0"
        done
    done
done