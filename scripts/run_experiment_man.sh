# Required environment variables:
# TAG: tag for the trail
# TYPE: finetune / prompt / prompt-demo  
# TASK: SST-2 / sst-5 / mr / cr / mpqa / subj / trec / CoLA / MNLI / SNLI / QNLI / RTE / MRPC / QQP / STS-B
# BS: batch size (recommendation: 2 / 4 / 8)
# LR: learning rate (recommendation: 1e-5 / 2e-5 / 5e-5)
# SEED: random seed (13 / 21 / 42 / 87 / 100)
# MODEL: pre-trained model name (roberta-*, bert-*), see Transformers model list

# Number of training instances per label
K=16

# Training steps
MAX_STEP=1000

# Validation steps
EVAL_STEP=100

# Task specific parameters
# The default length is 128 and the default number of samples is 16.
# For some tasks, we use longer length or double demo (when using demonstrations, double the maximum length).
# For some tasks, we use smaller number of samples to save time (because of the large size of the test sets).
# All those parameters are set arbitrarily by observing the data distributions.
TASK_EXTRA=""
case $TASK in
    CoLA)
        TEMPLATE=*cls**sent_0*_The_grammar_is*mask*.*sep+*
        MAPPING="{'0':'unacceptable','1':'acceptable'}"
        ;;
    SST-2)
        TEMPLATE=*cls**sent_0*_Overall_my_impression_is*mask*.*sep+*
        MAPPING="{'0':'bad','1':'good'}"
        ;;
    MRPC)
        TEMPLATE=*cls**sent_0*_and*+sent_1*_are_the*mask*.*sep+*
        MAPPING="{'0':'different','1':'same'}"
        ;;
    QQP)
        TEMPLATE=*cls**sent_0*_and*+sent_1*_are_the*mask*.*sep+*
        MAPPING="{'0':'different','1':'same'}"
        TASK_EXTRA="--num_sample 4"
        ;;
    STS-B)
        TEMPLATE=*cls**sent_0*_and*+sent_1*_are_the*mask*.*sep+*
        MAPPING="{'0':'different','1':'same'}"
        ;;
    MNLI)
        TEMPLATE=*cls*_Premise:*sent_1*_Hypothesis:*sent_0*_Label:*mask**sep+*
        MAPPING="{'contradiction':'no','entailment':'yes','neutral':'maybe'}"
        TASK_EXTRA="--max_seq_len 256 --num_sample 4"
        ;;
    SNLI)
        TEMPLATE=*cls*_Premise:*sent_1*_Hypothesis:*sent_0*_Label:*mask**sep+*
        MAPPING="{'contradiction':'no','entailment':'yes','neutral':'maybe'}"
        TASK_EXTRA="--max_seq_len 256 --num_sample 4"
        ;;
    QNLI)
        TEMPLATE=*cls*_Question:*sent_0*_Sentence:*sent_1*_Label:*mask**sep+*
        MAPPING="{'not_entailment':'no','entailment':'yes'}"
        ;;
    RTE)
        TEMPLATE=*cls*_Premise:*sent_0*_Hypothesis:*sent_1*_Label:*mask**sep+*
        MAPPING="{'not_entailment':'no','entailment':'yes'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
    mr)
        TEMPLATE=*cls**sent_0*_Overall_my_impression_is*mask*.*sep+*
        MAPPING="{0:'bad',1:'good'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50 --double_demo"
        ;;
    sst-5)
        TEMPLATE=*cls**sent_0*_Overall_my_impression_is*mask*.*sep+*
        MAPPING="{0:'very bad',1:'bad',2:'not bad',3:'good',4:'very good'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 20 --double_demo"
        ;;
    subj)
        TEMPLATE=*cls**sent_0*_The_sentence_is*mask*.*sep+*
        MAPPING="{0:'subjective',1:'objective'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50 --double_demo"
        ;;
    trec)
        TEMPLATE=*cls**sent_0*_The_question_is_about*mask*.*sep+*
        MAPPING="{0:'abbreviation',1:'entity',2:'description',3:'human',4:'location',5:'numeric'}"
        TASK_EXTRA="--first_sent_limit 110 --double_demo"
        ;;
    cr)
        TEMPLATE=*cls**sent_0*_Overall_my_impression_is*mask*.*sep+*
        MAPPING="{0:'bad',1:'good'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50 --double_demo"
        ;;
    mpqa)
        TEMPLATE=*cls**sent_0*_Overall_my_impression_is*mask*.*sep+*
        MAPPING="{0:'bad',1:'good'}"
        TASK_EXTRA="--first_sent_limit 110  --double_demo"
        ;;
    BoolQ)
        TEMPLATE=*cls*_Passage:*sent_0*_Question:*sent_1*_Answer:*mask*.*sep+*
        MAPPING="{False:'false',True:'true'}"
        TASK_EXTRA="--max_seq_len 512"
        ;;
    CB)
        TEMPLATE=*cls*_Premise:*sent_0*_Hypothesis:*sent_1*_Label:*mask**sep+*
        MAPPING="{'contradiction':'no','entailment':'yes','neutral':'maybe'}"
        TASK_EXTRA="--max_seq_len 512"
        ;;
    COPA)
        TEMPLATE=*cls*_Premise:*sent_2*_Question:*sent_3*_Choice1:*sent_0*_Choice2:*sent_1*_Answer:*mask*.*sep+*
        MAPPING="{0:'Choice1',1:'Choice2'}"
        TASK_EXTRA="--max_seq_len 256"
        ;;
    MultiRC)
        TEMPLATE=*cls*_Paragraph:*sent_0*_Question:*sent_1*_Answer:*sent_2*_Label:*mask**sep+*
        MAPPING="{0:'false',1:'true'}"
        TASK_EXTRA="--max_seq_len 512"
        ;;
    ReCoRD)
        TEMPLATE=*cls**sent_2**+sent_1**+sent_0**sep+*
        MAPPING="{0:'False',1:'True'}"
        TASK_EXTRA="--max_seq_len 512"
        ;;
    WiC)
        TEMPLATE="*cls*_'*sent_2*'_in*+sent_0*_and_'*sent_2*'_in*+sent_1*_are_the*mask*.*sep+*"
        MAPPING="{False:'different',True:'same'}"
        TASK_EXTRA="--max_seq_len 256"
        ;;
    WSC)
        TEMPLATE=*cls**sent_0**+sent_2*_is*mask*.*sep+*
        MAPPING="{False:'False',True:'True'}"
        TASK_EXTRA="--max_seq_len 256"
        ;;

esac

# Gradient accumulation steps
# For medium-sized GPUs (e.g., 2080ti with 10GB memory), they can only take 
# a maximum batch size of 2 when using large-size models. So we use gradient
# accumulation steps to achieve the same effect of larger batch sizes.
REAL_BS=2
GS=$(expr $BS / $REAL_BS)

# Use a random number to distinguish different trails (avoid accidental overwriting)
TRIAL_IDTF=$RANDOM
DATA_DIR=data/k-shot/$TASK/$K-$SEED

python run.py \
  --task_name $TASK \
  --data_dir $DATA_DIR \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy epoch \
  --model_name_or_path $MODEL \
  --few_shot_type $TYPE \
  --num_k $K \
  --max_seq_length 128 \
  --per_device_train_batch_size $REAL_BS \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps $GS \
  --learning_rate $LR \
  --max_steps $MAX_STEP \
  --logging_steps $EVAL_STEP \
  --eval_steps $EVAL_STEP \
  --num_train_epochs 0 \
  --output_dir result/$TASK-$TYPE-$K-$SEED-$MODEL-$TRIAL_IDTF \
  --seed $SEED \
  --tag $TAG \
  --template $TEMPLATE \
  --mapping "$MAPPING" \
  $TASK_EXTRA \
  $1 

# Delete the checkpoint 
# Since we need to run multiple trials, saving all the checkpoints takes 
# a lot of storage space. You can find all evaluation results in `log` file anyway.
rm -r result/$TASK-$TYPE-$K-$SEED-$MODEL-$TRIAL_IDTF \