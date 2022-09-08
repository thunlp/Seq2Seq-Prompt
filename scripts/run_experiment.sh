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
        TEMPLATE=*cls**sentu_0**mask**sep+*
        MAPPING="{'0':'incorrect','1':'correct'}"
        ;;
    SST-2)
        TEMPLATE=*cls**sentu_0**mask**sep+*
        MAPPING="{'0':'terrible','1':'great'}"
        ;;
    MRPC)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    QQP)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        TASK_EXTRA="--num_sample 4"
        ;;
    STS-B)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    MNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 256 --num_sample 4"
        ;;
    SNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 256 --num_sample 4"
        ;;
    QNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        ;;
    RTE)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
    mr)
        TEMPLATE=*cls**sentu_0**mask**sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50 --double_demo"
        ;;
    sst-5)
        TEMPLATE=*cls**sentu_0**mask**sep+*
        MAPPING="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 20 --double_demo"
        ;;
    subj)
        TEMPLATE=*cls**sentu_0**mask**sep+*
        MAPPING="{0:'subjective',1:'objective'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50 --double_demo"
        ;;
    trec)
        TEMPLATE=*cls**sentu_0**mask**sep+*
        MAPPING="{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}"
        TASK_EXTRA="--first_sent_limit 110 --double_demo"
        ;;
    cr)
        TEMPLATE=*cls**sentu_0**mask**sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50 --double_demo"
        ;;
    mpqa)
        TEMPLATE=*cls**sentu_0*?*mask**sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110  --double_demo"
        ;;
    BoolQ)
        TEMPLATE=*cls**sentu_1*?*mask*,*+sent_0**sep+*
        MAPPING="{False:'No',True:'Yes'}"
        TASK_EXTRA="--max_seq_len 512"
        ;;
    CB)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 512"
        ;;
    COPA)
        TEMPLATE=*cls**sent_2**+sentu_3*?*sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{0:'False',1:'True'}"
        TASK_EXTRA="--max_seq_len 256"
        ;;
    MultiRC)
        TEMPLATE=*cls**sent_1**mask*,*+sentl_2**+sent_0**sep+*
        MAPPING="{0:'No',1:'Yes'}"
        TASK_EXTRA="--max_seq_len 512"
        ;;
    ReCoRD)
        TEMPLATE=*cls**sent_0**+sent_2**sep+*
        MAPPING="{0:'False',1:'True'}"
        TASK_EXTRA="--max_seq_len 512"
        ;;
    WiC)
        TEMPLATE="*cls**sent_0**+sent_1*_'*sent_2*'*mask**sep+*"
        MAPPING="{False:'No',True:'Yes'}"
        TASK_EXTRA="--max_seq_len 256"
        ;;
    WSC)
        TEMPLATE=*cls**sentu_0**mask**sep+*
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
  --mapping $MAPPING \
  $TASK_EXTRA \
  $1 

# Delete the checkpoint 
# Since we need to run multiple trials, saving all the checkpoints takes 
# a lot of storage space. You can find all evaluation results in `log` file anyway.
rm -r result/$TASK-$TYPE-$K-$SEED-$MODEL-$TRIAL_IDTF \