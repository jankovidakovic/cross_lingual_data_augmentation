# data args
TRAIN_FILENAME="./data/docee/train_all.csv"
DEV_FILENAME="./data/docee/18091999/dev.csv"
NUM_LABELS=59
DATASET_TYPE="docee"

# model args
MODEL_TYPE="roberta"
PRETRAINED_MODEL_NAME_OR_PATH="roberta-base"
# CONFIG_NAME=
# TOKENIZER_NAME=

# finetuning hyperparams
MAX_SEQ_LENGTH=512
NUM_TRAIN_EPOCHS=3
# MAX_STEPS=
PER_DEVICE_TRAIN_BATCH_SIZE=32
PER_DEVICE_EVAL_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=2  # gives effective batch size of 128
LEARNING_RATE=2e-5
LR_SCHEDULER_TYPE="linear"
WEIGHT_DECAY=0.01
# WARMUP_STEPS=
WARMUP_RATIO=0.06
ADAM_EPSILON=1e-8
MAX_GRAD_NORM=1.0

# runtime meta-args
GPUS="0,1"
LOGGING_STEPS=10
EVAL_STEPS=10
SAVE_STEPS=10
SAVE_TOTAL_LIMIT=3
METRIC_FOR_BEST_MODEL="f1_macro"
DATALOADER_NUM_WORKERS=2

# directories
OUTPUT_DIR="./outputs/docee_roberta_proto"
CACHE_DIR="./pretrained_models"
LOG_DIR="./logs"

# misc
WANDB_PROJECT="Cross Lingual Data Augmentation"
WANDB_RUN="roberta_docee_proto"
SEED=18091999

python run_finetuning.py \
\
  --train_filename "$TRAIN_FILENAME" \
  --dev_filename "$DEV_FILENAME" \
  --num_labels "$NUM_LABELS" \
  --dataset_type "$DATASET_TYPE" \
\
  --model_type "$MODEL_TYPE" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
\
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --weight_decay "$WEIGHT_DECAY" \
  --warmup_ratio "$WARMUP_RATIO" \
  --adam_epsilon "$ADAM_EPSILON" \
  --max_grad_norm "$MAX_GRAD_NORM" \
\
  --gpus "$GPUS" \
  --logging_steps "$LOGGING_STEPS" \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --metric_for_best_model "$METRIC_FOR_BEST_MODEL" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
\
  --output_dir "$OUTPUT_DIR" \
  --cache_dir "$CACHE_DIR" \
  --log_dir "$LOG_DIR" \
\
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run "$WANDB_RUN" \
  --seed "$SEED" \
  --verbose
