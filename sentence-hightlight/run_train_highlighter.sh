# This code is used for training all highlighting models 
# Codes (1)-(4) provide differnet way to finetuen the highlgiht model 

# Dataset for finetunning 
TRAIN_ESNLI=esnli/esnli.train.sent_highlight.contradiction.jsonl
EVAL_ESNLI=esnli/esnli.dev.sent_highlight.contradiction.jsonl
TRAIN_FIN10K_CROSS=fin10k/fin10k.train.type2.segments.v4.0.r0.2.jsonl
TRAIN_FIN10K_FROMSCRATCH=fin10k/fin10k.train.type2.segments.v2.0.r0.2.jsonl
TRAIN_FIN10K_FURTHER=fin10k/fin10k.train.type2.segments.v1.0.r0.2.jsonl

# Heurtic labeling
TRAIN_FIN10K_FURTHER=fin10k/fin10k.train.type2.segments.v101.0.r1.jsonl
TRAIN_FIN10K_CROSS=fin10k/fin10k.train.type2.segments.v101.0.r1.jsonl
TRAIN_FIN10K_FROMSCRATCH=fin10k/fin10k.train.type2.segments.v101.0.r1.jsonl
EXP=$2

# (1) training with Esnli (zero-shot fin10k highlighter)
if [[ "$1" = 'zero-shot' ]]
then
    TYPE=esnli-zs-highlighter
    BS=8
    python3 train.py \
      --model_name_or_path bert-base-uncased \
      --config_name bert-base-uncased \
      --output_dir checkpoints/$TYPE \
      --train_file data/$TRAIN_ESNLI \
      --eval_file data/$EVAL_ESNLI \
      --per_device_train_batch_size $BS \
      --max_steps 12500 \
      --save_steps 2500 \
      --eval_steps 2500 \
      --max_seq_length 256 \
      --evaluation_strategy 'steps'\
      --evaluate_during_training \
      --do_train \
      --do_eval
fi

# (2) training with domain-transfer via mulit-task learning 
if [[ "$1" = 'cross-domain-transfer' ]]
then
    TYPE=cross-domain-transfer-$EXP
    BS=8
    MIXING_RATE=0.125 # indicate 8 * 0.125 instances out of 8
    python3 train.py \
      --model_name_or_path bert-base-uncased \
      --config_name bert-base-uncased \
      --output_dir checkpoints/$TYPE-$MIXING_RATE \
      --train_file data/$TRAIN_ESNLI \
      --train_file_2 data/$TRAIN_FIN10K_CROSS \
      --eval_file data/$EVAL_ESNLI \
      --per_device_train_batch_size $BS \
      --max_steps 12500 \
      --save_steps 2500 \
      --eval_steps 2500 \
      --max_seq_length 512 \
      --evaluation_strategy 'steps'\
      --evaluate_during_training \
      --do_train \
      --do_eval \
      --mixing_ratio $MIXING_RATE
fi

# (3) training with in-dmain weakly supervised learning from scratch
if [[ "$1" = 'from-scratch' ]]
then
    export CUDA_VISIBLE_DEVICES=1
    TYPE=from-scratch-$EXP
    BS=6
    python3 train.py \
      --model_name_or_path bert-base-uncased \
      --config_name bert-base-uncased \
      --output_dir checkpoints/$TYPE \
      --train_file data/$TRAIN_FIN10K_FROMSCRATCH \
      --eval_file data/$EVAL_ESNLI \
      --per_device_train_batch_size $BS \
      --max_steps 12500 \
      --save_steps 2500 \
      --eval_steps 2500 \
      --max_seq_length 512 \
      --evaluation_strategy 'steps'\
      --evaluate_during_training \
      --do_train \
      --do_eval
fi

# (4) Further training for out-domain transfer:w
if [[ "$1" = 'further-finetune' ]]
then
    export CUDA_VISIBLE_DEVICES=0
    TYPE=further-finetune-$EXP
    BS=6 # 10
    python3 train.py \
      --ignore_data_skip \
      --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-10000 \
      --model_name_or_path bert-base-uncased \
      --config_name bert-base-uncased \
      --output_dir checkpoints/$TYPE \
      --train_file data/$TRAIN_FIN10K_FURTHER \
      --eval_file data/$EVAL_ESNLI \
      --per_device_train_batch_size $BS \
      --max_steps 12500 \
      --save_steps 2500 \
      --eval_steps 2500 \
      --max_seq_length 512 \
      --evaluation_strategy 'steps'\
      --evaluate_during_training \
      --do_train \
      --do_eval 
fi
