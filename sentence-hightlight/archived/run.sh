# This code is used for training all highlighting models 
# Codes (1)-(4) provide differnet way to finetuen the highlgiht model 

# Dataset for finetunning 
TRAIN_ESNLI=data/esnli/esnli.train.highlight.contradiction.jsonl
EVAL_ESNLI=data/esnli/esnli.dev.highlight.contradiction.jsonl
# TRAIN_FIN10K=data/fin10k/fin10k.heuristic.synthetic.balance.soft.train.type2.jsonl
TRAIN_FIN10K=data/fin10k/fin10k.lexicon.synthetic.balance.train.type2.jsonl

export CUDA_VISIBLE_DEVICES=0
# further-finetune
TYPE=further-finetune-ll
BS=24 
python3 train.py \
  --ignore_data_skip \
  --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-15000 \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 20000\
  --save_steps 1500 \
  --eval_steps 1500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling false  \
  --tau 1 \
  --gamma 1 \
  --do_train \
  --do_eval 

# from-scratch
export CUDA_VISIBLE_DEVICES=0
TYPE=from-scratch-ll
BS=24
python3 train.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 10000 \
  --save_steps 3000 \
  --eval_steps 3000 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling false  \
  --tau 1 \
  --gamma 1 \
  --do_train \
  --do_eval
