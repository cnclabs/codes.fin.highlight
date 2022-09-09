# This code is used for training all highlighting models 
# Codes (1)-(4) provide differnet way to finetuen the highlgiht model 

# Dataset for finetunning 
TRAIN_ESNLI=data/esnli/esnli.train.highlight.contradiction.jsonl
EVAL_ESNLI=data/esnli/esnli.dev.highlight.contradiction.jsonl
TRAIN_FIN10K=data/fin10k/fin10k.heuristic.synthetic.balance.soft.train.type2.jsonl

# further-finetune
export CUDA_VISIBLE_DEVICES=0
# TYPE=further-finetune
# BS=24 
# python3 train.py \
#   --ignore_data_skip \
#   --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-10000 \
#   --model_name_or_path bert-base-uncased \
#   --config_name bert-base-uncased \
#   --output_dir checkpoints/$TYPE \
#   --train_file $TRAIN_FIN10K \
#   --eval_file $EVAL_ESNLI \
#   --per_device_train_batch_size $BS \
#   --max_steps 15000 \
#   --save_steps 2500 \
#   --eval_steps 2500 \
#   --max_seq_length 512 \
#   --evaluation_strategy 'steps'\
#   --evaluate_during_training \
#   --soft_labeling false  \
#   --do_train \
#   --do_eval 

TYPE=further-finetune-sl
BS=24 
python3 train.py \
  --ignore_data_skip \
  --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-10000 \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 15000 \
  --save_steps 2500 \
  --eval_steps 2500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling true  \
  --tau 1 \
  --gamma 0.5 \
  --do_train \
  --do_eval 

# TYPE=further-finetune-sl-tunned
# BS=24 
# python3 train.py \
#   --ignore_data_skip \
#   --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-10000 \
#   --model_name_or_path bert-base-uncased \
#   --config_name bert-base-uncased \
#   --output_dir checkpoints/$TYPE \
#   --train_file $TRAIN_FIN10K \
#   --eval_file $EVAL_ESNLI \
#   --per_device_train_batch_size $BS \
#   --max_steps 15000 \
#   --save_steps 2500 \
#   --eval_steps 2500 \
#   --max_seq_length 512 \
#   --evaluation_strategy 'steps'\
#   --evaluate_during_training \
#   --soft_labeling true  \
#   --tau 0.25 \
#   --gamma 0.1 \
#   --do_train \
#   --do_eval 

TYPE=further-finetune-sl-smooth
BS=24 
python3 train.py \
  --ignore_data_skip \
  --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-10000 \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 15000 \
  --save_steps 2500 \
  --eval_steps 2500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling true  \
  --tau 2 \
  --gamma 0.1 \
  --do_train \
  --do_eval 

# from-scratch
# export CUDA_VISIBLE_DEVICES=0
# TYPE=from-scratch
# BS=24
# python3 train.py \
#   --model_name_or_path bert-base-uncased \
#   --config_name bert-base-uncased \
#   --output_dir checkpoints/$TYPE \
#   --train_file $TRAIN_FIN10K \
#   --eval_file $EVAL_ESNLI \
#   --per_device_train_batch_size $BS \
#   --max_steps 10000 \
#   --save_steps 2500 \
#   --eval_steps 2500 \
#   --max_seq_length 512 \
#   --evaluation_strategy 'steps'\
#   --evaluate_during_training \
#   --soft_labeling false  \
#   --do_train \
#   --do_eval

# TYPE=from-scratch-sl
# BS=24
# python3 train.py \
#   --model_name_or_path bert-base-uncased \
#   --config_name bert-base-uncased \
#   --output_dir checkpoints/$TYPE \
#   --train_file $TRAIN_FIN10K \
#   --eval_file $EVAL_ESNLI \
#   --per_device_train_batch_size $BS \
#   --max_steps 10000 \
#   --save_steps 2500 \
#   --eval_steps 2500 \
#   --max_seq_length 512 \
#   --evaluation_strategy 'steps'\
#   --evaluate_during_training \
#   --soft_labeling true  \
#   --tau 1 \
#   --gamma 0.5 \
#   --do_train \
#   --do_eval

TYPE=from-scratch-sl-tunned
BS=24
python3 train.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 10000 \
  --save_steps 2500 \
  --eval_steps 2500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling true  \
  --tau 0.25 \
  --gamma 0.1 \
  --do_train \
  --do_eval

TYPE=from-scratch-sl-smooth
BS=24
python3 train.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 10000 \
  --save_steps 2500 \
  --eval_steps 2500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling true  \
  --tau 2 \
  --gamma 0.1 \
  --do_train \
  --do_eval

# # cross-domain-finetunning
# TYPE=cross-domain-transfer-$EXP
# BS=24
# MIXING_RATE=0.125 # indicate 8 * 0.125 instances out of 8
# python3 train.py \
#   --model_name_or_path bert-base-uncased \
#   --config_name bert-base-uncased \
#   --output_dir checkpoints/$TYPE-$MIXING_RATE \
#   --train_file $TRAIN_ESNLI \
#   --train_file_2 $TRAIN_FIN10K \
#   --eval_file $EVAL_ESNLI \
#   --per_device_train_batch_size $BS \
#   --max_steps 12500 \
#   --save_steps 2500 \
#   --eval_steps 2500 \
#   --max_seq_length 512 \
#   --evaluation_strategy 'steps'\
#   --evaluate_during_training \
#   --do_train \
#   --do_eval \
#   --mixing_ratio $MIXING_RATE
