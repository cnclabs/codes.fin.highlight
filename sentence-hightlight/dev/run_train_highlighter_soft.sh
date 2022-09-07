# This code is used for training all highlighting models 
# Codes (1)-(4) provide differnet way to finetuen the highlgiht model 

# Dataset for finetunning 
TRAIN_ESNLI=esnli/esnli.train.highlight.contradiction.jsonl
EVAL_ESNLI=esnli/esnli.dev.highlight.contradiction.jsonl
TRAIN_FIN10K_CROSS=fin10k/fin10k.train.type2.segments.v4.0.r0.2.jsonl
TRAIN_FIN10K_FROMSCRATCH=fin10k/fin10k.train.type2.segments.v2.0.r0.2.jsonl
TRAIN_FIN10K_FURTHER=fin10k/fin10k.train.type2.segments.v1.0.r0.2.jsonl

# Heurtic labeling
TRAIN_FIN10K_FURTHER=fin10k/fin10k.train.type2.segments.v101.0.r1.jsonl
TRAIN_FIN10K_CROSS=fin10k/fin10k.train.type2.segments.v101.0.r1.jsonl
TRAIN_FIN10K_FROMSCRATCH=fin10k/fin10k.train.type2.segments.v101.0.r1.jsonl

export CUDA_VISIBLE_DEVICES=0
TYPE=further-finetune-soft
BS=24 # 10
python3 train.py \
  --ignore_data_skip \
  --resume_from_checkpoint ../checkpoints/esnli-zs-highlighter/checkpoint-10000 \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir $TYPE \
  --remove_unused_columns false \
  --train_file fin10k.heuristic.synthetic.balance.soft.train.type2.jsonl \
  --eval_file ../data/$EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 12500 \
  --save_steps 2500 \
  --eval_steps 2500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling true \
  --do_train \
  --do_eval 
