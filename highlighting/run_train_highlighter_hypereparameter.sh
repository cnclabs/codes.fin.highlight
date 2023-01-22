# Abalation: hyperparameter setting (tau), fixed gamma at -1.1
TRAIN_ESNLI=data/esnli/esnli.train.highlight.contradiction.jsonl
EVAL_ESNLI=data/esnli/esnli.dev.highlight.contradiction.jsonl
TRAIN_FIN10K=data/fin10k/fin10k.heuristic.synthetic.balance.train.type2.jsonl
export CUDA_VISIBLE_DEVICES=1

TYPE=further-finetune-sl-smooth-0.25 # tau=2; gamma=0.1
BS=24
python3 train_ablation.py \
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
  --soft_labeling true  \
  --tau 0.25 \
  --gamma 0.1 \
  --do_train \
  --do_eval 

TYPE=further-finetune-sl-smooth-0.1 # tau=0.5, gamma = 0.1
BS=24 
python3 train_ablation.py \
  --ignore_data_skip \
  --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-15000 \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 20000 \
  --save_steps 1500 \
  --eval_steps 1500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling true  \
  --tau 0.1 \
  --gamma 0.1 \
  --do_train \
  --do_eval 


