TRAIN_ESNLI=esnli/esnli.train.highlight.contradiction.jsonl
EVAL_ESNLI=esnli/esnli.dev.highlight.contradiction.jsonl

TYPE=esnli-zs-highlighter-$EXP
BS=24
python3 train.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_ESNLI \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 12500 \
  --save_steps 2500 \
  --eval_steps 2500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --do_train \
  --do_eval
