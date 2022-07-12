TRAIN=fin10k_v4.0_rand_0.2
TYPE=cross-domain-transfer-new
BS=8 # 24

python3 train.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir ./checkpoints/${TYPE}-0.125 \
  --train_file ./data/esnli/esnli.train.sent_highlight.contradiction.jsonl \
  --train_file_2 ./data/fin10k/${TRAIN}/train.type2.segments.jsonl.bert_filtered \
  --eval_file ./data/esnli/esnli.dev.sent_highlight.contradiction.jsonl \
  --test_file ./data/esnli/esnli.test.sent_highlight.contradiction.jsonl \
  --per_device_train_batch_size 8 \
  --max_steps 12500 \
  --save_steps 2500 \
  --eval_steps 2500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --do_train \
  --do_eval \
  --mixing_ratio 0.125
