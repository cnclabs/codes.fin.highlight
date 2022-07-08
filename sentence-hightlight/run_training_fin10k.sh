export CUDA_VISIBLE_DEVICES=1
TRAIN=fin10k_v1.0_rand_0.2
TYPE=from-scratch-new
BS=3
python3 train.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir ./checkpoints/${TYPE} \
  --train_file ./data/fin10k/${TRAIN}/train.type2.segments.jsonl.bert_filtered \
  --eval_file ./data/esnli/esnli.dev.sent_highlight.contradiction.jsonl \
  --test_file ./data/fin10k/type2.cross_seg.eval.jsonl \
  --per_device_train_batch_size ${BS} \
  --max_steps 12500 \
  --save_steps 2500 \
  --eval_steps 2500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --do_train \
  --do_eval

