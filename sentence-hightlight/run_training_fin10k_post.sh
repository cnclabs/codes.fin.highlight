export CUDA_VISIBLE_DEVICES=0,1
TRAIN=fin10k_v2.0_rand_0.2
TYPE=further-finetune-new
BS=5 # 10
python3 train.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir ./checkpoints/${TYPE} \
  --train_file ./data/fin10k/${TRAIN}/train.type2.segments.jsonl.bert_filtered \
  --eval_file ./data/esnli/esnli.dev.sent_highlight.contradiction.jsonl \
  --test_file ./data/esnli/esnli.test.sent_highlight.contradiction.jsonl \
  --per_device_train_batch_size ${BS} \
  --max_steps 12500 \
  --save_steps 2500 \
  --eval_steps 2500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --do_train \
  --do_eval \
  --ignore_data_skip \
  --resume_from_checkpoint ./checkpoints/esnli-highlight/checkpoint-10000
