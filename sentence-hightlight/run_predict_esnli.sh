export CUDA_VISIBLE_DEVICES=0
BS=16

for MODEL in cross-domain-transfer-0.125  esnli-zs-highlighter from-scratch further-finetune;do
    EVAL=data/esnli/esnli.dev.sent_highlight.contradiction.jsonl
    OUTPUT=${EVAL##*/}
    python3 inference.py \
      --model_name_or_path checkpoints/$MODEL/checkpoint-12500/ \
      --config_name bert-base-uncased \
      --eval_file $EVAL \
      --output_file results/esnli/${OUTPUT/jsonl/results}-$MODEL \
      --remove_unused_columns false \
      --max_seq_length 512 \
      --per_device_train_batch_size $BS \
      --do_eval \
      --prob_aggregate_strategy max
done

