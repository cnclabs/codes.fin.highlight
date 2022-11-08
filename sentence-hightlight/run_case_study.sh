export CUDA_VISIBLE_DEVICES=2
BS=16
MODEL=further-finetune-sl-tunned

for CASE in case_study/*.jsonl;do
    CKPT=18000
    echo $MODEL
    python3 inference.py \
      --model_name_or_path checkpoints/$MODEL/checkpoint-$CKPT/ \
      --config_name bert-base-uncased \
      --eval_file $CASE \
      --output_file ${CASE/jsonl/highlighted.jsonl} \
      --remove_unused_columns false \
      --max_seq_length 512 \
      --per_device_eval_batch_size $BS \
      --do_eval \
      --prob_aggregate_strategy max
done

