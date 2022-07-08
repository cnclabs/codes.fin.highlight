eval_file=$1
# for ckpt in esnli-highlight esnli-fin10k-0.125 esnli-fin10k-0.25 esnli-fin10k_seg-0.125;do
for ckpt in esnli-fin10k_seg-0.5;do
    output=${eval_file//data/results}
    python3 inference.py \
      --model_name_or_path checkpoints/${ckpt}/checkpoint-12500/ \
      --config_name bert-base-uncased \
      --eval_file $eval_file \
      --result_json ${output/jsonl/results}-${ckpt}\
      --remove_unused_columns false \
      --max_seq_length 256 \
      --do_eval
done
