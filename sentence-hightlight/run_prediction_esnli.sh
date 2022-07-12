export HF_DATASETS_CACHE=/tmp2/jhju/cache/
export CUDA_VISIBLE_DEVICES=2

BS=16
STEPS=12500

for TYPE in esnli-zs-highlighter from-scratch-101;do
    for EVAL in esnli.dev.sent_highlight.contradiction.jsonl;do
        python3 inference.py \
          --model_name_or_path checkpoints/${TYPE}/checkpoint-${STEPS}/ \
          --config_name bert-base-uncased \
          --eval_file data/esnli/$EVAL \
          --result_json results/esnli/${EVAL/jsonl/results}-${TYPE} \
          --remove_unused_columns false \
          --max_seq_length 512 \
          --per_device_train_batch_size $BS \
          --do_eval \
          --prob_aggregate_strategy 'max'
    done
done

