export CUDA_VISIBLE_DEVICES=0
BS=16
STEPS=12500

# for TYPE in cross-domain-transfer-0.125 esnli-zs-highlighter from-scratch further-finetune;do
for TYPE in cross-domain-transfer-0.125;do
    for EVAL in fin10k.eval.type2.segments.jsonl;do
        python3 inference.py \
          --model_name_or_path checkpoints/${TYPE}/checkpoint-${STEPS}/ \
          --config_name bert-base-uncased \
          --eval_file data/fin10k/$EVAL \
          --result_json results/fin10k/${EVAL/jsonl/results}-${TYPE} \
          --remove_unused_columns false \
          --max_seq_length 512 \
          --per_device_train_batch_size $BS \
          --do_eval \
          --prob_aggregate_strategy 'max'
    done
done

