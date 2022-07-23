export CUDA_VISIBLE_DEVICES=0
BS=16
STEPS=12500
TYPE=esnli-zs-highlighter

# for TYPE in cross-domain-transfer-0.125 esnli-zs-highlighter from-scratch further-finetune;do
for FILE in data/fin10k-demo/*/ITEM*.type2;do
    OUTPUT_FILE=${FILE/data/results}
    if [ -s $FILE ]
    then
        python3 inference.py \
          --model_name_or_path checkpoints/${TYPE}/checkpoint-${STEPS}/ \
          --config_name bert-base-uncased \
          --eval_file $FILE \
          --result_json $OUTPUT_FILE.jsonl \
          --remove_unused_columns false \
          --max_seq_length 512 \
          --per_device_train_batch_size $BS \
          --do_eval \
          --prob_aggregate_strategy 'max'
    else
        echo File $FILE empty
        mkdir -p ${OUTPUT_FILE%/*}
        touch $OUTPUT_FILE
        echo Create empty file and skip processing.
    fi
done
