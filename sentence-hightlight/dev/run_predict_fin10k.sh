export CUDA_VISIBLE_DEVICES=0
BS=16

for MODEL in esnli-zs-highlighter;do
    for EVAL in data/fin10k/fin10k.eval*;do
        OUTPUT=${EVAL##*/}
        python3 inference.py \
          --model_name_or_path checkpoints/$MODEL/checkpoint-10000/ \
          --config_name bert-base-uncased \
          --eval_file $EVAL \
          --output_file results/fin10k/${OUTPUT/jsonl/results}-$MODEL \
          --remove_unused_columns false \
          --blind_predict true \
          --max_seq_length 512 \
          --per_device_train_batch_size $BS \
          --do_eval \
          --prob_aggregate_strategy max
      done
done

