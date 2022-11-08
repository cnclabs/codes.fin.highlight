export CUDA_VISIBLE_DEVICES=0
TYPE=type2
# TYPE=type1.easy
# TYPE=type1.hard
BS=16

CKPT=18000
for MODEL in checkpoints/*;do
    for EVAL in data/fin10k/fin10k.eval.${TYPE}*;do
        OUTPUT=${EVAL##*/}
        python3 inference.py \
          --model_name_or_path $MODEL/checkpoint-$CKPT/ \
          --config_name bert-base-uncased \
          --eval_file $EVAL \
          --output_file results/fin10k.eval/${TYPE}/${OUTPUT/jsonl/results}-${MODEL##*/} \
          --remove_unused_columns false \
          --max_seq_length 512 \
          --per_device_eval_batch_size $BS \
          --do_eval \
          --prob_aggregate_strategy max
    done
done

