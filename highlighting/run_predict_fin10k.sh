export CUDA_VISIBLE_DEVICES=2
BS=16

# CKPT=18000
# for MODEL in checkpoints/further*;do
#     for TYPE in type1.easy type1.hard type2;do
#         mkdir -p results/fin10k.eval/${TYPE}
#         EVAL=data/fin10k/fin10k.eval.${TYPE}.jsonl
#         OUTPUT=${EVAL##*/}
#         echo $MODEL
#         python3 inference.py \
#           --model_name_or_path $MODEL/checkpoint-$CKPT/ \
#           --config_name bert-base-uncased \
#           --eval_file $EVAL \
#           --output_file results/fin10k.eval/${TYPE}/${OUTPUT/jsonl/results}-${MODEL##*/} \
#           --remove_unused_columns false \
#           --max_seq_length 512 \
#           --per_device_eval_batch_size $BS \
#           --do_eval \
#           --prob_aggregate_strategy max
#     done
# done

CKPT=6000
for MODEL in checkpoints/from-scratch;do
    for TYPE in type1.easy type1.hard type2;do
        mkdir -p results/fin10k.eval/${TYPE}
        EVAL=data/fin10k/fin10k.eval.${TYPE}.jsonl
        OUTPUT=${EVAL##*/}
        echo $MODEL
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
