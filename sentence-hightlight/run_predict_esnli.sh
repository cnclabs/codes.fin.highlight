export CUDA_VISIBLE_DEVICES=0
BS=16

# CKPT=18000
# for MODEL in checkpoints/further*;do
#     for split in dev test;do
#         mkdir -p results/esnli.${split}/
#         EVAL=data/esnli/esnli.${split}.highlight.contradiction.jsonl
#         OUTPUT=${EVAL##*/}
#         python3 inference.py \
#           --model_name_or_path $MODEL/checkpoint-$CKPT/ \
#           --config_name bert-base-uncased \
#           --eval_file $EVAL \
#           --output_file results/esnli.${split}/${OUTPUT/jsonl/results}-${MODEL##*/} \
#           --remove_unused_columns false \
#           --max_seq_length 512 \
#           --per_device_eval_batch_size $BS \
#           --do_eval \
#           --prob_aggregate_strategy max
#     done
# done

CKPT=6000
for MODEL in checkpoints/from*ll;do
    for split in dev test;do
        mkdir -p results/esnli.${split}/
        EVAL=data/esnli/esnli.${split}.highlight.contradiction.jsonl
        OUTPUT=${EVAL##*/}
        python3 inference.py \
          --model_name_or_path $MODEL/checkpoint-$CKPT/ \
          --config_name bert-base-uncased \
          --eval_file $EVAL \
          --output_file results/esnli.${split}/${OUTPUT/jsonl/results}-${MODEL##*/} \
          --remove_unused_columns false \
          --max_seq_length 512 \
          --per_device_eval_batch_size $BS \
          --do_eval \
          --prob_aggregate_strategy max
    done
done

# CKPT=6000
# for MODEL in checkpoints/from-scratch*;do
#     for split in dev test;do
#         mkdir -p results/esnli.${split}/
#         EVAL=data/esnli/esnli.${split}.highlight.contradiction.jsonl
#         OUTPUT=${EVAL##*/}
#         python3 inference.py \
#           --model_name_or_path $MODEL/checkpoint-$CKPT/ \
#           --config_name bert-base-uncased \
#           --eval_file $EVAL \
#           --output_file results/esnli.${split}/${OUTPUT/jsonl/results}-${MODEL##*/} \
#           --remove_unused_columns false \
#           --max_seq_length 512 \
#           --per_device_eval_batch_size $BS \
#           --do_eval \
#           --prob_aggregate_strategy max
#     done
# done
