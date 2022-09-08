# export CUDA_VISIBLE_DEVICES=2
# BS=16
#
# for MODEL in cross-domain-transfer-0.125  esnli-zs-highlighter from-scratch further-finetune;do
#     for EVAL in data/fin10k/fin10k.eval*;do
#         OUTPUT=${EVAL##*/}
#         python3 inference.py \
#           --model_name_or_path checkpoints/$MODEL/checkpoint-12500/ \
#           --config_name bert-base-uncased \
#           --eval_file $EVAL \
#           --output_file results/fin10k/${OUTPUT/jsonl/results}-$MODEL \
#           --remove_unused_columns false \
#           --max_seq_length 512 \
#           --per_device_train_batch_size $BS \
#           --do_eval \
#           --prob_aggregate_strategy max
#       done
# done
#
export CUDA_VISIBLE_DEVICES=0
BS=16

# for MODEL in cross-domain-transfer-0.125  esnli-zs-highlighter from-scratch further-finetune;do
# for MODEL in further-finetune-soft from-scratch-soft;do
for MODEL in esnli-zs-highlighter-base;do
    for EVAL in data/fin10k/fin10k.eval*;do
        OUTPUT=${EVAL##*/}
        python3 inference.py \
          --model_name_or_path checkpoints/$MODEL/checkpoint-12500/ \
          --config_name bert-base-uncased \
          --eval_file $EVAL \
          --output_file results/fin10k/${OUTPUT/jsonl/results}-$MODEL \
          --remove_unused_columns false \
          --max_seq_length 512 \
          --per_device_train_batch_size $BS \
          --do_eval \
          --prob_aggregate_strategy max
      done
done

