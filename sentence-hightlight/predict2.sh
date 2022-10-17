export CUDA_VISIBLE_DEVICES=0
split=$1
BS=16

CKPT=18000
for MODEL in checkpoints/further-finetune-sl-soft;do
    for EVAL in data/esnli/esnli.${split}.highlight.contradiction.jsonl;do
        OUTPUT=${EVAL##*/}
        python3 inference.py \
          --model_name_or_path $MODEL/checkpoint-$CKPT/ \
          --config_name bert-base-uncased \
          --eval_file $EVAL \
          --output_file results/esnli.${split}/${OUTPUT/jsonl/results}-${MODEL##*/}-bp \
          --remove_unused_columns false \
          --max_seq_length 512 \
          --blind_predict \
          --per_device_eval_batch_size $BS \
          --do_eval \
          --prob_aggregate_strategy max
    done
done

CKPT=18000
for MODEL in checkpoints/further-finetune-sl-soft;do
    for EVAL in data/esnli/esnli.${split}.highlight.contradiction.jsonl;do
        OUTPUT=${EVAL##*/}
        python3 inference.py \
          --model_name_or_path $MODEL/checkpoint-$CKPT/ \
          --config_name bert-base-uncased \
          --eval_file $EVAL \
          --output_file results/esnli.${split}/${OUTPUT/jsonl/results}-${MODEL##*/}-sap \
          --remove_unused_columns false \
          --max_seq_length 512 \
          --same_predict \
          --per_device_eval_batch_size $BS \
          --do_eval \
          --prob_aggregate_strategy max
    done
done

CKPT=18000
for MODEL in checkpoints/further-finetune-sl-soft;do
    for EVAL in data/esnli/esnli.${split}.highlight.contradiction.jsonl;do
        OUTPUT=${EVAL##*/}
        python3 inference.py \
          --model_name_or_path $MODEL/checkpoint-$CKPT/ \
          --config_name bert-base-uncased \
          --eval_file $EVAL \
          --output_file results/esnli.${split}/${OUTPUT/jsonl/results}-${MODEL##*/}-shp \
          --remove_unused_columns false \
          --max_seq_length 512 \
          --shuffle_predict \
          --per_device_eval_batch_size $BS \
          --do_eval \
          --prob_aggregate_strategy max
    done
done
