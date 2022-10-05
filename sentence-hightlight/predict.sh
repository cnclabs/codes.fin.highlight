TYPE=$1

export CUDA_VISIBLE_DEVICES=1
BS=16

CKPT=18000
for MODEL in checkpoints/further-finetune-ll-smooth;do
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

CKPT=18000
for MODEL in checkpoints/further-finetune-sl-ko;do
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

CKPT=18000
for MODEL in checkpoints/further-finetune-sl-smooth;do
    for EVAL in data/fin10k/fin10k.eval.${TYPE}*;do
        OUTPUT=${EVAL##*/}
        python3 inference.py \
          --model_name_or_path $MODEL/checkpoint-$CKPT/ \
          --config_name bert-base-uncased \
          --eval_file $EVAL \
          --output_file results/fin10k.eval/${TYPE}/${OUTPUT/jsonl/results}-${MODEL##*/}-bp \
          --remove_unused_columns false \
          --max_seq_length 512 \
          --blind_predict \
          --per_device_eval_batch_size $BS \
          --do_eval \
          --prob_aggregate_strategy max
    done
done


