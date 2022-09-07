export CUDA_VISIBLE_DEVICES=0
BS=16

for MODEL in esnli-zs-highlighter;do
    EVAL=../data/fin10k/fin10k.heuristic.synthetic.balance.train.type2.jsonl
    OUTPUT=${EVAL##*/}
    python3 inference.py \
      --model_name_or_path ../checkpoints/$MODEL/checkpoint-10000/ \
      --config_name bert-base-uncased \
      --eval_file $EVAL \
      --output_file ${OUTPUT/train/soft.train} \
      --remove_unused_columns false \
      --max_seq_length 512 \
      --per_device_eval_batch_size $BS \
      --do_eval \
      --prob_aggregate_strategy max > log
done

