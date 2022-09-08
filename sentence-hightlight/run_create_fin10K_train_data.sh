# # Lexicon-based Labeling
# python3 tools/construct_fin10k_train_synthetic.py \
#     -input data/fin10k/fin10k.train.type2.jsonl \
#     -output data/fin10k/fin10k.lexicon.synthetic.train.type2.jsonl \
#     -synthetic lexicon-based \
#     -n_hard 0 \
#     -random 1 \
#     -neg_sampling \
#     -lexicon_sent 'LM.master_dictionary.sentiment.dict' \
#     -lexicon_stop 'LM.master_dictionary.stopwords.dict' 
#
# python3 scripts/get_dataset_stats.py \
#   -data data/fin10k/fin10k.lexicon.synthetic.train.type2.jsonl

# Heuristic Labeling (neg-sampling)
python3 tools/construct_fin10k_train_synthetic.py \
    -input data/fin10k/fin10k.train.type2.jsonl \
    -output data/fin10k/fin10k.heuristic.synthetic.balance.train.type2.jsonl \
    -synthetic heuristic \
    -n_hard 0 \
    -random 1 \
    -neg_sampling 3

python3 tools/get_dataset_stats.py \
  -data data/fin10k/fin10k.heuristic.synthetic.balance.train.type2.jsonl

# Heurisirc Labeling + Soft-labeling 
# Only need inferenced once
# for MODEL in esnli-zs-highlighter;do
#     EVAL=data/fin10k/fin10k.heuristic.synthetic.balance.train.type2.jsonl
#     OUTPUT=${EVAL##*/}
#     python3 inference.py \
#       --model_name_or_path checkpoints/$MODEL/checkpoint-10000/ \
#       --config_name bert-base-uncased \
#       --eval_file $EVAL \
#       --output_file ${OUTPUT/train/soft.train} \
#       --remove_unused_columns false \
#       --max_seq_length 512 \
#       --per_device_eval_batch_size $BS \
#       --do_eval \
#       --prob_aggregate_strategy max
# done
#
