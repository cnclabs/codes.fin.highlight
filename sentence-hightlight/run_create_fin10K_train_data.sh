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

# Heuristic Labeling
# python3 tools/construct_fin10k_train_synthetic.py \
#     -input data/fin10k/fin10k.train.type2.jsonl \
#     -output data/fin10k/fin10k.heuristic.synthetic.train.type2.jsonl \
#     -synthetic heuristic \
#     -n_hard 0 \
#     -random 1 \
#     -lexicon_sent 'LM.master_dictionary.sentiment.dict' \
#     -lexicon_stop 'LM.master_dictionary.stopwords.dict' 
#
# python3 tools/get_dataset_stats.py \
#   -data data/fin10k/fin10k.heuristic.synthetic.train.type2.jsonl

# Heuristic Labeling (neg-sampling)
python3 tools/construct_fin10k_train_synthetic.py \
    -input data/fin10k/fin10k.train.type2.jsonl \
    -output data/fin10k/fin10k.heuristic.synthetic.balance.train.type2.jsonl \
    -synthetic heuristic \
    -n_hard 0 \
    -random 1 \
    -neg_sampling 3 \
    -lexicon_sent 'LM.master_dictionary.sentiment.dict' \
    -lexicon_stop 'LM.master_dictionary.stopwords.dict' 

python3 tools/get_dataset_stats.py \
  -data data/fin10k/fin10k.heuristic.synthetic.balance.train.type2.jsonl

