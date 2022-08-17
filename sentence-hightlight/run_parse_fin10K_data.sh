EVAL_FOLDER=/tmp2/yshuang/fintext/new-data/eval.result/

# Evaluation data from type2
python3 tools/convert_text_to_jsonl.py \
    -input ${EVAL_FOLDER}/eval.type2.segments.all \
    -output data/fin10k/fin10k.eval.type2.jsonl \
    -type 2 \
    -nosep 

# Evaluation data from type1 (hard)
python3 tools/convert_text_to_jsonl.py \
    -input ${EVAL_FOLDER}/eval.type1.results.hard \
    -output data/fin10k/fin10k.eval.type1.hard.jsonl \
    -type 1 \
    -nosep 

# Evaluation data from type1 (easy)
python3 tools/convert_text_to_jsonl.py \
    -input ${EVAL_FOLDER}/eval.type1.results.easy \
    -output data/fin10k/fin10k.eval.type1.easy.jsonl \
    -type 1 \
    -nosep 

# Training data from type2 
# python3 tools/convert_text_to_jsonl.py \
#     -input $TRAIN_FILE \
#     -output data/fin10k/fin10k.train.type2.jsonl \
#     -type 2 \
#     -nosep 

# Training data from type2 (Lexicon labeling)
# python3 tools/convert_text_to_jsonl.py \
#     -input $TRAIN_FILE \
#     -output data/fin10k/fin10k.train.type2.lexicon.highlight.jsonl \
#     -type 2 \
#     -nosep 

