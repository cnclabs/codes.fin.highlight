# Fin10k evauation data 
# ---------------------
# include (1) type2 (2) type1 (easy) (3) type1 (hard)
# procudre (1) convert text to jsonl (2) filter overlegnth example
# ---------------------
EVAL_FOLDER=/tmp2/yshuang/fintext/new-data/eval.result/

# Evaluation data from type2
python3 tools/convert_text_to_jsonl.py \
    -input ${EVAL_FOLDER}/eval.type2.segments.all \
    -output data/fin10k/fin10k.eval.type2.jsonl \
    -type 2
python3 tools/filter_overlength_pair.py \
    -in data/fin10k/fin10k.eval.type2.jsonl \
    -out_ol data/fin10k/fin10k.eval.type2.overlength.jsonl &

# Evaluation data from type1 (hard)
python3 tools/convert_text_to_jsonl.py \
    -input ${EVAL_FOLDER}/eval.type1.results.hard \
    -output data/fin10k/fin10k.eval.type1.hard.jsonl \
    -type 1
python3 tools/filter_overlength_pair.py \
    -in data/fin10k/fin10k.eval.type1.hard.jsonl \
    -out_ol data/fin10k/fin10k.eval.type1.hard.overlength.jsonl &

# Evaluation data from type1 (easy)
python3 tools/convert_text_to_jsonl.py \
    -input ${EVAL_FOLDER}/eval.type1.results.easy \
    -output data/fin10k/fin10k.eval.type1.easy.jsonl \
    -type 1
python3 tools/filter_overlength_pair.py \
    -in data/fin10k/fin10k.eval.type1.easy.jsonl \
    -out_ol data/fin10k/fin10k.eval.type1.easy.overlength.jsonl &

