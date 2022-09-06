# Fin10k annation data 
# ---------------------
# include (1) type2 
# procudre (1) convert text to jsonl (2) filter overlegnth example
# ---------------------
EVAL_FOLDER=/tmp2/yshuang/fintext/new-data/eval.result/

# Evaluation data from type2
python3 tools/convert_text_to_jsonl.py \
    -input ${EVAL_FOLDER}/eval.type2.segments.annotation.highlight.all \
    -output data/fin10k/fin10k.annotation.type2.jsonl.raw \
    -type 2 

python3 tools/filter_overlength_pair.py \
    -in data/fin10k/fin10k.annotation.type2.jsonl.raw \
    -out_ol data/fin10k/fin10k.annotation.type2.jsonl.overlength
rm data/fin10k/fin10k.annotation.type2.jsonl.raw.bak

python3 tools/construct_fin10k_eval_annotation.py \
    -input data/fin10k/fin10k.annotation.type2.jsonl.raw \
    -output data/fin10k/fin10k.annotation.type2.jsonl

python3 tools/get_dataset_stats.py \
    -data data/fin10k/fin10k.annotation.type2.jsonl
