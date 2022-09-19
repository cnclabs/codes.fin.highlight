# first annotator
python3 tools/extract_fin10k_eval_annotation.py \
    -raw data/fin10k/fin10k.annotation.type2.jsonl \
    -input data/fin10k/our.fin10k.annotation.type2.csv \
    -output data/fin10k/fin10k.annotation.type2.jsonl.1 \
    -annotator 1

# third annotator
python3 tools/extract_fin10k_eval_annotation.py \
    -raw data/fin10k/fin10k.annotation.type2.jsonl \
    -input data/fin10k/our.fin10k.annotation.type2.csv \
    -output data/fin10k/fin10k.annotation.type2.jsonl.3 \
    -annotator 3
