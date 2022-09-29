## TYPE2
# first annotator
# python3 tools/extract_fin10k_eval_annotation.py \
#     -raw data/fin10k/fin10k.eval.type2.jsonl \
#     -input data/fin10k/our.fin10k.annotation.type2.csv \
#     -output data/fin10k/fin10k.annotation.type2.jsonl.1 \
#     -annotator 1
#
# # third annotator
# python3 tools/extract_fin10k_eval_annotation.py \
#     -raw data/fin10k/fin10k.eval.type2.jsonl \
#     -input data/fin10k/our.fin10k.annotation.type2.csv \
#     -output data/fin10k/fin10k.annotation.type2.jsonl.3 \
#     -annotator 3

## TYPE1 EASY
python3 tools/extract_fin10k_eval_annotation.py \
    -raw data/fin10k/fin10k.eval.type1.easy.jsonl \
    -input data/fin10k/our.fin10k.annotation.type1.easy.csv \
    -output data/fin10k/fin10k.annotation.type1.easy.jsonl.3 \
    -annotator 3

## TYPE1 HARD
python3 tools/extract_fin10k_eval_annotation.py \
    -raw data/fin10k/fin10k.eval.type1.hard.jsonl \
    -input data/fin10k/our.fin10k.annotation.type1.hard.csv \
    -output data/fin10k/fin10k.annotation.type1.hard.jsonl.3 \
    -annotator 3
