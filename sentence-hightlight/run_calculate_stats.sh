
# revised
python3 tools/calculate_rater_reliability.py \
    -truth data/fin10k/fin10k.annotation.type2.jsonl \
    --aggregate 1 --aggregate 2 --aggregate 3

# mismatched
python3 tools/calculate_rater_reliability.py \
    --files data/fin10k/fin10k.annotation.type1.easy.jsonl \
    --files data/fin10k/fin10k.annotation.type1.hard.jsonl \
    --aggregate 1 --aggregate 2 --aggregate 3

# separated kappa of mismatched pairs
# python3 tools/calculate_rater_reliability.py \
#     -truth data/fin10k/fin10k.annotation.type1.easy.jsonl \
#     --aggregate 1 --aggregate 2 --aggregate 3
# python3 tools/calculate_rater_reliability.py \
#     -truth data/fin10k/fin10k.annotation.type1.hard.jsonl \
#     --aggregate 1 --aggregate 2 --aggregate 3

# overall
python3 tools/calculate_rater_reliability.py \
    --files data/fin10k/fin10k.annotation.type2.jsonl \
    --files data/fin10k/fin10k.annotation.type1.easy.jsonl \
    --files data/fin10k/fin10k.annotation.type1.hard.jsonl \
    --aggregate 1 --aggregate 2 --aggregate 3
