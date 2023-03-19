# Other codes

This stage mainly includes the following process.
- FINAL data building


# Build FINAL data from scratch
You can run the following shell scripts.

```
# parsing
bash run_parse_fin10K_eval_data.sh
bash run_parse_fin10K_train_data.sh

# synthesize
bash run_create_fin10K_eval_data.sh
bash run_create_fin10K_train_data.sh
```

## Annoation reliability
Calculate the annoation reliabilty via Fleiss-kappa.
```
# revised 
python3 tools/calculate_rater_reliability.py \
    -truth data/fin10k/fin10k.annotation.type2.jsonl \
    --aggregate 1 --aggregate 2 --aggregate 3

# mismatched
python3 tools/calculate_rater_reliability.py \
    --files data/fin10k/fin10k.annotation.type1.easy.jsonl \
    --files data/fin10k/fin10k.annotation.type1.hard.jsonl \
    --aggregate 1 --aggregate 2 --aggregate 3

# overall
python3 tools/calculate_rater_reliability.py \
    --files data/fin10k/fin10k.annotation.type2.jsonl \
    --files data/fin10k/fin10k.annotation.type1.easy.jsonl \
    --files data/fin10k/fin10k.annotation.type1.hard.jsonl \
    --aggregate 1 --aggregate 2 --aggregate 3
```

