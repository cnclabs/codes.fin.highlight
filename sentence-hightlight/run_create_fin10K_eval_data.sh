# Heuristic Labeling (neg-sampling)
      # fin10k.eval.type1.easy.jsonl
      # fin10k.eval.type1.hard.jsonl
      # fin10k.eval.type2.jsonl
for EVAL_FILE in data/fin10k/fin10k.eval*;do
    OUTPUT_FILE=${EVAL_FILE/eval/heuristic.synthetic.balance.eval}
    python3 tools/construct_fin10k_train_synthetic.py \
        -input $EVAL_FILE \
        -output $OUTPUT_FILE \
        -synthetic heuristic \
        -n_hard 0 \
        -random 1 \
        -neg_sampling 3

    python3 tools/get_dataset_stats.py \
      -data $OUTPUT_FILE
done
