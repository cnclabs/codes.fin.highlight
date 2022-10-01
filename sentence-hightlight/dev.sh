# type2
python3 dev.py \
    -truth data/fin10k/fin10k.annotation.type2.jsonl.1 \
    -pred1 results/fin10k.eval/type2/fin10k.eval.type2.results-esnli-zs-highlighter \
    -pred2 results/fin10k.eval/type2/fin10k.eval.type2.results-further-finetune-sl-smooth
 
python3 dev.py \
    -truth data/fin10k/fin10k.annotation.type2.jsonl.3 \
    -pred1 results/fin10k.eval/type2/fin10k.eval.type2.results-esnli-zs-highlighter \
    -pred2 results/fin10k.eval/type2/fin10k.eval.type2.results-further-finetune-sl-smooth

# type1
python3 dev.py \
    -truth data/fin10k/fin10k.annotation.type1.easy.jsonl.3 \
    -pred1 results/fin10k.eval/type1.easy/fin10k.eval.type1.easy.results-esnli-zs-highlighter \
    -pred2 results/fin10k.eval/type1.easy/fin10k.eval.type1.easy.results-further-finetune-sl-smooth

# type1
python3 dev.py \
    -truth data/fin10k/fin10k.annotation.type1.hard.jsonl.3 \
    -pred1 results/fin10k.eval/type1.hard/fin10k.eval.type1.hard.results-esnli-zs-highlighter \
    -pred2 results/fin10k.eval/type1.hard/fin10k.eval.type1.hard.results-further-finetune-sl-smooth
