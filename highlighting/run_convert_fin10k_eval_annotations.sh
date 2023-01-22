## TYPE2
# first annotator
for i in 1 2 3;do
    echo Annotator $i type2
    echo -----------------
    python3 tools/extract_fin10k_eval_annotation.py \
        -raw data/fin10k/fin10k.eval.type2.jsonl \
        -input data/fin10k/our.fin10k.annotation.type2.csv \
        -output data/fin10k/fin10k.annotation.type2.jsonl.$i \
        -annotator $i
    python3 tools/get_dataset_stats.py \
        -data data/fin10k/fin10k.annotation.type2.jsonl.$i
    echo -----------------
done

## TYPE1 EASY
for i in 1 2 3;do
    echo Annotator $i type1 easy
    echo -----------------
    python3 tools/extract_fin10k_eval_annotation.py \
        -raw data/fin10k/fin10k.eval.type1.easy.jsonl \
        -input data/fin10k/our.fin10k.annotation.type1.easy.csv \
        -output data/fin10k/fin10k.annotation.type1.easy.jsonl.$i \
        -annotator $i
    python3 tools/get_dataset_stats.py \
        -data data/fin10k/fin10k.annotation.type1.easy.jsonl.$i
    echo -----------------
done

## TYPE1 HARD
for i in 1 2 3;do
    echo Annotator $i type1 hard
    echo -----------------
    python3 tools/extract_fin10k_eval_annotation.py \
        -raw data/fin10k/fin10k.eval.type1.hard.jsonl \
        -input data/fin10k/our.fin10k.annotation.type1.hard.csv \
        -output data/fin10k/fin10k.annotation.type1.hard.jsonl.$i \
        -annotator $i
    python3 tools/get_dataset_stats.py \
        -data data/fin10k/fin10k.annotation.type1.hard.jsonl.$i
    echo -----------------
done
