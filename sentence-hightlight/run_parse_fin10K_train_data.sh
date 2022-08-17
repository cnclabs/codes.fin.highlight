# Fin10k evauation data 
# ---------------------
# include: type2 
# procudre: convert text to jsonl 
# Synthetic dataset construction: (1) Heuristic labeling (2) Lexicon-based labling
# ---------------------
TRAIN_FILE=/tmp2/yshuang/fintext/new-data/type2.segments
TRAIN_FILE_TEMP=data/fin10k/type2.segments

cut -f1,2,3,4 $TRAIN_FILE > $TRAIN_FILE_TEMP

# Evaluation data from type2 
python3 tools/convert_text_to_jsonl.py \
    -input ${TRAIN_FILE_TEMP} \
    -output data/fin10k/fin10k.train.type2.jsonl \
    -type 2 \
    -spacy_sep

rm $TRAIN_FILE_TEMP
