# Setting sythentic datset parameters
# -------
# VERSION: v4 for cross-domain-transfer; v2 for from-scratch; v1 for further-finetune
# RAND: Randomness of sampling of other tokens
# N_HARD: No limitation on hard example (at least one positive) 
# -------

VERSION=$1 
RAND=0.2
N_HARD=0 
OUTPUT_FILE=fin10k.train.type2.segments.v$VERSION.$N_HARD.r$RAND.jsonl

echo "Createing data for bert...(train)" > fin10k.train.v$VERSION.dat
python3 scripts/create_fin10k_data.py \
    -input fin10k/train.type2.segments \
    -output fin10k/$OUTPUT_FILE \
    -model_type synthetic \
    -version $VERSION \
    -n_hard $N_HARD \
    -random $RAND \
    -lexicon_sent 'LM.master_dictionary.sentiment.dict' \
    -lexicon_stop 'LM.master_dictionary.stopwords.dict' >> fin10k.train.v$VERSION.dat

echo "Truncating overlenght sentence pair (>256)" >> fin10k.train.v$VERSION.dat
python3 scripts/filter_fin10k_overlength.py \
  -in fin10k/$OUTPUT_FILE \
  -tokenizer bert-base-uncased 

echo "Calculating statisitcs of dataset" >> fin10k.train.v$VERSION.dat
python3 scripts/get_dataset_stats.py \
  -data fin10k/$OUTPUT_FILE >> fin10k.train.v$VERSION.dat

