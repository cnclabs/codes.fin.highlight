if [[ "$1" == *train* ]]; then
    RAND=1
    N_HARD=0
    VERSION=$2
    FILE=v${VERSION}.${N_HARD}.r${RAND}
    OUTPUT=${1/fin10k\//fin10k/fin10k.}.${FILE}

    echo "Createing data for bert...(train)" > fin10k.train.v$VERSION.dat
    if [[ "$2" == *101* ]]; then
        python3 scripts/create_fin10k_data_new.py \
            -input $1 \
            -output $OUTPUT.jsonl \
            -model_type synthetic \
            -version ${VERSION} \
            -n_hard ${N_HARD} \
            -random ${RAND} \
            -lexicon_sent 'LM.master_dictionary.sentiment.dict' \
            -lexicon_stop 'LM.master_dictionary.stopwords.dict' >> fin10k.train.v$VERSION.dat
    fi

    if [[ "$2" == *102* ]]; then
        python3 scripts/create_fin10k_data_new.py \
            -input $1 \
            -output $OUTPUT.jsonl \
            -model_type synthetic \
            -version ${VERSION} \
            -n_hard ${N_HARD} \
            -random ${RAND} \
            -neg_sampling \
            -lexicon_sent 'LM.master_dictionary.sentiment.dict' \
            -lexicon_stop 'LM.master_dictionary.stopwords.dict' >> fin10k.train.v$VERSION.dat
    fi

    echo "Truncating overlenght sentence pair (>256)" >> fin10k.train.v$VERSION.dat

    python3 scripts/filter_fin10k_overlength.py \
      -in $OUTPUT.jsonl \
      -tokenizer naive 
    python3 scripts/filter_fin10k_overlength.py \
      -in $OUTPUT.jsonl \
      -tokenizer bert-base-uncased

    echo "Calculating statisitcs of dataset" >> fin10k.train.v$VERSION.dat
    python3 scripts/get_dataset_stats.py \
      -data $OUTPUT.jsonl >> fin10k.train.v$VERSION.dat

fi
