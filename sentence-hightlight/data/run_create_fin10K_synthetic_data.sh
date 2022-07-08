if [[ "$1" == *train* ]]; then
    RAND=0.2
    N_HARD=0
    VERSION=$2
    FOLDER=fin10k_v${VERSION}.${N_HARD}_rand_${RAND}

    echo "Createing data for bert...(train)" > fin10k.train.v$VERSION.dat
    python3 scripts/create_fin10k_data.py \
        -input $1 \
        -output ${1/fin10k/fin10k\/$FOLDER}.jsonl \
        -model_type synthetic \
        -version ${VERSION} \
        -n_hard ${N_HARD} \
        -random ${RAND} \
        -lexicon_sent 'LM.master_dictionary.sentiment.dict' \
        -lexicon_stop 'LM.master_dictionary.stopwords.dict' >> fin10k.train.v$VERSION.dat

    echo "Truncating overlenght sentence pair (>256)" >> fin10k.train.v$VERSION.dat

    python3 scripts/filter_fin10k_overlength.py \
      -in ${1/fin10k/fin10k\/$FOLDER}.jsonl \
      -tokenizer naive 
    python3 scripts/filter_fin10k_overlength.py \
      -in ${1/fin10k/fin10k\/$FOLDER}.jsonl \
      -tokenizer bert-base-uncased

    echo "Calculating statisitcs of dataset" >> fin10k.train.v$VERSION.dat
    python3 scripts/get_dataset_stats.py \
      -data ${1/fin10k/fin10k\/$FOLDER}.jsonl >> fin10k.train.v$VERSION.dat

    rm ${1/fin10k/fin10k\/$FOLDER}.jsonl.naive_filtered

fi

if [[ "$1" == *eval* ]]; then
    echo "Createing data for bert...(eval)" > fin10k.eval-wo.dat 
    python3 scripts/create_fin10k_data.py \
        -input $1 \
        -output $1.jsonl \
        -model_type bert \
        -version 1 \
        -random 0 \
        -nosep \
        -lexicon_sent 'LM.master_dictionary.sentiment.dict' \
        -lexicon_stop 'LM.master_dictionary.stopwords.dict' >> fin10k.eval-wo.dat
fi

if [[ "$1" == *annotation* ]]; then
    echo "Createing data for bert...(annotation)" 
    python3 scripts/create_fin10k_data.py \
        -annotation \
        -input $1 \
        -output $1.jsonl \
        -model_type bert \
        -version 1 \
        -random 0 \
        -nosep \
        -lexicon_sent 'LM.master_dictionary.sentiment.dict' \
        -lexicon_stop 'LM.master_dictionary.stopwords.dict'
fi
