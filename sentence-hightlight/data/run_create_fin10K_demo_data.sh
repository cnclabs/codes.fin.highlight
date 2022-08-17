# Type1 contain all one (highlight all)
# Type0 contain all zero (no highlight)
# Type2 contain all one (highlight all) --> (highlight partial)
echo "Createing data for bert...(demo)" > fin10k.demo.dat 

for FILE in fin10k-demo/data/*/*;do
    OUTFILE=${FILE/data\//}
    if [ -s $FILE ]
    then
        echo Convert $FILE into jsonl format
        python3 scripts/create_fin10k_data.py \
            -input $FILE \
            -output $OUTFILE \
            -type ${FILE##*type} \
            -nosep \
            -format jsonl \
            -model_type bert >> fin10k.demo.dat

        if [[ "$OUTFILE" = *type2* ]]; then
            echo Check overlength
            python3 scripts/filter_fin10k_overlength.py \
              -in $OUTFILE \
              -ol_out ${OUTFILE/type2/type1} \
              -tokenizer bert-base-uncased
        fi

    else
        echo File $FILE empty
        mkdir -p ${OUTFILE%/*}
        touch $OUTFILE
        echo Create empty file and skip processing.
    fi
done