# type2 in demo
echo start > log
for FILE in ../results/fin10k-demo/*/ITEM*.type2.jsonl;do
    OUTPUT_FILE=${FILE/\.jsonl/}
    if [ -s $FILE ]
    then
        # FILE=${FILE##*fin10k-demo/}
        echo $OUTPUT_FILE
        python3 hl_aggregate.py \
          -pred ${FILE} \
          -out ${OUTPUT_FILE} \
          -hl_on_a \
          -thres -1  >> log
    else
        echo File $FILE empty
        mkdir -p ${OUTPUT_FILE%/*}
        touch $OUTPUT_FILE
        echo Create empty file and skip processing.
    fi
done
