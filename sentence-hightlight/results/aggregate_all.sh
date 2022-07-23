# type0
# type2 in demo
for FILE in ../results/fin10k-demo/*/ITEM*.type2.jsonl;do
    OUTPUT_FILE=/tmp2/fin10k/agg/$COM_FILE.agg
    if [ -s $FILE ]
    then
        echo Aggregating prediction ${results##*/}
        COM_FILE=${FILE##*fin10k-demo/}
        python3 hl_aggregate.py \
          -pred ${FILE} \
          -out ${OUTPUT_FILE} \
          -hl_on_a \
          -thres -1  > log
    else
        echo File $FILE empty
        mkdir -p ${OUTPUT_FILE%/*}
        touch $OUTPUT_FILE
        echo Create empty file and skip processing.
    fi
done
# python3 hl_aggregate.py \
#   -pred fin10k/fin10k.eval.type2.segments.results-esnli-zs-highlighter \
#   -out ./aggregate.jsonl \
#   -hl_on_a \
#   -thres -1
