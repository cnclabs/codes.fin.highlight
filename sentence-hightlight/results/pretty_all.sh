if [ $1 == 'cross_seg' ]; then
    for result in fin10k/eval*segments*;do
        python3 hl_pretty.py \
          -truth ../data/fin10k/eval.type2.segments.jsonl \
          -pred $result \
          -out fin10k/pretty_topk/topk-segment
    done
fi

if [ $1 == 'sentence' ]; then
    for result in fin10k/eval*sentence*;do
        python3 hl_pretty.py \
          -truth ../data/fin10k/type2.sentence.eval.jsonl \
          -pred $result \
          -out fin10k/pretty_topk/topk-sentence
    done
fi
