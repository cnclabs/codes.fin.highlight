if [ $1 == 'cross_seg' ]; then
    for result in fin10k/fin10k.eval.type2.segments.results-*;do
        python3 hl_pretty.py \
          -truth ../data/fin10k/fin10k.eval.type2.segments.jsonl \
          -pred $result \
          -topk 5 \
          -out fin10k/pretty_topk/topk
    done
fi

# if [ $1 == 'sentence' ]; then
#     for result in fin10k/fin10k.eval.type2.segments.results-*;do
#         python3 hl_pretty.py \
#           -truth ../data/fin10k/fin10k.eval.type2.segments.jsonl \
#           -pred $result \
#           -out fin10k/pretty_topk/topk-segment
#     done
# fi
