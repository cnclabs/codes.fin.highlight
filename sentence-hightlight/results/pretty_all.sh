for RESULT in fin10k/fin10k.eval.type2*;do
    python3 hl_pretty.py \
      -truth ../data/fin10k/fin10k.annotation.type2.jsonl \
      -pred $RESULT \
      -thres 0 \
      -topk 5 \
      -out fin10k/pretty_topk/top5
done

for RESULT in fin10k/fin10k.eval.type1.easy*;do
    python3 hl_pretty.py \
      -truth ../data/fin10k/fin10k.eval.type1.easy.jsonl \
      -pred $RESULT \
      -thres 0 \
      -topk 5 \
      -out fin10k/pretty_topk/top5
done

# for result in esnli/esnli.dev.sent_highlight.contradiction.results-*;do
#     python3 hl_pretty.py \
#       -truth ../data/esnli/esnli.dev.sent_highlight.contradiction.jsonl \
#       -pred $result \
#       -topk 3 \
#       -out esnli/pretty_topk/topk
# done

