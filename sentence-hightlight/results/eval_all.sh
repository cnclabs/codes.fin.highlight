# for results in esnli/esnli*dev*;do
#     echo Loading prediction ${results##*/} > ${results##*/}.log
#     python3 hl_eval.py \
#       -split dev \
#       -truth ../data/esnli/esnli.dev.sent_highlight.contradiction.jsonl \
#       -pred $results \
#       --verbose \
#       -thres 0.5 --topk 3 >> ${results##*/}.log
# done

for results in fin10k/fin10k.eval.type2.segments.results-*;do
    echo Loading prediction ${results##*/} > ${results##*/}.log
    python3 hl_eval.py \
      -truth ../data/fin10k/eval.type2.segments.annotation.highlight.jsonl \
      -pred $results \
      --verbose \
      -thres 0.5 --topk 5 >> ${results##*/}.log
done

