# for RESULT in fin10k/fin10k.eval.type2*;do
#     echo Loading prediction ${RESULT##*/} > ${RESULT##*/}.log
#     python3 hl_eval.py \
#       -truth ../data/fin10k/fin10k.annotation.type2.jsonl \
#       -pred $RESULT --verbose \
#       -thres 0 >> ${RESULT##*/}.log
# done

# for RESULT in fin10k/fin10k.eval.type1.easy*;do
#     echo Loading prediction ${RESULT##*/} > ${RESULT##*/}.log
#     python3 hl_eval.py \
#       -truth ../data/fin10k/fin10k.eval.type1.easy.jsonl \
#       -pred $RESULT --verbose \
#       -thres 0 >> ${RESULT##*/}.log
# done

for RESULT in esnli/esnli*dev*;do
    echo Loading prediction ${RESULT##*/} > ${RESULT##*/}.log
    python3 hl_eval.py \
      -truth ../data/esnli/esnli.dev.highlight.contradiction.jsonl \
      -pred $RESULT \
      --verbose >> ${RESULT##*/}.log
      # -topk 2 \
      # -thres 0.01 \
done

