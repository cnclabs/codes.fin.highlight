# fin10k type2 super easy
for RESULT in results/fin10k/fin10k.eval.type2*;do
    echo Loading prediction ${RESULT##*/} > ${RESULT##*/}.log
    python3 tools/judge_highlights.py \
      -truth data/fin10k/fin10k.heuristic.synthetic.balance.eval.type2.jsonl \
      -pred $RESULT \
      --verbose >> ${RESULT##*/}.log
      # -topk 2
      # -thres 0 
done

# fin10k type2 easy
for RESULT in results/fin10k/fin10k.eval.type2*;do
    echo Loading prediction ${RESULT##*/} > logs/${RESULT##*/}.log
    python3 tools/judge_highlights.py \
      -truth data/fin10k/fin10k.annotation.type2.jsonl \
      -pred $RESULT \
      --verbose >> logs/${RESULT##*/}.log
      # -topk 2
      # -thres 0 
done

# # fin10k type1 easy
# for RESULT in results/fin10k/fin10k.eval.type1.easy*;do
#     echo Loading prediction ${RESULT##*/} > logs/${RESULT##*/}.log
#     python3 tools/judge_highlights.py \
#       -truth data/fin10k/fin10k.heuristic.synthetic.balance.eval.type1.easy.jsonl \
#       -pred $RESULT \
#       --verbose >> logs/${RESULT##*/}.log
#       # -thres 0 
# done
#
# # fin10k type1 hard
# for RESULT in results/fin10k/fin10k.eval.type1.hard*;do
#     echo Loading prediction ${RESULT##*/} > logs/${RESULT##*/}.log
#     python3 tools/judge_highlights.py \
#       -truth data/fin10k/fin10k.heuristic.synthetic.balance.eval.type1.hard.jsonl \
#       -pred $RESULT \
#       --verbose >> logs/${RESULT##*/}.log
#       # -thres 0 
# done

# for RESULT in results/esnli/esnli*results*;do
#     echo Loading prediction ${RESULT##*/} > logs/${RESULT##*/}.log
#     python3 tools/judge_highlights.py \
#       -truth data/esnli/esnli.dev.highlight.contradiction.jsonl \
#       -pred $RESULT \
#       --verbose >> logs/${RESULT##*/}.log
#       # -topk 2 \
#       # -thres 0.01 \
# done
#
