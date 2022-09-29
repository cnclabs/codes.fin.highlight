#############################################
# Automatic Evaluation 
# (1) Fin10k Type2 # 200
# (2) Fin10k Type1 # 100 (easy)
# (3) Fin10k Type1 # 100 (hard)
# (4) e-SNLI Contradiction # 3278
# (5) e-SNLI Contradiction # 3278
#############################################

# fin10k type2 
# for RESULT in results/fin10k/type2/fin10k.eval.type2*;do
#     echo Loading prediction ${RESULT##*/} > logs/fin10k.eval/${RESULT##*/}.log.1
#     python3 tools/judge_highlights.py \
#       -truth data/fin10k/fin10k.annotation.type2.jsonl.1 \
#       -pred $RESULT \
#       --verbose >> logs/fin10k.eval/${RESULT##*/}.log.1
#       # -topk 2
#       # -thres 0 
# done
# for RESULT in results/fin10k/type2/fin10k.eval.type2*;do
#     echo Loading prediction ${RESULT##*/} > logs/fin10k.eval/${RESULT##*/}.log.3
#     python3 tools/judge_highlights.py \
#       -truth data/fin10k/fin10k.annotation.type2.jsonl.3 \
#       -pred $RESULT \
#       --verbose >> logs/fin10k.eval/${RESULT##*/}.log.3
# done
#
# fin10k type1 easy
for RESULT in results/fin10k/type1/fin10k.eval.type1.easy*;do
    echo Loading prediction ${RESULT##*/} > logs/fin10k.eval/${RESULT##*/}.log.3
    python3 tools/judge_highlights.py \
      -truth data/fin10k/fin10k.annotation.type1.easy.jsonl.3 \
      -pred $RESULT \
      --verbose >> logs/fin10k.eval/${RESULT##*/}.log.3
done

# fin10k type1 hard
for RESULT in results/fin10k/type1/fin10k.eval.type1.hard*;do
    echo Loading prediction ${RESULT##*/} > logs/fin10k.eval/${RESULT##*/}.log.3
    python3 tools/judge_highlights.py \
      -truth data/fin10k/fin10k.annotation.type1.hard.jsonl.3 \
      -pred $RESULT \
      --verbose >> logs/fin10k.eval/${RESULT##*/}.log.3
done

# for RESULT in results/esnli/esnli*dev*results*;do
#     echo Loading prediction ${RESULT##*/} > logs/esnli.dev/${RESULT##*/}.log
#     python3 tools/judge_highlights.py \
#       -truth data/esnli/esnli.dev.highlight.contradiction.jsonl \
#       -pred $RESULT \
#       --verbose >> logs/esnli.dev/${RESULT##*/}.log
# done
#
# for RESULT in results/esnli/esnli*test*results*;do
#     echo Loading prediction ${RESULT##*/} > logs/esnli.test/${RESULT##*/}.log
#     python3 tools/judge_highlights.py \
#       -truth data/esnli/esnli.test.highlight.contradiction.jsonl \
#       -pred $RESULT \
#       --verbose >> logs/esnli.test/${RESULT##*/}.log
# done
