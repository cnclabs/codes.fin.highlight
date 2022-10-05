#############################################
# Automatic Evaluation 
# (1) Fin10k Type2 # 200
# (2) Fin10k Type1 # 100 (easy)
# (3) Fin10k Type1 # 100 (hard)
# (4) e-SNLI Contradiction # 3278
# (5) e-SNLI Contradiction # 3278
#############################################
LOG=results-good-read

# fin10k type2 
for RESULT in results/fin10k.eval/type2/fin10k.eval.type2*;do
    echo Loading prediction ${RESULT##*/} > ${LOG}/fin10k.eval/type2/${RESULT##*/}.log.1
    python3 tools/judge_highlights.py \
      -truth data/fin10k/fin10k.annotation.type2.jsonl.1 \
      -pred $RESULT \
      --verbose >> ${LOG}/fin10k.eval/type2/${RESULT##*/}.log.1

    echo Loading prediction ${RESULT##*/} > ${LOG}/fin10k.eval/type2/${RESULT##*/}.log.3
    python3 tools/judge_highlights.py \
      -truth data/fin10k/fin10k.annotation.type2.jsonl.3 \
      -pred $RESULT \
      --verbose >> ${LOG}/fin10k.eval/type2/${RESULT##*/}.log.3
done

# fin10k type1 easy
for RESULT in results/fin10k.eval/type1.easy/fin10k.eval.type1.easy*;do
    echo Loading prediction ${RESULT##*/} > ${LOG}/fin10k.eval/type1.easy/${RESULT##*/}.log.3
    python3 tools/judge_highlights.py \
      -truth data/fin10k/fin10k.annotation.type1.easy.jsonl.3 \
      -pred $RESULT \
      --verbose >> ${LOG}/fin10k.eval/type1.easy/${RESULT##*/}.log.3
done

# fin10k type1 hard
# for RESULT in results/fin10k.eval/type1.hard/fin10k.eval.type1.hard*;do
#     echo Loading prediction ${RESULT##*/} > ${LOG}/fin10k.eval/type1.hard/${RESULT##*/}.log.3
#     python3 tools/judge_highlights.py \
#       -truth data/fin10k/fin10k.annotation.type1.hard.jsonl.3 \
#       -pred $RESULT \
#       --verbose >> ${LOG}/fin10k.eval/type1.hard/${RESULT##*/}.log.3
# done
#
# for RESULT in results/esnli.dev/*;do
#     echo Loading prediction ${RESULT##*/} > ${LOG}/esnli.dev/${RESULT##*/}.log
#     python3 tools/judge_highlights.py \
#       -truth data/esnli/esnli.dev.highlight.contradiction.jsonl \
#       -pred $RESULT \
#       --verbose >> ${LOG}/esnli.dev/${RESULT##*/}.log
# done

for RESULT in results/esnli.test/*;do
    echo Loading prediction ${RESULT##*/} > ${LOG}/esnli.test/${RESULT##*/}.log
    python3 tools/judge_highlights.py \
      -truth data/esnli/esnli.test.highlight.contradiction.jsonl \
      -pred $RESULT \
      --verbose >> ${LOG}/esnli.test/${RESULT##*/}.log
done
