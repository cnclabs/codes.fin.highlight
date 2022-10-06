#############################################
# Automatic Evaluation 
# (1) Fin10k Type2 # 200
# (2) Fin10k Type1 # 100 (easy)
# (3) Fin10k Type1 # 100 (hard)
# (4) e-SNLI Contradiction # 3278
# (5) e-SNLI Contradiction # 3278
#############################################
LOG=results-good-read

# FIN10K
# fin10k type2 
for RESULT in results/fin10k.eval/type2/fin10k.eval.type2;do
    echo Loading prediction ${RESULT##*/} > ${LOG}/fin10k.eval/type2/${RESULT##*/}.log
    python3 tools/judge_highlights.py \
      -truth data/fin10k/fin10k.annotation.type2.jsonl \
      --aggregate 1 --aggregate 2 --aggregate 3 \
      -pred $RESULT \
      --verbose > ${LOG}/fin10k.eval/type2/${RESULT##*/}.log
done
# fin10k type1 easy
for result in results/fin10k.eval/type1.easy/fin10k.eval.type1.easy;do
    echo loading prediction ${result##*/} > ${LOG}/fin10k.eval/type1.easy/${result##*/}.log
    python3 tools/judge_highlights.py \
      -truth data/fin10k/fin10k.annotation.type1.easy.jsonl \
      --aggregate 1 --aggregate 2 --aggregate 3 \
      -pred $result \
      --verbose >> ${LOG}/fin10k.eval/type1.easy/${result##*/}.log
done
# fin10k type1 hard
for RESULT in results/fin10k.eval/type1.hard/fin10k.eval.type1.hard;do
    echo Loading prediction ${RESULT##*/} > ${LOG}/fin10k.eval/type1.hard/${RESULT##*/}.log
    python3 tools/judge_highlights.py \
      -truth data/fin10k/fin10k.annotation.type1.hard.jsonl \
      --aggregate 1 --aggregate 2 --aggregate 3 \
      -pred $RESULT \
      --verbose >> ${LOG}/fin10k.eval/type1.hard/${RESULT##*/}.log
done

# # e-SNLI
# # dev
# for RESULT in results/esnli.dev/*;do
#     echo Loading prediction ${RESULT##*/} > ${LOG}/esnli.dev/${RESULT##*/}.log
#     python3 tools/judge_highlights.py \
#       -truth data/esnli/esnli.dev.highlight.contradiction.jsonl \
#       -pred $RESULT \
#       --verbose >> ${LOG}/esnli.dev/${RESULT##*/}.log
# done
# # test
# for RESULT in results/esnli.test/*;do
#     echo Loading prediction ${RESULT##*/} > ${LOG}/esnli.test/${RESULT##*/}.log
#     python3 tools/judge_highlights.py \
#       -truth data/esnli/esnli.test.highlight.contradiction.jsonl \
#       -pred $RESULT \
#       --verbose >> ${LOG}/esnli.test/${RESULT##*/}.log
# done
