#############################################
# Automatic Evaluation 
# (4) e-SNLI Contradiction # 3278
# (5) e-SNLI Contradiction # 3278
#############################################
LOG=results-good-read
select=$1

mkdir -p ${LOG}/esnli.dev
mkdir -p ${LOG}/esnli.test

# e-SNLI
# dev
for RESULT in results/esnli.dev/*${select}*;do
    echo Loading prediction ${RESULT##*/} > ${LOG}/esnli.dev/${RESULT##*/}.log
    python3 tools/judge_highlights.py \
      -truth data/esnli/esnli.dev.highlight.contradiction.jsonl \
      -pred $RESULT >> ${LOG}/esnli.dev/${RESULT##*/}.log
done
test
for RESULT in results/esnli.test/*${select}*;do
    echo Loading prediction ${RESULT##*/} > ${LOG}/esnli.test/${RESULT##*/}.log
    python3 tools/judge_highlights.py \
      -truth data/esnli/esnli.test.highlight.contradiction.jsonl \
      -pred $RESULT >> ${LOG}/esnli.test/${RESULT##*/}.log
done
