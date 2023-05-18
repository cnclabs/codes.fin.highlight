#############################################
# Automatic Evaluation 
# (1) Fin10k Type2 # 200
# (2) Fin10k Type1 # 100 (easy)
# (3) Fin10k Type1 # 100 (hard)
#############################################
LOG=results-good-read
select=$1 # specify the model setting to infernece the results

mkdir -p ${LOG}/fin10k.eval
mkdir -p ${LOG}/fin10k.eval/type2
mkdir -p ${LOG}/fin10k.eval/type1.easy
mkdir -p ${LOG}/fin10k.eval/type1.hard

# FIN10K
# fin10k type2 
for RESULT in results/fin10k.eval/type2/fin10k.eval.type2*${select}*;do
    echo Loading prediction ${RESULT##*/} > ${LOG}/fin10k.eval/type2/${RESULT##*/}.log
    python3 tools/judge_highlights.py \
      -truth data/fin10k/fin10k.annotation.type2.jsonl \
      --aggregate 1 --aggregate 2 --aggregate 3 \
      -pred $RESULT \
      --verbose >> ${LOG}/fin10k.eval/type2/${RESULT##*/}.log
done

# fin10k type1 easy
for RESULT in results/fin10k.eval/type1.easy/fin10k.eval.type1.easy*${select}*;do
    echo loading prediction ${RESULT##*/} > ${LOG}/fin10k.eval/type1.easy/${RESULT##*/}.log
    python3 tools/judge_highlights.py \
      -truth data/fin10k/fin10k.annotation.type1.easy.jsonl \
      --aggregate 1 --aggregate 2 --aggregate 3 \
      -pred $RESULT \
      --verbose >> ${LOG}/fin10k.eval/type1.easy/${RESULT##*/}.log
done

# fin10k type1 hard
for RESULT in results/fin10k.eval/type1.hard/fin10k.eval.type1.hard*${select}*;do
    echo Loading prediction ${RESULT##*/} > ${LOG}/fin10k.eval/type1.hard/${RESULT##*/}.log
    python3 tools/judge_highlights.py \
      -truth data/fin10k/fin10k.annotation.type1.hard.jsonl \
      --aggregate 1 --aggregate 2 --aggregate 3 \
      -pred $RESULT \
      --verbose >> ${LOG}/fin10k.eval/type1.hard/${RESULT##*/}.log
done
