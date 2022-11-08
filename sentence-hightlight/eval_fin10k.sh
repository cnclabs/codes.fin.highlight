#############################################
# Automatic Evaluation 
# (1) Fin10k Type2 # 200
# (2) Fin10k Type1 # 100 (easy)
# (3) Fin10k Type1 # 100 (hard)
# (4) e-SNLI Contradiction # 3278
# (5) e-SNLI Contradiction # 3278
#############################################
LOG=results-good-read
select=$1

mkdir -p ${LOG}/fin10k.eval
mkdir -p ${LOG}/fin10k.eval/type2
mkdir -p ${LOG}/fin10k.eval/type1.easy
mkdir -p ${LOG}/fin10k.eval/type1.hard

# fin10k type2 
for RESULT in results/fin10k.eval/type2/fin10k.eval.type2*${select}*;do
    echo Loading prediction ${RESULT##*/} > ${LOG}/fin10k.eval/type2/${RESULT##*/}.log.sep
    for i in 1 2 3;do
        python3 tools/judge_highlights.py \
          -truth data/fin10k/fin10k.annotation.type2.jsonl.$i \
          -pred $RESULT >> ${LOG}/fin10k.eval/type2/${RESULT##*/}.log.sep
    done
done

# fin10k type1 easy
for RESULT in results/fin10k.eval/type1.easy/fin10k.eval.type1.eas*${select}*;do
    echo loading prediction ${RESULT##*/} > ${LOG}/fin10k.eval/type1.easy/${RESULT##*/}.log.sep
    for i in 1 2 3;do
        python3 tools/judge_highlights.py \
          -truth data/fin10k/fin10k.annotation.type1.easy.jsonl.$i \
          -pred $RESULT >> ${LOG}/fin10k.eval/type1.easy/${RESULT##*/}.log.sep
    done
done

# fin10k type1 hard
for RESULT in results/fin10k.eval/type1.hard/fin10k.eval.type1.hard*${select}*;do
    echo Loading prediction ${RESULT##*/} > ${LOG}/fin10k.eval/type1.hard/${RESULT##*/}.log.sep
    for i in 1 2 3;do
        python3 tools/judge_highlights.py \
          -truth data/fin10k/fin10k.annotation.type1.hard.jsonl.$i \
          -pred $RESULT >> ${LOG}/fin10k.eval/type1.hard/${RESULT##*/}.log.sep
    done
done
