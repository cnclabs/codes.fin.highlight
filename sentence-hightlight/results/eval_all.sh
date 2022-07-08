# echo "****************\nsetence highlight\n****************" > esnli.dev.log
# for results in esnli/esnli*dev*;do
#     echo ${results##*/} >> esnli.dev.log
#     python3 hl_eval.py -split dev -pred $results >> esnli.dev.log
# done
#
# echo "****************\nsetence highlight\n****************" > esnli.test.log
# for results in esnli/esnli*test*;do
#     echo ${results##*/} >> esnli.test.log
#     python3 hl_eval.py -split test -pred $results >> esnli.test.log
# done

echo --------fin10k-------- > fin10k.type2.cross_seg.log
for results in fin10k/eval.type2.segments.results-*;do
    echo Loading prediction ${results##*/} >> fin10k.type2.cross_seg.log
    python3 hl_eval.py \
      -truth ../data/fin10k/annotation/eval.type2.segments.annotation.jsonl \
      -pred $results \
      --verbose \
      -thres 0 --topk 20 >> fin10k.type2.cross_seg.log
done

# echo "****************\nsetence highlight\n****************" > fin10k.type2.sentence.log
# for results in fin10k/eval.type2.sentences.results-*;do;
#     echo ${results##*/} >> fin10k.type2.sentence.log
#     python3 hl_eval.py \
#       -truth ../data/fin10k/type2.sentence.eval.jsonl \
#       -pred $results >> fin10k.type2.sentence.log
# done
