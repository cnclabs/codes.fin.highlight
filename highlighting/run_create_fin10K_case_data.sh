# Fin10k case studies
# ---------------------
# include (1) CIK 1001250 (2) 97745
# ---------------------
for CIK in 1001250;do

    CASE=/tmp2/yshuang/fintext/new-data/case_study/${CIK}

    # Revised relation
    cut -f2,3,4,5 $CASE/revised.relations > removeme
    # python3 tools/convert_text_to_jsonl.py \
    #     -input removeme \
    #     -output case_study/$CIK-revised.jsonl \
    #     -type 2 #revised
    # python3 tools/filter_overlength_pair.py \
    #     -in case_study/$CIK-revised.jsonl \
    #     -out_ol case_study/$CIK-revised.overlength.jsonl
    # rm case_study/$CIK-revised.jsonl.bak

    # Mismatched relation
    cat $CASE/uncorrelated.relations > removeme
    python3 tools/convert_text_to_jsonl.py \
        -input removeme \
        -output case_study/$CIK-mismatched.jsonl \
        -type 1 # mismatched
    python3 tools/filter_overlength_pair.py \
        -in case_study/$CIK-mismatched.jsonl \
        -out_ol case_study/$CIK-mismatched.overlength.jsonl
    rm case_study/$CIK-mismatched.jsonl.bak

done

