echo "Createing data for bert...(eval)" > fin10k.eval.dat 

# Standard version 
python3 scripts/create_fin10k_data.py \
    -input fin10k/eval.type2.segments.all \
    -output fin10k/fin10k.eval.type2.segments.jsonl \
    -model_type bert \
    -version 1 \
    -random 0 \
    -nosep \
    -lexicon_sent 'LM.master_dictionary.sentiment.dict' \
    -lexicon_stop 'LM.master_dictionary.stopwords.dict' >> fin10k.eval.dat

# Windowed version
python3 scripts/create_fin10k_data.py \
    -global \
    -input fin10k/eval.type2.segments.all.window \
    -output fin10k/fin10k.eval.type2.segments.all.window.jsonl \
    -model_type bert \
    -version 1 \
    -random 0 \
    -nosep \
    -lexicon_sent 'LM.master_dictionary.sentiment.dict' \
    -lexicon_stop 'LM.master_dictionary.stopwords.dict' >> fin10k.eval.dat


# # Annotation tsv
# python3 scripts/create_fin10k_data.py \
#     -annotation \
#     -format tsv \
#     -input fin10k/eval.type2.segments.annotation.highlight \
#     -output fin10k/eval.type2.segments.annotation.highlight.csv \
#     -model_type bert \
#     -version 1 \
#     -random 0 \
#     -nosep \
#     -lexicon_sent 'LM.master_dictionary.sentiment.dict' \
#     -lexicon_stop 'LM.master_dictionary.stopwords.dict' >> fin10k.eval.dat
#
# # Annotation jsonl
# python3 scripts/create_fin10k_data.py \
#     -annotation \
#     -format jsonl \
#     -input fin10k/eval.type2.segments.annotation.highlight \
#     -output fin10k/eval.type2.segments.annotation.highlight.jsonl \
#     -model_type bert \
#     -version 1 \
#     -random 0 \
#     -nosep \
#     -lexicon_sent 'LM.master_dictionary.sentiment.dict' \
#     -lexicon_stop 'LM.master_dictionary.stopwords.dict' >> fin10k.eval.dat
# #