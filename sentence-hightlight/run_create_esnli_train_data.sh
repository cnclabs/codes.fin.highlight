# e-SNLI annation data 
# ---------------------
# include (1) Contradiction
# procudre (1) convert text to jsonl (2) filter overlegnth example
# dataset (1) train (2) dev (3) test
# ---------------------
SPLIT=train
# python3 tools/convert_text_to_jsonl.py \
#     -input data/esnli_${SPLIT}.csv \
#     -output data/esnli/esnli.${SPLIT}.parsed.contradiction.jsonl \
#     -type 2 \
#     -spacy_sep
python3 tools/construct_esnli_data.py \
    -input data/esnli/esnli.${SPLIT}.parsed.contradiction.jsonl \
    -output data/esnli/esnli.${SPLIT}.highlight.contradiction.jsonl &

# for SPLIT in dev test;do
#     python3 tools/convert_text_to_jsonl.py \
#         -input data/esnli_${SPLIT}.csv \
#         -output data/esnli/esnli.${SPLIT}.parsed.contradiction.jsonl \
#         -type 2 \
#         -spacy_sep
#
#     python3 tools/construct_esnli_data.py \
#         -input data/esnli/esnli.${SPLIT}.parsed.contradiction.jsonl \
#         -output data/esnli/esnli.${SPLIT}.highlight.contradiction.jsonl &
# done
