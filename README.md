# FinText
Phase          | Descrption
---            | --- 
Document segmetation | |
Segment alignment    | | 
Sentence highlighting     | | 

## Document Segmentation
TBD

## Segments Alignment
TBD 

## Sentence Highlighting
* Data preparation: Transform the sentence pair into BERT input
> [Note] default text format is a s follow: 
> docidA sentenceA |-| docidB setenceB |-| scoreX scoreY
```
cd data 
python3 scripts/create_fin10k_data.py \
    -input_path <path-to-aligned-file> \
    -output_path <path-to-output-json-file> \
    -model_type bert
cd ..
```

* Doanload PyTorch checkpoint (we've trained in advance, or you can train it your self, see below)
```
PATH: cfda2:/tmp2/jhju/codes.fin.contextualized/sentence-highlighting/checkpoints/*
```

* Predict token importance: Predict the probabilities of each financial sentence pair.
```
python3 inference.py \
    --model_name_or_path checkpoints/bert-base-uncased/checkpoint-10000 \
    --output_dir checkpoints/bert-base-uncased \
    --config_name bert-base-uncased \
    --eval_file <path-to-output-json-file> \
    --result_json results/fin10k/bert-seq-labeling.${filecode}.highlights.jsonl \
    --max_seq_length 128 \
    --do_eval
```
