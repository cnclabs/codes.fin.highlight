# Signal highlighting stage 

This stage mainly includes the following process.
- Out-domain fine-tuning 
- Pseudo-labeling process
- In-domain fine-tuning

Also, we provide other codes of preprocessing and evaluaion codes in [OTHERS.md](OTHERS.md)

---

## Data preparation (e-SNLI)

- Download 
The required dataset is [e-SNLI](https://github.com/OanaMariaCamburu/e-SNLI/tree/master/dataset/). 
You should first donwload and save the data in [data/](data/).
> Note there are 2 esnli trainining files; we merge the two into one standalone file.

- Preprocessing
Parse the e-SNLI dataset into the following formats.
```
$ bash run_create_esnli_data.sh
```

## S2: Out-domain fine-tuning

### Baseline model 1: Zero-shot
The `esnli-zs-highlighter` is the baseline highlighting models.
Fine-tuned on the compiled e-SNLI dataset which labels are `contradiction`.

The `esnli-zs-highligher` model checkpoints can be found in [huggingface](#)(TBA).

```
TRAIN_ESNLI=data/esnli/esnli.train.highlight.contradiction.jsonl
EVAL_ESNLI=data/esnli/esnli.eval.highlight.contradiction.jsonl

TYPE=esnli-zs-highlighter
BS=24
python3 train.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_ESNLI \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 15000 \
  --save_steps 5000 \
  --eval_steps 5000 \
  --max_seq_length 258 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --do_train \
  --do_eval
```

## S2+: In-domain fine-tuning

### Pseduo-labeling (hard-label)
We first use the heuristic hard-label process the generate a training dataset (FINAL-train).

### Pseduo-labeling (soft-label)
Once obtaining the `esnli-zh-highlighter`, we can use it to generate the pseduo-labeled dataset.

### Baseline model 2: Pseudo few-shot
The `pseudo-zs-highlighter` is the baseline highlighting models.
Fine-tuned on the compiled e-SNLI dataset which labels are `contradiction`.

```
TRAIN_FIN10K=data/fin10k/fin10k.heuristic.synthetic.balance.train.type2.jsonl
EVAL_ESNLI=data/esnli/esnli.eval.highlight.contradiction.jsonl

TYPE=pseudo-few-shot
BS=24
python3 train.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 10000 \
  --save_steps 1500 \
  --eval_steps 1500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling false  \
  --tau 1 \
  --gamma 1 \
  --do_train \
  --do_eval
```


export CUDA_VISIBLE_DEVICES=2
BS=16
MODEL=further-finetune-sl-tunned

for CASE in case_study/*.jsonl;do
    CKPT=18000
    echo $MODEL
    python3 inference.py \
      --model_name_or_path checkpoints/$MODEL/checkpoint-$CKPT/ \
      --config_name bert-base-uncased \
      --eval_file $CASE \
      --output_file ${CASE/jsonl/highlighted.jsonl} \
      --remove_unused_columns false \
      --max_seq_length 512 \
      --per_device_eval_batch_size $BS \
      --do_eval \
      --prob_aggregate_strategy max
done

## TYPE2
# first annotator
for i in 1 2 3;do
    echo Annotator $i type2
    echo -----------------
    python3 tools/extract_fin10k_eval_annotation.py \
        -raw data/fin10k/fin10k.eval.type2.jsonl \
        -input data/fin10k/our.fin10k.annotation.type2.csv \
        -output data/fin10k/fin10k.annotation.type2.jsonl.$i \
        -annotator $i
    python3 tools/get_dataset_stats.py \
        -data data/fin10k/fin10k.annotation.type2.jsonl.$i
    echo -----------------
done

## TYPE1 EASY
for i in 1 2 3;do
    echo Annotator $i type1 easy
    echo -----------------
    python3 tools/extract_fin10k_eval_annotation.py \
        -raw data/fin10k/fin10k.eval.type1.easy.jsonl \
        -input data/fin10k/our.fin10k.annotation.type1.easy.csv \
        -output data/fin10k/fin10k.annotation.type1.easy.jsonl.$i \
        -annotator $i
    python3 tools/get_dataset_stats.py \
        -data data/fin10k/fin10k.annotation.type1.easy.jsonl.$i
    echo -----------------
done

## TYPE1 HARD
for i in 1 2 3;do
    echo Annotator $i type1 hard
    echo -----------------
    python3 tools/extract_fin10k_eval_annotation.py \
        -raw data/fin10k/fin10k.eval.type1.hard.jsonl \
        -input data/fin10k/our.fin10k.annotation.type1.hard.csv \
        -output data/fin10k/fin10k.annotation.type1.hard.jsonl.$i \
        -annotator $i
    python3 tools/get_dataset_stats.py \
        -data data/fin10k/fin10k.annotation.type1.hard.jsonl.$i
    echo -----------------
done
# e-SNLI annation data 
# ---------------------
# include (1) Contradiction
# procudre (1) convert text to jsonl (2) filter overlegnth example
# dataset (1) train (2) dev (3) test
# ---------------------
SPLIT=train

# for SPLIT in train dev test;do
for SPLIT in train;do
    # python3 tools/convert_text_to_jsonl.py \
    #     -input data/esnli_${SPLIT}.csv \
    #     -output data/esnli/esnli.${SPLIT}.parsed.contradiction.jsonl \
    #     -type 2 \
    #     -spacy_sep 

    python3 tools/construct_esnli_data.py \
        -input data/esnli/esnli.${SPLIT}.parsed.contradiction.jsonl \
        -output data/esnli/esnli.${SPLIT}.highlight.contradiction.jsonl &
done
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

# Fin10k annation data 
# ---------------------
# include (1) type2 
# procudre (1) convert text to jsonl (2) filter overlegnth example
# ---------------------
EVAL_FOLDER=/tmp2/yshuang/fintext/new-data/eval.result/

# Evaluation data from type2
python3 tools/convert_text_to_jsonl.py \
    -input /tmp2/yshuang/fintext/new-data/eval.result/eval.type2.segments.annotation.highlight.final \
    -output data/fin10k/fin10k.annotation.type2.jsonl \
    -type 2 
python3 tools/filter_overlength_pair.py \
    -in data/fin10k/fin10k.annotation.type2.jsonl \ # the original input will be replaced
    -out_ol data/fin10k/fin10k.annotation.type2.overlength.jsonl # the overlength sentence pairs
rm data/fin10k/fin10k.annotation.type2.jsonl.bak # bak is the original input data

python3 tools/construct_fin10k_eval_annotation.py \
    -input data/fin10k/fin10k.annotation.type2.jsonl \
    -output data/fin10k/fin10k.annotation.type2.jsonl \
    --output_csv 
python3 tools/get_dataset_stats.py \
    -data data/fin10k/fin10k.annotation.type2.jsonl
# Heuristic Labeling (neg-sampling)
# python3 tools/construct_fin10k_train_synthetic.py \
#     -input data/fin10k/fin10k.train.type2.jsonl \
#     -output data/fin10k/fin10k.heuristic.synthetic.balance.train.type2.jsonl \
#     -synthetic heuristic \
#     -n_hard 0 \
#     -random 1 \
#     -neg_sampling 3
#
# python3 tools/get_dataset_stats.py \
#   -data data/fin10k/fin10k.heuristic.synthetic.balance.train.type2.jsonl

# Heurisirc Labeling + Soft-labeling (Only need inferenced once)
# export CUDA_VISIBLE_DEVICES=0
# BS=16
# for MODEL in esnli-zs-highlighter;do
#     EVAL=data/fin10k/fin10k.heuristic.synthetic.balance.train.type2.jsonl
#     python3 inference.py \
#       --model_name_or_path checkpoints/$MODEL/checkpoint-10000/ \
#       --config_name bert-base-uncased \
#       --eval_file $EVAL \
#       --output_file ${EVAL/train/soft.train} \
#       --remove_unused_columns false \
#       --max_seq_length 512 \
#       --per_device_eval_batch_size $BS \
#       --do_eval \
#       --prob_aggregate_strategy max
# done

# Lexicon-based Labeling
# python3 tools/construct_fin10k_train_synthetic.py \
#     -input data/fin10k/fin10k.heuristic.synthetic.balance.train.type2.jsonl \
#     -output data/fin10k/fin10k.lexicon.synthetic.balance.train.type2.jsonl \
#     -synthetic lexicon-based \
#     -n_hard 0 \
#     -random 1 \
#     -neg_sampling 3 \
#     -lexicon_sent tools/lexicons/LM.master_dictionary.sentiment.dict \
#     -lexicon_stop tools/lexicons/LM.master_dictionary.stopwords.dict

# python3 tools/get_dataset_stats.py \
#   -data data/fin10k/fin10k.lexicon.synthetic.balance.train.type2.jsonl
#############################################
# Automatic Evaluation 
# (1) e-SNLI Dev Contradiction # 3278
# (2) e-SNLI Test Contradiction # 3278
#############################################
LOG=results-good-read
select=$1

mkdir -p ${LOG}/esnli.dev
mkdir -p ${LOG}/esnli.test

# e-SNLI
# dev
# for RESULT in results/esnli.dev/*${select}*;do
#     echo Loading prediction ${RESULT##*/} > ${LOG}/esnli.dev/${RESULT##*/}.log
#     python3 tools/judge_highlights.py \
#       -truth data/esnli/esnli.dev.highlight.contradiction.jsonl \
#       -pred $RESULT  >> ${LOG}/esnli.dev/${RESULT##*/}.log
# done
# test
for RESULT in results/esnli.test/*${select}*;do
    echo Loading prediction ${RESULT##*/} > ${LOG}/esnli.test/${RESULT##*/}.log
    python3 tools/judge_highlights.py \
      -truth data/esnli/esnli.test.highlight.contradiction.jsonl \
      -pred $RESULT >> ${LOG}/esnli.test/${RESULT##*/}.log
done
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

# Fin10k evauation data 
# ---------------------
# include (1) type2 (2) type1 (easy) (3) type1 (hard)
# procudre (1) convert text to jsonl (2) filter overlegnth example
# ---------------------
EVAL_FOLDER=/tmp2/yshuang/fintext/new-data/eval.result/

# Evaluation data from type2 (use `run_create_fin10K_annotation_data.sh` instead)
# python3 tools/convert_text_to_jsonl.py \
#     -input ${EVAL_FOLDER}/eval.type2.segments.all \
#     -output data/fin10k/fin10k.eval.type2.jsonl \
#     -type 2
# python3 tools/filter_overlength_pair.py \
#     -in data/fin10k/fin10k.eval.type2.jsonl \
#     -out_ol data/fin10k/fin10k.eval.type2.overlength.jsonl &

# Evaluation data from type1 (hard)
python3 tools/convert_text_to_jsonl.py \
    -input ${EVAL_FOLDER}/eval.type1.results.hard \
    -output data/fin10k/fin10k.eval.type1.hard.jsonl \
    -type 1
python3 tools/filter_overlength_pair.py \
    -in data/fin10k/fin10k.eval.type1.hard.jsonl \
    -out_ol data/fin10k/fin10k.eval.type1.hard.overlength.jsonl
python3 tools/create_fin10k_annotation_sheet.py \
    -input data/fin10k/fin10k.eval.type1.hard.jsonl \
    -output data/fin10k/fin10k.eval.type1.hard.tsv

# Evaluation data from type1 (easy)
python3 tools/convert_text_to_jsonl.py \
    -input ${EVAL_FOLDER}/eval.type1.results.easy \
    -output data/fin10k/fin10k.eval.type1.easy.jsonl \
    -type 1
python3 tools/filter_overlength_pair.py \
    -in data/fin10k/fin10k.eval.type1.easy.jsonl \
    -out_ol data/fin10k/fin10k.eval.type1.easy.overlength.jsonl
python3 tools/create_fin10k_annotation_sheet.py \
    -input data/fin10k/fin10k.eval.type1.easy.jsonl \
    -output data/fin10k/fin10k.eval.type1.easy.tsv

rm data/fin10k/*eval*bak
# Fin10k train data 
# ---------------------
# include: type2 
# procudre: convert text to jsonl 
# Synthetic dataset construction: (1) Heuristic labeling (2) Lexicon-based labling
# ---------------------
# TRAIN_FILE=/tmp2/yshuang/fintext/new-data/type2.segments
TRAIN_FILE_TEMP=data/fin10k/type2.segments

cut -f1,2,3,4 $TRAIN_FILE > $TRAIN_FILE_TEMP

# Evaluation data from type2 
python3 tools/convert_text_to_jsonl.py \
    -input ${TRAIN_FILE_TEMP} \
    -output data/fin10k/fin10k.train.type2.jsonl \
    -type 2 \
    -spacy_sep
#
# rm $TRAIN_FILE_TEMP
python3 tools/filter_overlength_pair.py \
    -in data/fin10k/fin10k.train.type2.jsonl \
    -out_ol data/fin10k/fin10k.train.overlength.type2.jsonl \
    -is_train &

export CUDA_VISIBLE_DEVICES=0
BS=16

# CKPT=18000
# for MODEL in checkpoints/further*;do
#     for split in dev test;do
#         mkdir -p results/esnli.${split}/
#         EVAL=data/esnli/esnli.${split}.highlight.contradiction.jsonl
#         OUTPUT=${EVAL##*/}
#         python3 inference.py \
#           --model_name_or_path $MODEL/checkpoint-$CKPT/ \
#           --config_name bert-base-uncased \
#           --eval_file $EVAL \
#           --output_file results/esnli.${split}/${OUTPUT/jsonl/results}-${MODEL##*/} \
#           --remove_unused_columns false \
#           --max_seq_length 512 \
#           --per_device_eval_batch_size $BS \
#           --do_eval \
#           --prob_aggregate_strategy max
#     done
# done

CKPT=6000
for MODEL in checkpoints/from*ll;do
    for split in dev test;do
        mkdir -p results/esnli.${split}/
        EVAL=data/esnli/esnli.${split}.highlight.contradiction.jsonl
        OUTPUT=${EVAL##*/}
        python3 inference.py \
          --model_name_or_path $MODEL/checkpoint-$CKPT/ \
          --config_name bert-base-uncased \
          --eval_file $EVAL \
          --output_file results/esnli.${split}/${OUTPUT/jsonl/results}-${MODEL##*/} \
          --remove_unused_columns false \
          --max_seq_length 512 \
          --per_device_eval_batch_size $BS \
          --do_eval \
          --prob_aggregate_strategy max
    done
done

# CKPT=6000
# for MODEL in checkpoints/from-scratch*;do
#     for split in dev test;do
#         mkdir -p results/esnli.${split}/
#         EVAL=data/esnli/esnli.${split}.highlight.contradiction.jsonl
#         OUTPUT=${EVAL##*/}
#         python3 inference.py \
#           --model_name_or_path $MODEL/checkpoint-$CKPT/ \
#           --config_name bert-base-uncased \
#           --eval_file $EVAL \
#           --output_file results/esnli.${split}/${OUTPUT/jsonl/results}-${MODEL##*/} \
#           --remove_unused_columns false \
#           --max_seq_length 512 \
#           --per_device_eval_batch_size $BS \
#           --do_eval \
#           --prob_aggregate_strategy max
#     done
# done
export CUDA_VISIBLE_DEVICES=2
BS=16

# CKPT=18000
# for MODEL in checkpoints/further*;do
#     for TYPE in type1.easy type1.hard type2;do
#         mkdir -p results/fin10k.eval/${TYPE}
#         EVAL=data/fin10k/fin10k.eval.${TYPE}.jsonl
#         OUTPUT=${EVAL##*/}
#         echo $MODEL
#         python3 inference.py \
#           --model_name_or_path $MODEL/checkpoint-$CKPT/ \
#           --config_name bert-base-uncased \
#           --eval_file $EVAL \
#           --output_file results/fin10k.eval/${TYPE}/${OUTPUT/jsonl/results}-${MODEL##*/} \
#           --remove_unused_columns false \
#           --max_seq_length 512 \
#           --per_device_eval_batch_size $BS \
#           --do_eval \
#           --prob_aggregate_strategy max
#     done
# done

CKPT=6000
for MODEL in checkpoints/from-scratch;do
    for TYPE in type1.easy type1.hard type2;do
        mkdir -p results/fin10k.eval/${TYPE}
        EVAL=data/fin10k/fin10k.eval.${TYPE}.jsonl
        OUTPUT=${EVAL##*/}
        echo $MODEL
        python3 inference.py \
          --model_name_or_path $MODEL/checkpoint-$CKPT/ \
          --config_name bert-base-uncased \
          --eval_file $EVAL \
          --output_file results/fin10k.eval/${TYPE}/${OUTPUT/jsonl/results}-${MODEL##*/} \
          --remove_unused_columns false \
          --max_seq_length 512 \
          --per_device_eval_batch_size $BS \
          --do_eval \
          --prob_aggregate_strategy max
    done
done
# Abalation: hyperparameter setting (tau), fixed gamma at -1.1
TRAIN_ESNLI=data/esnli/esnli.train.highlight.contradiction.jsonl
EVAL_ESNLI=data/esnli/esnli.dev.highlight.contradiction.jsonl
TRAIN_FIN10K=data/fin10k/fin10k.heuristic.synthetic.balance.train.type2.jsonl
export CUDA_VISIBLE_DEVICES=1

TYPE=further-finetune-sl-smooth-0.25 # tau=2; gamma=0.1
BS=24
python3 train_ablation.py \
  --ignore_data_skip \
  --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-15000 \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 20000\
  --save_steps 1500 \
  --eval_steps 1500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling true  \
  --tau 0.25 \
  --gamma 0.1 \
  --do_train \
  --do_eval 

TYPE=further-finetune-sl-smooth-0.1 # tau=0.5, gamma = 0.1
BS=24 
python3 train_ablation.py \
  --ignore_data_skip \
  --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-15000 \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 20000 \
  --save_steps 1500 \
  --eval_steps 1500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling true  \
  --tau 0.1 \
  --gamma 0.1 \
  --do_train \
  --do_eval 


# Ablation: lexicon 
TRAIN_ESNLI=data/esnli/esnli.train.highlight.contradiction.jsonl
EVAL_ESNLI=data/esnli/esnli.dev.highlight.contradiction.jsonl
TRAIN_FIN10K_LEXICON=data/fin10k/fin10k.lexicon.synthetic.balance.train.type2.jsonl

export CUDA_VISIBLE_DEVICES=0
TYPE=further-finetune-ll
BS=24 
python3 train.py \
  --ignore_data_skip \
  --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-15000 \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K_LEXICON \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 20000 \
  --save_steps 1500 \
  --eval_steps 1500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling false  \
  --tau 1 \
  --gamma 1 \
  --do_train \
  --do_eval 

TYPE=further-finetune-ll-balance
BS=24 
python3 train.py \
  --ignore_data_skip \
  --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-15000 \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K_LEXICON \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 20000 \
  --save_steps 1500 \
  --eval_steps 1500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling true  \
  --tau 1 \
  --gamma 0.5 \
  --do_train \
  --do_eval 
# Main domain adaptive fine-tuning
TRAIN_ESNLI=data/esnli/esnli.train.highlight.contradiction.jsonl
EVAL_ESNLI=data/esnli/esnli.dev.highlight.contradiction.jsonl
TRAIN_FIN10K=data/fin10k/fin10k.heuristic.synthetic.balance.train.type2.jsonl
TRAIN_FIN10K_SOFT=data/fin10k/fin10k.heuristic.synthetic.balance.soft.train.type2.jsonl
TRAIN_FIN10K_LEXICON=data/fin10k/fin10k.lexicon.synthetic.balance.train.type2.jsonl

export CUDA_VISIBLE_DEVICES=0

# Setting Zero-shot
TYPE=esnli-zs-highlighter
BS=24
python3 train.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_ESNLI \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 15000 \
  --save_steps 5000 \
  --eval_steps 5000 \
  --max_seq_length 258 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --do_train \
  --do_eval

# Setting from-scratch
TYPE=from-scratch
BS=24
python3 train.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 10000 \
  --save_steps 1500 \
  --eval_steps 1500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling false  \
  --tau 1 \
  --gamma 1 \
  --do_train \
  --do_eval

# Setting soft-labeling (hard)
TYPE=further-finetune
BS=24 
python3 train.py \
  --ignore_data_skip \
  --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-15000 \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 20000\
  --save_steps 1500 \
  --eval_steps 1500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling false  \
  --tau 1 \
  --gamma 1 \
  --do_train \
  --do_eval 


# Setting soft-labeling (balance)
TYPE=further-finetune-sl-balance # tau=1, gamma=0.5
BS=24
python3 train.py \
  --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-15000 \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K_SOFT \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 20000 \
  --save_steps 1500 \
  --eval_steps 1500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling true  \
  --tau 1 \
  --gamma 0.5 \
  --do_train \
  --do_eval

# Setting soft-labeling (tunned)
TYPE=further-finetun-sl-tunned # tau=1, gamma=0.1
BS=24
python3 train.py \
  --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-15000 \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K_SOFT \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 20000 \
  --save_steps 1500 \
  --eval_steps 1500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling true  \
  --tau 1 \
  --gamma 0.1 \
  --do_train \
  --do_eval

# Setting soft-labeling (soft)
TYPE=further-finetune-soft # tau=1, gamma=0
BS=24
python3 train.py \
  --resume_from_checkpoint checkpoints/esnli-zs-highlighter/checkpoint-15000 \
  --model_name_or_path bert-base-uncased \
  --output_dir checkpoints/$TYPE \
  --train_file $TRAIN_FIN10K_SOFT \
  --eval_file $EVAL_ESNLI \
  --per_device_train_batch_size $BS \
  --max_steps 20000 \
  --save_steps 1500 \
  --eval_steps 1500 \
  --max_seq_length 512 \
  --evaluation_strategy 'steps'\
  --evaluate_during_training \
  --soft_labeling true  \
  --tau 1 \
  --gamma 0 \
  --do_train \
  --do_eval

