# Signal highlighting stage 

This stage mainly includes the following process.
- Out-domain fine-tuning 
- Pseudo-labeling
- In-domain fine-tuning

Also, we provide other codes of preprocessing and evaluaion codes (See [OTHERS.md](OTHERS.md) for detail).

---

## Data preparation (e-SNLI)

- Download 
The required dataset is [e-SNLI](https://github.com/OanaMariaCamburu/e-SNLI/tree/master/dataset/). 
You should first donwload and save the data in [data/](data/).
> Note there are 2 esnli trainining files; we merge the two into one standalone file.

- Preprocessing (e-SNLI)
Parse the e-SNLI dataset into the following formats.
```
bash run_create_esnli_data.sh
```

- Preprocessing (FINAL)
Prepare the FINAL dataset with the following codes. 
The preprocessed data has been already done; the data can be found in the [document](../README.md).
```
unzip final_v1.zip
  ...
  creating: final_v1/
  inflating: final_v1/fin10k.annotation.mismatched.jsonl.truth
  inflating: final_v1/fin10k.heuristic.synthetic.balance.train.revised.jsonl
  inflating: final_v1/fin10k.annotation.revised.jsonl.truth
```

## S2: Out-domain fine-tuning

### Baseline model 1: Zero-shot (esnli-zs-highligher)
The `esnli-zs-highlighter` is the baseline highlighting models.
Fine-tuned on the compiled e-SNLI dataset which labels are `contradiction`.

Such model checkpoints can be found in [huggingface](#)(TBA).

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
We repor the detail instruction here. However, we have already done the following process; the files are save in the `final_v1.zip` file.
That is, `final_v1/fin10k.heuristic.synthetic.balance.train.revised.jsonl`


### Pseduo-labeling (hard-label)
We first use the heuristic hard-label process the generate a binary labels (`label` training dataset, FINAL-train).
See [(Build FINAL data from scratch)](OTHERS.md/#build-final-data-from-scratch).

### Pseduo-labeling (soft-label)
Once obtaining the `esnli-zh-highlighter`, we can use it to generate the pseduo-labeled dataset.
Inference the probability as psuedo soft-label.

```
# Generate soft-labeling (Only need inferenced once)
BS=16
FILE=data/final_v1/fin10k.heuristic.synthetic.balance.train.type2.jsonl
python3 inference.py \
  --model_name_or_path checkpoints/esnli-zs-highlighter/checkpoint-10000/ \
  --config_name bert-base-uncased \
  --eval_file $FILE \
  --output_file ${FILE/train/soft.train} \
  --remove_unused_columns false \
  --max_seq_length 512 \
  --per_device_eval_batch_size $BS \
  --do_eval \
  --prob_aggregate_strategy max
```

### Baseline model 2: Pseudo few-shot
The `pseudo-zs-highlighter` is the baseline highlighting models.
Fine-tuned on the compiled e-SNLI dataset which labels are `contradiction`.

```
TRAIN_FIN10K=data/final_v1/fin10k.heuristic.synthetic.balance.train.type2.jsonl
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

### Proposed model: In-domain fine-tuning (domain-adpative)
The `domain-adpative` is our proposed highlighting models.
Fine-tuned on the compiled FINAL_v1 dataset which is generated with psuedo hard-labeling and soft-labeling.

```
TRAIN_FIN10K=data/final_v1/fin10k.heuristic.synthetic.balance.train.type2.jsonl
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


### Prediction: Inference
Predict the total 400 pairs we have labeled, including 
(1) mismatched (hard) with 200 sample pairs. 
(2) revised (easy) with 200 sample pairs.
```
CKPT=<int, the fine-tunned steps>
for MODEL in checkpoints/pseudo-few-shot;do
    for TYPE in type1.easy type1.hard type2;do
        mkdir -p results/fin10k.eval/${TYPE}
        EVAL=data/final_v1/fin10k.eval.${TYPE}.jsonl
        OUTPUT=${EVAL##*/}
        echo $MODEL

        python3 inference.py \
          --model_name_or_path $MODEL/checkpoint-$CKPT/ \
          --config_name bert-base-uncased \
          --eval_file $EVAL \
          --output_file results/final_v1/${TYPE}/${OUTPUT/jsonl/results}-${MODEL##*/} \
          --remove_unused_columns false \
          --max_seq_length 512 \
          --per_device_eval_batch_size $BS \
          --do_eval \
          --prob_aggregate_strategy max
    done
done
```
