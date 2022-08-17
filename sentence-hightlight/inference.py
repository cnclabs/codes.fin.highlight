import json
import sys
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorWithPadding
)

from datasets import load_dataset, Dataset
from models import BertForHighlightPrediction
from torch.utils.data import DataLoader
from datacollator import Fin10KDataCollator

# Arguments: (1) Model arguments (2) DataTraining arguments (3)
@dataclass
class OurModelArguments:

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default='./models')
    model_type: Optional[str] = field(default='bert-base-uncased')
    config_name: Optional[str] = field(default='bert-base-uncased')
    tokenizer_name: Optional[str] = field(default='bert-base-uncased')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)

@dataclass
class OurDataArguments:

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    # Customized arguments
    eval_file: Optional[str] = field(default="")
    test_file: Optional[str] = field(default="")
    output_file: Optional[str] = field(default="")
    max_seq_length: Optional[int] = field(default=512)

@dataclass
class OurTrainingArguments(TrainingArguments):

    output_dir: str = field(default='./models')
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    do_test: bool = field(default=False)
    save_steps: int = field(default=1000)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    weight_decay: float = field(default=0.0)
    logging_dir: Optional[str] = field(default='./logs')
    warmup_steps: int = field(default=1000)
    resume_from_checkpiint: Optional[str] = field(default=None)
    remove_unused_columns: Optional[bool] = field(default=False)
    # Customeized argument for inferencing
    prob_aggregate_strategy: Optional[str] = field(default='first')

def main():
    parser = HfArgumentParser((OurModelArguments, OurDataArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # config and tokenizers
    # [TODO] If the additional arguments are fixed for putting in the function,
    tokenizer_kwargs = {"use_fast": model_args.use_fast_tokenizer }
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)

    config_kwargs = {"output_hidden_states": True, }
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)

    # model 
    model_kwargs = {"cache_dir": model_args.cache_dir,}
    model = BertForHighlightPrediction.from_pretrained(
            model_args.model_name_or_path,
            config=config, 
            model_args=model_args
    )

    # evaluation dataset
    dataset = Dataset.from_json(data_args.eval_file)

    # ## preprocessing function (batch tokenization and wordid)
    # datacollator
    data_collator = Fin10KDataCollator(
            tokenizer=tokenizer,
            padding=True,
            truncation='only_first',
            max_length=data_args.max_seq_length,
    )

    # loader
    dataloader = DataLoader(
            dataset,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator
    )

    f = open(data_args.output_file, 'w')
    # run prediction
    for b, batch in enumerate(dataloader):

        predictions = {}
        output, info = model.inference(batch)

        # output
        word_ids = info.pop('word_ids')
        word_probs = (output['probabilities'] * output['active_tokens']).cpu().tolist()
        word_labels = output['active_predictions'].cpu().tolist()

        # merge info
        for i in range(len(word_ids)):

            # for each example
            predictions = {k: info[k][i] for k in info}
            sosB = info['probs'][i][1:].index(-1) + 1
            probs_holder = []
            labels_holder = []
            spec = 0

            for j, (w_i, p, l) in enumerate(zip(word_ids[i], word_probs[i], word_labels[i])):
                if w_i==None:
                    if (spec==1) or (spec==2): # start of sentB and end of sentB
                        probs_holder.append(-1)
                        labels_holder.append(-1)
                    spec += 1
                elif (spec==2) and (word_ids[i][j-1] != w_i):
                    probs_holder.append(p)
                    labels_holder.append(l)
                elif (spec==2) and (word_ids[i][j-1] == w_i):
                    # [TODO] implement mean aggregation ?
                    probs_holder[-1] = max(p, probs_holder[-1])
                    labels_holder[-1] = max(l, labels_holder[-1])
                
            # allocate prob/label 
            predictions['probs'][sosB:] = probs_holder
            predictions['labels'][sosB:] = labels_holder

            assert len(predictions['words']) == len(predictions['probs']),\
                    "Inconsistent length of words and probs."
            f.write(json.dumps(predictions) + '\n')

        if b % 10 == 0:
            print(f"Output post-processing {b} batch...")
            print("Output: {}".format(
                [(w, p, l) for w, p, l in zip(
                    predictions['words'][sosB:],
                    predictions['probs'][sosB:],
                    predictions['labels'][sosB:]
                )]
            ))

    f.close()


if __name__ == '__main__':
    main()
