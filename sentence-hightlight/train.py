import os
import sys
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    Trainer,
    HfArgumentParser,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding
)

from datasets import load_dataset, DatasetDict, concatenate_datasets
from models import BertForHighlightPrediction
from trainers import BertTrainer

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
    # Cutomized arguments
    pooler_type: str = field(default="cls")
    temp: float = field(default=0.05)
    num_labels: float = field(default=2)
    label_smoothing: float = field(default=0)


@dataclass
class OurDataArguments:

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=True)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    # Customized arguments
    train_file: Optional[str] = field(default="")
    eval_file: Optional[str] = field(default="")
    test_file: Optional[str] = field(default="")
    train_file_2: Optional[str] = field(default="")
    max_seq_length: Optional[int] = field(default=512)
    pad_to_max_length: bool = field(default=False)

@dataclass
class OurTrainingArguments(TrainingArguments):
    output_dir: str = field(default='./models')
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    save_steps: int = field(default=1000)
    eval_steps: int = field(default=1000)
    evaluate_during_training: bool = field(default=False)
    evaluation_strategy: Optional[str] = field(default='no')
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    weight_decay: float = field(default=0.0)
    logging_dir: Optional[str] = field(default='./logs')
    warmup_steps: int = field(default=1000)
    resume_from_checkpoint: Optional[str] = field(default=None)
    mixing_ratio: Optional[float] = field(default=0)
    subset_index: Optional[int] = field(default=None)


def main():
    # Parseing argument for huggingface packages
    parser = HfArgumentParser((OurModelArguments, OurDataArguments, OurTrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_datalcasses()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # [CONCERN] Deprecated? or any parser issue.
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # config and tokenizers
    # [TODO] If the additional arguments are fixed for putting in the function,
    # make it consistent to the function calls.
    config_kwargs = {
            "num_labels": model_args.num_labels,
            "output_hidden_states": True
    }
    tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir, 
            "use_fast": model_args.use_fast_tokenizer
    }
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)


    # model 
    model_kwargs = {
            "cache_dir": model_args.cache_dir,
    }
    model = BertForHighlightPrediction.from_pretrained(
            model_args.model_name_or_path,
            config=config, 
            model_args=model_args
    )

    # Dataset 
    # (1) loading the jsonl file
    # (2) Preprocessing 
    def preprare_esnli_seq_labeling(examples):
    
        def merge_list(word_labels, word_id_list):
            """aggregate the word_lables to token_labels."""
            token_labels = word_labels
            for i, idx in enumerate(word_id_list):
                if idx == None:
                    token_labels.insert(i, -100)
                elif word_id_list[i-1] == word_id_list[i]:
                    token_labels.insert(i, -100)

            return token_labels

        size = len(examples['wordsA'])

        features = tokenizer(
            examples['wordsA'], examples['wordsB'],
            is_split_into_words=True, # allowed pre-tokenization process, to match the seq-order
            max_length= data_args.max_seq_length,
            truncation=True,
            padding=True
        )   

        # 1) transforme the label to token-level
        # 2) Preserve the word ids (for evaluation)
        features['labels'] = [None] * size
        features['word_ids'] = [None] * size

        for b in range(size):
            features['labels'][b] = merge_list(
                word_labels=examples['labels'][b], 
                word_id_list=features.word_ids(b)
            )
            features['word_ids'][b] = features.word_ids(b)

        return features

    ## Loading form json
    dataset = DatasetDict.from_json({
        "train": data_args.train_file,
        "dev": data_args.eval_file,
    })


    ## Preprocessing: training dataset
    dataset['train'] = dataset['train'].map(
        function=preprare_esnli_seq_labeling,
        batched=True,
        remove_columns=['sentA', 'sentB', 'keywordsA', 'keywordsB', 'labels', 'wordsA', 'wordsB'],
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=not data_args.overwrite_cache,
    )
    dataset['train'].remove_columns('word_ids')
    
    ## Preprocessing: dev dataset (preseve the words and word_ids)
    dataset['dev'] = dataset['dev'].map(
        function=preprare_esnli_seq_labeling,
        batched=True,
        remove_columns=['sentA', 'sentB', 'keywordsA', 'keywordsB', 'labels'],
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=not data_args.overwrite_cache,
    )
    dataset['dev'].remove_columns('word_ids')

    if training_args.mixing_ratio > 0 and data_args.train_file_2 is not None: # add fin10k dataset
        dataset_additional = dataset.from_json(data_args.train_file_2)
        dataset_additional = dataset_additional.map(
            function=preprare_esnli_seq_labeling,
            batched=True,
            remove_columns=['sentA', 'sentB', 'keywordsA', 'keywordsB', 'labels', 'wordsA', 'wordsB'],
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=not data_args.overwrite_cache,
        )
        dataset_additional.remove_columns('word_ids')
        training_args.subset_index = len(dataset['train'])


    # Dataset
    # (2) data collator: transform the datset into the training mini-batch
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        return_tensors="pt",
        max_length=data_args.max_seq_length,
        padding='max_length'
    )

    # Trainer
    if training_args.mixing_ratio > 0:
        trainer = BertTrainer(
                model=model, 
                args=training_args,
                train_dataset=concatenate_datasets([dataset['train'], dataset_additional]),
                eval_dataset=dataset['dev'],
                data_collator=data_collator,
        )
    else:
        trainer = BertTrainer(
                model=model, 
                args=training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['dev'],
                data_collator=data_collator,
        )
    
    # ***** strat training *****
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
            # model_path=model_pathk
    )

    return results

if __name__ == '__main__':
    main()
