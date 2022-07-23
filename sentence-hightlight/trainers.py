"""
Customized trainer for setnece highlight
"""
from transformers import Trainer
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
import math
import copy
import time
import json
import collections
import multiprocessing
import torch
from torch.utils.data import DataLoader, Sampler, ConcatDataset
from transformers.trainer_utils import has_length

_is_torch_generator_available = False

class BertTrainer(Trainer):


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    return DualTasksRandomSampler(
                            self.train_dataset, 
                            generator=generator,
                            subset_index=self.args.subset_index,
                            mixing_ratio=self.args.mixing_ratio
                    )
                return DualTasksRandomSampler(
                        self.train_dataset,
                        generator=generator,
                        subset_index=self.args.subset_index,
                        mixing_ratio=self.args.mixing_ratio
                )
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )

    def inference(self,
                  output_jsonl='results.jsonl',
                  eval_dataset=None, 
                  prob_aggregate_strategy='first',
                  save_to_json=True):

        f = open(output_jsonl, 'w')
        output_dict = collections.defaultdict(dict)
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        words, word_ids = [], []
        probs, labels = [], []
        # other unused columns
        unused = collections.defaultdict(list)

        for b, batch_idx in enumerate(range(0, len(eval_dataset), self.args.per_device_eval_batch_size)):
            batch = eval_dataset[batch_idx: batch_idx+self.args.per_device_eval_batch_size]
            word_ids += batch.pop('word_ids')
            # unused_batch = [{}] * len(batch)

            # remove unused and pass to unused columns
            for k in list(batch.keys()):
                if k in ['attention_mask', 'input_ids', 'labels', 'token_type_ids']:
                    batch[k] = torch.tensor(batch[k]).to(self.args.device)
                else:
                    unused[k] += batch.pop(k)
                    # for i, v in enumerate(batch.pop(k)):
                    #     unused_batch[i].update({k: v})

            # unused += [unused_batc[i] for i in range(len(un]
            output = self.model.inference(batch)
            probs += (output['probabilities'] * output['active_tokens']).cpu().tolist()
            labels += output['active_predictions'].cpu().tolist()

            if b % 100 == 0:
                print(f'Inferecning {b} batchs...')

        # per example in batch
        for i, (word_id, prob, label) in enumerate(zip(word_ids, probs, labels)):
            predictions = collections.defaultdict(list)
            # predictions['word'] += words[i]
            # Add unsed columns
            for k in unused.keys():
                predictions[k] = unused[k][i]

            # intialized probs and labesl
            predictions['probs'] = []
            predictions['labels'] = []

            # in a example (sentence pairs)
            for j, word_i  in enumerate(word_id):
                # outsides loop of word ids and words
                if word_i == None:
                    predictions['labels'].append(-1)
                    predictions['probs'].append(-1)
                elif word_id[j-1] == word_i:
                    if prob_aggregate_strategy == 'max':
                        predictions['probs'][-1] = max(prob[j], predictions['probs'][-1])
                    if prob_aggregate_strategy == 'mean':
                        dist = (j - len(prob) - 1)
                        predictions['probs'][-1] = \
                                (predictions['probs'][-1] * dist + prob[j]) / (dist + 1)
                    else: 
                        pass 
                else:
                    predictions['labels'].append(label[j])
                    predictions['probs'].append(prob[j])

            if save_to_json:
                f.write(json.dumps(predictions) + '\n')

            if i % 100 == 0:
                print(f"Output post-processing {i} example...")
                sosB = predictions['words'].index('<tag2>')
                print("Output: {}".format(
                    [(w, p, l) for w, p, l in zip(
                        predictions['words'][sosB:], 
                        predictions['probs'][sosB:], 
                        predictions['labels'][sosB:]
                    )]
                ))


class DualTasksRandomSampler(Sampler[int]):
    data_source: Sized
    replacement: bool

    def __init__(self, 
                 data_source: Sized, 
                 subset_index: int = None,
                 mixing_ratio: float = 0.5,
                 replacement: bool = False, 
                 num_samples: Optional[int] = None,
                 generator=None,
                 batch_size=8) -> None:

        self.data_source = data_source
        self.subset_index = subset_index # length of the first task
        self.mixing_ratio = mixing_ratio
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.batch_size = batch_size

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        # RandomSampler
        # return iter(range(n))
        n = len(self.data_source)

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # permutation
        if self.mixing_ratio > 0 and self.mixing_ratio < 1:
            dualtask_list = list()
            first_bs = math.ceil(self.batch_size * (1-self.mixing_ratio))
            second_bs = self.batch_size - first_bs
            first_list = torch.randperm(self.subset_index, generator=generator).tolist()
            second_list = (torch.randperm(n-self.subset_index, generator=generator)+self.subset_index).tolist()

            for i in range(self.subset_index // first_bs):
                dualtask_list += first_list[(i*first_bs):((i+1)*first_bs)]
                j = i % (n-self.subset_index)
                dualtask_list += second_list[(j*second_bs):((j+1)*second_bs)]
            yield from dualtask_list
            # a = torch.randperm(n, generator=generator).tolist()
            # return iter(a)
        else:
            yield from torch.randperm(n, generator=generator).tolist()

