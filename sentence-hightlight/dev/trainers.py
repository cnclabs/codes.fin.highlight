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
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

_is_torch_generator_available = False

class BertTrainer(Trainer):


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. 
        By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

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