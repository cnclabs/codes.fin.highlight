from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import (
        PaddingStrategy,
        PreTrainedTokenizerBase
)
from transformers.data.data_collator import DataCollatorForTokenClassification

@dataclass
class TokenHighlightDataCollator(DataCollatorForTokenClassification):
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    sentA_removal: bool = False
    sentA_consistent: bool = False

    def torch_call(self, features):
        "labels", "features", "word_ids", "probs"
        import torch

        labels = [ft['labels'] for ft in features]
        probs = [ft['probs'] for ft in features]

        features_input = []
        for ft in features:
            features_input.append({
                    k: v for k, v in ft.items() \
                            if k in ['attention_mask', 'input_ids', 'token_type_ids']
            })

        batch = self.tokenizer.pad(
            features_input,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch['labels'] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) \
                        for label in labels
            ]
            batch_probs = [
                list(prob) + [0] * (sequence_length - len(prob)) \
                        for prob in probs
            ]
        else:
            batch['labels'] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) \
                        for label in labels
            ]
            batch_probs = [
                [0] * (sequence_length - len(prob)) + list(prob) \
                        for prob in probs
            ]

        for ft in features:
            assert len(ft['labels']) == len(ft['probs']), 'X'

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        batch['probs'] = torch.tensor(batch_probs, dtype=torch.float32)
        return batch

@dataclass
class Fin10KDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    sentA_removal: bool = False
    sentA_consistent: bool = False
    sentA_shuffle: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        n = len(features)
        wordsA_features = [ft['wordsA'] for ft in features]
        wordsB_features = [ft['wordsB'] for ft in features]

        if self.sentA_removal:
            wordsA_features = [["[PAD]"] for ft in features]

        if self.sentA_consistent:
            wordsA_features = [ft['wordsB'] for ft in features]

        if self.sentA_shuffle:
            import random
            shuffled_indices = random.sample(range(len(features)), len(features))
            wordsA_features = [features[i]['wordsA'] for i in shuffled_indices]

        # process input
        features_input = self.tokenizer(
            wordsA_features, wordsB_features,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors=self.return_tensors
        )

        # process info
        features_info = {}
        for k in features[0]:
            features_info[k] = [ft[k] for ft in features]

        features_info['word_ids'] =  [features_input.word_ids(i) for i in range(n)]

        return features_input, features_info
