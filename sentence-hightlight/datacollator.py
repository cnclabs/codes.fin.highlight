from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import (
        PaddingStrategy,
        PreTrainedTokenizerBase
)

@dataclass
class Fin10KDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        n = len(features)
        wordsA_features = [ft['wordsA'] for ft in features]
        wordsB_features = [ft['wordsB'] for ft in features]

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
