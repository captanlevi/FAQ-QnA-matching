from torch.utils.data.dataset import Dataset
import torch
from sentence_transformers import InputExample
from typing import List, Union , Dict, Tuple, Any
import random
import numpy as np




class SoftmaxDataset(Dataset):
    def __init__(self, examples : List[InputExample], model):
        self.model = model
        self.examples = examples
        for example in self.examples:
            example.texts_tokenized = self.tokenizeExample(example=example)





    def tokenizeExample(self, example : InputExample):
        if example.texts_tokenized is not None:
            return example.texts_tokenized

        return [self.model.tokenize(text) for text in example.texts]


    def __len__(self) -> int:
        return len(self.examples)
    

    def __getitem__(self, idx: int) -> Any:
        return self.examples[idx].texts_tokenized , torch.tensor(self.examples[idx].label)

