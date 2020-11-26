from numpy.core.fromnumeric import size
from sentence_transformers import InputExample
from typing import List, Union , Dict, Tuple, Any
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import torch


class BatchHardDataset(Dataset):
    def __init__(self, examples : List[InputExample],  model):
        for example in examples:
            assert len(example.texts) == 1
        
        self.model = model
        self.examples : List[InputExample] = examples

        self._build_dict()
        for example in self.examples:
            example.texts_tokenized = self.tokenizeExample(example=example)


    def _build_dict(self):
        self.labelToIndices : Dict[int , List[int]] = dict()

        for i,example in enumerate(self.examples):
            label = example.label
            if(label not in self.labelToIndices):
                self.labelToIndices[label] = []
            
            self.labelToIndices[label].append(i)

    
    def tokenizeExample(self, example : InputExample):
        if example.texts_tokenized is not None:
            return example.texts_tokenized

        return [self.model.tokenize(text) for text in example.texts]
    
    
    def __getitem__(self, idx : int) -> Tuple[List[Any], torch.Tensor]:
        return self.examples[idx].texts_tokenized , torch.tensor(self.examples[idx].label)


    def __len__(self) -> int:
        return len(self.examples)



class PKSampler(Sampler):
    def __init__(self, dataSource : BatchHardDataset, p=4, k=4):
        super().__init__(dataSource)
        assert p > 2 and k > 2 , "batch hard condition enforced, increase batch size"
        self.p = p
        self.k = k
        self.dataSource = dataSource

    def __iter__(self):
        pkCount = len(self) // (self.p * self.k)
        for _ in range(pkCount):
            labels = np.random.choice(list(self.dataSource.labelToIndices.keys()), self.p, replace=False)
            for l in labels:
                indices = self.dataSource.labelToIndices[l]
                replace = True if len(indices) < self.k else False

                for i in np.random.choice(indices, self.k, replace=replace):
                    yield i

    def __len__(self):
        pk = self.p * self.k
        samples = ((len(self.dataSource) - 1) // pk + 1) * pk
        return samples


        

        
