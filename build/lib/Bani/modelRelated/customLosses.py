import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from enum import Enum

class SoftmaxLayerLoss(nn.Module):
    def __init__(self, model , numLabels):
        super().__init__()
        self.model = model
        self.numlabels = numLabels
        self.modelOutDim = self.model.get_sentence_embedding_dimension()
        self.classifier = nn.Linear(self.modelOutDim,numLabels, bias= False)


    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor):
        features = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features][0]
        output = self.classifier(features)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(output, labels.view(-1))
        print(loss)
        return loss