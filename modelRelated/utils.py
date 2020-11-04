from sentence_transformers.readers import InputExample
from common.FAQ import FAQ
from common.exceptions import *
from typing import List
import numpy as np


def cosineSim(v,V):
    """
    computes cosine sim between v,V
    where v and V are  2D np matrices (n,E) (N,E)
    output is of the shape (n,N)
    """

    n1 = np.linalg.norm(v, axis = -1)
    n2 = np.linalg.norm(V, axis = -1)
    dot = np.expand_dims(v,1)*np.expand_dims(V,0)
    # shape (n,N,E)
    dot = dot.sum(axis = -1)
    ans = dot/n1.reshape(-1,1)
    ans = ans/n2.reshape(1,-1)
    return ans



def convertForBatchHardTripletLoss(FAQ : FAQ, minSameLabels = 2) -> List[InputExample]:
    """
    Must check if the data has atleast minSameLabels(atleast 2) examples for each label !!!
    """
    assert minSameLabels >= 2
    faq = FAQ.FAQ

    counter = dict()
    for unit in faq:
        label = unit.label

        if(label not in counter):
            counter[label] = 0
        counter[label] += 1

    for L,count in counter.items():
        if(count < minSameLabels):
            raise TrainDataInvalid("Attempted to use batchhardtriplet loss but label {} has only {} exmaples !!! At least {} required".format(L,count,minSameLabels))
        
    # now converting
    inputs = []
    for unit in faq:
        inputs.append(InputExample(texts = [unit.question.text] , label= unit.label))

    
    return inputs


