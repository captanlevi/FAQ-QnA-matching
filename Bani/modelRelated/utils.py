from sentence_transformers.readers import InputExample
from ..core.FAQ import FAQ, FAQUnit
from ..core.exceptions import *
from typing import Dict, List
import random
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



def minLabelChecker(lossName : str , FAQ : FAQ, minSameLabels : int):
    faq = FAQ.FAQ

    counter = dict()
    for unit in faq:
        label = unit.label

        if(label not in counter):
            counter[label] = 0
        counter[label] += 1

    for L,count in counter.items():
        if(count < minSameLabels):
            raise TrainDataInvalid("Attempted to use {} loss but label {} has only {} exmaples !!! At least {} required".format(lossName,L,count,minSameLabels))




def convertForBatchHardTripletLoss(FAQ : FAQ, minSameLabels = 2) -> List[InputExample]:
    """
    Must check if the data has atleast minSameLabels(atleast 2) examples for each label !!!
    """
    assert minSameLabels >= 2
    minLabelChecker(lossName= "batchHardTriplet",FAQ = FAQ,minSameLabels= minSameLabels)
    faq = FAQ.FAQ    
    # now converting
    inputs = []
    for unit in faq:
        inputs.append(InputExample(texts = [unit.question.text] , label= unit.label))

    
    return inputs



convertForTripletLoss =  convertForBatchHardTripletLoss



def convertForContrastiveLoss(FAQ : FAQ , minSameLabels : int = 2, numIters : int = 1):
    """
    Gets data for training on a cosine similirity loss, each sentence must have atleaast one paraphrase !!!
    numIters is number of times we pick up negative examples
    """
    assert minSameLabels >= 2 and numIters >= 1
    minLabelChecker(lossName= "ContrastiveLoss",FAQ = FAQ,minSameLabels= minSameLabels)

    faq = FAQ.FAQ

    inputs = []

    labelToQuestions : Dict[int, List[str]] = dict()


    for unit in faq:
        label = unit.label
        if(label not in labelToQuestions):
            labelToQuestions[label] = []
        labelToQuestions[label].append(unit.question.text)

    for label in list(labelToQuestions.keys()):
        # positives
        positives = labelToQuestions[label]
        
        for i in range(len(positives)):
            for j in range(i+ 1, len(positives)):
                s1, s2 = positives[i], positives[j]
                inputs.append(InputExample(texts= [s1,s2], label= 1))



    for _ in range(numIters):
        labelList = list(labelToQuestions.keys())
        for i in range(len(labelList)):
            # negatives
            for j in range(i+1,len(labelList)):
                label1, label2 = labelList[i],labelList[j]
                if(label2 == label1):
                    continue


                s1 = random.choice(labelToQuestions[label1])
                s2 = random.choice(labelToQuestions[label2])

                inputs.append(InputExample(texts= [s1,s2], label= 0))

        

    return inputs





def convertForSoftmaxLoss(FAQ : FAQ):
    """
    This loss adds an softmax layer on top of the sentence transformer
    """

    # making faq labels to loss label mapping

    labelTolabel : Dict[int,int] = dict()
    labelToQuestions : Dict[int,List[str]] = dict()   # Contains mapping from the loss labels to questions 
    faq = FAQ.FAQ
    index = 0
    for unit in faq:
        label = unit.label
        if(label not in labelTolabel):
            labelTolabel[label] = index
            index += 1

        lossLabel = labelTolabel[label]
        if(lossLabel not in labelToQuestions):
            labelToQuestions[lossLabel] = []
        labelToQuestions[lossLabel].append(unit.question.text)



    inputs = []

    for label , questions in labelToQuestions.items():
        for question in questions:
            inputs.append(InputExample(texts= [question], label= label))
    
    random.shuffle(inputs)
    return inputs , len(labelToQuestions)




    