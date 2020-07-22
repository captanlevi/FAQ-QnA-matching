import sys 
sys.path.append('../')
import warnings 
import json
import pickle
import numpy as np
import pandas as pd
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn,Tensor
from typing import Union, Tuple, List, Iterable, Dict
from torch.utils.data import SequentialSampler



class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str, texts: List[str], label: Union[int, float]):
        """
        Creates one InputExample with the given texts, guid and label
        str.strip() is called on both texts.
        :param guid
            id for the example
        :param texts
            the texts for the example
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = [text.strip() for text in texts]
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))


class Question_sampler(Dataset): 
    def __init__(self,questions,generated_ques,model,labels = None, bs = 32 , n = 4):
        """
        questions --> array of string
        labels --> array of labels
        generated_ques --> a dict from ques to array of generated ques
        bs --> batch size
        n  --> number of classes to pick , from 
        """
       
        self.questions = questions
        if(labels is None):
            labels = [x for x in range(len(questions))]
            
        assert len(questions) == len(labels), 'length of ques is {} but len of labels is {}'.format(len(questions), len(labels))
        self.labels = labels
        
        self.model = model
        self.data = {}
        # data is a dict from label to array of questions
        self.min_gen = 100000000000
        self.classes = len(set(self.labels))
        for i,que in enumerate(questions):
            self.min_gen = min(self.min_gen , len(generated_ques[que]))
            if(self.labels[i] not in self.data):
                self.data[self.labels[i]] = []
            self.data[self.labels[i]] += [que] + generated_ques[que]
             
        self.sequence = []
        self.make_sequence(total_examples = len(questions), batch_size = bs,n = n)
    def __len__(self):
        return len(self.sequence[0])
    
    
    def __getitem__(self, index):
        inp , lab = self.sequence
        return [inp[index]], lab[index]
    def sample_triplet(self):
        """
        To sample triplet from a sentence
        returns a pair of triplet
        [Anchor , positive , negative]
        """
        total_ques = len(self.questions)
        anchor_data_index,neg_data_index = np.random.choice(len(self.data), size = 2,replace = False).tolist()
        pos_list = self.data[anchor_data_index]
        anchor_index , pos_index = np.random.choice(len(pos_list), size = 2,replace = False).tolist()
        anchor , pos = pos_list[anchor_index].strip() , pos_list[pos_index].strip() 
        neg_list = self.data[neg_data_index]
        neg_index = np.random.choice(len(neg_list))
        neg = neg_list[neg_index]
        
        
        return [anchor, pos, neg]
    
    
    
    
    def make_sequence_util(self, n , k):
        """
        to make a batch of n classes
        where each class has k examples
        """
        if(n <= self.classes):
            class_indices = np.random.choice(self.labels , size = n , replace = False)
        else:
            warnings.warn('n is greater than number of classes') 
            class_indices = np.random.choice(self.labels , size = n , replace = True)
        
        
        batch_inps = []
        batch_labels = []
        for i in class_indices:
            label = i
            ques = self.data[i]
            num_ques = len(ques)
            if(k <= num_ques):
                que_indices = np.random.choice(num_ques, size = k, replace = False)
            else:
                warnings.warn('k is greater than than number of generated ques')
                que_indices = np.random.choice(num_ques, size = k, replace = True)
                
            for j in que_indices:
                batch_inps.append(self.model.tokenize(ques[j]))
                batch_labels.append(label)
        return [batch_inps, batch_labels]
            
    def make_sequence(self,total_examples,batch_size,n):
        """
        total_examples number of batches are formed --> making the seq batch_size*total_examples long
        n  class are drawn in each batch 
        batch_size//n number of examples for each classes
        """
        assert batch_size%n == 0 , 'batch_size should be divisible by n'
        self.sequence = []
        batch_inps = []
        batch_labels = []
        
        for i in range(total_examples):
            x,y = self.make_sequence_util(n,batch_size//n)
            # sampled a batch , of size batch_size
            batch_inps += x
            batch_labels += y
            
            
            
        self.sequence = [batch_inps, torch.tensor(batch_labels)]
        
        
def get_dataloader(questions,generated_ques,model,labels = None, bs = 32 , n = 4):
    """
        questions --> array of string
        labels --> array of labels
        generated_ques --> a dict from ques to array of generated ques
        bs --> batch size
        n  --> number of classes to pick , from 
    """

    Q = Question_sampler(questions ,generated_ques,model,labels, bs = bs , n = n)
    train_data = Q
    train_dataloader = DataLoader(train_data,  batch_size= bs, sampler= SequentialSampler(train_data))
    return train_dataloader



if __name__ == '__main__':
    pass
