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
from utils import get_dataloader
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers import losses
from sentence_transformers.evaluation import TripletEvaluator



class modelInterface:
    def __init__(self, model_path = None):
        """ 
        Either mention the model path to previously saved model ,
        or let it be , none
        when model path is None , the model will be a transformer model, with roBERTa base
        """
        if(model_path == None):
            model_path = 'roberta-base-nli-mean-tokens'
        self.model = SentenceTransformer(model_path)
        self.current_faq = {}
        # current data is to be filled using the fit_FAQ function call
        # it has 3 keys 1) embeddings  (a np array) 2) labels 3) label_to_answer dict
    def train(self,data, model_save_path):
        """
        data --> a dict {'questions' : list of questions , 'labels': 'list of labels',
                        'generated_ques': dict mapping from question(string) to an list of generated_questions
                        'bs': batch_size for training
                        'n' : num_of_classes to sample in a batch (bs%n ==0), 
                        }
        model_save_path = path of folder to save the model 

        This function will fit you data using a batch hard triplet loss, and save the model to the folder specified
        the folder should be empty or not created in the  beginning!!!
        """
        data['model'] = self.model
        train_dataloader = get_dataloader(**data)
        train_loss = losses.BatchHardTripletLoss(sentence_embedder=self.model)

        self.model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        evaluator = None,
        output_path= model_save_path,
        )

    def cosine_sim(self,v,V):
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

    def evaluate(self,data, mode = 'mean'):
        """
        Will evaluate model , on a given test_set
        data --> dict
        keys are 
            1) test_questions_labels_dict --> a dict {'question' : label_number, ...}  is a mapping from test questions to labels
            2) data_questions_label_dict --> dict {'question', label_number , ....} is a mapping of FAQ questions to labels

            make sure that the labels are in sync, 
            so the label will map test questions to another question in the test set
        """

        # converting to array
        test_questions = []
        test_labels = []

        for q ,l in data['test_questions_labels_dict'].items():
            test_questions.append(q)
            test_labels.append(int(l))

        test_questions = np.array(self.model.encode(test_questions))
        test_labels = np.array(test_labels)
        # Now test_questions is a np.ndarray (N , embedding_dim) , and test_labels is a np.ndarray (N,)

        data_questions = []
        data_labels = []

        for q,l in data['data_questions_labels_dict'].items():
            data_questions.append(q)
            data_labels.append(int(l))

        data_questions = np.array(self.model.encode(data_questions))
        data_labels = np.array(data_labels)

        unique_labels = np.unique(data_labels)

        index_to_label = {}
        mean_vector = np.empty((0,data_questions.shape[1]))
        for i,label in enumerate(unique_labels):
            index_to_label[i] = label

            # picking out all the question with label == label
            ques = data_questions[data_labels == label].mean(axis = 0).reshape(1,-1)
            # now of shape (1,embedding_dim)
            mean_vector = np.append(mean_vector , ques , axis= 0)

        # now that we have label to index , lets compute ,cosine similirity

        cosine_sims = self.cosine_sim(test_questions , mean_vector)
        # of shape (num_test , num_unique_label)

        indices = np.argmax(cosine_sims,axis = -1)

        pred_labels = np.array([index_to_label[i] for i in indices ])
        return ((test_labels == pred_labels).sum()) /len(test_labels)


    

    def fit_FAQ(self,question_to_label , answer_to_label, labels = None):
        """
        A function to fit any given FAQ
        data is a dict, has keys
            1) questions --> a dict ('string question' : label)
            2) answers --> a dict ('string answer' : label)
        """

        questions = []
        labels = []
        for q,l in question_to_label.items():
            questions.append(q.replace('\n','').strip())
            labels.append(int(l))
        questions = np.array(questions)
        labels = np.array(labels)

        # Now inverting answer_to_label
        label_to_answer  = {}
        for answer , label in answer_to_label.items():
            if(label in label_to_answer):
                assert False , 'multiple answers have the same labels'
            label_to_answer[label] = answer

            if(label not in labels):
                warnings.warn('some labels in answers are not present in questions , you might not have labels in Sync')
            
        for l in labels:
            if(l not in label_to_answer):
                 warnings.warn('some labels in question are not present in answers ,this might cause runtime errors later, you might not have labels in Sync')
            


        self.current_faq  = {'embeddings' : self.model.encode(questions) , 'labels': labels, 'label_to_answer': label_to_answer}


    def answer_question(self, question):
        """
            This is where you ask the question , and a approropriate answer is returned,
            must call fit_FAQ before this

            question ==> string
            returns ==> string
        """
        if (self.current_faq is None):
            assert False , 'Need to fit_FAQ before calling answer_question'
        
        embeddings = self.current_faq['embeddings']
        question_labels = self.current_faq['labels']
        label_to_answer = self.current_faq['label_to_answer']

        question = self.model.encode([question])[0].reshape(1,-1)
        # question is now a np.ndarray of shape (1,embedding_dim)
        
        cosine_sim = self.cosine_sim(question, embeddings)[0]
        #cosine_sim --> shape (N,)

        max_ind = np.argmax(cosine_sim)

        max_val = cosine_sim[max_ind]
        print(max_val)
        if(max_val < .2):
            return "out of set question"
        else:
            label = question_labels[max_ind]
            if(label not in label_to_answer):
                return 'No answer corrosponding to the label {} , this means your question--label--answer dict is faulty '.format(label)
            return label_to_answer[label]        









        



        






    