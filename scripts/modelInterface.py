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
import nlpaug
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import random
import pickle

def save_dict(obj, path ):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class AUG():
    def __init__(self):
        aug0 = naw.RandomWordAug()
        aug1 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
        aug2 = naw.SynonymAug(aug_src='wordnet')
        aug3 = naw.SplitAug()
        aug4 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")

        self.augs = [aug0, aug1,aug2,aug3,aug4]
        
    def __call__(self,sent,n= 1):
        return self.augment(sent,n)
        
    
    def augment(self,sent ,n = 1):
        ans = []
        sent = re.sub(r'[^a-zA-Z0-9_ ]', '', sent)
        for _ in range(n):
            aug = random.choice(self.augs)
            ans.append(aug.augment(sent))
                    
        return ans



class modelInterface:
    def __init__(self, faq_name ,faq_data = None,model_path = None):
        """ 
        Either mention the model path to previously saved model ,
        or let it be , none
        when model path is None , the model will be a transformer model, with roBERTa base

        faq_name is the name of the faq , generated questions and answers , if it already exists , we will used the
        processed questions and answers , otherwise , we have to create a new one

        if faq name , has not been processed atleast once , you must provide faq_data
        faq_data --> dict has two keys , question_to_label , and answer_to_label

        1) question_to_label
        2) answer_to_label

        question_to_labels is again a dictionary from questions : label(int)  can have multiple questions for same label
        answer_to_labels is a dictionary from answers to label : one label per answer (strict !!!)
        """
        if(model_path == None):
            model_path = 'roberta-base-nli-stsb-mean-tokens'

        self.model = SentenceTransformer(model_path)
        self.current_faq = None
        self.faq_path = os.path.join(os.getcwd(),"FAQs", faq_name) 
        self.augment_rushi = AUG()
        self.question_to_label = {} # contans all the augmented and orignal questions mapped to their labels
        self.answer_to_label = {} #  contains mapping form answer to labels
        # current data is to be filled using the fit_FAQ function call
        # it has 3 keys 1) embeddings  (a np array) 2) labels 3) label_to_answer dict

        if(self.check_faq_path()):
            print("found preexisiting faq data , loading dicts from the same")
            que_path = os.path.join(self.faq_path, "questions.pkl")
            self.question_to_label = load_dict(que_path)
            
            ans_path = os.path.join(self.faq_path, "answers.pkl")
            self.answer_to_label = load_dict(ans_path)

        else:
            assert not faq_data is None , "Did not find and preexisting of {} so you must provide faq_data".format(faq_name)
            self.make_faq(faq_data)
            
    
    def check_faq_path(self):
        if(os.path.exists(self.faq_path) == False):
            return False

        files = [ "questions.pkl" ,  "answers.pkl"]

        for f in files:
            pth = os.path.join(self.faq_path, f)
            if(not os.path.exists(pth)):
                return False
        return True

    def destroy_faq(self):
        if(os.path.exists(self.faq_path) == False):
            return
        
        files = [ "questions.pkl" ,  "answers.pkl"]

        for f in files:
            pth = os.path.join(self.faq_path, f)
            if(os.path.exists(pth)):
                os.remove(pth)
        os.rmdir(self.faq_path)


    def make_faq(self,FAQ):
        """
        FAQ is a dictionary has 2 keys.....
        1) question_to_label
        2) answer_to_label

        question_to_labels is again a dictionary from questions : label(int)  can have multiple questions for same label
        answer_to_labels is a dictionary from answers to label : one label per answer (strict !!!)
        """
        self.destroy_faq()
        os.mkdir(self.faq_path)
        question_to_label  = FAQ['question_to_label']
        answer_to_label = FAQ['answer_to_label']

        q_labels = set()
        a_labels = set()

        for q , l in question_to_label.items():
            q_labels.add(l)
        

        for a, l in answer_to_label.items():

            if(l not in q_labels):
                warnings.warn("Some labels in answers are not in the questions, these answers will never be a part of answers from the FAQ !!!")
                print("label {} not in questions".format(l))

        
        aug_question_to_label = dict()
        
        for q,l in question_to_label.items():
            # Damien , here also incorporate the other pipeline ....
            gen_ques = self.augment_rushi(q,10)
            # gen_ques are generated questions , a list , you need to append other gengerated ques to this list
            """
                invoke your function for augmentation pipeline here....
                and append results to the gen_ques list..

                Thank you
            """
            aug_question_to_label[q] = l
            for a_q in gen_ques:
                aug_question_to_label[a_q] = l
            #  note that ifthe augmentation yields the same question, as a result it will not be added....

        self.question_to_label = aug_question_to_label
        self.answer_to_label = answer_to_label
        save_dict(self.question_to_label, os.path.join(self.faq_path,"questions.pkl"))
        save_dict(self.answer_to_label , os.path.join(self.faq_path, 'answers.pkl'))

                




    
    def train(self,model_save_path, data =None):
        """

        questions = ['Q1', 'Q2', 'Q3', ....]
        labels = [1,2,3,1,4,8,9,10]

        generated_ques = {'Q1' : ['GQ1-1', 'GQ1-2', ...] , 'Q2' : ['GQ2-1', ...]}
        bs : 32
        n : 4

        model_save_path : './models/model_first'
        data --> a dict {'question_to_label' : mapping from question to label,
                        'bs': batch_size for training
                        'n' : num_of_classes to sample in a batch (bs%n ==0), 
                        }
        model_save_path = path of folder to save the model 

        This function will fit you data using a batch hard triplet loss, and save the model to the folder specified
        the folder should be empty or not created in the  beginning!!!

        if data is NONE 
        we will just use the presaved data
        """
        if(data is None):
            data = {'question_to_label' : self.question_to_label ,"bs": 32 , "n" : 4}
            

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


    

    def fit_FAQ(self,question_to_label = None , answer_to_label = None):
        """
        dataset
        Q1 --> A1
        Q2 --> A1 ....

        dct1 --> {'Q1' : 1, 'Qn': 1}

        dct2 --> {'A1': 1}

        A function to fit any given FAQ
        data is a dict, has keys
            1) question_to_label --> a dict ('string question' : label)
            2) answer_to_label --> a dict ('string answer' : label)
        

        if you provide None , then it will just use the data stored in the current faqs name

        """
        if(question_to_label is None):
            question_to_label = self.question_to_label
        if(answer_to_label is None):
            answer_to_label = self.answer_to_label

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
                print("{} label present in answer but not in question".format(label))
                warnings.warn('some labels in answers are not present in questions , you might not have labels in Sync')
            
        for l in labels:
            if(l not in label_to_answer):
                 warnings.warn('some labels in question are not present in answers ,this might cause runtime errors later, you might not have labels in Sync')
            


        self.current_faq  = {'embeddings' : self.model.encode(questions) , 'labels': labels, 'label_to_answer': label_to_answer, 'question_to_label' : question_to_label}


    def answer_question(self, question, K = 1 , cutoff = .3):
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
        question_to_label = self.current_faq['question_to_label']

        question = self.model.encode([question])[0].reshape(1,-1)
        # question is now a np.ndarray of shape (1,embedding_dim)
        
        cosine_sim = self.cosine_sim(question, embeddings)[0]
        #cosine_sim --> shape (N,)

        cosine_sim = cosine_sim.tolist()
        inds = [x for x in range(len(cosine_sim))]
        inds.sort(reverse = True, key = lambda x : cosine_sim[x])
        inds = inds[:K]
        # we need to pick the top k answers

        max_val = cosine_sim[inds[0]]
        print(max_val)
        if(max_val < cutoff):
            return "out of set question"
        
        labels = [question_labels[x] for x in inds]
        majority  = {}

        for label in labels:
            if(label not in majority):
                majority[label] = 0

            majority[label] += 1    
        
        cnt = -1
        ans = -1

        for label, count in majority.items():
            if(count > cnt):
                cnt = count
                ans = label



       
        if(label not in label_to_answer):
            return 'No answer corrosponding to the label {} , this means your question--label--answer dict is faulty '.format(label)
        
        for que, lab in question_to_label.items():
            if(lab == label):
                print("Answering {}".format(que))
        return label_to_answer[label]        









        



        






    
