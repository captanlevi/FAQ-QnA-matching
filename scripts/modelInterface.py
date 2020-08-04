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



def save_dict(obj, path ):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class modelInterface:
    def __init__(self, faq_path ,faq_data = None,model_path = None):
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
        self.faq_path = faq_path
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
            assert not faq_data is None , "Did not find and preexisting of {} so you must provide faq_data".format(faq_data)
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
        
        files = [ "questions.pkl" ,  "answers.pkl", "fit.pkl"]

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
            gen_ques = self.augment_rushi(q,6)
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

    def evaluate(self,data, K = 5, cutoff = .6):
        """
        Will evaluate model , on a given test_set
        data is a dict from test_questions to labels 
        MAKE SURE THAT THE LABELS ARE IN SYNC WITH THE ONES YOU USED FIT MODEL ON!!!
        """
        correct = 0
        for q,l in data.items():
            _, predicted_label = self.answer_question(q, verbose = False,K = K, cutoff = cutoff)
            if(int(predicted_label) == int(l) ):
                correct += 1

        return correct/len(data)

        # converting to array



    def unfit_FAQ(self):
        savepath = os.path.join(self.faq_path, "fit.pkl")
        if(os.path.exists(savepath)):
            os.remove(savepath)
        

    

    def fit_FAQ(self):
        """
        Will calculate the vectors of all the questions
        and store them in a file "fit.pkl",
        if the file already exists then , will directely fetch data from there.....

        To make changes , is 
            IF YOU HAVE TRAINED A NEW MODEL AND WANT TO FIT AGAIN....
            PLEASE CALL UNFIT_MODEL FIRST...
        """

        save_path = os.path.join(self.faq_path, "fit.pkl")
        if(os.path.exists(save_path)):
            warnings.warn("Found existing fit.pkl loading diles from there .....  if you have trained the model recently and want to use that model to fit, please call unfit_model... ")
            self.current_faq = load_dict(save_path)
            return



        question_to_label = self.question_to_label
        answer_to_label = self.answer_to_label

        questions = []
        labels = []
        for q,l in question_to_label.items():
            questions.append(q)
            labels.append(int(l))
        

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
            


        self.current_faq  = {'embeddings' : np.array(self.model.encode(questions)) , 'labels': labels, 'label_to_answer': label_to_answer, 'question_to_label' : question_to_label}
        save_dict(self.current_faq, save_path)

    def answer_question(self, question, K = 1 , cutoff = .3, verbose = True):

        """
            This is where you ask the question , and a approropriate answer is returned,
            must call fit_FAQ before this
            question ==> string
            returns ==> (string , int)   ------ the answer and the label to the question the answer belongs to.....
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

        if(max_val < cutoff):
            return "out of set question" ,-1
        
        labels = [question_labels[x] for x in inds]
        confs = [cosine_sim[x] for x in inds]
        label_to_conf  = {}

        ans = -1
        mx = -1
        for l ,conf in zip(labels,confs):
            if(l not in label_to_conf):
                label_to_conf[l] = 0
            label_to_conf[l] += conf
            if(label_to_conf[l] > mx):
                mx = label_to_conf[l]
                ans = l


        """
        print(labels)
        print(confs)
        majority  = {}

        for label in labels:
            if(label not in majority):
                majority[label] = 0
            
            majority[label] += 1
        
        cnt = -1
        ans = -1

        print(majority)
        for label, count in majority.items():
            if(count > cnt):
                cnt = count
                ans = label



        """
        if(ans not in label_to_answer):
            return 'No answer corrosponding to the label {} , this means your question--label--answer dict is faulty '.format(ans)
        
        if(verbose):
            print(max_val)
            print(label_to_conf)
            print("MAX label is {}".format(ans))
        for que, lab in question_to_label.items():
            if(lab == ans):
                if(verbose):
                    print("Answering {}".format(que))
        return label_to_answer[ans] , ans       
