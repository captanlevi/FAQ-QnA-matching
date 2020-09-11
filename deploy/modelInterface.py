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

sys.path.append("./generation")
sys.path.append("./generation/rajat_work")
sys.path.append("./generation/rajat_work/qgen")

from generation.base_generator import RushiAUG, RushiEDA, RushiFuzzy, RushiSymsub,RushiBroken,multiProcessControl


producer_classes = [(RushiAUG,2),(RushiEDA,3),(RushiFuzzy,4),(RushiSymsub,4),(RushiBroken,4)]




def save_dict(obj, path ):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class LabelsSyncException(Exception):
    """To raise exception when labels are not in sync (some labels in questions not in answers, as it will
    cause runtime errors 
    )"""

class modelInterface:
    def __init__(self, faq_path : str ,faq_data : dict = None,model_path : str = None):
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
        example
        
            q2l = {"How are you doing " : 1 , "where are you ? ": 3}
            a2l = {"I am fine" : 1 , "I am in India": 3}
            faq_data = {"questiontolabel" : q2l , "answertolabel" : a2l}

        
        """
        if(model_path == None):
            model_path = 'roberta-base-nli-stsb-mean-tokens'

        self.model = SentenceTransformer(model_path)
        self.current_faq = None
        self.faq_path = faq_path
        self.question_to_label = {} # contans all the augmented and orignal questions mapped to their labels
        self.answer_to_label = {} #  contains mapping form answer to labels
        # current data is to be filled using the fit_FAQ function call
        # it has 3 keys 1) embeddings  (a np array) 2) labels 3) label_to_answer dict
        self.label_to_answer = {}
        self.label_to_orignal = {}

        if(self.check_faq_path()):
            print("found preexisiting faq data , loading dicts from the same")
            question_to_label_path = os.path.join(self.faq_path, "question_to_label.pkl")
            self.question_to_label = load_dict(question_to_label_path)
            
            answer_to_label_path = os.path.join(self.faq_path, "answer_to_label.pkl")
            self.answer_to_label = load_dict(answer_to_label_path)

            label_to_answer_path = os.path.join(self.faq_path , "label_to_answer.pkl")
            self.label_to_answer = load_dict(label_to_answer_path)

            label_to_orignal_path = os.path.join(self.faq_path, "label_to_orignal.pkl")
            self.label_to_orignal = load_dict(label_to_orignal_path)

        else:
            print("Did not find preexisting faq data, this can also mean that the preexisting data is corrupt or has some files missing !!!")
            self.destroy_faq()
            assert not faq_data is None , "Did not find and preexisting of {} so you must provide faq_data".format(faq_data)
            self.make_faq(faq_data)
            
    
    def check_faq_path(self):
        if(os.path.exists(self.faq_path) == False):
            return False

        files = [ "question_to_label.pkl" ,  "answer_to_label.pkl", "label_to_answer.pkl","label_to_orignal.pkl"]

        for f in files:
            pth = os.path.join(self.faq_path, f)
            if(not os.path.exists(pth)):
                return False
        return True

    def destroy_faq(self):
        if(os.path.exists(self.faq_path) == False):
            return
        
        files = os.listdir(self.faq_path)

        for f in files:
            pth = os.path.join(self.faq_path, f)
            if(os.path.exists(pth)):
                os.remove(pth)
        os.rmdir(self.faq_path)


    def make_faq(self,FAQ : dict):
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

        # creating inverse mapping
        label_to_answer = dict()
        
        for q , l in question_to_label.items():
            q_labels.add(l)

        for a, l in answer_to_label.items():
            a_labels.add(l)
            if(l not in q_labels):
                warnings.warn("Some labels in answers are not in the questions, these answers will never be a part of answers from the FAQ !!!")
                print("label {} not in questions".format(l))

            if(l in label_to_answer):
                raise LabelsSyncException("Multiple labels for same answer, {} ,{}".format(a,l))
            label_to_answer[l] = a

      
        
        for l in q_labels:
            if(l not in a_labels):
                raise LabelsSyncException('some labels in question are not present in answers ,this might cause runtime errors later, you might not have labels in Sync ------ label --> {}'.format(l))
                  
        
        aug_question_to_label = dict()
        label_to_orignal = dict()
        
        generated_dict = multiProcessControl(producer_classes, list(question_to_label))  # this is a mapping from question to a list of generated questions

        for que, label in question_to_label.items():
            aug_question_to_label[que] = label
            if(que not in generated_dict):
                print("Some of the questions in the FAQ are missing after generation")
                continue
            gens = generated_dict[que]
            label_to_orignal[label] = que
            for q in gens:
                aug_question_to_label[q] = label


        
        
       
        self.question_to_label = aug_question_to_label
        self.answer_to_label = answer_to_label
        self.label_to_orignal = label_to_orignal
        self.label_to_answer = label_to_answer




        save_dict(self.question_to_label, os.path.join(self.faq_path,"question_to_label.pkl"))
        save_dict(self.answer_to_label , os.path.join(self.faq_path, 'answer_to_label.pkl'))
        save_dict(self.label_to_answer , os.path.join(self.faq_path, 'label_to_answer.pkl'))
        save_dict(self.label_to_orignal , os.path.join(self.faq_path, 'label_to_orignal.pkl'))

                




    
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
            _, predicted_label,_ = self.answer_question(q, verbose = False,K = K, cutoff = cutoff)
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
        label_to_answer = self.label_to_answer
        
        questions = []
        labels = []
        for q,l in question_to_label.items():
            questions.append(q)
            labels.append(int(l))
        

        # Now inverting answer_to_label
        """
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
                raise LabelsSyncException('some labels in question are not present in answers ,this might cause runtime errors later, you might not have labels in Sync')
            
        """

        # I am saving labels so that the ordered is remembered , as the dict in python cant be trusted for the order !!!
        self.current_faq  = {'embeddings' : np.array(self.model.encode(questions)) , 'labels': labels}
        save_dict(self.current_faq, save_path)

    def answer_question(self, question, K = 1 , cutoff = .3, verbose = False):

        """
            This is where you ask the question , and a approropriate answer is returned,
            must call fit_FAQ before this
            question ==> string
            returns ==> (answer : string , status : int , [list of similar questions])   ------ the answer and the label to the question the answer belongs to.....
            if the status is -1 then the answer is out of set 
            else the status is the label of the question (for debugginh and testing only)
        """
        if (self.current_faq is None):
            assert False , 'Need to fit_FAQ before calling answer_question'
        
        embeddings = self.current_faq['embeddings']
        question_labels = self.current_faq['labels']
        

        question = self.model.encode([question])[0].reshape(1,-1)
        # question is now a np.ndarray of shape (1,embedding_dim)
        
        cosine_sim = self.cosine_sim(question, embeddings)[0]
        #cosine_sim --> shape (N,)
        
        cosine_sim = cosine_sim.tolist()
        inds = [x for x in range(len(cosine_sim))]
        inds.sort(reverse = True, key = lambda x : cosine_sim[x])

        """
        Writing code for suggested questions , ie the most similar questions.....
        Getting most similar 5  questions
        """
        ########################################################
        similar_labels = set()
        for index in inds:
            if(question_labels[index] not in similar_labels and question_labels[index] != question_labels[0]):
                similar_labels.add(question_labels[index])
            if(len(similar_labels) > 5):
                break
        
        similar_questions = [self.label_to_orignal[l] for l in similar_labels]
        
        ########################################################



        inds = inds[:K]
        # we need to pick the top k answers

        max_val = cosine_sim[inds[0]]

        if(max_val < cutoff):
            return "out of set question" ,-1, similar_questions
        
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
        if(ans not in self.label_to_answer):
            return ("label {} not found in label_to_answer , this shoud not have happened unless you misused the code , please redo the whole FAQ" , -1, [])
        
        if(verbose):
            print(max_val)
            print(label_to_conf)
            print("MAX label is {}".format(ans))
        for que, lab in self.question_to_label.items():
            if(lab == ans):
                if(verbose):
                    print("Answering {}".format(que))
        return self.label_to_answer[ans] , ans , similar_questions      




if __name__ == "__main__":



    """
    covid_df = pd.read_csv("../data/covid19data/msf_covid19.csv", header = None)
    covid_questions = {}
    covid_answers = {}
    label = 0

    for q,a in zip(covid_df[1], covid_df[2]):
        q = q.split('\n')[0]
        a = a.split('\n')[0]
        if(a in covid_answers):
            l = covid_answers[a]
            covid_questions[q] = l
        else:
            covid_questions[q] = label
            covid_answers[a] = label
            label += 1

    
    covid_data =  {"question_to_label" : covid_questions ,"answer_to_label" : covid_answers}
    """
    
    orignal = load_dict("../Orignal_FAQs/comcare_orignal.pkl")
    modelInter = modelInterface(faq_path = "../FAQs/comcare", faq_data = orignal)

    

