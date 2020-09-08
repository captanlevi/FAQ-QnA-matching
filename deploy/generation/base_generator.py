import os
import sys
import itertools
import random
import nlpaug
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import re
from rajat_work.qgen.generator.symsub import SymSubGenerator
from rajat_work.qgen.generator.fpm.fpm import FPMGenerator
#from rajat_work.qgen.encoder.universal_sentence_encoder import USEEncoder
from rajat_work.qgen.encoder.dummy import dummyEN
from rajat_work.qgen.generator.eda import EDAGenerator
from broken_english.broken_english_generator import BrokenEnglishGen
import multiprocessing as mp
import pandas as pd
import numpy as np

class AUG():
    def __init__(self):
        #aug0 = naw.RandomWordAug()
        aug1 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
        aug2 = naw.SynonymAug(aug_src='wordnet')
        #aug3 = naw.SplitAug()
        aug4 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")

        self.augs = [aug1,aug2,aug4]
        
    def __call__(self,sent,n= 1):
        return self.augment(sent,n)
        
    
    def augment(self,sent : list ,n : int = 10) -> dict:
        ans = []
        sent = re.sub(r'[^a-zA-Z0-9_ ]', '', sent)
        for _ in range(n):
            aug = random.choice(self.augs)
            ans.append(aug.augment(sent))
                    
        return ans

    def batch_generate(self,questions : list) -> dict:
        result = dict()

        for q in questions:
            result[q] = self.augment(q)

        return result



class QuestionGenerator:
    def __init__(self, name : str, producer):
        self.name = name
        self.producer = producer

    def generate_n(self,questions: list,n : int) -> list:
        """
        takes as input a list of questions

        the producer is an object that has a method , batch_generate

        batch generate takes in a list of questions as input and 
        outputs a dict .....
        the dict maps each orignal question to a list of generated questions
        """
        print("working with {} pipeline".format(self.name))
        if("exact_batch_generate" in dir(self.producer)):
            result_dict = self.producer.exact_batch_generate(questions,n)
        else:
            result_dict = self.producer.batch_generate(questions)
        if(result_dict is None):
            answer = dict()
            for q in questions:
                answer[q] = []
            return answer

        for orignal_question in result_dict:
            result_dict[orignal_question] = result_dict[orignal_question][:n]

        return result_dict

        
        
    



USE_PATH = os.path.join('model_cache','universal_sentence_encoder')
AUG_cache = AUG()

class RushiSymsub(QuestionGenerator):
    def __init__(self):
        """
        encoder can be any sentence encoder that returns , a vector for a sentence
        and has the method "get_vectors"
        this method takes in a list of string and returns a list of vectors
        """
        super().__init__("symsub model",SymSubGenerator(dummyEN("lite")))
        #USEEncoder(USE_PATH)


class RushiFuzzy(QuestionGenerator):
    def __init__(self):
        super().__init__("FPM matcher", FPMGenerator())
    
  

class RushiEDA(QuestionGenerator):
    def __init__(self):
        super().__init__("EDA gen",EDAGenerator())
    

class RushiAUG(QuestionGenerator):
    def __init__(self):
        super().__init__("AUG", AUG())
        self.aug = AUG_cache
    

class RushiBroken(QuestionGenerator):
    def __init__(self):
        super().__init__("BrokenEnglish", BrokenEnglishGen())

def worker_function(producer_class : QuestionGenerator ,questions: list,to_generate : int):
    """
    making the actual multiprocessing work , by instancitaing producer_object and starting the generate process
    """
    producer_object = producer_class()
    #Q.put(producer_object.generate_n(questions = questions,n = to_generate))
    return producer_object.generate_n(questions=questions, n = to_generate)

def multiProcessControl(producer_classes : list, questions : list):
    """
    IMPORTANT producer classes is a list of lists , [Class, number of quetions to generate] classes and not objects, the objects will be instiantiated later
    also make the classes so that, they do not require any argument in init 

    questions is just a list of strings 
    """


    args = []
    for p in producer_classes:
        args.append((p[0],questions,p[1]))
    with mp.Pool(len(producer_classes)) as pool:
        results = pool.starmap(worker_function, args)

   
    
    merged_dict = dict()
    for dct in results:
        for que,gen_ques in dct.items():
            if(que not in merged_dict):
                merged_dict[que] = []
            merged_dict[que].extend(gen_ques)
            
    
    return merged_dict



if __name__ == "__main__":
    #p1 = [RushiSymsub,10]
    #p2 = [RushiAUG,12]
    #p3 = [RushiFuzzy,6]
    p4 = [RushiBroken,6]
    covid_df = pd.read_csv("../../data/covid19data/msf_covid19.csv", header = None)
    covid_questions = {}
    covid_answers = {}
    label = 0
    nm = 10
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
        if(nm == 0):
            break
        nm -= 1
    questions = list(covid_questions.keys())

    #with mp.Pool(2) as pool:
    #    results = pool.starmap(worker_function, [(p1[0],questions,5), (p2[0],questions,4)])
    
    print(multiProcessControl([p4],questions))
    #print(results)