import os
from typing import List, Tuple, Dict

from numpy.testing._private.utils import assert_raises
from .core.FAQ import Answer, FAQ, Question , FAQUnit, FAQOutput
import numpy as np
from sentence_transformers.readers import InputExample
from sentence_transformers import  SentenceTransformer,SentencesDataset
from .core.exceptions import *
from .modelRelated.utils import cosineSim
from .modelRelated.lossHandler import LossHandler
import warnings

class FAQWrapper:
    def __init__(self,id : int ,  FAQ : FAQ):
        self.FAQ = FAQ
        self.id = id
        self.vectors = self.getVectors()


    def getVectors(self) -> np.ndarray:
        faq = self.FAQ.FAQ
        vectors = []
        for unit in faq:
            vector = unit.vectorRep
            if(vector is None):
                raise VectorNotAssignedException()
            vectors.append(vector)
        

        return np.array(vectors)



    def _getClosestQuestions(self,rankedIndices : List[int] ,K : int, topAnswer : str):
        includedSet = set()
        includedSet.add(topAnswer)

        closestQuestions = []
        for ind in rankedIndices:
            currentUnit = self.FAQ.FAQ[ind]
            currentOrignal = currentUnit.orignal.text

            if(currentOrignal not in includedSet):
                includedSet.add(currentOrignal)
                closestQuestions.append(currentOrignal)

            if(len(closestQuestions) == K):
                break

        return closestQuestions

            


        

    def solveForQuery(self,queryVector : np.ndarray, K : int, topSimilar : int = 5)  -> FAQOutput:
        # queryVector has shape (1,emeddingDim)
        if(len(queryVector.shape) == 1):
            queryVector = queryVector.reshape(1,-1)

        assert queryVector.shape[0] == 1 

        cosineScores = cosineSim(queryVector, self.vectors)[0]

        cosineScores = cosineScores.tolist()
        rankedIndices  = [x for x in range(len(cosineScores))]
        rankedIndices.sort(reverse = True, key = lambda x : cosineScores[x])

        maxScore = cosineScores[rankedIndices[0]]
        # Now rankedIndices hold the order of indices with highest to lowest similirity !!!


        competeDict = dict()
        for ind in rankedIndices[:K]:
            # using top K results !!!
            currentlabel = self.FAQ.FAQ[ind].label
            if(currentlabel not in competeDict):
                competeDict[currentlabel] = 0
            competeDict[currentlabel] += cosineScores[ind]

        
        competeList = [(label,score) for label , score in competeDict.items()]
        competeList.sort(key= lambda x : x[1] , reverse= True)

        bestScore = competeList[0][1]
        bestLabel = competeList[0][0]

        bestAnswer = self.FAQ.getAnswerWithLabel(bestLabel)
        bestMatchQuestion = self.FAQ.getQuestionWithLabel(bestLabel)

        return  FAQOutput(faqId= self.id,faqName= self.FAQ.name, answer = bestAnswer,
            question= bestMatchQuestion , score= bestScore,
            similarQuestions=self._getClosestQuestions(rankedIndices,topSimilar,bestMatchQuestion.text) , maxScore= maxScore)
    

        
        




class Bani:
    def __init__(self,FAQs : List[FAQ], modelPath : str = None, assignVectors : bool = True):
        if(modelPath == None):
            modelPath = 'roberta-base-nli-stsb-mean-tokens'
        self.model : SentenceTransformer = self._getModel(modelPath)
        self.FAQs : List[FAQWrapper] = []
        self.idToFAQ : Dict[int,FAQWrapper] = dict()
        self.assignVectors = assignVectors

        self._registerFAQs(FAQs = FAQs)


    def _registerFAQs(self,FAQs : List[FAQ]):
        """
        registers all the faqs given and then extracts vectors , and forms a 
        gobal index and vector to use for combined question answering 
        """

        assert len(FAQs) > 0
        # All FAQs should be Usable !!!
        for faq in FAQs:
            if(faq.isUsable() == False):
                raise ValueError("All faqs passed to chadBot must be Usable !!!! please build FAQ again or load from preexisting one")
        

        for faq in FAQs:
            if(faq.hasVectorsAssigned() == True  and self.assignVectors == False):
                warnings.warn("Vectors already assigned to {} FAQ , if you want to reassign using the current model please clear the vectors using resetAssigned vectors".format(faq.name))
            elif(faq.hasVectorsAssigned() == True and self.assignVectors == True):
                print("OverWritting the vectors of FAQ named {} , it will name save the generated vectors , to do that use the saveFAQ/s feature".format(faq.name))
                faq._assignVectors(model = self.model)
            elif(faq.hasVectorsAssigned() == False and self.assignVectors == True):
                print("Assigning vectors to {} faq , , it will name save the generated vectors , to do that use the saveFAQ/s feature".format(faq.name))
                faq._assignVectors(model = self.model)
            else:
                raise VectorNotAssignedException("assignVectors is False , but the vectors are not stored in faq named {}".format(faq.name))
                
               


        id = 0
        for faq in FAQs:
            newFAQ = FAQWrapper(id,faq)
            self.FAQs.append(newFAQ)
            self.idToFAQ[id] = newFAQ
            id += 1



    def findClosest(self,query : str,  K : int = 3 , topSimilar : int = 5) -> List[FAQOutput]:
        """
        Here we find the closest from each faq and then compare of the 
        top contenders from different faqs are not dangerouusly similar

        """

        competeList : List[FAQOutput] = []
        queryVector = self.model.encode([query])[0].reshape(1,-1)

        for faq in self.FAQs:
            competeList.append(faq.solveForQuery(queryVector=queryVector, K = K, topSimilar= topSimilar))


        competeList.sort(key = lambda x : x.score, reverse= True)
        # competeList now has answer from each faq in the descending order
        return competeList

         
    def findClosestFromFAQ(self,faqId : int, query : str, K : int = 3, topSimilar : int = 5) -> FAQOutput:
        assert faqId in self.idToFAQ , "The id {} not in faqId only {} are available".format(faqId,list(self.idToFAQ.keys()))
        faq = self.idToFAQ[faqId]
        queryVector = self.model.encode([query])[0].reshape(1,-1)
        return faq.solveForQuery(queryVector= queryVector, K = K,topSimilar= topSimilar)


    def _tester(self,faqId : int, questions : List[str],  labels : List[int] , K : int = 3):
        correct = 0
        for question, label in zip(questions,labels):
            answer = self.findClosestFromFAQ(faqId,question)
            if(answer.question.label == label):
                correct += 1

        return correct/len(questions)



    def test(self,faqId : int,testData : List[Tuple[str,str]], K : int = 3) -> float:
        """
        Interface to test any given faq , expects a list of tuples of size 2
        first element is the orignal question and second is the paraphrased version.
        All the orignal question should ideally match the questions in the FAQ
        """
        assert faqId in self.idToFAQ , "The id {} not in faqId only {} are available".format(faqId,list(self.idToFAQ.keys()))
        questions : List[str] = []
        labels : List[str] = []

        nonMatched : List[str] = []
        testFAQ = self.idToFAQ[faqId].FAQ
        for testPoint in testData:
            orignal = testPoint[0]
            reframed = testPoint[1]
            flag = 0
            for unit in testFAQ.FAQ:
                if(unit.orignal.text.strip().lower() == orignal.strip().lower()):
                    questions.append(reframed)
                    labels.append(unit.label)
                    flag = 1
                    break
            if(flag == 0):
                nonMatched.append(orignal)

        if(nonMatched):
            warnings.warn("{} questions in the test set did not match any orignal questions".format(len(nonMatched)))

        print("Running test on {} questions".format(len(questions)))
        return self._tester(faqId= faqId,questions= questions, labels= labels,K= K)


    
    
    
    def train(self,outputPath : str,batchSize = 16, epochs : int = 1, lossName : str = "batchHardTriplet",**kwargs):
        """
        Trains the model using batch hard triplet loss , 
        for the other kwargs take a look at the documentation for sentencetransformers
        """
        os.makedirs(outputPath, exist_ok=True)
        trainingObjectives = [] # training each faq on a different objective
        
        for faq in self.FAQs:
            lossInstance = LossHandler(lossName= lossName, model= self.model, FAQ = faq.FAQ, batchSize= batchSize)
            trainDataloader = lossInstance.trainDataLoader
            trainLoss = lossInstance.trainLoss
            trainingObjectives.append((trainDataloader, trainLoss))

        self.model.fit(train_objectives=  trainingObjectives,epochs= epochs, save_best_model= False,output_path= outputPath, **kwargs)
        self.saveModel(outputPath)
        self.model = SentenceTransformer(outputPath)
        for faq in self.FAQs:
            print("Assigning vectors from the trained model to FAQ {}".format(faq.FAQ.name))
            faq.FAQ._assignVectors(self.model)



    def getFAQWithId(self, id : int) -> FAQ:
        if(id not in self.idToFAQ):
            raise KeyError("FAQ with id {} does not exist".format(id))

        return self.idToFAQ[id].FAQ

    def saveModel(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save(path)


    def _getModel(self, path):
        return SentenceTransformer(path)

    def saveFAQs(self, rootDirPath : str):
        for faq in self.FAQs:
            coreFaQ = faq.FAQ
            coreFaQ.save(rootDirPath)
    def saveFAQ(self, id : int, rootDirPath : str):
        assert id in self.idToFAQ
        self.idToFAQ[id].FAQ.save(rootDirPath)

    
        


if __name__ == "__main__":
    print("HEllo")







        





       





    









        
        
        


    
    


    




    

    











