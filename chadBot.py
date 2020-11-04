import os
from typing import List, Tuple, Dict

from numpy.testing._private.utils import assert_raises
from core.FAQ import Answer, FAQ, Question , FAQUnit, FAQOutput
import numpy as np
from sentence_transformers.readers import InputExample
from sentence_transformers import  SentenceTransformer,SentencesDataset,losses
from core.exceptions import *
from modelRelated.utils import cosineSim  , convertForBatchHardTripletLoss
from torch.utils.data import DataLoader
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
            similarQuestions=self._getClosestQuestions(rankedIndices,topSimilar,bestMatchQuestion.text) )
    

        
        




class ChadBot:
    def __init__(self,FAQs : List[FAQ], modelPath : str = None):
        if(modelPath == None):
            modelPath = 'roberta-base-nli-stsb-mean-tokens'
        self.model : SentenceTransformer = SentenceTransformer(modelPath)
        self.FAQs : List[FAQWrapper] = []
        self.idToFAQ : Dict[int,FAQWrapper] = dict()

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
            if(faq.hasVectorsAssigned()):
                warnings.warn("Vectors already assigned to {} FAQ , if you want to reassign using the current model please clear the vectors using resetAssigned vectors".format(faq.name))
            else:
                print("Assigning vectors to {} faq".format(faq.name))
                faq._assignVectors(model = self.model)


        id = 0
        for faq in FAQs:
            newFAQ = FAQWrapper(id,faq)
            self.FAQs.append(newFAQ)
            self.idToFAQ[id] = newFAQ
            id += 1


    def findClosest(self,query : str,  K : int = 3 , topSimilar : int = 5) -> FAQOutput:
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
        assert faqId in self.idToFAQ
        faq = self.idToFAQ[faqId]
        queryVector = self.model.encode([query])[0].reshape(1,-1)
        return faq.solveForQuery(queryVector= queryVector, K = K,topSimilar= topSimilar)



    def train(self,outputPath : str, batchSize = 16, epochs : int = 1, **kwargs):
        """
        Trains the model using batch hard triplet loss , 
        for the other kwargs take a look at the documentation for sentencetransformers
        """
        assert batchSize > 4 and epochs > 0 and os.path.exists(outputPath)
        trainingObjectives = [] # training each faq on a different objective
        for faq in self.FAQs:
            trainExamples = convertForBatchHardTripletLoss(faq.FAQ)
            trainDataset = SentencesDataset(trainExamples,self.model)
            trainDataloader = DataLoader(trainDataset, shuffle=True, batch_size= batchSize)
            trainLoss = losses.BatchHardTripletLoss(model= self.model)
            trainingObjectives.append((trainDataloader, trainLoss))

        self.model.fit(train_objectives=  trainingObjectives, warmup_steps= 100,epochs= epochs, 
            output_path= outputPath, **kwargs)


    def saveFAQs(self, rootDirPath : str):
        for faq in self.FAQs:
            coreFaQ = faq.FAQ
            coreFaQ.save(rootDirPath)
    def saveFAQ(self, id : int, rootDirPath : str):
        assert id in self.idToFAQ
        self.idToFAQ[id].FAQ.save(rootDirPath)

    
        







        





       




if __name__ == "__main__":
    faq = FAQ(name = "babyBonus")
    faq.load('./')
    print(len(faq))
    bot = ChadBot(FAQs = [faq])
    bot.train(outputPath= "./checkpoints/dummy")
    print(bot.findClosest("is my child eligible for baby bonus ?").answer)
    









        
        
        


    
    


    




    

    











