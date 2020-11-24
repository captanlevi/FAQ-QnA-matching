from torch import mode
from .utils import convertForBatchHardTripletLoss, convertForContrastiveLoss, convertForTripletLoss
from .batchHardDataset import BatchHardDataset, PKSampler
from sentence_transformers import losses , SentenceLabelDataset, SentencesDataset
from torch.utils.data import DataLoader



class LossHandler:
    def __init__(self, lossName : str, FAQ, model, batchSize : int):
        supportedLosses = ("batchHardTriplet", "contrastiveLoss", "tripletLoss")
        assert lossName in supportedLosses  , "Error {} loss not supported , supported losses are {}".format(lossName, supportedLosses)

        if(lossName == supportedLosses[0]):
            assert batchSize > 4 and batchSize%4 == 0, "needed for the pk sampler (p ,k > 2)"

            self.trainExamples = convertForBatchHardTripletLoss(FAQ= FAQ)
            self.dataset = BatchHardDataset(examples = self.trainExamples , model = model)
            self.trainLoss = losses.BatchHardTripletLoss(model= model)
            sampler = PKSampler(dataSource= self.dataset, p = 4 , k = batchSize//4)
            self.trainDataLoader =  DataLoader(self.dataset, batch_size= batchSize, sampler= sampler)
        elif(lossName == supportedLosses[1]):
            self.trainExamples = convertForContrastiveLoss(FAQ = FAQ)
            self.dataset = SentencesDataset(self.trainExamples, model= model) 
            self.trainLoss = losses.ContrastiveLoss(model = model)
            self.trainDataLoader = DataLoader(self.dataset, shuffle=True, batch_size= batchSize)

        elif(lossName == supportedLosses[2]):
            self.trainExamples = convertForTripletLoss(FAQ = FAQ)
            self.dataset = SentenceLabelDataset(self.trainExamples , model= model)
            self.trainLoss = losses.TripletLoss(model= model , triplet_margin= 1)
            self.trainDataLoader = DataLoader(self.dataset, shuffle=True, batch_size= batchSize)
        
        

   