from torch import mode
from .utils import convertForBatchHardTripletLoss, convertForContrastiveLoss, convertForTripletLoss, convertForSoftmaxLoss
from .batchHardDataset import BatchHardDataset, PKSampler
from .softmaxDataset import SoftmaxDataset
from sentence_transformers import losses , SentenceLabelDataset, SentencesDataset
from torch.utils.data import DataLoader
from .customLosses import SoftmaxLayerLoss



class LossHandler:
    def __init__(self, lossName : str, FAQ, model, batchSize : int):
        supportedLosses = ("batchHardTriplet", "contrastiveLoss", "tripletLoss", "softmaxLayerLoss")
        assert lossName in supportedLosses  , "Error {} loss not supported , supported losses are {}".format(lossName, supportedLosses)

        if(lossName == supportedLosses[0]):
            assert batchSize > 8 and batchSize%8 == 0, "needed for the pk sampler (p ,k > 2), must give large batches for batch hard"

            self.trainExamples = convertForBatchHardTripletLoss(FAQ= FAQ)
            self.dataset = BatchHardDataset(examples = self.trainExamples , model = model)
            self.trainLoss = losses.BatchHardTripletLoss(model= model)
            sampler = PKSampler(dataSource= self.dataset, p = 8 , k = batchSize//8)
            self.trainDataLoader =  DataLoader(self.dataset, batch_size= batchSize, sampler= sampler)
        elif(lossName == supportedLosses[1]):
            assert batchSize > 4 and batchSize%4 == 0, "needed for good contrastive loss"
            
            self.trainExamples = convertForContrastiveLoss(FAQ = FAQ)
            self.dataset = SentencesDataset(self.trainExamples, model= model) 
            self.trainLoss = losses.ContrastiveLoss(model = model)
            self.trainDataLoader = DataLoader(self.dataset, shuffle=True, batch_size= batchSize)

        elif(lossName == supportedLosses[2]):
            self.trainExamples = convertForTripletLoss(FAQ = FAQ)
            self.dataset = SentenceLabelDataset(self.trainExamples , model= model)
            self.trainLoss = losses.TripletLoss(model= model , triplet_margin= 1)
            self.trainDataLoader = DataLoader(self.dataset, shuffle=True, batch_size= batchSize)
        elif(lossName == supportedLosses[3]):
            self.trainExamples, numLabels = convertForSoftmaxLoss(FAQ = FAQ)
            self.dataset = SoftmaxDataset(self.trainExamples, model= model)
            self.trainLoss = SoftmaxLayerLoss(model= model, numLabels= numLabels)
            self.trainDataLoader = DataLoader(self.dataset, shuffle= True, batch_size= batchSize)
            

        
        

   