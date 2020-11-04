import nlpaug
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import re
import random
from tqdm import tqdm

class AUG():
    def __init__(self):
        #aug0 = naw.RandomWordAug()
        aug1 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
        aug2 = naw.SynonymAug(aug_src='wordnet')
        #aug3 = naw.SplitAug()
        aug4 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")

        self.augs = [aug1,aug2,aug4]
        

        
    
    def augment(self,sent : list ,n : int = 10) -> dict:
        ans = []
        sent = re.sub(r'[^a-zA-Z0-9_ ]', '', sent)
        for _ in range(n):
            aug = random.choice(self.augs)
            ans.append(aug.augment(sent))
                    
        return ans

    def exact_batch_generate(self,questions : list, n : int) -> dict:
        result = dict()

        for q in tqdm(questions):
            result[q] = self.augment(q, n)

        return result