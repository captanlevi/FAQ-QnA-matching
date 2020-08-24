import numpy as np
from googletrans import Translator
from .ngram_combine import sortByScore, getProb
from .utils import joinEnSeg, getTranslator
import time
import string

# random combine for one list of segments
def randomCombineForOneSeg(chSeg, seed):
    rng = np.random.RandomState(seed)
    # at least it leaves 2 segmentations, at most it leaves len-1 segmentations
    segNum = rng.choice(np.array(range(2, len(chSeg))))
    indices = sorted(rng.choice(len(chSeg), segNum, replace=False))
    ret = []
    for i,idx in enumerate(indices):
        # extract the segmentations that need to be combined
        if i==0:
            tmp = chSeg[:idx+1]
        else:
            tmp = chSeg[indices[i-1]+1:idx+1]
        ret.append(''.join(tmp))

        if i==len(indices)-1 and idx!=len(chSeg)-1:
            # if it was the last idx and it hadn't reached the end of chSeg
            # then combine the rest together.
            tmp = chSeg[idx+1:]
            ret.append(''.join(tmp))
    
    return ret

def randomCombine(chSeg, lm, permNum):
    # return the random combination of chinese characters
    randomCombinations = []
    for i in range(permNum):
        randomSeg = randomCombineForOneSeg(chSeg, i+len(chSeg))
        prob = getProb(randomSeg, lm)
        randomCombinations.append((randomSeg, prob))

    # sort combinations and sentences by their probabilities
    # ret1 and ret2 are both list that contains many tuples
    ret, randomCombinations, scores = sortByScore(randomCombinations)

    # return broken_english_sentences, randomCombinations
    return ret, randomCombinations




if __name__ == "__main__":
    import string
    from googletrans import Translator
    import kenlm
    import jieba
    
    punc = string.punctuation
    chineseStr = 'zh-cn'
    englishStr = 'en'
    
    translator = Translator(service_urls=['translate.google.cn'])
    
    lm_path = './lm/lm3.arpa'
    model = kenlm.Model(lm_path)
    ch = '我被裁了。我如何申请更多的资金和贷款帮助'
    print('Chinese translation: '+ch)
    
    chSeg = list(jieba.cut(ch)) # every element of this list is a chinese word group
    print('Chinese translation after cutting: '+str(chSeg))
    broken_english_sentences, segs_combinations = randomCombine(model, translator, chSeg, punc, chineseStr, englishStr,permNum=2)