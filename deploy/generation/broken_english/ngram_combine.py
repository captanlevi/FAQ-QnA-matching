import jieba
import pickle
import math
import re
from .utils import getNgramWordList
import time

# form a big list of words
def formListOfWords(path, from_scratch=False):
    if from_scratch:
        chUnigramStart = 21311
        chUnigramEnd = 284082
        chBigramStart = 683422
        chBigramEnd = 11608957
        ret = getNgramWordList(path, 5000, chUnigramStart, chUnigramEnd)
        ret2 = getNgramWordList(path, 50000, chBigramStart, chBigramEnd)
        ret.update(ret2)
        with open('C:\\Users\\rjkin\\Desktop\\NTU_thesis\\cluster-model\\deploy\\generation\\broken_english\\pkl\\ngramWordsList.pkl','wb') as f:
            pickle.dump(ret, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('C:\\Users\\rjkin\\Desktop\\NTU_thesis\\cluster-model\\deploy\\generation\\broken_english\\pkl\\ngramWordsList.pkl','rb') as f:
            ret = pickle.load(f)

    return ret

# get probability of a list of segments
def getProb(chSeg, lm):
    prob = 1.0
    for seg in chSeg:
        prob *= math.pow(10, lm.score(seg))
    return math.log10(prob)

# sort (segs,scores) tuples by their scores
def sortByScore(segs_scores_list):
    scores = []
    segs = []
    segs_sorted = []
    ret = []
    for t in segs_scores_list:
        segs.append(t[0])
        scores.append(t[1])
    s = sorted(scores, reverse=True)
    indices = [scores.index(i) for i in s]
    for idx in indices:
        ret.append((segs[idx],scores[idx]))
        segs_sorted.append(segs[idx])
    # ret is a list containing many tuples
    return ret, segs_sorted, scores

# given a list of segments and indices of segments that need to be combined, return a new segment
def updateSeg(chSeg, indices):
    if indices == ():
        return chSeg
    
    seg = []
    for i,idx in enumerate(indices):
        if i==0:
            for j in range(idx):
                seg.append(chSeg[j])
            seg.append(chSeg[idx]+chSeg[idx+1])

        if i!=0 and i!=len(indices)-1:
            for j in range(indices[i-1]+2,idx):
                seg.append(chSeg[j])
            seg.append(chSeg[idx]+chSeg[idx+1])

        if i==len(indices)-1:
            if len(indices)>1:
                for j in range(indices[i-1]+2,idx):
                    seg.append(chSeg[j])
                seg.append(chSeg[idx]+chSeg[idx+1])
            for j in range(idx+2,len(chSeg)):
                seg.append(chSeg[j])

    return seg

def ngramCombine(chSeg, ngram_words, lm):
    ret = [] # the final combination results. it's a list containing many tuples. elements of these tuples are new word segs and probability
    comb = [] # for example, if chSeg=['a','b','c'] then comb=['ab','bc']
    indices = [] # indices of words in comb that exist in ngram words list
    for i in range(len(chSeg)-1):
        w = chSeg[i]+chSeg[i+1]
        comb.append(w)
        if w in list(ngram_words.keys()):
            indices.append(i)

    if indices==[]:
        # all combinations are not in ngram words list
        # print('indices=[]')
        return (chSeg, None), False
    else:
        toCombine = tuple(indices) # combine all word pairs appearing in ngram words list
        # update the original word seg based on the combination index toCombine
        new_seg = updateSeg(chSeg, toCombine)
        prob = getProb(new_seg, lm)
        ret.append((new_seg, prob))
        # recursively find all the candidate combinations
        r, sign = ngramCombine(new_seg, ngram_words, lm)
        if sign:
            ret = ret + r

            # delete the repeated segs
            ret2 = []
            for i in ret:
                sign2 = True
                for j in ret2:
                    if j[0] == i[0]:
                        sign2 = False
                        break
                if sign2:
                    ret2.append(i)
        else:
            ret2 = ret
        
        # sort by score
        ret, segs, scores = sortByScore(ret2)

        return ret, True


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
    ch = '我一下子就好了起来。我怎么这么好看！'
    print('Chinese translation: '+ch)
    
    chSeg = list(jieba.cut(ch)) # every element of this list is a chinese word group
    print('Chinese translation after cutting: '+str(chSeg))
    ret, sign = ngramCombine(chSeg, formListOfWords('C:\\Users\\rjkin\Desktop\\NTU_thesis\\cluster-model\\deploy\\generation\\broken_english\\lm\\lm3.txt',from_scratch=False), model)
    print(sign)