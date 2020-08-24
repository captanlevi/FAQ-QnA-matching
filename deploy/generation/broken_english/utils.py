import jieba
import pickle
import math
import re
from googletrans import Translator

# setup the translator
def getTranslator(set_proxy=True):
    if set_proxy:
        # translator = Translator(service_urls=['translate.google.com', 'translate.google.co.kr','translate.google.cn'], proxies={'http':'127.0.0.1:19180','https':'127.0.0.1:19180'})
        translator = Translator(service_urls=['translate.google.com', 'translate.google.co.kr'])
    else:
        translator = Translator(service_urls=['translate.google.com', 'translate.google.co.kr'])
    return translator

# read the source english questions
def readTexts(path):
    with open(path, 'r', encoding='UTF-8') as f:
        texts = f.readlines()

    texts = [t.replace('\n',str()) for t in texts]
    return texts

# use the unigram words to extend jieba' dictionary
def extendJieba():
    path = 'C:\\Users\\rjkin\\Desktop\\NTU_thesis\\cluster-model\\deploy\\generation\\broken_english\\lm\\lm3.txt'
    chUnigramStart = 21311
    chUnigramEnd = 284082
    save_buffer = getNgramWordList(path, 5000, chUnigramStart, chUnigramEnd)
    # add them to jieba
    for k in save_buffer.keys():
        jieba.suggest_freq(k, True)

# choose maxLength words from the ngram model
def getNgramWordList(path, maxLength, ngramStart, ngramEnd):
    save_buffer = {}
    # maxLength = 10000 # choose 2000 compound syllables and add them into Jieba

    minVal = 100.0
    minKey = 0.0

    i = 0 # which row the file pointer points to
    with open(path, 'r', encoding='UTF-8') as f:
        while i < ngramStart-1:
            f.readline()
            i += 1

        # for the first one
        row = f.readline().split('\t')
        prob = float(row[0])
        syllable = ''.join(row[1].split(' '))
        minVal = prob
        minKey = syllable
        i += 1

        while i < ngramEnd:
            row = f.readline().split('\t')
            prob = float(row[0])
            syllable = ''.join(row[1].split(' '))
            if '<s>' in syllable or '</s>' in syllable or '\n' in syllable or bool(re.search('[a-z]',syllable)):
                i += 1
                continue
            if len(save_buffer) < maxLength:
                save_buffer[syllable] = prob
                if prob < minVal:
                    minVal = prob
                    minKey = syllable
            else:
                # replace the syllable that has the lowest probability
                if prob > minVal:
                    del save_buffer[minKey]
                    save_buffer[syllable] = prob
                    # update minVal and minKey
                    minVal = min(save_buffer.values())
                    idx = list(save_buffer.values()).index(minVal)
                    minKey = list(save_buffer.keys())[idx]
            i += 1
    return save_buffer

# join some english segments into a english sentence
def joinEnSeg(enSeg, punc):
    # join these segmentations
    # return a string
    tmp = str()
    for i, k in enumerate(enSeg):
        if i==0 or k in punc:
            # for the first word or a punctuation mark, insert it without space
            tmp += k
        elif enSeg[i-1] in punc:
            # in this case, the word k isn't a punctuation mark and the previous word is a punctuation mark

            # for punctuation marks that suggest the end of previous words,
            # insert space between them and the following word
            if enSeg[i-1] in ",.?!);:":
                tmp = tmp+" "+k
            # for other punctuation marks like (/ etc.,
            # there is no space between them and the following word.
            else:
                tmp += k
        else:
            tmp = tmp+" "+k
    return tmp




if __name__ == "__main__":
    # readTexts('english_questions.txt')
    # extendJieba()
    # a = ['我','一','下','子','就','好','了']
    # a = ['我', '被', '裁', '了', '。', '我', '如何', '申请', '更', '多', '的', '资金', '和', '贷款', '帮助']
    # ret = ngramCombine(a,formListOfWords('lm3.txt',False))
    # print(ret)
    formListOfWords('lm3.txt',True)
