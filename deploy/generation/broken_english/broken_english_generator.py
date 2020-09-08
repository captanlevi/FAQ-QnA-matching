from googletrans import Translator
import jieba
from .utils import readTexts, extendJieba, joinEnSeg, getTranslator
import string
import time
import pickle
import numpy as np
import kenlm
from .ngram_combine import ngramCombine, formListOfWords, sortByScore
from .random_combine import randomCombine
import json
import os
import threading
import time

class BrokenEnglishGen():

    def __init__(self):
        self.directorypath = os.path.dirname(os.path.abspath(__file__))
        lm_path = os.path.join(self.directorypath, 'lm', 'lm3.arpa')
        self.model = kenlm.Model(lm_path)

    def translate(self, ret):
        punc = string.punctuation
        chineseStr = 'zh-cn'
        englishStr = 'en'
        broken_english_sentences = []
        broken_english_segments = []
        ch_segments = []
        for r in ret:
            segs = r[0]
            ch_segments.append(segs)

            translator = getTranslator(set_proxy=True)
            enSeg = translator.translate(segs, src=chineseStr, dest=englishStr)
            enSeg = [e.text for e in enSeg]
            broken_english_segments.append(enSeg)
            broken_english_sentences.append(joinEnSeg(enSeg, punc))

            wait_time = 0.1
            # print('wait {}s'.format(str(wait_time)))
            time.sleep(wait_time)

        return broken_english_sentences, broken_english_segments, ch_segments

    def exact_batch_generate(self, questions: list, n: int) -> list:
        # questions: source english questions
        # n: number of generated questions

        results_dict = {}
        threads = []

        # add some words into jieba
        extendJieba()

        for sentence in questions:
            thread = threading.Thread(target=self.generate, args=(sentence, n, results_dict))
            threads.append(thread)
            thread.start()
            time.sleep(1.5)

        for t in threads:
            t.join()

        return results_dict

    def generate(self, sentence, n, results_dict):
        punc = string.punctuation
        chineseStr = 'zh-cn'
        englishStr = 'en'
        translator = getTranslator(set_proxy=True)
        broken_english_sentences = []

        # print()
        # print('English source sentence: ' + sentence)
        translator = getTranslator(set_proxy=True)
        # translate it into chinese and cut it
        ch = translator.translate(sentence, src=englishStr, dest=chineseStr).text
        # ch = '这里有更多的人'
        # print('Chinese translation: ' + ch)
        chSeg = list(filter(" ".__ne__, jieba.cut(ch))) # every element of this list is a chinese word group
        # print('Chinese translation after cutting: ' + str(chSeg))
        lm3txt = os.path.join(self.directorypath, 'lm', 'lm3.txt')
        ret_ngram, sign = ngramCombine(chSeg, formListOfWords(lm3txt, False), self.model)
        if sign == False:
            # no permutations produced by ngram combine, then use random combine
            combine_type = 'random'
            ret, rc_list = randomCombine(chSeg, self.model, permNum=n)
        else:
            if len(ret_ngram) < n:
                combine_type = 'ngram and random'
                # the number of segs that ngram combine produces is smaller than n
                ret_rc, rc_list = randomCombine(chSeg, self.model, permNum=n-len(ret_ngram))
                ret = ret_rc + ret_ngram
                ret, _, _ = sortByScore(ret)
            else:
                combine_type = 'ngram'
                # the number of segs is bigger than n, then choose the top ones
                ret_ngram = ret_ngram[:n]
                ret = ret_ngram

        # TODO: translate the chinese combinations into english
        broken_english_sentences, broken_english_segments, combinations = self.translate(ret)
        # print(combine_type)
        results_dict[sentence] = broken_english_sentences
        return
