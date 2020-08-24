# broken english generator

## Introduction

The idea is to firstly get the Chinese translations of input English questions and then use jieba to cut Chinese into segments. Next try to combine more Chinese segments together and finally translate them back to English by segments.

We've tried 2 ways to combine more word segments together. One is random combination and the other is to use ngram word list for combination. In my code, ngram combine has priority over random combine. **If ngram combine can't produce any permutation or the number of permutations is smaller than what we expected, it will use random combine as a supplement**. 

The number of permutations that random combine can produce can be set manually. But the number of permutations that ngram combine can produce is uncertain. It depends on the number of word combinations that exist in the word list. For example, there are 5 segments w1,w2,w3,w4,w5. If combination of w1w2 appears in the word list, then the segments becomes w1w2,w3,w4,w5 and we continue to check if any combination appears in word list. If w1w2w3 exist in word list, then the segments becomes w1w2w3,w4,w5. Repeat the process above until there is no word combination appearing in the word list. In the example above, ngram combine produces 2 permutations, w1w2,w3,w4,w5 and w1w2w3,w4,w5.

As for ngram word list, I singled out 5000 unigram words and 50000 bigram words with the highest probabilities as ngram words list. If you want to extend the words list, you can modify **formListOfWords** function in ngram_combine.py


## Installation

Supporting labraries contains jieba, kenlm and googletrans. Use the following command to install them.

```pip install jieba```
```pip install https://github.com/kpu/kenlm/archive/master.zip```
```pip install googletrans```


## Setup

### Getting input data ready

Input question files should be put in the folder named questions. Each line of input file is supposed to be an English question. And instead of putting all questions in one file, having several input question files is recommended. And remember to change the ```txt_paths``` and ```save_paths``` of the main function in broken_sentence_generate.py

```
def main():
    # source file
    txt_path = './questions/example_english_questions.txt'
    save_path = './results/example_result.json'
    # readTexts return a list and it contains all of the english questions
    texts = readTexts(txt_path)
    broken_generate(texts,save_path=save_path)
```

### Proxy

The python package named "googletrans" is totally free but not stable. Due to the frequent requests, Google may ban your ip address for a while and thus raise a json.decoder.JSONDecodeError error when running the code. Through setting proxies, I fixed this JSONDecodeError error. So you will need a **VPN** and put ip address of its server in getTranslator function of utils.py.

```
def getTranslator(set_proxy=True):
    if set_proxy:
        translator = Translator(service_urls=['translate.google.com', 'translate.google.co.kr','translate.google.cn'], proxies={'http':'127.0.0.1:19180','https':'127.0.0.1:19180'})
    else:
        translator = Translator(service_urls=['translate.google.com', 'translate.google.co.kr','translate.google.cn'])
    return translator
```

### The number of Permutations

This is the amount of sentences you want to generate for each input english question. You can set it in broken_generate function of broken_sentence_generate.py

```
def broken_generate(texts, save_path):
    ......
    perm_num = 20
```


## Usage

After finishing setup, **Just run broken_sentence_generate.py**

You can use **example_english_questions.txt** for test. There are 55 english questions in it. It will take about 1.5 hours to generate 20 permutations for each of these 55 questions.


## Output

Output json file will be generated in the folder named results. The format of one line of the output json file is shown below.

```
ret = {'english source sentence':sentence, 
       'chinese translation':ch, 
       'chinese translation after cutting':chSeg, 
       'combinations':combinations, 
       'broken english segments':broken_english_segments,
       'broken english sentences':broken_english_sentences}
``` 

The ```broken english sentences``` field contains the broken english sentences and it's a 1-dimensional list with many strings in it. It looks like this.
```["How to contact Singapore Boys' Home?", "how is it Contact Singapore boys hospital?", ......]```

The ```'combinations'``` field contains the chinese segments after combination phase. It's a 2-dimensional list and it looks like this.
```[["如何联系新加坡", "男童院", "？"], ["如何", "联系新加坡男童", "院？"], ......]```

The ```broken english segments``` field contains english segments that are translated from chinese segments. It's also a 2-dimensional list and it looks like this.
```[["How to contact Singapore", "Boys' Home", "?"],["how is it", "Contact Singapore boys", "hospital?"], ......]```

By the way, sentences are all sorted by scores(probabilities) of chinese segments after combination, from high to low.

The whole output json file looks like this.
```
{ret1}
{ret2}
{ret3}
{ret4}
{ret5}
......
```

You can use code below to decode output file.
```
ret = []
f = open('./results/example_result.json', 'r', encoding='utf-8')
for line in f.readlines():
    ret.append(json.loads(line))
```