# Bani
This package aims to provide an easy way to set up a question answering system,  
Taking as input just raw text question answer pairs.

## Installation 
#### Install with pip
```
pip install Bani
python -m spacy download en_core_web_md
```
This will install all the necessary packages , including the correct version of sentence transformers and transformers. 
#### Copy the source code
Clone or download the source and then 
```
python -m spacy download en_core_web_md
cd Bani ; pip install -r requirements
```


### Getting Started
See the [tutorial](https://github.com/captanlevi/Bani/blob/master/Tutorial.ipynb) notebook for a quick introduction to the usage of the package.

### Docs

#### FAQ
```
class FAQ (self,name : str,questions : List[str] = None, answers : List[str] = None)
```
All the user supplied FAQs are stored in the FAQ class, The FAQ class further runs sanity checks on the faqs ,and provides interface to  
generate questions and assign vectors.  

##### Parameters  

    1. name : The name of an FAQ , all FAQs must have unique names.  
    2. questions : list of questions or None.  
    3. answers : list of corrosponding answers or None.  
    (if questions are None answers must also be None , and the FAQ will be empty , you can load this empty faq with another presaved FAQ)


## Adding your own producers(sentence_generator)
The quality of the FAQ is directely related to the quality of questions produced, As such Bani comes with a default  
question generation pipeline , but also gives full freedom to customize or add your own **producers**.
A producer is an instance of any class that implements either batch_generate method or exact_batch_generate
```
class MyProducer1:
    def __init__(self):
        pass
    
    def batch_generate(questions : List[str]) -> Dict[str, List[str]]:
        """
        Takes list of questions and returns a dict , with each question 
        mapped to the list of generated questions
        """
        
        resultDict = dict()
        for question in questions:
            resultDict[question] = ["generated1", "generated2", "and so on"]
        
        return resultDict
```

The objects that implement exact_batch_generate will produce at most **n** questions for a given question. 

```
class MyProducer2:
    def __init__(self):
        pass
    
    def exact_batch_generate(questions : List[str], num : int) -> Dict[str, List[str]]:
        """
        Takes list of questions and returns a dict , with each question 
        mapped to the list of generated questions , for each question at most num questions are generated
        """
        
        resultDict = dict()
        for question in questions:
            resultDict[question] = ["generated1", "generated2", "and so on"]
        
        return resultDict
```

Each of the producers are registered in a GenerateManager , with their names and how many questions to generate at max from  
the producer.

```
from Bani.core.generation import GenerateManager

names = ["myProducer1_name", "myProducer2_name"]
toGenerate = [3,5] # At max generate 3 for first producer and 5 for second
producers = [MyProducer1(), MyProducer2()]

myGenerateManager = GenerateManager(producers = producers , names = names , nums = toGenerate)

# Or you can register the producers one by one

myGenerateManager.addProducer(producer = myProducer3, name = "myProducer3Name", togenerate = 5)
```




