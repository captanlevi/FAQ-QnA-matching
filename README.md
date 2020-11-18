# Bani
This package aims to provide an easy way to set up a question answering system,  
Taking as input just raw text question answer pairs. The principal used is question similirity, ie the most similar question to a  
given query is found and the answer corrosponding to the said question is answered. For this purpose KNN algorithm is used, And Batch hard Tripet  
Loss is used to train a sentence transformer model.

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
     name : The name of an FAQ , all FAQs must have unique names.  
     questions : list of questions or None.  
     answers : list of corrosponding answers or None.  
    (if questions are None answers must also be None , and the FAQ will be empty , you can load this empty faq with another presaved FAQ)

##### Methods
     getAnswerWithLabel(self, label : int) -> Answer  
     getQuestionWithLabel(self, label : int) -> Question  
     buildFAQ(self,generator : GenerateManager,model = None) : this method will generate questions using the given generator , and   
                                                             if the model is also provided , it will assign the vectors to questions as well.  
     isEmpty(self) -> bool : Returns true if the FAQ is empty  
     isUsable(self) -> bool : Returns true if buildFAQ has been called and questions are generated.
     hasVectorsAssigned(self) -> bool : Returns true if all the questions have vectors assigned.  
     load(self,rootDirPath) -> None : Loads the FAQ with the name as self.name  within the root directory.  
     save(self,rootDirPath) -> None : Saves the current object (self) as (self.name).pkl in the root directory.  
     resetAssignedVectors -> None : Resets all the FAQ's assigned vectors to None.  
     resetFAQ -> None : Resets the FAQ to an empty FAQ.  

#### GenerateManager  
```
class GenerateManager (self , producers : List[Any], names : List[str] = None, nums : List[int] = None)
```
The GenerateManager is the interface where the user can register their own sentence prodicers. The class takes care of  
how to run the producers (multi processing , multi threading or single process).  

##### Parameters  
     producers : list of producers (A producer is an instance of any class that implements either batch_generate method or exact_batch_generate).  
     names : list of names of the producers , each producer must have a unique name.
     nums : list of numbers , each number indicates the max number of questions to generate from the producer.  
 
##### Methods  
     addProducer(self,producer , name : str , toGenerate : int) : adding producer , the name must be different from the preexisting ones.  
     producerList(self) -> Tuple[List[str],List[int],List[Any]] : returns the names,nums and producers that are registered.  
     removeProducer(self, name) -> None : remove a producer from the generateManager.  
     

#### Bani

```
class Bani(self,FAQs : List[FAQ], modelPath : str = None, assignVectors : bool = True):
```
The class that acts as the chatbot , It registers any number of FAQs , trains a model on the FAQs and then answers the questions on these FAQs.  

##### Parameters  
     FAQs : list of instances of FAQ class. (each FAQ is given a unique id)
     modelPath : The path to a pretrained model , or any model from the sentence transformers models , if None then the roberta model is pulled.  
     assignVectors : Whether to assign vectors wrt the new model, if true every question in all FAQs are passed through the current model , and new  
                     vectors are assigned, if false then all the FAQs should have re assigned vectors.  


##### Methods  
     train(self,outputPath : str,batchSize = 16, epochs : int = 1, **kwargs) : method to train the model , after training  the new model is loaded and  
                                                                               the FAQ vectors are reassigned using this model.  
                                                                              
     saveFAQs(self, rootDirPath : str) : method to save the FAQ with vectors assigned to rootDirPath , so that the next time you can set,  
                                         assignVectors to False, if you are loading these FAQs (Just to save time).  
     
     getFAQWithId(self, id : int) -> FAQ: method to get the faq wrt the given id , the indexing starts from 0.  
     
     
     findClosestFromFAQ(self,faqId : int, query : str, K : int = 3, topSimilar : int = 5) -> FAQOutput : Takes in a user query and runs the knn algo over it.  
                                         with K as K, and returns a FAQOutput object, whick topSimilar number of closest questions. The query is processed only  
                                         over the 'faqId'  FAQ.
     findClosest(self,query : str,  K : int = 3 , topSimilar : int = 5) -> List[FAQOutput] : The same as findClosestFromFAQ, but here the query is run over all the,  
                                        FAQs and the result is a list of FAQOutputs , the length of the list is the same as the number of FAQs.
     
     test(self,faqId : int,testData : List[Tuple[str,str]], K : int = 3) -> float:    Interface to test any given faq , expects a list of tuples of size 2
                                        first element is the orignal question and second is the paraphrased version. All the orignal question should ideally match the                                                 questions in the FAQ , if not you will be warned about it.


#### FAQOutput
    The user will get this , or a list of FAqOutput ,as the output for any query. It contains.  

     answer : Answer : The actual answer
     question : Question : The question that is being answered. (A generated question may be being answered, but only orignal question is given here)
     faqName : str,      : name of the faq the answer is from. 
     faqId : int,        : Id of the FAQ wrt the Bani object. 
     score : float       : Combined KNN score
     similarQuestions : List[str]  : Similar questions to the query asked , from the said FAQ.
     maxScore : float    : The question with maximum similirity with the query.
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




