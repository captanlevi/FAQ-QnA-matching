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


## Adding your own producers(sentence_generator)



