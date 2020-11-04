from core.FAQ import *
import os
from typing import List , Tuple, Dict
import pandas as pd





def csvReader(path : str) -> Tuple[List[Question], List[Answer]]:
    """
    the csv is assumed to have questions on column 1 and answers in column 2 
    WITH NO HEADER
    """

    df = pd.read_csv(path, header= None)
    questions = df.iloc[:,0]
    answers = df.iloc[:,1]

    return questions, answers




    
