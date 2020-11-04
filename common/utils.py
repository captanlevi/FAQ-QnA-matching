import pickle
import numpy as np
from typing import List, Union, Dict

def save_dict(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)









