from .base import BaseEncoder
import numpy as np

class dummyEN(BaseEncoder):
    def __init__(self, model_path):
        super().__init__("dummy", 300, "lite")

    def get_vector(self, sentence):
        return np.random.rand(300)

    def get_vectors(self,sentences):
        return np.random.rand(len(sentences),100)