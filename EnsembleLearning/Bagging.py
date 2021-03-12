import numpy as np
import random
from DecisionTree.Tree import *
#a = treemaker()

class baggedTree(treemaker):
    @staticmethod
    def randomSample(data : np.array, sampleCount : int):
        result = random.sample(data, sampleCount)
        return np.array(result)
