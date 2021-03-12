import numpy as np
from DecisionTree.Tree import *
class adaboost(treemaker):
    def generateTree(self, S, maxdepth=5, process=True, mode=0):
        trees = []
        attributes, label = self.getAttributes(S)
        tree = None
        error = -1
        weightVector = np.ones(S.shape[0]) / S.shape[0]
        for t in range(maxdepth):
            weightVector = self.weights(S, weightVector, tree, error)
            tree = self.id3(S,attributes,label,1)
            error = 1 - self.accuracy(S, tree)
            trees.append(tuple([tree, weightVector.copy(), error]))

        

        return None
    def weights(self, S, weights, tree, error):
        if tree == None:
            return np.ones(S.shape[0]) / S.shape[0]
        output = weights.copy()
        alpha = 0.5 * np.log((1 - error) / error)
        for i in range(S.shape[0]):
            result = self.evalTree(tree, S[i])
            sign = 1 if S[i][len(S[i])-1].equals(result) else -1
            output[i] *= np.exp(-alpha * sign)
        output /= np.sum(output)
        return output


        