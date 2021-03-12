import numpy as np
from DecisionTree.Tree import treemaker, loadfile
from EnsembleLearning.AdaBoost import adaboost

a = adaboost()

trainingdata = loadfile("DecisionTree/bank/train.csv")
testdata = loadfile("DecisionTree/bank/test.csv")
a.generateTree(trainingdata)