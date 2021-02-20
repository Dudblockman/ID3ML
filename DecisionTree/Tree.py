import numpy as np
from collections import Counter

def entropy(arr: np.array):
    a = (arr.T / np.sum(arr, axis=1)).T
    b = np.ma.log2(a)
    c = np.sum(-np.multiply(a, b), axis=1)
    d = (np.sum(arr, axis=1)/np.sum(arr))
    return np.sum(np.multiply(c,d))

def gain(arr: np.array):
    return entropy(arr.T)-entropy(arr)

def bestGainIndex(S, attributes, label):
    coalesced = []
    lastIndex = len(attributes)
    for i in range(lastIndex):
        coalesced.append( np.zeros( (len(attributes[i]), len(label)), dtype=np.float ) )
    for terms in data:
        for i in range(lastIndex):
            j = attributes[i].index(terms[i])
            k = label.index(terms[lastIndex])
            coalesced[i][j][k] += 1

    bestIndex = -1
    bestGain = 0
    for i in range(lastIndex):
        if gain(coalesced[i]) > bestGain:
            bestGain = gain(coalesced[i])
            bestIndex = i
    return bestIndex

def loadfile(filename):
    data = []
    with open (filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            data.append(terms)
    return np.array(data)

def getAttributes(data):
    attributes = [list(set(x)) for x in data.T]
    [v.sort() for v in attributes]
    return attributes[0:len(attributes)-1], attributes[len(attributes)-1]

def preprocess(data, terms):
    return

def id3(S, attributes, label):
    node = {}
    lastIndex = len(attributes)

    commonList = Counter(S.T[lastIndex]).most_common()
    mostCommon = commonList[0][0]
    if len(commonList) < 2:
        return mostCommon

    bestIndex = bestGainIndex(S, attributes, label)
    node["attribute"] = bestIndex
    node["leaves"] = {}
    for leaf in attributes[bestIndex]:
        subset = np.array([x for x in S if x[bestIndex] == leaf])
        if len(subset) == 0:
            node["leaves"][leaf] = mostCommon
        else:
            newAttributes = attributes.copy()
            newAttributes[bestIndex] = []
            node["leaves"][leaf] = None#id3(subset, newAttributes)

    return node

data = loadfile("DecisionTree/car/train.csv")
attributes, label = getAttributes(data)
print(getAttributes(data))
id3(data, attributes, label)