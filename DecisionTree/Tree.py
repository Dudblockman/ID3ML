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

def bestGainIndex(S, attributes):
    coalesced = []
    lastIndex = len(attributes)-1
    for i in range(lastIndex):
        coalesced.append( np.zeros( (len(attributes[i]), len(attributes[lastIndex])), dtype=np.float ) )
    for terms in data:
        for i in range(lastIndex):
            j = attributes[i].index(terms[i])
            k = attributes[lastIndex].index(terms[lastIndex])
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
    attributes = []
    for i in range(len(data[0])):
        attributes.append(set())
    for terms in data:
        for i in range(len(terms)):
            attributes[i].add(terms[i])
    attributes = [list(v) for v in attributes]
    [v.sort() for v in attributes]
    return attributes

def preprocess(data, terms):
    return

def id3(S, attributes):
    node = {}

    lastIndex = len(attributes)-1

    print(Counter(S.T[lastIndex]).most_common(1)[0][0])

    bestIndex = bestGainIndex(S, attributes)
    node["attribute"] = bestIndex
    node["leaves"] = {}
    for leaf in attributes[bestIndex]:
        subset = [x for x in S if x[bestIndex] == leaf]
        newAttributes = attributes.copy()
        newAttributes[bestIndex] = None
        node["leaves"][leaf] = id3(subset, newAttributes)

    return node

data = loadfile("DecisionTree/car/train.csv")
attributes = getAttributes(data)
id3(data, attributes)