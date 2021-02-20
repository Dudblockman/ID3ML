import numpy as np
import re
from collections import Counter

def entropy(arr: np.array):
    if len(arr.shape) > 1:
        a = (arr.T / np.sum(arr, axis=1)).T
        b = np.ma.log2(a)
        c = np.sum(-np.multiply(a, b), axis=1)
        d = (np.sum(arr, axis=1)/np.sum(arr))
        e = np.sum(np.multiply(c,d))
        return e
    else: 
        a = (arr.T / np.sum(arr)).T
        b = np.ma.log2(a)
        c = np.sum(-np.multiply(a, b))
        d = (np.sum(arr)/np.sum(arr))
        e = np.sum(np.multiply(c,d))
        return e

def gain(arr: np.array, arr2: np.array):
    return entropy(arr2)-entropy(arr)

def bestGainIndex(S, attributes, label):
    coalesced = {}
    lastIndex = attributes["count"]
    for k,v in attributes.items():
        if hasattr(v, "__len__"):
            coalesced[k] = np.zeros( (len(v), len(label)), dtype=np.float ) 

    for terms in S:
        for i in range(lastIndex):
            if i in coalesced:
                j = attributes[i].index(terms[i])
                k = label.index(terms[lastIndex])
                coalesced[i][j][k] += 1
    counts = Counter(S.T[len(S[0])-1])

    arr2 = np.array([counts[x] for x in label]).T

    bestIndex = -1
    bestGain = 0
    for k in coalesced.keys():
        if gain(coalesced[k], arr2) > bestGain:
            bestGain = gain(coalesced[k], arr2)
            bestIndex = k
    
    return bestIndex

def loadfile(filename, process=True):
    data = []
    with open (filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            data.append(terms)
    return np.array(data)

def getAttributes(data):
    dat = [list(set(x)) for x in data.T]
    [v.sort() for v in dat]
    attributes = dat[0:len(dat)-1]
    attributes = {i: attributes[i] for i in range(len(attributes))}
    attributes["count"] = len(dat)-1
    label = dat[len(dat)-1]
    return attributes, label

def preprocess(data):
    data2 = data.copy()
    terms, _ = getAttributes(data)
    for key, term in terms.items():
        if type(term) == list:
            numeric = True
            for element in term:
                try:
                    float(element)
                except ValueError:
                    numeric = False
                    break
            if numeric == True:
                median = np.round(np.median(np.array(term).astype(np.float))).astype(np.int)
                for i in range(len(data)):
                    data2[i, int(key)] = ">="+str(median) if float(data[i, int(key)]) >= median else "<"+str(median)
    return data2

def id3(S, attributes, label, maxdepth):
    node = {}

    commonList = Counter(S.T[len(S[0])-1]).most_common()
    mostCommon = commonList[0][0]

    bestIndex = bestGainIndex(S, attributes, label)
    if len(commonList) < 1 or bestIndex == -1:
        return mostCommon
    node["attribute"] = bestIndex
    node["leaves"] = {}
    for leaf in attributes[bestIndex]:
        subset = np.array([x for x in S if x[bestIndex] == leaf])
        if len(subset) == 0:
            node["leaves"][leaf] = mostCommon
        elif maxdepth <= 1:
            node["leaves"][leaf] = Counter(subset.T.tolist()[len(S[0])-1]).most_common()[0][0]
        else:
            newAttributes = attributes.copy()
            del newAttributes[bestIndex]
            node["leaves"][leaf] = id3(subset, newAttributes, label, maxdepth - 1)
    return node

def generateTree(S, maxdepth=100, process=True):
    if process:
        S = preprocess(S)
    else:
        S = S.copy()
    attributes, label = getAttributes(S)
    return id3(S, attributes, label, maxdepth)

def evalTree(T, V):
    while type(T) == dict:
        index = T["attribute"]
        index = V[index]
        if index in T["leaves"]:
            T = T["leaves"][index]
        else:
            for k,v in T["leaves"].items():
                m = re.search("^([<>]=?)(-?[0-9]+)$", k)
                if m != None:
                    if m.groups()[0] == ">=":
                        if int(index) >= int(m.groups()[1]):
                            T = T["leaves"][k]
                            break
                    if m.groups()[0] == "<":
                        if int(index) < int(m.groups()[1]):
                            T = T["leaves"][k]
                            break
    return T

def accuracy(S, T):
    correct = 0
    for case in S:
        if case[len(case)-1] == evalTree(T, case):
            correct+=1
    return correct / len(S)


trainingdata = loadfile("DecisionTree/bank/train.csv")
testdata = loadfile("DecisionTree/bank/test.csv")

tree = generateTree(trainingdata, 16)

print(tree)
print(accuracy(trainingdata, tree), accuracy(testdata, tree))



'''
trainingdata = loadfile("DecisionTree/car/train.csv")
testdata = loadfile("DecisionTree/car/test.csv")

attributes, label = getAttributes(trainingdata)

tree = id3(trainingdata, attributes, label, 3)

print(tree)
print(accuracy(trainingdata, tree), accuracy(testdata, tree))

'''
'''

trainingdata = loadfile("DecisionTree/tennis.csv")

attributes, label = getAttributes(trainingdata)

tree = id3(trainingdata, attributes, label, 4)
output = str(tree)
output = output.replace("0","Outlook").replace("1","Temperature").replace("2","Humidity").replace("3","Windy")

print(output)
print(accuracy(trainingdata, tree))
'''