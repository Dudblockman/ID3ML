import numpy as np
import re
from collections import Counter

class treemaker:
    mode = 0
    @staticmethod
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

    @staticmethod
    def gain(arr: np.array, arr2: np.array):
        return treemaker.entropy(arr2)-treemaker.entropy(arr)

    @staticmethod
    def informationGain(S, attributes, label, coalesced, mastercounts):
        bestIndex = -1
        bestGain = 0
        for k in coalesced.keys():
            g = treemaker.gain(coalesced[k], mastercounts)
            if g > bestGain:
                bestGain = g
                bestIndex = k
        return bestIndex

    @staticmethod
    def majorityError(S, attributes, label, coalesced, mastercounts):
        return 0

    @staticmethod
    def gini(A):
        summ = np.sum(A)
        g = 0
        for i in range(A.shape[0]):
            g += (1 - np.sum(np.square(A[i] / np.sum(A[i])))) * (np.sum(A[i]) / summ)
        return g

    @staticmethod
    def giniIndex(S, attributes, label, coalesced, mastercounts):
        bestIndex = -1
        bestGain = 1
        for k in coalesced.keys():
            g = treemaker.gini(coalesced[k])
            if g < bestGain:
                bestGain = g
                bestIndex = k
        return bestIndex

    def bestGainIndex(self, S, attributes, label):
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

        mastercounts = np.array([counts[x] for x in label]).T
        if self.mode == 0:
            return treemaker.informationGain(S, attributes, label, coalesced, mastercounts)
        if self.mode == 1:
            return treemaker.majorityError(S, attributes, label, coalesced, mastercounts)
        if self.mode == 2:
            return treemaker.giniIndex(S, attributes, label, coalesced, mastercounts)
        return 0


    def getAttributes(self, data):
        dat = [list(set(x)) for x in data.T]
        [v.sort() for v in dat]
        attributes = dat[0:len(dat)-1]
        attributes = {i: attributes[i] for i in range(len(attributes))}
        attributes["count"] = len(dat)-1
        label = dat[len(dat)-1]
        return attributes, label

    def preprocess(self, data):
        data2 = data.copy()
        terms, _ = self.getAttributes(data)
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

    def id3(self, S, attributes, label, maxdepth):
        node = {}

        commonList = Counter(S.T[len(S[0])-1]).most_common()
        mostCommon = commonList[0][0]

        bestIndex = self.bestGainIndex(S, attributes, label)
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
                node["leaves"][leaf] = self.id3(subset, newAttributes, label, maxdepth - 1)
        return node

    def generateTree(self, S, maxdepth=100, process=True, mode=0):
        self.mode = mode
        if process:
            S = self.preprocess(S)
        else:
            S = S.copy()
        attributes, label = self.getAttributes(S)
        return self.id3(S, attributes, label, maxdepth)

    @staticmethod
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
    
    @staticmethod
    def accuracy(S, T):
        correct = 0
        for case in S:
            if case[len(case)-1] == treemaker.evalTree(T, case):
                correct+=1
        return correct / len(S)

def loadfile(filename, process=True):
    data = []
    with open (filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            data.append(terms)
    return np.array(data)


'''
treegen = treemaker()

cartrainingdata = loadfile("DecisionTree/bank/train.csv")
cartestdata = loadfile("DecisionTree/bank/test.csv")

banktrainingdata = loadfile("DecisionTree/bank/train.csv")
banktestdata = loadfile("DecisionTree/bank/test.csv")


for mode in range(0,3,2):
    for depth in range(1,7):
        tree = treegen.generateTree(cartrainingdata, depth, mode=mode)
        print(mode, depth, accuracy(cartrainingdata, tree), accuracy(cartestdata, tree))

for mode in range(0,3,2):
    for depth in range(1,17):
        tree = treegen.generateTree(banktrainingdata, depth, mode=mode)
        print(depth, "&" ,accuracy(banktrainingdata, tree), "&", accuracy(banktestdata, tree), "\\\\")

''
trainingdata = loadfile("DecisionTree/tennis.csv")


tree = treegen.generateTree(trainingdata,mode=2)
output = str(tree)
output = output.replace("0","Outlook").replace("1","Temperature").replace("2","Humidity").replace("3","Windy")
print(output)
print(accuracy(trainingdata, tree))'''