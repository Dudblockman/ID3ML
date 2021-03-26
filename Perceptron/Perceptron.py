
import numpy as np
from DecisionTree.Tree import *

def perceptron(filename, T):

    data = loadfile(filename=filename).astype(np.float)
    examples = data[:, 0:len(data[0])-1]

    labels = data[:, len(data[0])-1]

    label_val = -1
    label_map = {}
    for label in labels:
        if label not in label_map:
            label_map[label] = label_val
            label_val += 2
            if label_val > 1:
                break

    labels = [label_map[l] for l in labels]

    weight = np.zeros(len(examples[0, :]))
    weights = [label_map]
    learning_rate = 1 / (10**3)
    correct_count = 0

    for t in range(T):
        for i in range(len(examples)):
            predictor = weight.T @ np.array(examples[i, :])
            prediction = -1 if predictor < 0 else 1
            if prediction != labels[i]:
                weight = np.array(weight + learning_rate * labels[i] * examples[i, :])
                weights.append(tuple([weight, correct_count]))
                correct_count = 0
            else:
                correct_count += 1
    weights.append(tuple([weight, correct_count]))

    return weights

    
def getAttributes(data):
    dat = [list(set(x)) for x in data.T]
    [v.sort() for v in dat]
    attributes = dat[0:len(dat)-1]
    attributes = {i: attributes[i] for i in range(len(attributes))}
    attributes["count"] = len(dat)-1
    label = dat[len(dat)-1]
    return attributes, label


def evaluate(weights, data, mode):
    mapper = weights.pop(0)
    if mode == "standard":
        predictor = np.array(weights[-1][0]).T
        return -1 if predictor @ data < 0 else 1
    if mode == "voted":
        result = 0
        for predictor in weights:
            weight = predictor[0].T
            result += predictor[1] * (weight @ data)
        return -1 if result < 0 else 1
    if mode == "average":
        average = np.zeros(len(weights[0][0]))
        for predictor in weights:
            average += predictor[0]
        result = average.T @ data

        if mapper[0] == -1 if result < 0 else 1:
            return mapper[0]
        else:
            return mapper[1]

def test(weights, test_file, mode):
    data = loadfile(filename=test_file).astype(np.float)
    examples = data[:, 0:len(data[0])-1]
    labels = data[:, len(data[0])-1]

    label_val = -1
    label_map = {}
    for label in labels:
        if label not in label_map:
            label_map[label] = label_val
            label_val += 2
            if label_val > 1:
                break

    labels = [label_map[l] for l in labels]

    evaluated = []
    for example in examples:
        evaluated.append(evaluate(list(weights), example, mode))
    count = 0
    for i in range(len(labels)):
        if labels[i] * evaluated[i] > 0:
            count += 1

    return count / len(labels)

    