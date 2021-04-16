import numpy as np
from DecisionTree.Tree import *


def parallel_shuffle(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def primal_svm(filename, T, rate_functor, c):
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
    learning_rate = 0.9

    for t in range(T):
        parallel_shuffle(labels, examples)
        for i in range(len(examples)):
            predictor = weight.dot(examples[i, :])
            if predictor * labels[i] <= 1:
                weight = np.array((1 - learning_rate) * weight + learning_rate * c * len(examples) * labels[i] * examples[i, :])
            else:
                weight[:-1] = (1 - learning_rate) * weight[:-1]
        learning_rate = rate_functor(learning_rate, t)
    weights.append(weight)

    return weights

def get_label(hypothesis, example, label_map):
    result = -1 if hypothesis[1].dot(example) < 0 else 1
    return label_map[0] if label_map[0] == result else label_map[1]

def test_primal(weights, test_file):
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
    predictions = []
    label_map = weights[0]

    for example in examples:
        predictions.append(get_label(weights, example, label_map))

    correct = 0
    for i in range(len(labels)):
        if labels[i] * predictions[i] > 0:
            correct += 1

    return tuple([correct, len(labels)])