import numpy as np
from DecisionTree.Tree import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidP(x):
    return x * (1 - x)

def parallel_shuffle(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def gaussian_weights(example_len, hidden_layers, layer_width):
    weights = []
    weights.append(np.random.normal( size=(example_len, layer_width)))
    layer_weight_size = (layer_width+1, layer_width)
    for _ in range(hidden_layers-1):
        weights.append(np.random.normal( size=layer_weight_size))
    weights.append(np.random.normal( size=(layer_width+1, 1)))
    return weights

def zero_weights(example_len, hidden_layers, layer_width):
    weights = []
    weights.append(np.zeros( shape=(example_len, layer_width)))
    layer_weight_size = (layer_width+1, layer_width)
    for _ in range(hidden_layers-1):
        weights.append(np.zeros( shape=layer_weight_size))
    weights.append(np.zeros( shape=(layer_width+1, 1)))
    return weights

def forward(inputs, weights):
    layers = [inputs] 
    gradient = []  
    bias = np.array([1])

    for i in range(len(weights)-1):
        l = sigmoid(layers[i].dot(weights[i]))
        l = np.concatenate((l, bias))
        layers.append(np.array(l))

        gradient.append(np.outer(layers[i], sigmoidP(layers[-1][:-1])))

    prediction = np.float64(layers[-1].dot(weights[-1]))
    layers.append(prediction)

    return layers, gradient

def back_prop(layers, forward_cache, weights, true_label, learning_rate):
    err = layers[-1] - true_label
    last_layer = np.array([weights[-1][:-1]])
    backprop_cache = learning_rate * err * last_layer  
    if len(backprop_cache.shape) > 2:
        backprop_cache = np.squeeze(backprop_cache, axis=2)
    last_hidden = (learning_rate * err * layers[-2]).transpose()
    weights[-1] = np.squeeze(weights[-1]) - last_hidden  
    for i in range(len(forward_cache)-1, -1, -1):
        cur_layer_weights = np.delete(weights[i], -1, 0)  
        rows = np.repeat(backprop_cache, repeats=len(layers[i]), axis=0)
        cache = np.multiply(forward_cache[i], np.array(rows))
        weights[i] = weights[i] - cache
        if i > 0:
            backprop_cache = backprop_cache @ cur_layer_weights

def train_network(filename, epochs, rate_schedule, layer_width, weight_init):

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

    if weight_init == "gaussian":
        weight_list = gaussian_weights(examples.shape[1], 2, layer_width)
    else:
        weight_list = zero_weights(examples.shape[1], 2, layer_width)

    learning_rate = 0.01  # gamma

    for t in range(epochs):
        parallel_shuffle(examples, labels)
        for i in range(len(examples)):
            forward_result, cache = forward(examples[i, :], weight_list)
            back_prop(forward_result, cache, weight_list, labels[i], learning_rate)

        learning_rate = rate_schedule(learning_rate, t, d=2)

    return label_map, weight_list