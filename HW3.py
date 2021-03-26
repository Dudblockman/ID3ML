from Perceptron.Perceptron import perceptron, test
import numpy as np

training = "Perceptron/bank-note/train.csv"
testing = "Perceptron/bank-note/test.csv"

weights = perceptron(training, 10)

print("vector=", weights[-1][0])



standardresult = test(weights, testing, "standard")

print("standard", standardresult)





votedresult = test(weights, testing, "voted")

for i in range(1, len(weights)):
    print(i, ":", weights[i][0], "Correctness:", weights[i][1])
print("voted", votedresult)





averageresult = test(weights, testing, "average")

average = np.zeros(len(weights[1][0]))
for predictor in weights:
    average += predictor[0]

print("average vector", average)

print("average", averageresult)