from SVM.SVM import *
import numpy as np

training = "Perceptron/bank-note/train.csv"
testing = "Perceptron/bank-note/test.csv"
C = [100/873, 500/873, 700/873]
for c in C:
    d = 0.9
    functor = lambda y0, t: y0 / ( 1 + t * y0 / d)
    weights = primal_svm(training, 100, functor, c)
    print("C=", c)
    print("vector=", weights[1])

    standardresult = test_primal(weights, training)
    print("training =", 1 - standardresult[0]/standardresult[1])

    standardresult = test_primal(weights, testing)

    print("test =", 1 - standardresult[0]/standardresult[1])


for c in C:
    functor = lambda y0, t: y0 / ( 1 + t )
    weights = primal_svm(training, 100, functor, c)
    print("C=", c)
    print("vector=", weights[1])
    
    standardresult = test_primal(weights, training)
    print("training =", 1 - standardresult[0]/standardresult[1])

    standardresult = test_primal(weights, testing)

    print("test =", 1 - standardresult[0]/standardresult[1])




