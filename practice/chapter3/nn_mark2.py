"""
    Refactor Mark 1 to make use of numpy.
    Will yield the same result but we make use of the numpy lib
"""


import numpy as np


def neural_network(input, weights):
    hidden_layer = input.dot(weights[0])
    prediction = hidden_layer.dot(weights[1])
    return prediction

weights_l1 = np.array([
    [0.1, 0.2, 0.0],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
]).T
weights_l2 = np.array([0.1, 0.1, 0.1])
data = np.array([10, 25, 2])

prediction = neural_network(data, [weights_l1, weights_l2])
print(f"Our prediction: {prediction}")
