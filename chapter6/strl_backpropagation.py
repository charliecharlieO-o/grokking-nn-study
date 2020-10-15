"""
    Chapter 6, Our first NN with
    backpropagation and conditional correlation
"""

import numpy as np


np.random.seed(1)


def relu(x):
    # Our Relu method for conditional correlation in the middle layer
    # it will set all negative numbers to 0, "turning off" middle nodes.
    return (x > 0) * x


def relu_deriv(output):
    # Dervicatve of relu, returns 1 for input > 0, 0 otherwise
    return output > 0


alpha = 0.2
hidden_size = 4


streetlights = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
])
walk_vs_stop = np.array([
    [1, 1, 0, 0],
]).T    # transpose 1x4 vector


# Two set of weights now to connect the three layers (randomly initialized)
weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

for iteration in range(60):
    layer_2_error = 0
    for i in range(len(streetlights)):
        layer_0 = streetlights[i:i + 1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))            # Pass input to weights (dot prod) then apply relu
        layer_2 = np.dot(layer_1, weights_1_2)                  # where negative values become 0 -> becomes input for the next layer

        # Squared error calculation of layer 2
        layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i + 1]) ** 2)

        layer_2_delta = (walk_vs_stop[i:i + 1] - layer_2)

        # The following line computes the delta at layer_1 given the delta at layer_2 by taking
        # the layer_2 delta and multiplying it by it's connecting weights
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu_deriv(layer_1)

        # Updating our weights based on our previously calculated deltas
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

        if (iteration % 10 == 9):
            print(f"Error: {layer_2_error}")
