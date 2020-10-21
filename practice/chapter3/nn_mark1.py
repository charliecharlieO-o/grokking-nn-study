"""
    V1:
    Here we demonstrate how the dot product works on a neural network
    with only one set of weights, 3 inputs and 1 output. It doesn't
    work because we are not correcting (for starters).
    # will predict 6.0 well out of bounds
    V2:
    Updated the NN to have one hidden layer that then merges into one
    output.
    # will predict 1.34, well that's an improvement, but still dumb
"""

def dot_product(a, b):
    assert len(a) == len(b)
    output = 0
    for idx, n in enumerate(a):
        output += (n * b[idx])
    return output

def vector_matrix_mult(vector, matrix):
    assert len(vector) == len(matrix)
    output = [0, 0, 0]
    for idx, _ in enumerate(vector):
        output[idx] = dot_product(vector, matrix[idx])
    return output

def neural_network(input, weights):
    hidden_layer = vector_matrix_mult(input, weights[0])
    prediction = dot_product(hidden_layer, weights[1][0])
    return prediction

data = [
  [10, 25, 2],
  [15, 5, 4],
  [3, 10, 5],
  [30, 1, 0]
]
weights_l1 = [
    [0.1, 0.2, 0.0],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
]
weights_l2 = [
    [0.1, 0.1, 0.1],
]

pred = neural_network(data[0], [weights_l1, weights_l2])
print(f"Our prediction: {pred}")
