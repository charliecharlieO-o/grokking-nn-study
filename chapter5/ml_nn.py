"""
    MultiLayer I/O NN
"""


# Empty network with 3 inputs and 3 outputs

            # toeas %win fans
weights = [ [0.1, 0.1, -0.3],   # hurt?
            [0.1, 0.2, 0.0],    # win?
            [0.0, 1.3, 0.1]]    # sad?

def w_sum(a, b):
    assert len(a) == len(b)
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output

def vect_mat_mul(vector, matrix):
    assert len(vector) == len(matrix)
    output = [0, 0, 0]
    for i in range(len(vector)):
        output[i] = w_sum(vector, matrix[i])
    return output

def neural_network(input, weights):
    pred = vect_mat_mul(input, weights)
    return pred

# Making a prediction, calculating error and delta

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

hurt = [0.1, 0.0, 0.0, 0.1]
win = [1, 1, 0, 1]
sad = [0.1, 0.0, 0.1, 0.2]

alpha = 0.01

input = [toes[0], wlrec[0], nfans[0]]
true = [hurt[0], win[0], sad[0]]

pred = neural_network(input, weights)

error = [0, 0, 0]
delta = [0, 0, 0]

for i in range(len(true)):
    error[i] = (pred[i] - true[i]) ** 2
    delta[i] = pred[i] - true[i]

# Calculating each weight and putting it on each weight

def zeros_matrix(len_a, len_b):
    output = []
    for i in range(len_a):
        output.append([0 for x in range(len_b)])
    return output

def outer_prod(vec_a, vec_b):
    out = zeros_matrix(len(vec_a), len(vec_b))
    for i in range(len(vec_a)):
        for j in range(len(vec_b)):
            out[i][j] = vec_a[i] * vec_b[i]
    return out

weight_deltas = outer_prod(input, delta)

# Learn, updating the weights

for i in range(len(weights)):
    for j in range(len(weights[0])):
        weights[i][j] -= alpha * weight_deltas[i][j]
