"""
    Gradient descent with multiple inputs
"""


def w_sum(a, b):
    assert len(a) == len(b)
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output


def neural_network(input, weights):
    pred = w_sum(input, weights)
    return pred


def ele_mul(number, vector):
    output = [0, 0, 0]
    assert len(output) == len(vector)

    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output


weights = [0.1, 0.2, -0.1]

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

win_or_lose_binary = [1, 1, 0, 1]

true = win_or_lose_binary[0]
input = [toes[0], wlrec[0], nfans[0]]

alpha = 0.01

for iter in range(3):

    pred = neural_network(input, weights)

    error = (pred - true) ** 2
    delta = pred - true

    weight_deltas = ele_mul(delta, input)

    print("="*100)
    print(f"Iteration: {iter + 1}")
    print(f"Pred: {pred}")
    print(f"Error: {error}")
    print(f"Delta: {delta}")
    print(f"Weights: {weights}")
    print(f"Weight Deltas: {weight_deltas}")

    # Update weights
    for i in range(len(weights)):
        weights[i] -= alpha * weight_deltas[i]
