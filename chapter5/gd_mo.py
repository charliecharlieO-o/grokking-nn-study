"""
    Using gradient descent on a NN with multiple outputs
"""


def scalar_ele_mul(number, vector):
    output = [0, 0, 0]
    assert len(output) == len(vector)
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output


def ele_mul(number, vector):
    output = [0, 0, 0]
    assert len(output) == len(vector)

    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output


# Instead of predicting if a team won or lost we are
# also predicting hurt, win or sad

weights = [0.3, 0.2, 0.9]

def neural_network(input, weights):
    pred = ele_mul(input, weights)
    return pred

# Making a prediction and calculating error and delta
wlrec = [0.65, 1.0, 1.0, 0.9]

hurt = [0.1, 0.0, 0.0, 0.1]
win = [1, 1, 0, 1]
sad = [0.1, 0.0, 0.1, 0.2]

input = wlrec[0]
true = [hurt[0], win[0], sad[0]]

pred = neural_network(input, weights)

error = [0, 0, 0]
delta = [0, 0, 0]

for i in range(len(true)):
    error[i] = (pred[i] - true[i])**2   # Error squared
    delta[i] = pred[i] - true[i]        # Delta, error difference
weight_deltas = scalar_ele_mul(input, weights)      # Calculating each weight delta

# Updating the weights

alpha = 0.1     # Learning rate

for i in range(len(weights)):
    weights[i] -= (weight_deltas[i] * alpha)

print(f"Weights: {weights}")
print(f"weight Deltas: {weight_deltas}")
