"""
  Predicting with multiple inputs and outputs.
    - 3 weights going into each output node
"""


        # toes  wins  fans
weights = [
          [0.1, 0.1, -0.3], # hurt?
          [0.1, 0.2, 0.0],  # win?
          [0.0, 1.3, 0.1]   # sad?
]

def w_sum(a, b):
    # Performing a weighted a sum of inputs
    assert len(a) == len(b)
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output

# VECTOR MATRIX MULTIPLICATION
def vect_mat_mul(vect, matrix):
    # For each output we are performing a weighted sum of inputs
    # this function iterates through each row of weigths and makes
    # a prediction using w_sum
    assert len(vect) == len(matrix)

    output = [0, 0, 0]

    for i in range(len(vect)):
        output[i] = w_sum(vect, matrix[i])

    return output

def neural_network(input, weights):
    prediction = vect_mat_mul(input, weights)
    return prediction

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.64, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

input = [toes[0], wlrec[0], nfans[0]]

prediction = neural_network(input, weights)

print(prediction)
