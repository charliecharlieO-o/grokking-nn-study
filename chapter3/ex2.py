"""
    Making a prediction with multiple inputs
    what does it do?
        - multiplies three inputs by weights and sums them
        - this is a weighted sum (DOT PRODUCT)
"""


weights = [0.1, 0.2, 0]

def w_sum(a, b):
    # Performing a weighted a sum of inputs
    assert len(a) == len(b)
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output

def neural_network(input, weights):
    # Empty network with multiple inputs
    prediction = w_sum(input, weights)
    return prediction

toes = [8.5, 9.5, 9.0, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

input = [toes[0], wlrec[0], nfans[0]]
pred = neural_network(input, weights)
print(pred)
