"""
    Making a prediction with multiple outputs

    Notice the three predictions are completely separate.
    unlike NN's with multiple inputs and a single output
    where prediction is undeniably connected, this NN behaves
    as three independent components.
"""


weights = [0.3, 0.2, 0.9]

def ele_mul(number, vector):
    output = [0, 0, 0]

    assert len(output) == len(vector)

    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output

def neural_network(input, weights):
    prediction = ele_mul(input, weights)
    return prediction

wlrec = [0.65, 0.8, 0.8, 0.9]
input = wlrec[0]
prediction = neural_network(input, weights)
print(prediction)
