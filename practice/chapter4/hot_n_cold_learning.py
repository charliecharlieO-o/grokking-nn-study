"""
    The simplest form of neural learning is the hot & cold approach
    we wiggle weights either up or down and see which reduces error
    to move in that direction
"""

import numpy as np


weight = 0.1
lr = 0.01


def neural_network(input, weight):
    prediction = input.dot(weight)
    return prediction

number_of_toes = np.array([8.5])
win_or_lose_binary = np.array([1])

# Make a prediction and evaluate error

prediction = neural_network(number_of_toes, weight)

error = (prediction - win_or_lose_binary[0])**2
print(f"Squared Error: {error}")

# Make a prediction with a higher weight and evaluate error

prediction_a = neural_network(number_of_toes, weight + lr)

error_a = (prediction_a - win_or_lose_binary[0])**2
print(f"Squared Error Up: {error_a}")

# Make a prediction with a lower weight and evaluate error

prediction_b = neural_network(number_of_toes, weight - lr)

error_b = (prediction_b - win_or_lose_binary[0])**2
print(f"Squared Error Down: {error_b}")


# Compare and learn

error_a = error_a[0]
error_b = error_b[0]

if (error > error_b) or (error > error_a):
    if error_b < error_a:
        print("Decreasing weight")
        weight -= lr
    if error_a < error_b:
        print("Increasing weight")
        weight += lr

print(f"New weight {weight}")
