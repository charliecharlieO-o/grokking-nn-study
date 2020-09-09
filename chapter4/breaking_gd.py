"""
    When the input is too big we the weight update also increases,
    this combined with a small error will result in overcorrecting.
    This phenomena is called divergence.

    This is where gradient descent's alpha comes from, by multiplying
    the weight by a fraction we make it smaller.
"""

weight = 0.5
goal_pred = 0.8
input = 2

for iteration in range(20):
    pred = input * weight
    error = (pred - goal_pred)**2   # Squared error
    delta = pred - goal_pred    # Error
    weight_delta = input * delta
    weight = weight - weight_delta
    print(f"Error: {error} Prediction: {pred}")
