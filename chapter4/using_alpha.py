"""
    Implementing alpha as mentioned in breaking_gd.py
"""

alpha = 0.1
weight = 0.5
goal_pred = 0.8
input = 2

for iteration in range(20):
    pred = input * weight
    error = (pred - goal_pred)**2   # Squared error
    derivative = input * (pred - goal_pred)
    weight = weight - (alpha * derivative)
    print(f"Error: {error} Prediction: {pred}")
