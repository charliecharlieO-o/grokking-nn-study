"""
    Squared Error amplifies big errors and reduces small errors
    while keeping the error positive in value.
"""

knob_weight = 0.5
input = 0.5
goal_pred = 0.8

pred = input * knob_weight
error = (pred - goal_pred)**2

print(f"Squared Error: {error}")
