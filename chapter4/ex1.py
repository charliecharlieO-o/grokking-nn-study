alpha = 0.1
weight = 0.2    # Usually called theta in GD
goal_pred = 1
input = 0.5

for iteration in range(20):
    pred = input * weight
    error = (pred - goal_pred)**2
    derivative = input * (pred - goal_pred)
    weight = weight - (alpha * derivative)
    print(f"Error: {error} Prediction: {pred}")
