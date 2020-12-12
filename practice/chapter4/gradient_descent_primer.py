weight = 0.5
goal_pred = 0.8
input = 0.5

for iteration in range(20):
    pred = input * weight
    error = (pred - goal_pred)**2
    # Pure error times input scaling to ge theta
    direction_and_amount = (pred - goal_pred) * input
    # Weight update
    weight = weight - direction_and_amount

    print(f"Error: {error} Prediction: {pred}")
