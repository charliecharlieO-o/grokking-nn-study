
weight, goal_pred, input = (0.0, 0.8, 1.1)

for iteration in range(4):
    print(f"----- Weight: {weight} -----")
    prediction = input * weight
    error = (prediction - goal_pred)**2
    delta = prediction - goal_pred
    weight_delta = delta * input
    weight = weight - weight_delta
    print(f"Error: {error} Prediction: {prediction}")
    print(f"Delta: {delta} Weight Delta: {weight_delta}")
