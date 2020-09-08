weight = 0.5
input = 0.5
goal_prediction = 0.8

step_amount = 0.001

for iteration in range(1101):
    prediction = input * weight
    error = (prediction - goal_prediction)**2

    # Gradient Descent -> Addresses scaling, negative reversal and stopping
    direction_and_amount = (prediction - goal_prediction) * input
    weight = weight - direction_and_amount

    print(f"Error: {error} Prediction: {prediction}")
