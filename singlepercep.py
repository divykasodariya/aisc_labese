
import numpy as np


# Function to train the perceptron
def train(X, y, lr=0.2, epochs=50):
    # Initialize all weights and bias to zero
    # X.shape[1] gives the number of input features
    w = np.zeros(X.shape[1])
    b = 0

    # Repeat training for maximum number of epochs
    for epoch in range(epochs):

        errors = 0

        # Take one input sample and its actual output at a time
        for x, t in zip(X, y):

            # Calculate weighted sum and predict output
            # Output is 1 if sum >= 0, otherwise 0
            p = int(np.dot(x, w) + b >= 0)

            # Calculate difference between actual and predicted output
            update = lr * (t - p)

            # Adjust weights and bias
            w += update * x
            b += update

            # Count incorrect predictions
            if update != 0:
                errors += 1

        # Stop training if there are no errors
        if errors == 0:
            break

    # Return the final trained weights and bias
    return w, b


# Function to predict output for new inputs
def predict(X, w, b):

    predictions = []

    # Calculate output for every test sample
    for x in X:

        # Apply the same weighted sum and activation function
        p = int(np.dot(x, w) + b >= 0)

        predictions.append(p)

    return np.array(predictions)


if __name__ == "__main__":

    print("EXPERIMENT 4 - SINGLE LAYER PERCEPTRON")

    # Training input data (6 samples, 2 features each)
    X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ])

    # Actual output labels for training data
    y = np.array([1, 1, 1 ,0])

    # Train the perceptron
    w, b = train(X, y)

    # New data on which we want to make predictions
    test = np.array([
        [0, 0],
        [1, 1]
    ])

    # Display trained weights and bias
    print("Weights:", np.round(w, 3))
    print("Bias:", round(b, 3))

    # Predict output for test data
    print("Predictions:", predict(test, w, b))
