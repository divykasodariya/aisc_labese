
import numpy as np

# Sigmoid activation function (output 0 to 1)
def sig(x):
    return 1/(1+np.exp(-np.clip(x,-50,50)))


# Training input: 4 features per sample
X = np.array([[.1,.1,.1,.1],[.2,.1,.2,.2],
              [.4,.3,.4,.4],[.7,.8,.6,.7],
              [.8,.7,.8,.8],[.9,.9,.9,.9]])

# Actual output (target labels)
y = np.array([[0],[0],[0],[1],[1],[1]])

# Random weights initialize (42 = fixed random seed)
rng = np.random.default_rng(42)

# Input to hidden layer: 4 inputs, 6 neurons
W1 = rng.normal(0,.5,(4,6))

# Hidden layer bias (initially zero)
b1 = np.zeros((1,6))

# Hidden to output layer: 6 neurons, 1 output
W2 = rng.normal(0,.5,(6,1))

# Output layer bias (initially zero)
b2 = np.zeros((1,1))

# Learning rate
lr = .8

# Training for 5000 epochs
for _ in range(5000):

    # Forward propagation: calculate hidden layer output
    a1 = sig(X @ W1 + b1)

    # Forward propagation: calculate final output
    out = sig(a1 @ W2 + b2)

    # Output layer error (backpropagation)
    d2 = (out-y)*out*(1-out)

    # Hidden layer error
    d1 = (d2 @ W2.T)*a1*(1-a1)

    # Update output layer weights
    W2 -= lr*a1.T@d2/len(X)

    # Update output layer bias
    b2 -= lr*d2.mean(0,keepdims=True)

    # Update hidden layer weights
    W1 -= lr*X.T@d1/len(X)

    # Update hidden layer bias
    b1 -= lr*d1.mean(0,keepdims=True)

# New test input (4 features)
test = np.array([[.8,.8,.7,.8]])
# Predict probability using trained weights and bias
result = sig(sig(test@W1+b1)@W2+b2)
# Print final probability (4 decimal places)
print("High-risk probability:", round(float(result[0,0]),4))
