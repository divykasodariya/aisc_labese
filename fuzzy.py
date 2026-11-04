
import numpy as np

# ---------------- DATA GENERATION ----------------

np.random.seed(42)

n = 1000

# Generate input data
temp = np.random.uniform(10, 40, n)
humidity = np.random.uniform(20, 90, n)
occupancy = np.random.randint(0, 7, n)
appliances = np.random.uniform(1, 5, n)

# Calculate energy consumption
energy = (2 + 0.1*temp + 0.02*humidity +
          0.5*occupancy + 2*appliances +
          np.random.normal(0, 0.5, n))

# Combine all inputs
X = np.column_stack((temp, humidity, occupancy, appliances))

# Normalize inputs
X = (X-X.min(axis=0)) / (X.max(axis=0)-X.min(axis=0))

# Split training and testing data
idx = np.random.permutation(n)
split = int(0.8*n)

X_train = X[idx[:split]]
X_test = X[idx[split:]]

y_train = energy[idx[:split]]
y_test = energy[idx[split:]]


# ---------------- MEMBERSHIP FUNCTIONS ----------------

# Two Gaussian functions for each of four inputs
centers = np.array([
    [.25, .25, .25, .25],
    [.75, .75, .75, .75]
])

sigma = np.ones((2,4)) * 0.5


# Generate 16 rules
rules = []

for a in range(2):
    for b in range(2):
        for c in range(2):
            for d in range(2):
                rules.append([a,b,c,d])


# Calculate rule firing strengths
def firing_strength(x):

    # Calculate membership values
    mu = np.zeros((2,4))

    for i in range(2):
        for j in range(4):
            mu[i,j] = np.exp(-((x[j]-centers[i,j])**2) /
                              (2*sigma[i,j]**2))

    # Calculate firing strength of each rule
    firing = np.ones(16)

    for i in range(16):
        for j in range(4):
            firing[i] *= mu[rules[i][j],j]

    # Normalize firing strengths
    firing = firing / (np.sum(firing) + 1e-8)

    return firing


# ---------------- TRAINING ----------------

# 16 rules, each having 5 consequent parameters
weights = np.random.randn(16,5) * 0.1

lr = 0.01

for epoch in range(1000):

    gradients = np.zeros((16,5))

    for x,t in zip(X_train, y_train):

        # Calculate firing strengths
        w = firing_strength(x)

        # Add bias (last value is 1)
        f = np.append(x, 1)

        # Calculate output of each rule
        rule_output = weights @ f

        # Calculate final prediction
        pred = np.sum(w * rule_output)

        # Calculate error
        error = pred - t

        # Calculate gradients for all rules
        for i in range(16):
            gradients[i] += 2 * error * w[i] * f

    # Update weights after processing all samples
    gradients = gradients / len(X_train)
    weights = weights - lr * gradients


# ---------------- PREDICTION ----------------

def predict(x):

    w = firing_strength(x)
    f = np.append(x, 1)

    rule_output = weights @ f

    return np.sum(w * rule_output)


predicted = []

for x in X_test:
    predicted.append(predict(x))

predicted = np.array(predicted)


# ---------------- PERFORMANCE ----------------

mse = np.mean((y_test-predicted)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test-predicted))

r2 = 1 - np.sum((y_test-predicted)**2) / np.sum(
    (y_test-y_test.mean())**2
)

print("Final Weights:")
print(np.round(weights, 3))

print("\nRMSE:", round(rmse, 3))
print("MAE:", round(mae, 3))
print("R2 Score:", round(r2, 3))
