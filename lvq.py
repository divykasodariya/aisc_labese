
import numpy as np

# Training data
X = np.array([
    [.1, .1], [.2, .1], [.2, .2], [.7, .8],
    [.8, .7], [.9, .9], [.6, .7], [.3, .2]
])

y = np.array([0, 0, 0, 1, 1, 1, 1, 0])

# Initial prototype weights
w = np.array([
    [.2, .2],
    [.8, .8]
])


# Assign labels to initial weights
labels = []

for p in w:

    # Find distance between prototype and all training points
    distances = np.linalg.norm(X - p, axis=1)

    # Find nearest training point
    j = np.argmin(distances)

    # Assign its label to the prototype
    labels.append(y[j])


# Train the LVQ model
for epoch in range(300):

    for x, t in zip(X, y):

        # Find distance from input to each prototype
        distances = np.linalg.norm(w - x, axis=1)

        # Find the nearest prototype
        j = np.argmin(distances)

        # Move prototype towards input if label matches
        if labels[j] == t:
            w[j] = w[j] + 0.2 * (x - w[j])

        # Move prototype away if label does not match
        else:
            w[j] = w[j] - 0.2 * (x - w[j])


# Test the model
test = np.array([[.75, .7]])

# Find nearest prototype for test input
distances = np.linalg.norm(w - test[0], axis=1)
j = np.argmin(distances)

# Display results
print("LVQ Final Weights:")
print(np.round(w, 3))
print("Prediction:", labels[j])
