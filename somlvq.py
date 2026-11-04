
import numpy as np
import matplotlib.pyplot as plt


# Training dataset
X = np.array([
    [.1,.1], [.2,.1], [.2,.2], [.7,.8],
    [.8,.7], [.9,.9], [.6,.7], [.3,.2]
])

y = np.array([0,0,0,1,1,1,1,0])


# ---------------- SOM ----------------

# Initialize 2x2 grid with 2 weights per neuron
w = np.random.default_rng(7).random((2,2,2))


# Find Best Matching Unit
def BMU(x):

    # Calculate distance from input to all neurons
    dist = np.linalg.norm(w - x, axis=2)

    # Find index of nearest neuron
    j = np.argmin(dist)

    # Convert index into row and column
    return np.unravel_index(j, (2,2))


# Train SOM
for epoch in range(200):

    # Decrease learning rate
    lr = 0.5 * (1 - epoch/200)

    for x in X:

        # Find nearest neuron
        r, c = BMU(x)

        # Move its weights towards input
        w[r,c] = w[r,c] - lr * (w[r,c] - x)


print("SOM Final Weights:")
print(np.round(w, 3))

# Find BMU for each training sample
for i, x in enumerate(X):
    print("Location", i+1, ": BMU =", BMU(x))


# ---------------- LVQ ----------------

# Initialize prototype weights
p = np.array([
    [.2,.2],
    [.8,.8]
])


# Assign labels to prototypes
labels = []

for q in p:

    # Find nearest training sample
    dist = np.linalg.norm(X - q, axis=1)
    j = np.argmin(dist)

    # Assign its class label
    labels.append(y[j])


# Find nearest prototype
def nearest(x):

    dist = np.linalg.norm(p - x, axis=1)
    return np.argmin(dist)


# Train LVQ
for epoch in range(30):

    for x, t in zip(X, y):

        # Find nearest prototype
        j = nearest(x)

        # If labels match, move towards input
        if labels[j] == t:
            p[j] = p[j] - 0.2 * (p[j] - x)

        # If labels differ, move away from input
        else:
            p[j] = p[j] + 0.2 * (p[j] - x)


# Test the model
test = np.array([[.75,.7]])

j = nearest(test[0])

print("\nLVQ Final Weights:")
print(np.round(p, 3))
print("Prediction:", labels[j])


# ---------------- GRAPHS ----------------

plt.figure(figsize=(10,4))


# SOM graph
plt.subplot(1,2,1)

# Plot training data
plt.scatter(X[:,0], X[:,1], c=y)

# Plot SOM neurons
plt.scatter(w[:,:,0], w[:,:,1], marker='s', s=120, c='red')

plt.title("SOM")


# LVQ graph
plt.subplot(1,2,2)

# Plot training data
plt.scatter(X[:,0], X[:,1], c=y)

# Plot LVQ prototypes
plt.scatter(p[:,0], p[:,1], marker='*', s=200, c='red')

plt.title("LVQ")


# Display both graphs
plt.show()
