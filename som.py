
import numpy as np

# Input dataset
X = np.array([
    [.1,.1], [.2,.1], [.2,.2], [.7,.8],
    [.8,.7], [.9,.9], [.6,.7], [.3,.2]
])

# Initialize 2x2 grid with 2 weights per neuron
w = np.random.default_rng(7).random((2,2,2))


# Find Best Matching Unit (BMU)
def winner(x):
    # Calculate distance from input to all neurons
    dist = np.linalg.norm(w - x, axis=2)
    # Find index of neuron with minimum distance
    j = np.argmin(dist)
    # Convert index into row and column
    r, c = np.unravel_index(j, (2,2))
    return r, c


# Train SOM
for epoch in range(200):
    # Decrease learning rate with each epoch
    lr = 0.5 * (1 - epoch/200)
    for x in X:
        # Find nearest neuron
        r, c = winner(x)
        # Move winning neuron's weights towards input
        w[r,c] = w[r,c] - lr * (w[r,c] - x)


# Display final weights
print("SOM Final Weights:")
print(np.round(w, 3))


# Find BMU for each input
for i, x in enumerate(X):

    print("Location", i+1, ": BMU =", winner(x))
