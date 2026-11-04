
import numpy as np

X = np.array([
    [.1, .1], [.2, .1], [.2, .2], [.7, .8],
    [.8, .7], [.9, .9], [.6, .7], [.3, .2]
])

y = np.array([0, 0, 0, 1, 1, 1, 1, 0])

# Initialize prototype weights
w = np.array([
    [0.1, 0.3],
    [0.4, 0]
])

# Assign labels to prototypes
label = []

for p in w:
    dist = np.linalg.norm(X - p, axis=1)
    j = np.argmin(dist)
    label.append(y[j])


# Train LVQ
for epoch in range(500):

    for x, t in zip(X, y):

        # Find nearest prototype
        dist = np.linalg.norm(w - x, axis=1)
        j = np.argmin(dist)

        # If correct, move prototype towards input
        if label[j] == t:
            w[j] = w[j] - 0.2 * (w[j] - x)

        # If incorrect, move prototype away from input
        else:
            w[j] = w[j] + 0.2 * (w[j] - x)


# Test the model
test = np.array([[.75, .7]])

dist = np.linalg.norm(w - test[0], axis=1)
j = np.argmin(dist)

pred = label[j]

print("The weights are:")
print(np.round(w, 3))
print("The prediction is:", pred)
