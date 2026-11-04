import numpy as np 

# Input dataset
X = np.array([
    [.1,.1], [.2,.1], [.2,.2], [.7,.8],
    [.8,.7], [.9,.9], [.6,.7], [.3,.2]
])


w = np.random.default_rng(7).random((2,2,2))

def winner(x):

    dist = np.linalg.norm(w-x,axis=2)
    j = np.argmin(dist)
    r,c= np.unravel_index(j,(2,2))
    return r,c


for epoch in range(200) :
    for x in X:
        lr = 0.5 * (1-epoch/200)
        r,c = winner(x)
        w[r,c] = w[r,c]- lr*(w[r,c]-x)


train = np.array([0.1,0.3])

print(winner(train))