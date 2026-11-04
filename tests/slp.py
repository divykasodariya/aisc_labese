import numpy as np 

def train(X , y , lr=0.1 ,epochs=50 ):
    w = np.zeros(X.shape[1])
    b=0
    for _ in range(epochs):
        errors=0
        for x , t in zip(X,y):
            predicted = int(np.dot(x,w) + b >=0)
            diff = t-predicted
            if(diff==0):
                continue
            errors+=1
            update= lr * diff 
            w+=x*update 
            b+=update
        if errors==0:
            break
    return w,b

def predict (x,w,b):
    return int(np.dot(x,w)+b >=0 )



X= np.array([
    [1,1],
    [2,2],
    [3,3],
    [0,1],
    [4,3],
    [5,5]
]
)

y=np.array([-1,1,1,1,1,1])

w,b = train(X,y)
testdata= np.array([
    [1,1],
    [0,1],
    [100,100]
]
)
for x in testdata:
    print(f"input : {x} , prediction : {predict(x,w,b)}")