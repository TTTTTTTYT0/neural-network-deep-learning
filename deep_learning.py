import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score



X,y = make_blobs(n_samples=100, n_features=5, centers=2, random_state=0)
y= y.reshape((y.shape[0], 1))

print('dimension de X',X.shape)
print('dimension de Y',y.shape)


def initialisation(X):
    w= np.random.randn(X.shape[1],1)
    b= np.random.randn(1)
    return (w,b)


def model(X,w,b):
    Z= X.dot(w) + b 
    A= 1/(1+np.exp(-Z))
    return A


def log_loss(A,y):
    return 1/ len(y) * np.sum(-y*np.log(A)-(1-y)*np.log(1-A))


def gradients(A,X,y):
    dw = 1/len(y)* np.dot(X.T,A-y)
    db = 1/len(y)* np.sum(A-y)
    return (dw,db)


def update(dw,db, w, b, pas):
    w=w -pas*dw
    b=b-pas*db
    return(w,b)


def predict(X,w,b):
    A=model(X,w,b)
    print(A)
    return A>=0.5


def A_N (X,y, pas=0.1, n_iter=100):
    #initialisé w,b
    w,b=initialisation(X)
    loss=[]
    
    for i in range (n_iter):
        A=model(X,w,b)
        loss.append(log_loss(A,y))
        dw,db=gradients(A,X,y)
        w,b=update(dw,db,w,b,pas)
    
    y_pred= predict(X,w,b)
    print(accuracy_score(y, y_pred))
    
    plt.plot(loss)
    plt.show()
    
    return(w,b)

w,b = A_N(X,y)

new=np.array([2,1])

x0 = np.linspace(-1, 4,100)
x1 = (-w[0]*x0- b)/w[1]

plt.scatter(X[:,0], X[:,1], c=y, cmap='summer')
plt.scatter(new[0],new[1], c='r')
plt.plot(x0,x1,c='orange',lw=3)
plt.show()

print(predict(new,w,b))