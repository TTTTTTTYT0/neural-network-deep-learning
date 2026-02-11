import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import h5py
from tqdm import tqdm


def initialisation(X):
    w= np.random.randn(X.shape[1],1)
    b= np.random.randn(1)
    return (w,b)


def model(X,w,b):
    Z= X.dot(w) + b 
    A= 1/(1+np.exp(-Z))
    return A


def log_loss(A,y):
    ep= 1e-15
    return 1/ len(y) * np.sum(-y*np.log(A + ep)-(1-y)*np.log(1-A + ep))


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
    return A>=0.5


def A_N (X_t,y_t,X_te,y_te, pas=0.01, n_iter=1000):
    #initialisé w,b
    w,b=initialisation(X_t)
    t_loss=[]
    t_acc=[]
    te_loss=[]
    te_acc=[]
    
    for i in tqdm(range (n_iter)):
        
        A_t=model(X_t,w,b)
        
        if i%10==0:
            #train
            t_loss.append(log_loss(A_t,y_t))
            y_pred= predict(X_t,w,b)
            t_acc.append(accuracy_score(y_t,y_pred))
            #test
            A_te=model(X_te,w,b)
            te_loss.append(log_loss(A_te,y_te))
            y_pred= predict(X_te,w,b)
            te_acc.append(accuracy_score(y_te,y_pred))
            
        
        dw,db=gradients(A_t,X_t,y_t)
        w,b=update(dw,db,w,b,pas)
        
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(t_loss, label='train loss')
    plt.plot(te_loss, label='test loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(t_acc, label='train accuracy')
    plt.plot(te_acc, label='train accuracy')
    plt.legend()
    plt.show()
    
    return(w,b)

#w,b = A_N(X,y)

def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    
    return X_train, y_train, X_test, y_test

X_t,y_t,X_te,y_te = load_data()
'''
print(X_t.shape)
print(y_t.shape)
print(np.unique(y_t,return_counts=True))

print(X_te.shape)
print(y_te.shape)
print(np.unique(y_te, return_counts=True))

plt.figure(figsize=(16,8))
for i in range(1,10):
    plt.subplot(4,5,i)
    plt.imshow(X_t[i],cmap='gray')
    plt.title(y_t[i])
    plt.tight_layout()
plt.show()
'''

X_t = X_t.reshape(X_t.shape[0],-1) / X_t.max()
X_te= X_te.reshape(X_te.shape[0],-1)/ X_te.max()

w,b = A_N(X_t,y_t,X_te,y_te)

