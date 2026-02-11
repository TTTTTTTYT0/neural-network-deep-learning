import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score,log_loss
import h5py
from tqdm import tqdm
#scikit-learn

def initialisation(n0,n1,n2):
    
    W1= np.random.randn(n1,n0)
    b1= np.random.randn(n1,1)
    W2= np.random.randn(n2,n1)
    b2= np.random.randn(n2,1)
    
    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2        
    }
    return parametres

def forward_propagation(X, parametres):
    
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']
    
    Z1=W1.dot(X)+b1 
    A1= 1/(1+np.exp(-Z1))
    Z2=W2.dot(A1)+b2 
    A2= 1/(1+np.exp(-Z2))
    
    activations = {
        'A1': A1,
        'A2': A2,        
    }
    
    return activations

def back_propagation(X, y, activations, parametres):
    
    A1= activations['A1']
    A2= activations['A2']
    W2= parametres['W2']
    
    m=y.shape[1]
    
    dZ2= A2-y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1= np.dot(W2.T,dZ2)*A1*(1-A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {
        'dW1' : dW1,
        'db1' : db1,
        'dW2' : dW2,
        'db2' : db2
    }
    
    return gradients

def update( parametres, gradients, pas):
    W1=parametres['W1']
    W2=parametres['W2']
    b1=parametres['b1']
    b2=parametres['b2']
    
    dW1=gradients['dW1']
    dW2=gradients['dW2']
    db1=gradients['db1']
    db2=gradients['db2']
    
    W1=W1 -pas*dW1
    b1=b1-pas*db1
    W2=W2 -pas*dW2
    b2=b2-pas*db2
    
    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2        
    }
    return parametres

def predict(X,parametres):
    activations =forward_propagation(X, parametres)
    A2= activations['A2']
    return A2 >= 0.5

def neural_network (X_train,y_train,X_test,y_test,n1, pas=0.01, n_iter=5000):
    n0 = X_train.shape[0]
    n2= y_train.shape[0]
    parametres =initialisation(n0,n1,n2)
    
    train_loss=[]
    train_acc=[]
    test_loss=[]
    test_acc=[]
    
    for i in tqdm(range (n_iter)):
        
        activations = forward_propagation(X_train, parametres)
        gradients = back_propagation(X_train,y_train,activations, parametres)
        parametres = update(parametres,gradients, pas)
        
        if i%10==0:
            #train
            train_loss.append(log_loss(y_train,activations['A2']))
            y_pred= predict(X_train,parametres)
            current_accuracy = accuracy_score(y_train.flatten(),y_pred.flatten())
            train_acc.append(current_accuracy)
            #test
            activations = forward_propagation(X_test, parametres)
            test_loss.append(log_loss(y_test,activations['A2']))
            y_pred= predict(X_test,parametres)
            current_accuracy = accuracy_score(y_test.flatten(),y_pred.flatten())
            test_acc.append(current_accuracy)
        
        
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(train_acc, label='train accuracy')
    plt.plot(test_acc, label='test accuracy')
    plt.legend()
    
    plt.show()
    
    return parametres

def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    
    return X_train, y_train, X_test, y_test

X_train,y_train,X_test,y_test = load_data()

X_train=X_train.T
X_train= X_train.reshape(-1,X_train.shape[-1]) / X_train.max()
X_test=X_test.T
X_test= X_test.reshape(-1, X_test.shape[-1])/ X_test.max()

y_train=y_train.T
y_test=y_test.T

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

parametres = neural_network(X_train,y_train,X_test,y_test, 32, 0.01, 8000)