import numpy as np
import pandas as pd

def euc(x,y):
    return np.sqrt(np.sum((np.array(x)-np.array(y))**2))

def alldist(X):
    n = X.shape[0]
    D = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1, n):
            D[i,j] = euc(X.iloc[i], X.iloc[j])
    return D+D.transpose()

def achmat(D,d):
    return np.sign((np.random.randn(D,d) <0)-0.5)

def reduce(X, d):
    A=achmat(X.shape[1],d)
    return np.dot(X,A)/np.sqrt(d)

def distortion(dm1, dm2):
    A = dm2/dm1
    return A[np.triu_indices(len(A),k=1)]
