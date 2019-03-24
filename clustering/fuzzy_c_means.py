# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import random

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

#plt.scatter(X[:, 0], X[:, 1], s=50);


class FuzzyCMeans:
    
    def __init__(self,K, X, m, iterations):
        self.K = max(K,1)
        self.X = X
        self.N = X.shape[0]
        self.M = X.shape[1]
        self.m = m
        self.U = self.init_U()
        self.centroids = self.init_centroids()
        self.Y = np.zeros((self.N,1))
        self.iterations = iterations
    
    def init_U(self):
        num_cols = self.N
        num_rows = self.K
        matrix = np.random.rand(num_rows,num_cols)
        matrix /= np.sum(matrix, axis=1).reshape(-1,1)
        return matrix
        
    def init_centroids(self):
        min_xi = np.amin(X,0)
        max_xi = np.amax(X,0)
        centroids = np.random.rand(self.K, self.M)
        centroids = centroids*(max_xi-min_xi) + min_xi
        return centroids
    
        
    def recalc_U(self):
        c_r = self.centroids.ravel()
        X_r = np.tile(X,(1,self.K))
        X2 = (X_r - c_r)
        X3 = np.reshape(X2,(-1,self.M))
        X3 = np.linalg.norm(X3,axis=1).reshape((-1,1))
        X3 = np.reshape(X3,(-1,self.K))
        X3 = X3 ** (2/(self.m-1))
        X4 = X3*((1/X3).sum(1).reshape(-1,1))
        X4 = 1/X4
        self.U = X4.T
        
            
    def compute_error(self):
        c_r = self.centroids.ravel()
        X_r = np.tile(X,(1,self.K))
        X2 = (X_r - c_r)
        X3 = np.reshape(X2,(-1,self.M))
        X3 = np.linalg.norm(X3,axis=1)
        X3 = X3**2
        X4 = np.reshape(X3,(-1,self.K))
        X4 = X4*self.U.T
        J = X4.sum()
        return J
        
    def recalc_centroids(self):        
        for i in range(self.K):
            u = self.U[i,:].reshape((-1,1))
            u = u**self.m
            numerator = np.nansum(u * X,0)
            denominator = np.nansum(u)
            ci = numerator/denominator
            self.centroids[i,:] = ci
                
    def run(self):
        for i in range(self.iterations):
            self.recalc_centroids()
            self.recalc_U()
        self.plot()            
    def plot(self):
        colors = ['b','g','r','c','m','y','w']
        cs = [colors[np.argmax(i)] for i in self.U.T]
        afs = [ min(1,i[np.argmax(i)]) for i in self.U.T]
        for idx,i in enumerate(self.X):
            plt.scatter(self.X[idx,0],self.X[idx,1],color = cs[idx], alpha=afs[idx])
        plt.scatter(self.centroids[:,0],self.centroids[:,1],color="k",marker="x")
        plt.show()
    
np.random.seed(100)
cm = FuzzyCMeans(4, X, 2, 10)
cm.run()