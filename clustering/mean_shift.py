# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import random

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)



class MeanShift:
    
    def __init__(self, X, epsilon, iterations):
        self.X_original = np.copy(X)
        self.X = np.copy(X)
        self.N = X.shape[0]
        self.M = X.shape[1]
        self.K= 1
        self.Y = np.zeros((self.N,self.K))
        self.centroids = np.random.rand(self.K,self.M)
        self.past_X = []
        self.epsilon = epsilon,
        self.iterations = iterations
    
    def euclid_distance(self, xi, xj):
        return np.linalg.norm(xi-xj)
    
    def find_neighbors(self, xi):
        # Compute euclidian distance
        dists = np.linalg.norm(self.X - xi,axis=1)
        # Get all points within one epsilon radius
        neighbors = self.X[np.where(dists<=self.epsilon)]
        return neighbors
            
    
    def compute_rbf_kernel(self, xi, xj, bandwidth= 0.5):
        kernel = np.linalg.norm(xi - xj.T)**2 * -bandwidth
        kernel = np.exp(kernel)
        return kernel
    
    def run(self):
        for i in range(self.iterations):
            for idx,xi in enumerate(self.X):
                neighbors = self.find_neighbors(xi)
                numerator = 0
                denominator = 0
                for neighbor in neighbors:
                    weight = self.compute_rbf_kernel(neighbor,xi)
                    numerator += (weight * neighbor)
                    denominator += weight
                    
                new_x = numerator/denominator
                
                self.X[idx,:] = new_x
            
            self.past_X.append(np.copy(self.X))
            self.assign_centroids()
        self.plot()
                    
    def plot(self):
        colors = ['b','g','m','c','r','y','b','w']
        cs = [colors[int(i)] for i in self.Y]
        plt.scatter(self.X_original[:,0],self.X_original[:,1], color=cs)
        plt.scatter(self.centroids[:,0],self.centroids[:,1],color="r")
        plt.show()
    
    def compress_centroids(self, dec=5):
        self.centroids = np.unique(self.X.round(dec),axis=0)
        return self.centroids
    
    def assign_centroids(self):
        self.centroids = self.compress_centroids()
        self.K = np.max(self.centroids.shape)
        X_r = np.tile(self.X_original,(1,self.K))
        centroids_r = self.centroids.ravel()       
        X2 = (X_r - centroids_r)**2
        X3 = np.reshape(X2,(-1,self.M))
        X3 = np.nansum(X3,1)
        X3 = np.reshape(X3,(-1,self.K))
        Y = np.argmin(X3,1)
        self.Y = Y
        return Y
        
    

np.random.seed(100)    
epsilon = 2
iterations = 10
ms = MeanShift(X, epsilon, iterations)
ms.plot()
ms.run()

        