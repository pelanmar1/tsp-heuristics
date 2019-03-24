
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import random

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

#plt.scatter(X[:, 0], X[:, 1], s=50);


class KMeans:
    
    def __init__(self,K, X, iterations):
        self.K = K
        self.X = X
        self.N = X.shape[0]
        self.M = X.shape[1]
        self.centroids = self.init_centroids()
        self.Y = np.zeros((self.N,1))
        self.iterations = iterations
    
    def init_centroids(self):
        min_xi = np.amin(X,0)
        max_xi = np.amax(X,0)
        centroids = np.random.rand(self.K, self.M)
        centroids = centroids*(max_xi-min_xi) + min_xi
        return centroids
    
    def assign_centroids(self):
        X_r = np.tile(self.X,(1,self.K))
        centroids_r = self.centroids.ravel()       
        X2 = (X_r - centroids_r)**2
        X3 = np.reshape(X2,(-1,self.M))
        X3 = np.nansum(X3,1)
        #X3 = np.sqrt(X3)
        X3 = np.reshape(X3,(-1,self.K))
        Y = np.argmin(X3,1)
        self.Y = Y
        
    def recalc_centroids(self):
        for i in range(self.K):
            self.centroids[i] = X[np.where(self.Y==i)].mean(0)
    
    def run(self):
        self.plot()
        for i in range(self.iterations):
            self.assign_centroids()
            self.plot()

            self.recalc_centroids()
        self.plot()
            
    def plot(self):
        colors = ['b','g','m','c','r','y','b','w']
        cs = [colors[int(i)] for i in self.Y]
        plt.scatter(self.X[:,0],self.X[:,1],color = cs)
        plt.scatter(self.centroids[:,0],self.centroids[:,1],color="r")
        plt.show()
        

np.random.seed(100)

km = KMeans(4, X, 100)

Y = km.run()