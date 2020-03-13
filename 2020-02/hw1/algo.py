import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_openml, fetch_20newsgroups

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score




class MyKNeighborsClassifier(BaseEstimator):
    
    def __init__(self, n_neighbors, algorithm='brute'):
        self.engine = {'brute':self.brute_engine, 'kd_tree':self.kd_engine}[algorithm](n_neighbors)

    
    class kd_engine:
        pass

    class brute_engine:
        def __init__(self, NN):
            self.NN=NN

        def fit (self,X,y):
            self.Neibs = np.asarray(X)
            self.Targets = np.asarray(y,dtype = int)
        
        def predict(self, X):
            predicted = np.asarray([],dtype=int)
            Distns = sp.spatial.distance.cdist(X, self.Neibs)
            for i in range(Distns.shape[0]):
                predicted = np.append(predicted, np.argpartition(Distns[i],-self.NN)[-self.NN:])
            predicted = self.Targets[predicted]
            predicted = predicted.reshape(Distns.shape[0], self.NN)

            solution = np.asarray([ np.argmax(np.bincount(i)) for i in predicted] )
            return solution




    def fit(self, X, y):
        self.engine.fit(X,y)
    
    def predict(self, X):
        return self.engine.predict(X)
    