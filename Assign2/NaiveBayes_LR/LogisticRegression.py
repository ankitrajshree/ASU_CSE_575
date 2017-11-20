# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:24:16 2017

@author: Ankit
"""
import numpy as np
import math as math
from sklearn.feature_extraction.text import  CountVectorizer

class LogisticRegression:
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.W_Vector = None
        self.learnRate = None
        self.iterations = None
        
    def getFrequencyAndLabels(self, X_train):

        vectorizer = CountVectorizer(input='content', token_pattern=u'(?u)\\b\\w+\\b')  # , analyzer='word')
        dtm = vectorizer.fit_transform(X_train)
        vocab = vectorizer.get_feature_names()
        vocab = np.array(vocab)  # list
        dtm = dtm.toarray();
        b = 20
        return [dtm,vocab]
    
    
    '''The sigmoid function adjusts the cost function hypotheses to adjust 
    the algorithm proportionally for worse estimations   '''   
    
    def Sigmoid(self,z):
        muZ = float(1.0 / float((1.0 + math.exp(-1.0*z))))
        return muZ
    
    def gradient(self,pi,dmat):
        ydata = self.Y_train
        grad = np.dot(dmat.T,(ydata-pi));
        return grad
                        
    def trainModel(self, X_train, Y_train,iterations,learning):
        self.X_train = X_train
        self.Y_train = Y_train
        self.iterations = iterations;
        self.learnRate = learning
        offset = 0 ;
        datamatrix , vocab = self.getFrequencyAndLabels(X_train)
        weightVect = np.zeros(len(vocab));
        for i in range(iterations):
            z = 0;
            for j in range(datamatrix.shape[0]):
                z += np.dot(datamatrix[j],weightVect);
            z = z+offset;
            pi = self.Sigmoid(z);
            grd = self.gradient(pi,datamatrix);
            weightVect = weightVect + learning*grd;
        self.W_Vector = weightVect;
        v = 20
        pass
        
    def predict(self,X_Test):
        predictedValues = []
        fweight = self.W_Vector;
        for testExample in X_Test:
            p = np.dot(X_Test[testExample],fweight)
            if p < 0.5:
                predictedValues.append(0)
            else:
                predictedValues.append(1)
        return predictedValues
    
    def accuracy(self, test_label, predictedValues):
        sum = 0
        for i in range(0,test_label.__len__()):
            if test_label[i] == predictedValues[i]:
                sum += 1

        return float(sum)/test_label.__len__()
        pass
    
    
    
    
    
    