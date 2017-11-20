# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:06:23 2017

@author: Ankit
"""
from collections import Counter
from sklearn.metrics import accuracy_score
import numpy as np
import operator
class KNN:
    def __init__(self):
        pass
        
    def predict(self,xtrain,ytrain,tdata,k):
        distVector = [];
        output = [];
        for i in range(xtrain.shape[0]):
            squares = np.square(tdata - (xtrain[i]))
            l2dist = np.sqrt(np.sum(squares))
            #Storing the distance in distance list with the data sample
            label = int(ytrain[i])
            #print(label)
            distVector.append([l2dist,label]);    
        distVector = sorted(distVector,key=operator.itemgetter(0) , reverse=True);
        for i in range(k):
            classlabel = distVector[i][1]
            output.append(classlabel)
        #print((Counter(output).most_common(k)))
        return (Counter(output).most_common(1)[0][0])
    
    def KNearestNeighbours(self,xtrain,ytrain,xtest,ytest,k):
        #k = 1;
        #print(xtrain[0]);
        count = 1
        predictedLabel = []*len(xtest)
        for data in xtest:
            pLab = self.predict(xtrain,ytrain,data,k);
            predictedLabel.append(pLab)
            print('DataPoint :',count)
            count +=1
        #print('Predicted Labels',predictedLabel)
        score = accuracy_score(ytest,predictedLabel);
        return score