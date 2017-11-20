# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 01:23:17 2017

@author: Ankit
"""
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
import KNN as knn
import matplotlib.pyplot as plt



def findPCA(trainData):
    X_std = StandardScaler().fit_transform(trainData)
    covMatTdata = np.cov(X_std.T)
    eigenVal, eigenVect = np.linalg.eig(covMatTdata)
    eigenPairs = [(np.abs(eigenVal[i]), eigenVect[:,i]) for i in range(len(eigenVal))]
    eigenPairs.sort(key=lambda x: x[0], reverse=True)
    eigVector50 = [eigenPairs[i][1] for i in range(0,50)]
    npEigVect50 = (np.asarray(eigVector50)).T
    reducedTdata = X_std.dot(npEigVect50)
    return reducedTdata

def splitDataforCV(trainData,trainLabel):
    tSetIdxList = []*5
    splittedTrainData = []*5
    splittedTrainLabel = []*5
    indices = np.random.permutation(trainData.shape[0])
    for i in range(0,5):
        tSet_idx = indices[1000*i:1000*(i+1)]
        tSetIdxList.append(tSet_idx)
    for indexs in tSetIdxList:
        sTData = [data for i,data in enumerate(trainData) if i in indexs]
        sTLabel = [data for i,data in enumerate(trainLabel) if i in indexs]
        splittedTrainData.append(sTData)
        splittedTrainLabel.append(sTLabel)
    return splittedTrainData,splittedTrainLabel



mat = sio.loadmat('knn_data', squeeze_me=True);
#print(sio.whosmat('knn_data'))
testData = mat['test_data']
trainData = mat['train_data']
testLabel = mat['test_label']
trainLabel = mat['train_label']
reducedTrainData = findPCA(trainData)
knn = knn.KNN()
k_accuracyDict = {};
splittedTrainData,splittedTrainLabel = splitDataforCV(reducedTrainData,trainLabel);
KVal = [1, 3, 5, 7, 9, 11, 13, 15, 17]
for ks in KVal:
    score = 0
    for i in range(0,5):
        xtrain = []*4
        xlabel = []*4
        for j in range(0,i):
            xtrain.append(splittedTrainData[j])
            xlabel.append(splittedTrainLabel[j])
        xtest = splittedTrainData[i]
        ytest = splittedTrainLabel[i]
        for k in range(i+1,5):
            xtrain.append(splittedTrainData[k])
            xlabel.append(splittedTrainLabel[k])
        xtrain = np.asarray(xtrain).reshape(4000,50)
        xlabel = np.asarray(xlabel).reshape(4000,1)
        xtest = np.asarray(xtest)
        ytest = np.asarray(ytest)
        score = score + knn.KNearestNeighbours(xtrain,xlabel,xtest,ytest,ks)
    k_accuracyDict[ks] = score/5
accuracyList = []
for k in KVal:
    accuracyList.append(k_accuracyDict[k] * 100)
plt.plot(KVal,accuracyList)
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("CV after PCA plot (K value VS Accuracy")
plt.show()















