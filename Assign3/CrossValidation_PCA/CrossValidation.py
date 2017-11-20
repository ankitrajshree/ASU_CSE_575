# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:15:28 2017

@author: Ankit
"""
import KNN as knn
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

'''def splitDataforCV(trainData,trainLabel):
    np.random.shuffle(trainData)
    splitDataLst = []*5
    splitLabelLst = []*5
    for i in range(0,5):
        indices = numpy.random.permutation(trainData.shape[0])
        splitDataPart = trainData.ix[rows]
        splitLabelPart = trainLabel.ix[rows]
        trainData = trainData.drop(rows)
        splitDataLst.append(splitDataPart)
        splitLabelLst.append(splitLabelPart)
    return  splitDataLst,splitLabelLst'''
    
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
'''print(trainData[3023])
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
#tSet1_idx,tSet2_idx,tSet3_idx,tSet4_idx,tSet5_idx = indices[0:1000], indices[1000:]
#np.random.shuffle(trainData)
#splittedTrainSet = np.array_split(trainData, 5, axis=0)
#trainDataDf = pd.DataFrame(trainData)
#trainLabelDf = pd.DataFrame(trainData)
#print(trainDataDf.index)
#k=1'''
knn = knn.KNN()
k_accuracyDict = {};
splittedTrainData,splittedTrainLabel = splitDataforCV(trainData,trainLabel);
KVal = [1, 3, 5, 7, 9, 11, 13, 15, 17]
'''for ks in KVal:
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
        xtrain = np.asarray(xtrain).reshape(4000,166)
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
plt.title("CV plot (K value VS Accuracy")
plt.show()
    
finalK = 5
score = knn.KNearestNeighbours(trainData,trainLabel,testData,testLabel,finalK)
#print(xlabel[0][5])
#print(score)'''









