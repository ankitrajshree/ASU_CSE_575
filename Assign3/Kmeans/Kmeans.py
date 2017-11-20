# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 23:56:45 2017

@author: Ankit
"""

import scipy.io as sio
import numpy as np
#import math;
import random;
import matplotlib.pyplot as plt
#from scipy.spatial.distance import cdist

def L2Norm(center,dataPoint):
    distance = 0
    for i in range(len(center)):
        distance += (center[i] - dataPoint[i]) ** 2;
    distance = np.sqrt(distance);
    return distance;
    
def findMembership(centers,dataPoint):
    distDict = {};
    for i in range(len(centers)):
        #print(centers[i])
        #print(dataPoint)
        dist = L2Norm(centers[i],dataPoint);
        distDict[dist] = i;
    sortedDist = sorted(distDict.keys());
    #print(sortedDist);
    memberShip = distDict.get(sortedDist[0])
    distance = sortedDist[0]**2;
    return memberShip,distance;

def genRandCenter(k,totDataPt):
    KIndices = random.sample(range(totDataPt),k)
    return KIndices;

def updateCenters(centers,dataSet,dataMship):
    newCenters = []*len(centers)
    for i in range(len(centers)):
        indices = [index for index, x in enumerate(dataMship) if x == i];
        members = [dataset for index, dataset in enumerate(dataSet) if index in indices];
        newCenter = (sum(members))/float(len(members))
        newCenters.append(newCenter);
    return newCenters;

def compareCenters(oldCenters,newCenters):
    converge = False
    convergeCount = 0
    for i in range(len(oldCenters)):
        centerDiff = L2Norm(oldCenters[i],newCenters[i])
        if centerDiff == 0:
            convergeCount += 1
    if convergeCount == len(oldCenters):
        converge = True;
    return converge

mat = sio.loadmat('kmeans_data', squeeze_me=True);
dataSet = mat['data'];
clusterSize = [2,3,4,5,6,7,8,9,10];
lossFunction = []*len(clusterSize);
for k in clusterSize:    
    centers = []*k;
    centerIndices = genRandCenter(k,7195);
    for index in centerIndices:
        centers.append(dataSet[index]);
    count = 1;
    oldCenters = centers
    while (True):
        dataMemships = []*dataSet.shape[0];
        dataDistances = []*dataSet.shape[0];
        for datapoint in dataSet:
            memShip,distance = findMembership(oldCenters,datapoint);
            dataMemships.append(memShip);
            dataDistances.append(distance);
        #print("Loss Function in iteration :",count,"is",sum(dataDistances))
        #print(sum(dataDistances));
        #print("OldCenter :",oldCenters);
        newCenters = updateCenters(oldCenters,dataSet,dataMemships);
        #print("NewCenter :",newCenters)
        if compareCenters(oldCenters,newCenters):
            lossFunction.append(sum(dataDistances))
            break
        oldCenters = newCenters
        count +=1
plt.plot(clusterSize,lossFunction)
plt.xlabel("Cluster Size")
plt.ylabel("Loss Function")
plt.title("K Means plot (Cluster Size VS Loss Function")
plt.show()



















