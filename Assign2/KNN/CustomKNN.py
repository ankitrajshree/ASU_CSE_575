# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 00:03:40 2017

@author: Ankit
"""


import numpy as np
import pandas as pd 
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#def loadMNISTImages(filename)

def loadMNISTLabels(self,filename):
    fp = open('X:/study/ASU/FALL_2017/SML/Assignment/Assignment2/DigitsRecognition/train-labels.idx1-ubyte', 'rb');
    magic = fp.read(4);
    magic = int.from_bytes(magic,byteorder='big')
    print(magic)
    if (magic != 2049):
        print('Bad magic number');
        return;
    filebytes = fp.read(4)
    labelCount = int.from_bytes(filebytes, byteorder='big');
    labels = []*labelCount;
    #print(labelCount)
    while True:
        labelsByte = fp.read(1);
        if not labelsByte:
            break;
        label = int.from_bytes(labelsByte, byteorder='big')
     #   print(label)
        labels.append(label);
    return labels;


def loadMNISTImages(self,filename):
    fp = open('X:/study/ASU/FALL_2017/SML/Assignment/Assignment2/DigitsRecognition/train-images.idx3-ubyte', 'rb');
    magic = fp.read(4);
    magic = int.from_bytes(magic,byteorder='big')
    #print(magic)
    if (magic != 2051):
        print('Bad magic number');
        return;
    filebytes = fp.read(4)
    imageCount = int.from_bytes(filebytes, byteorder='big');
    #print(imageCount);
    filebytes = fp.read(4)
    imagesRow = int.from_bytes(filebytes, byteorder='big');
    #print(imagesRow);
    filebytes = fp.read(4)
    imagesCol = int.from_bytes(filebytes, byteorder='big');
    #print(imagesCol);
    imageData = []*(imageCount*imagesCol*imagesRow);
    #print(labelCount)
    while True:
        labelsByte = fp.read(1);
        if not labelsByte:
            break;
        label = int.from_bytes(labelsByte, byteorder='big')
        #print(label)
    imageData.append(label);
    imageMatrix = np.reshape(imageData, (imageCount,imagesRow,imagesCol));
    imageMatrix = float(imageMatrix) / 255;
    return imageMatrix

def getData(filename):
        dataFrm = pd.read_csv(filename);
        return dataFrm;
        
    
def predict(xtrain,ytrain,xtest,k):
        #Distance storing list
        distVector = [];
        output = [];
        
        for i in range(len(xtrain)):
            #calcuate the distance of testdata with every training data
            l2dist = np.sqrt(np.square(xtest-xtrain[i,:]))
            #Storing the distance in distance list with the data sample
            distVector.append([l2dist,i]);
            
        distVector = sorted(distVector);
        
        for i in range(k):
            classlabel = ytrain(distVector[i][1]);
            output.append(classlabel);
            
        return Counter(output).most_common(1)[0][0];
    
def KNearestNeighbours():
        predictedLabels = [];
        k=50;
        Xtrainfile = 'X:/study/ASU/FALL_2017/SML/Assignment/Assignment2/DigitsRecognition/train-images.idx3-ubyte';
        Ytrainfile = 'X:/study/ASU/FALL_2017/SML/Assignment/Assignment2/DigitsRecognition/train-labels.idx1-ubyte'
        Xtestfile = 'X:/study/ASU/FALL_2017/SML/Assignment/Assignment2/DigitsRecognition/t10k-images.idx3-ubyte';
        Ytestfile = 'X:/study/ASU/FALL_2017/SML/Assignment/Assignment2/DigitsRecognition/t10k-labels.idx1-ubyte';
        xtrain = loadMNISTImages(Xtrainfile);
        ytrain = loadMNISTLabels(Ytrainfile);
        xtest = loadMNISTImages(Xtestfile);
        ytest = loadMNISTLabels(Ytestfile);
        scores = []*10;
        for k in [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]:
            for i in range(len(xtest)):
                predictLabel = predict(xtrain,ytrain,xtest,k);
                predictedLabels.append(predictLabel);
                score = accuracy_score(ytest,predictLabel);
                scores.append(score)
                print("Custom KNN Score :",score)
        plt.plot([1, 3, 5, 10, 30, 50, 70, 80, 90, 100],scores)
        plt.xlabel('K Values');
        plt.ylabel('Score');
        plt.title('KNN Accuracy');
        plt.show();
        
                
                
            
            