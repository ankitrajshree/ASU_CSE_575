# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:06:52 2017

@author: Ankit
"""
from sklearn.feature_extraction.text import CountVectorizer as cvect
#import pandas as pd 
import numpy as np

feature = [];
ads = [];
#Remove the duplicate words from the line
def uniqueString(inputLine):
    output = [];
    seen = set();
    for word in line.split():
        if word not in seen:
            output.append(word);
            seen.add(word);
    return output;

fp = open('X:/study/ASU/FALL_2017/SML/Assignment/Assignment2/Farmland_Data/test.txt');
while 1:
    line = fp.readline();
    if not line:
        break
    #remove the \n character and leading spaces
    line = line.lstrip();
    line = line.rstrip();
    #Creating an list of all the adds
    ads.append(line);
    #Remove the duplicate words
    output = uniqueString(line);
    for i in range(len(output)):
        feature.append(output[i]);
features = list(set(feature));
#print(features);
#print(len(features))

cv = cvect(vocabulary = features);
arr = cv.fit_transform(ads).toarray();
arrNumpy = np.asarray(arr);
np.savetxt("X:/study/ASU/FALL_2017/SML/Assignment/Assignment2/Python/testData.csv" ,arrNumpy.astype(int), delimiter=",")
#arr = arr.transpose();
#print(arr);
#df = pd.DataFrame(arr)
#df.to_csv("X:/study/ASU/FALL_2017/SML/Assignment/Assignment2/Python/testData.csv",index=False,header=False)



















