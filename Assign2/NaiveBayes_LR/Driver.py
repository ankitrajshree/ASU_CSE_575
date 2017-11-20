
from sklearn.feature_extraction.text import  CountVectorizer

from sklearn.model_selection import train_test_split
import NaiveBayesClassifier as NBC
import LogisticRegression as LR
import numpy as np
import csv
import datetime
import matplotlib.pyplot as plt


def vectorize(filepath):
    fp = open(filepath, "r")
    lines = fp.readlines()
    vectorizer = CountVectorizer(input='content', token_pattern=u'(?u)\\b\\w+\\b')#, analyzer='word')
    dtm = vectorizer.fit_transform(lines)
    vocab = vectorizer.get_feature_names()
    dtm = dtm.toarray()
    vocab = np.array(vocab)
    print(len(dtm))
    f = open('data.csv', 'w')
    writer = csv.writer(f)
    for index,row in enumerate(dtm.T):
        li = []
        li.append(vocab[index])
        li.extend(row)
        writer.writerow(li)
    f.close()


def getDataAndLabels(datafilename, labefilename):
    fp = open(datafilename, "r")
    lines = fp.readlines()
    fp.close()
    labeldata = np.loadtxt(labelfilename, dtype=np.int64)

    labeldata = np.hsplit(labeldata, 2)

    return [lines, labeldata[1]]


# def getFrequencyAndLabels(datafilename, labefilename):
#     fp = open(datafilename, "r")
#     lines = fp.readlines()
#     vectorizer = CountVectorizer(input='content', token_pattern=u'(?u)\\b\\w+\\b')  # , analyzer='word')
#     dtm = vectorizer.fit_transform(lines)
#     vocab = vectorizer.get_feature_names()
#     dtm = dtm.toarray() #4143 * 43624
#
#     #dtm = dtm.T
#
#     vocab = np.array(vocab) #list
#     #print(len(dtm))
#
#     fp.close()
#
#     #traindata = np.loadtxt('data_words.csv', dtype=None)
#
#     labeldata = np.loadtxt(labelfilename, dtype=np.int64)
#
#     labeldata = np.hsplit(labeldata, 2)
#
#
#     #dtm.(4143, 43624)
#     #newDataMatrix = np.concatenate(traindata, labeldata[1])
#
#
#     b = 20
#
#     return [dtm, labeldata[1], vocab]

if __name__ =='__main__':

    print(datetime.datetime.now())

    datafilename = 'X:/study/ASU/FALL_2017/SML/Assignment/Assignment2/Farmland_Data/farm-ads.txt'
    labelfilename = "X:/study/ASU/FALL_2017/SML/Assignment/Assignment2/Farmland_Data/farm-ads-label.txt"
    #a = vectorize(datafilename)
    b = 20

    #Read Data
    data, labels = getDataAndLabels(datafilename, labelfilename)
    accuracysNB = []*6;
    accuracysLR = []*6;
    for splitRatio in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]:

        print("Iteration "+str(splitRatio))
        # Split the data into the training and test set
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size=splitRatio)



        
        #Naive Bayes Implementation

        # Call Naive Bayes
        nb = NBC.NaiveBayesClassifier()
        nb.trainModel(X_train, Y_train)

        print(datetime.datetime.now())

        #test
        predictedValues = nb.predict(X_test)

        print(datetime.datetime.now())

        #Accuracy
        accuracy = nb.accuracy(Y_test, predictedValues)
        accuracysNB.append(accuracy)
        print("Split Ratio = " + str(splitRatio) + "Accuracy = " + str(accuracy))

        print(datetime.datetime.now())
        
        
        
        #Logistic Regression Implementation
        
        lr = LR.LogisticRegression()
        lr.trainModel(X_train, Y_train,20,0.5)

        predicted_labels = lr.predict(X_test)

        #Accuracy
        accuracy = nb.accuracy(Y_test, predictedValues)
        accuracysLR.append(accuracy)
        print("Split Ratio = " + str(splitRatio) + "Accuracy = " + str(accuracy))

        print("Accuracy of model is {0}%", accuracy)
        
        
    plt.plot([0.1, 0.3, 0.5, 0.7, 0.8, 0.9],accuracysNB)
    plt.xlabel('Split ratio');
    plt.ylabel('Accuracy');
    plt.title('Naive Bayes Accuracy');
    plt.show();




    cv = 20