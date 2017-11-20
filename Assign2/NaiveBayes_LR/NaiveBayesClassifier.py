



import numpy as np
from sklearn.feature_extraction.text import  CountVectorizer

#import math

class NaiveBayesClassifier:
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.vocab = None
        self.PriorProb_Rej = None #0
        self.PriorProb_Acc = None #1
        self.Prob_W_Y_Dict = None


        pass

    def getFrequencyAndLabels(self, X_train):

        vectorizer = CountVectorizer(input='content', token_pattern=u'(?u)\\b\\w+\\b')  # , analyzer='word')
        dtm = vectorizer.fit_transform(X_train)
        vocab = vectorizer.get_feature_names()
        dtm = dtm.toarray()  # 4143 * 43624

        # dtm = dtm.T

        vocab = np.array(vocab)  # list

        b = 20

        return [dtm, vocab]


    def Probability_Of_W_Given_Y(self, DocWordMatrix, label, vocab):
        #X_train is d * w matrix (document * words)
        WordDocMatrix = DocWordMatrix.T

        #create vocab freq list for [word,freq in class 0, freq in class 1]
        VocabFreqDict = {}


        rows, cols = WordDocMatrix.shape

        for wordIndex in range(0, vocab.__len__()):
            FreqListPerWord = [0,0,vocab[wordIndex]]

            #FreqListPerWord1 = {vocab[wordIndex]:[0,0]}

            for docIndex in range(0, cols):
                #for each word, go through all the documents and get the freq value. Add this value to FreqListPerWord.
                #label[docIndex] has labels for the correcponding doc( Class 0 or Class 1)
                #based on the label, change the FreqListPerWord for class 0 or class 1
                #print(label[docIndex][0])
                indexToAddTo = int(label[docIndex][0])
                freqToAdd = int(WordDocMatrix[wordIndex][docIndex])
                #FreqListPerWord[label[docIndex+1][0]] = str( int(FreqListPerWord[label[docIndex+1][0]]) + int(WordDocMatrix[wordIndex][docIndex]))

                FreqListPerWord[indexToAddTo] = FreqListPerWord[indexToAddTo] + freqToAdd

                #FreqListPerWord1[vocab[wordIndex][0]][indexToAddTo] = FreqListPerWord[vocab[wordIndex][0]][indexToAddTo] + freqToAdd

                    #freqWord = freqWord + int(WordDocMatrix[wordIndex][docIndex])
            VocabFreqDict[vocab[wordIndex]] = FreqListPerWord


        Prob_W_Y_Dict = dict(VocabFreqDict)

        V = VocabFreqDict.__len__()

        SumClass0 = sum( classList[0] for str, classList in VocabFreqDict.items())
        SumClass1 = sum( classList[1] for str, classList in VocabFreqDict.items())

        for str, classList in Prob_W_Y_Dict.items():
            classList[0] = ((int(classList[0]) + 1) / (SumClass0 + V))
            classList[1] = ((int(classList[1]) + 1) / (SumClass1 + V))

            #classList[0] = ((int(classList[0])) / (SumClass0))
            #classList[1] = ((int(classList[1])) / (SumClass1))

        return Prob_W_Y_Dict

        pass

    def trainModel(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train


        #find dataMatrix and Vocab
        dataMatrix, vocab = self.getFrequencyAndLabels(X_train)

        self.vocab = vocab

        #find prior probability
        label, count = np.unique(self.Y_train, return_counts=True)
        labelCountDict = dict(zip(label, count))
        totalCount = len(self.Y_train)
        self.PriorProb_Rej, self.PriorProb_Acc = [ float(labelFreq)/float(totalCount) for labelVal, labelFreq in labelCountDict.items()]


        #P(w = wi | Y=yk)
        #find Probability of word = w given class 0 or 1
        self.Prob_W_Y_Dict = self.Probability_Of_W_Given_Y(dataMatrix, Y_train, vocab)

        v = 20
        pass


    def predict(self, X_test):


        predictedValues = []
        for testExample in X_test:

            # find dataMatrix and Vocab
            dataMatrix, vocab = self.getFrequencyAndLabels([testExample])

            #find the Numerator of Naive Bayes for Positive and Negative classes
            NrClass0Prob = self.PriorProb_Rej
            NrClass1Prob = self.PriorProb_Acc
            for word in vocab:
                if word in self.Prob_W_Y_Dict.keys():
                    NrClass0Prob *= self.Prob_W_Y_Dict[word][0]
                    NrClass1Prob *= self.Prob_W_Y_Dict[word][1]

                    #NrClass0Prob += math.log(self.Prob_W_Y_Dict[word][0])
                    #NrClass1Prob += math.log(self.Prob_W_Y_Dict[word][1])

            if NrClass0Prob > NrClass1Prob:
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