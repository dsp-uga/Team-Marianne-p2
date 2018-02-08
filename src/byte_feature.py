from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
from sys import argv
import numpy as np
import re
import urllib2
import csv
from nltk.util import ngrams
import itertools
from collections import Counter

class preprocessor(object):
    '''
    preprocess byte data for microsoft malware detection project
      convert byte file and labels into a sparse matrix
      parameters:
        - X: pyspark rdd, with (id, rawDoc) format
        - y: pyspark rdd, with (id, label) format
    '''

    def __init__(self, grams=[1, 2], freqThreshold=10):
        self.grams = grams
        self.freqThreshold = freqThreshold

    # helper methods
    def stripFileNames(self, stringOfName):
        splits = stringOfName.split("/")
        name = splits[-1][:20]
        return name

    # token a document, only keep 2-digit code, and its grams
    def tokenEachDoc(self, aDoc):
        '''
        return a dictionary of item-freq, here items are single words and grams
        '''
        tmpWordList = [x for x in re.sub('\\\\r\\\\n', ' ', aDoc).split() if len(x) == 2 and x != '??']
        tmpGramList = []
        for i in xrange(len(self.grams)):
            tmpGramList.append([''.join(x) for x in list(ngrams(tmpWordList, self.grams[i]))])

        # here tmpGramList is a list of list, here we should remove the inner lists
        sumGramList = list(itertools.chain.from_iterable(tmpGramList))  # this is a very long list, depends on the gram numbers
        sumGramDict = dict(Counter(sumGramList))

        for keys in sumGramDict.keys():
            if sumGramDict[keys] < self.freqThreshold:
                del sumGramDict[keys]
        return sumGramDict

    def byteFeatureGenerator(self, X, y):  # format, (id, dictionary of items)
        '''
        return an rdd of (id, (freq dictionary of items, label))
        '''
        tokenizedX = X \
            .map(lambda x: (self.stripFileNames(x[0]), self.tokenEachDoc(x[1]))) \
            .join(y)

        return tokenizedX

    def convertHexToInt(self, hexStr):
        '''
        convert all hex number to 1-65792, which is one by one.
        '''
        if len(hexStr) == 2:
            return (int(str(hexStr), 16)+1)
        else:
            return (int('1' + str(hexStr), 16)-65279)

    def convertDict(self, textDict):
        tmp = {}
        for oldKey, value in textDict.items():
            tmp[self.convertHexToInt(oldKey)] = value
        return tmp

    def convertToSVMFormat(self, Str):
        '''
        convert each line to the SVM format which can be loaded by MLUtils.loadLibSVMFile()
        '''
        newStr = re.sub("\), \(", " ", Str)
        newStr = re.sub("\, ", ":", newStr)
        newStr = re.sub("\)\]", "", newStr)
        newStr = re.sub("\[\(", " ", newStr)
        return newStr

    def convertToSVMFormat2(self, Str):
        '''
        prepare to combine the byte training/test files with asm training/test files.
        '''
        newStr = re.sub("'", "", Str)
        newStr = re.sub("\,  ", "#", newStr)
        newStr = re.sub('[\(\)]', '', newStr)
        return newStr



sc= SparkContext.getOrCreate()
prc = preprocessor()
byteFiles = sc.wholeTextFiles("TrainBytes/")
byteTestFiles = sc.wholeTextFiles("TestBytes/")

trainFileIndex = sc.textFile("X_vs_train.txt").zipWithIndex()\
        .map(lambda x: (x[1], x[0]))
trainFileLabel = sc.textFile("y_vs_train.txt").zipWithIndex()\
        .map(lambda x: (x[1], x[0]))

testFileIndex = sc.textFile("X_vs_test.txt").zipWithIndex()\
        .map(lambda x: (x[1], x[0]))

labelRDD = trainFileIndex.join(trainFileLabel)\
        .map(lambda x: (x[1][0], x[1][1]))
labelRDDTest = testFileIndex.map(lambda x: (x[1], '1'))

trainData = byteFiles.map(lambda x: (prc.stripFileNames(x[0]), prc.tokenEachDoc(x[1])))\
        .map(lambda x: (x[0], prc.convertDict(x[1])))\
        .join(labelRDD)\
        .map(lambda x: (str(x[1][1]) + '-' + str(x[0]), sorted(x[1][0].iteritems(), key = lambda d:d[0])))\
        .map(lambda x: (x[0], prc.convertToSVMFormat(str(x[1]))))\
        .map(lambda x: prc.convertToSVMFormat2(str(x)))

testData = byteTestFiles.map(lambda x: (prc.stripFileNames(x[0]), prc.tokenEachDoc(x[1])))\
        .map(lambda x: (x[0], prc.convertDict(x[1])))\
        .join(labelRDDTest)\
        .map(lambda x: (str(x[1][1]) + '-' + str(x[0]), sorted(x[1][0].iteritems(), key = lambda d:d[0])))\
        .map(lambda x: (x[0], prc.convertToSVMFormat(str(x[1]))))\
        .map(lambda x: prc.convertToSVMFormat2(str(x)))

#write into a file
resTrain = trainData.collect()
f2 = open('trainDataTotal', 'w')
for i in xrange(len(resTrain)):
    f2.write(str(resTrain[i]) + '\n')
f2.close()

resTest = testData.collect()
f3 = open('testDataTotal', 'w')
for i in xrange(len(resTest)):
    f3.write(str(resTest[i]) + '\n')
f3.close()

sc.stop()
