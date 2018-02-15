from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
from sys import argv
import numpy as np
import re
#import urllib2
import csv
from nltk.util import ngrams
import itertools
from collections import Counter
from src.preprocess import preprocess

class ByteFeatures:
    '''
    preprocess byte data for microsoft malware detection project
      convert byte file and labels into a sparse matrix
      parameters:
        - X: pyspark rdd, with (id, rawDoc) format
        - y: pyspark rdd, with (id, label) format
    '''

    def __init__(self, ctx):
        self.sc = ctx

    def byteFeatureGenerator(self, X, y):  # format, (id, dictionary of items)
        '''
        return an rdd of (id, (freq dictionary of items, label))
        '''
        tokenizedX = X \
            .map(lambda x: (self.stripFileNames(x[0]), self.tokenEachDoc(x[1]))) \
            .join(y)

        return tokenizedX

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

    def load_data(self, data_path, labels_path=None):
        '''
        Load the training and testing data from given path
        '''
        data = self.sc.textFile(data_path)
        data = data.zipWithIndex().map(lambda x: (x[1], x[0])) # (id, hash)
        if labels_path is not None:
            labels = self.sc.textFile(labels_path)
            labels = labels.zipWithIndex().map(lambda x: (x[1], x[0])) # (id, label)
        else:
            labels = None
        return data, labels

    def extract_data(self, hash_list, byte_files_path):
        '''Extracts byte files of hash number from the path provided and adds
        all up in one single rdd
        '''
        if byte_files_path[len(byte_files_path)-1] != '/':    # making sure bytes_file_path ends with '/'
            byte_files_path += '/'

        byte_files = self.sc.wholeTextFiles(str(byte_files_path)+str(hash_list[0])+'.bytes')
        for hash in hash_list[1:]:
            byte_files += self.sc.wholeTextFiles(str(byte_files_path)+str(hash)+'.bytes')
        return byte_files #(hash, byte file)

    def write_to_file(self, data, path):
        resTrain = data.collect()
        f = open(path, 'w')
        numDocs = data.count()
        for i in range(numDocs):
            f.write(str(resTrain[i]) + '\n')
        f.close()

    def transform_data(self, data, byte_files_path, labels=None):
        '''Loads the actual data, Extracts features out of it and maps labels with file names i.e. hash
        '''
        if labels is not None:
            labels = data.join(labels) # (id, (hash, label))
            labels = labels.map(lambda x: (x[1][0], x[1][1])) # (hashm label)
        else:
            labels = None
        #self.ngrams = self.sc.broadcast(self.grams)
        hash_list = data.values().collect()
        byte_files = self.extract_data(hash_list, byte_files_path)

        def stripFileNames(stringOfName):
            splits = stringOfName.split("/")
            name = splits[-1][:20]
            return name

        def tokenEachDoc(aDoc):
            '''
            return a dictionary of item-freq, here items are single words and grams
            '''
            tmpWordList = [x for x in re.sub('\\\\r\\\\n', ' ', aDoc).split() if len(x) == 2 and x != '??' """and x != '00'""" ]  # try with last 2 conditions removed
            tmpGramList = []
            grams = [1,2]
            for i in grams:
                # tmpGramList.append([''.join(x) for x in list(ngrams(tmpWordList, grams[i]))])
                tmpGramList +=  list(ngrams(tmpWordList, i))

            # here tmpGramList is a list of list, here we should remove the inner lists
            sumGramList = list(itertools.chain.from_iterable(tmpGramList))  # this is a very long list, depends on the gram numbers
            sumGramDict = dict(Counter(tmpGramList))
            # for x in sumGramDict:
            #     for key in x:
            #         print("\n\n\n")
            #         print(key)
            #         print(type(key))
            #         print("\n\n\n")
            #         x[key] = np.array(x[key])
            retDec = {}                     # ---------- CLEAN THIS PART --------------
            for keys in sumGramDict.keys():
                if sumGramDict[keys] < 10:
                    #del sumGramDict[keys]
                    retDec[keys] = sumGramDict[keys]
            return sumGramDict
        data = byte_files.map(lambda x: (stripFileNames(x[0]), tokenEachDoc(x[1]))) # (hash, bigrams_dict)

        def convertHexToInt(hexStr):
            '''
            convert all hex number to 1-65792, which is one by one.
            '''
            if len(hexStr) == 2:
                return (int(str(hexStr), 16)+1)
            else:
                return (int('1' + str(hexStr), 16)-65279)

        def convertDict(textDict):
            tmp = {}
            for oldKey, value in textDict.items():
                tmp[convertHexToInt(oldKey)] = value
            return tmp
#        data = data.map(lambda x: (x[0], convertDict(x[1]))) # (hash, bigrams_dict)-- bigram_dict's key is integer now
        return data, labels


    def zip_rdds(self, rdd1, rdd2):
        """
        this function joins two rdds
        """
        r1 = rdd1.zipWithIndex().map(lambda x: (x[1], x[0]))   # (index, data)
        r2 = rdd2.zipWithIndex().map(lambda x: (x[1], x[0]))   # (index, data)
        rdd_join = r1.join(r2).map(lambda x: x[1])  # excluding indexes
        return rdd_join

    def process_byte_data(self, byte_data):
        """
        this function excludes all the irrelevant information from a file
        returns only the hexadecimals seperated with a whitespace
        """
        single_string = re.sub(r'\b\w{3,}\b', '', byte_data)
        single_string = re.sub('\\\\r\\\\n', ' ', single_string)
        return single_string

    def get_bytesrdd(self, hash_list, labels_list, bytes_data_path):
        """
        this function returns the byte file documents mapped with corresponding class labels
        returns rdd of ("file_data", label)
        """
        hashes = self.sc.textFile(hash_list)    # hashes rdd
        labels = self.sc.textFile(labels_list)  # labels rdd
        hashlist = hashes.collect()    # list of hashes
        # returns data in each file in form of (filepath, string) RDD[("file1", "file2", "file3", ... ]
        byte_data = self.extract_data(hashlist, bytes_data_path)
        byte_data = byte_data.values()  # has only the bytes files data
        byte_data = byte_data.map(preprocess.process_byte_data)
#        processed_byte_data = byte_data.mapValues(self.process_byte_data)
        labeled_documents = self.zip_rdds(byte_data, labels)    # RDD[("file1", label), ("file2", label), ... ]
        return labeled_documents