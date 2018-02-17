from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
from sys import argv
import numpy as np
import re
import urllib.request
import csv
from nltk.util import ngrams
import itertools
from collections import Counter
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors, SparseVector, _convert_to_vector
from pyspark.sql.types import StructType, StructField, StringType

class ByteFeatures:
    '''
    preprocess byte data for microsoft malware detection project
      convert byte file and labels into a sparse matrix
      parameters:
        - X: pyspark rdd, with (id, rawDoc) format
        - y: pyspark rdd, with (id, label) format
    '''
    def __init__(self, ctx,  grams=[1, 2], freqThreshold=10):
        self.grams = grams
        self.freqThreshold = freqThreshold #frequency selected by 10
        self.sc = ctx

    def loadLibSVMFile(self, sc, data, numFeatures=-1, minPartitions=None, multiclass=None):
    	'''This is function used to load a file and convert it in LabeledPoint RDD
    	   I tweaked this method to directly process my RDD.
    	   refer to : https://spark.apache.org/docs/1.6.3/api/python/_modules/pyspark/mllib/util.html for details
    	'''
    	from pyspark.mllib.regression import LabeledPoint
    	if multiclass is not None:
    		warnings.warn("deprecated", DeprecationWarning)

    	lines = data
    	def _parse_libsvm_line(line, multiclass=None):
    		"""
    		Parses a line in LIBSVM format into (label, indices, values).
    		This method was tweaked so that labels of my training set can be set
    		to be from 0 to 8 instead of default 1 to 9.
    		refer to : https://spark.apache.org/docs/1.6.3/api/python/_modules/pyspark/mllib/util.html for details
    		"""
    		if multiclass is not None:
    			warnings.warn("deprecated", DeprecationWarning)
    		items = line.split(None)
    		label = float(items[0])-1
    		nnz = len(items) - 1
    		indices = np.zeros(nnz, dtype=np.int32)
    		values = np.zeros(nnz)
    		for i in range(nnz):
    			index, value = items[1 + i].split(":")
    			indices[i] = int(index) - 1
    			values[i] = float(value)
    		return label, indices, values
    	parsed = lines.map(lambda l: _parse_libsvm_line(l))
    	if numFeatures <= 0:
    		parsed.cache()
    		numFeatures = parsed.map(lambda x: -1 if x[1].size == 0 else x[1][-1]).reduce(max) + 1
    	return parsed.map(lambda x: LabeledPoint(x[0], Vectors.sparse(numFeatures, x[1], x[2])))

    def convert_to_dataframe(self, sql_context, data, labels):
        data_rdd = data.join(labels) \
        .map(lambda x: (x[0], x[1][0], x[1][1])) # (hashid, feature, label)
        # Creating id, feature, label schema
        schema = StructType([StructField('id', StringType(), True),\
        StructField('feature', StringType(), True),\
        StructField('label', StringType(), True)])
        data = sql_context.createDataFrame(data_rdd, schema)
        return data

    def convert_svmlib(self, sc, args, train_data, train_labels, test_data, test_labels):
        '''This prepares and converts the training and testing data to svmlib format
        '''
        def convertToSVMFormat(Str):
            '''convert each line to the SVM format which can be loaded by MLUtils.loadLibSVMFile()
            '''
            Str = re.sub("\)\, \(", " ", Str)
            Str = re.sub("\, ", ":", Str)
            Str = re.sub("\)\]", "", Str)
            Str = re.sub("\[\(", "", Str)
            s = re.sub("\(\'","",Str)
            s = re.sub("\'\,","",s)
            s = re.sub("\(\'","",s)
            s = re.sub("\'\(","",s)
            s = re.sub("\)\'\)","",s)
            s = re.sub("\'","",s)
            s = re.sub("\)","",s)
            s = re.sub("\(","",s)
            return s
        #Convert train data to svmlib file format
        train_data = train_data.join(train_labels)\
    	     .mapValues(lambda x: (x[1][1], sorted(x[1][0].items(), key = lambda d:d[0])))\
             .mapValues(lambda x: convertToSVMFormat(str(x[1])))\
    		 .map(lambda x:(x[0],( x[1][0], x[1][0]+' '+x[1][1])))
        train_data = ByteFeatures(sc).loadLibSVMFile(sc, train_data.values())
    	# Convert test data to svmlib format
        test_data = test_data.join(test_labels)\
    	     .mapValues(lambda x: (x[1][1], sorted(x[1][0].items(), key = lambda d:d[0])))\
             .mapValues(lambda x: convertToSVMFormat(str(x[1])))\
    		 .map(lambda x: (x[0],(x[1][0], x[1][0]+' '+x[1][1])))
        test_data = ByteFeatures(sc).loadLibSVMFile(sc, test_data.values())
        return train_data, test_data

    def byteFeatureGenerator(self, X, y):  # format, (id, dictionary of items)
        '''
        return an rdd of (id, (freq dictionary of items, label))
        '''
        tokenizedX = X \
            .map(lambda x: (self.stripFileNames(x[0]), self.tokenEachDoc(x[1]))) \
            .join(y)

        return tokenizedX

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

    def write_to_file(self, data, path):
        '''Writes the rdd given in data to the path specified in Path.
           Purely for development purposes
        '''
        resTrain = data.collect()
        f = open(path, 'w')
        for i in range(len(resTrain)):
            f.write(str(resTrain[i]) + '\n')
        f.close()
        return path

    def extract_data_in_rdd(self, hashIds, byte_files_path):
        '''This method extracts the google stotrage based files in single rdd (Much more efficient)
        '''
        def make_urls(hash_id):
            return byte_files_path+'/'+hash_id+'.bytes'
        hashURLs = hashIds.map(lambda x:make_urls(x))
        hashURLs = hashURLs.reduce(lambda x, y: x+','+y)
        print('hashURLs are -----', hashURLs)
        return self.sc.wholeTextFiles(hashURLs)

    def transform_data(self, data, byte_files_path, labels=None):
        '''Loads the actual data, Extracts features out of it and maps labels with file names i.e. hash
        '''
        if labels is not None:
            labels = data.join(labels) # (id, (hash, label))
            labels = labels.map(lambda x: (x[1][0], x[1][1])) # (hashid, label)
        else:
            labels = data.map(lambda x: (x[1], '1'))

        def extract_data(byte_files_path, a):
            '''Extracts byte files of hash number from the path provided
            '''
            if 'http' in byte_files_path:
                with urllib.request.urlopen(byte_files_path+'/'+a+'.bytes') as url:
                    byte_file = url.read().decode('utf-8')
            else:
                file = open(byte_files_path+'/'+a+'.bytes', 'rb')
                byte_file = file.read().decode('utf-8')
            return byte_file #(hash, byte file)
        #byte_files = data.map(lambda x:(x[1], extract_data(byte_files_path,x[1]))) #(hashid, byte_file)
        def stripFileNames(stringOfName):
            splits = stringOfName.split("/")
            name = splits[-1][:20]
            return name
        byte_files = ByteFeatures(self.sc).extract_data_in_rdd (data.values(), byte_files_path)
        byte_files = byte_files.map(lambda x: (stripFileNames(x[0]), x[1]))

        def tokenEachDoc(aDoc):
            '''
            return a dictionary of item-freq, here items are single words and grams
            '''
            tmpWordList = [x for x in re.sub('\\\\r\\\\n', ' ', aDoc).split() if len(x) == 2 and x != '??']
            tmpGramList = []
            grams = [1,2]
            for i in range(len([1,2])):
                tmpGramList.append([''.join(x) for x in list(ngrams(tmpWordList, grams[i]))])

            # here tmpGramList is a list of list, here we should remove the inner lists
            sumGramList = list(itertools.chain.from_iterable(tmpGramList))  # this is a very long list, depends on the gram numbers
            sumGramDict = dict(Counter(sumGramList))
            retDec = {}
            for keys in sumGramDict.keys():
                if sumGramDict[keys] < 10:
                    #del sumGramDict[keys]
                    retDec[keys] = sumGramDict[keys]
            return retDec
        #data = byte_files.map(lambda x: (x[0], tokenEachDoc(x[1]))) # (hash, bigrams_dict)
        def convert_to_feature(doc):
            '''Convert the byte file to feature
            '''
            tmpWordList = [x for x in re.sub('\\\\r\\\\n', ' ', doc).split() if len(x) == 2 and x != '??' and x!='00']
            s= ''.join(str(f)+' ' for f in tmpWordList)
            return s
        data = byte_files.mapValues(lambda x: convert_to_feature(x)) # (hash, bigrams_dict)

        def convertHexToInt(hexStr):
            '''
            convert all hex number to 1-65792(upper limit value), which is one by one.
            '''
            if len(hexStr) == 2:
                return (int(str(hexStr), 16)+1)
            else:
                return (int('1' + str(hexStr), 16)-65279) #

        def convertDict(textDict):
            tmp = {}
            for oldKey, value in textDict.items():
                tmp[convertHexToInt(oldKey)] = value
            return tmp
        #data = data.map(lambda x: (x[0], convertDict(x[1]))) # (hash, bigrams_dict)-- bigram_dict's key is integer now
        return data, labels
