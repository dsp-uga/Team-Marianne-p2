
import argparse
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel, LogisticRegressionWithLBFGS, LogisticRegressionModel
import re
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from sys import argv
import numpy as np
import urllib.request
import csv
from nltk.util import ngrams
import itertools
from collections import Counter
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors, SparseVector
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, NGram, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructType, StructField, StringType

class SparkDFMl:
    '''This class provides functions to be used to implement ML models in dataframes
    '''
    def __init__(self, sc):
        self.sc = sc

    def featurize_data(sc, data):
        '''This method converts the raw bigram features into tfidf vectors
        '''
        # Tokenize the strings
        tokenizer = Tokenizer(inputCol='feature', outputCol='feature_tokens')
        tokenized_train_df = tokenizer.transform(data)
        # Extract bigrams out of strings
        ngram = NGram(n=2, inputCol='feature_tokens', outputCol='ngrams')
        bigram_train_df = ngram.transform(tokenized_train_df)
        # Converting to hashing features

        hashing_tf = HashingTF(inputCol='ngrams', outputCol='features')
        tf = hashing_tf.transform(bigram_train_df)
        tf.show()
        string_indexer = StringIndexer(inputCol='label', outputCol='label_numeric')
        rescaled_data_numeric = string_indexer.fit(tf).transform(tf)
        ret_df = rescaled_data_numeric.selectExpr('id as id', 'label_numeric as label', 'features as features')
        # Converting to counts to IDF
        #idf = IDF(inputCol='rawFeatures', outputCol='features')
        #df_model = idf.fit(tf)
        #rescaled_data = idf_model.transform(tf)
        # multiplying tf and idf scores
        #tfidf = tf.join(rescaled_data, ['id', 'label']).select('id', 'label', (tf.rawFeatures*rescaled_data.features_idf).alias('feature'))
        # Converting label (which was StringValue ) to Numeric value
        #string_indexer = StringIndexer(inputCol='label', outputCol='label_numeric')
        #rescaled_data_numeric = string_indexer.fit(tfidf).transform(rescaled_data)
        # renaming the label column
        #ret_df = rescaled_data_numeric.selectExpr('id as id', 'label_numeric as label', 'features as features')
        return ret_df

    def naive_bayes(sc, train_df, test_df):
        '''This is implementation of Naive Bayes Algorithm using dataframes
        '''
        train_data = SparkDFMl(sc).featurize_data(train_df)
        test_data = SparkDFMl(sc).featurize_data(test_df)
        nb = NaiveBayes(smoothing=1.0, modelType='multinomial')
        nb_model = nb.fit(train_data)
        predictions = nb_model.transform(test_data)
        predictions.select('prediction').show(truncate = False)

        evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
        accuracy = evaluator.evaluate(predictions)
        print('Accuracy is---' + str(accuracy))

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
                if len(items[1 + i].split(":")) is 2:
                    index, value = items[1 + i].split(":")
                    indices[i] = int(index) - 1
                    values[i] = float(value)
                else:
                    print('error is --- ', items[1 + i])
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
            Str = re.sub("\: ", "", Str)
            Str = re.sub(" \:", "", Str)
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
    	     .map(lambda x: (x[1][1], sorted(x[1][0].items(), key = lambda d:d[0])))\
             .mapValues(lambda x: convertToSVMFormat(str(x)))\
    		 .map(lambda x:(x[0], x[0]+' '+x[1]))
        train_data = ByteFeatures(sc).loadLibSVMFile(sc, train_data.values())
    	# Convert test data to svmlib format
        test_data = test_data.join(test_labels)\
    	     .map(lambda x: (x[1][1], sorted(x[1][0].items(), key = lambda d:d[0])))\
             .mapValues(lambda x: convertToSVMFormat(str(x)))\
    		 .map(lambda x: (x[0], x[0]+' '+x[1]))
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
        def make_urls(hash_id):
            return byte_files_path+'/'+hash_id+'.bytes'
        hashURLs = hashIds.map(lambda x:make_urls(x))
        hashURLs = hashURLs.reduce(lambda x, y: x+','+y)
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
        #byte_files = byte_files.map(lambda x: (stripFileNames(x[0]), x[1]))

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
        data = byte_files.map(lambda x: (stripFileNames(x[0]), tokenEachDoc(x[1]))) # (hash, bigrams_dict)
        def convert_to_feature(doc):
            '''Convert the byte file to feature
            '''
            tmpWordList = [x for x in re.sub('\\\\r\\\\n', ' ', doc).split() if len(x) == 2 and x != '??' and x!='00']
            s= ''.join(str(f)+' ' for f in tmpWordList)
            return s
        #data = byte_files.mapValues(lambda x: convert_to_feature(x)) # (hash, bigrams_dict)

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
        data = data.map(lambda x: (x[0], convertDict(x[1]))) # (hash, bigrams_dict)-- bigram_dict's key is integer now
        return data, labels

def loadLibSVMFile(sc, data, numFeatures=-1, minPartitions=None, multiclass=None):
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

class SparkRDDMl:
    ''' This class provides functions to be used to implement ML models in rdds
    '''
    def __init__(self, sc) :
        self.sc = sc

    def random_forest_classification(self, sc, args, train_data, test_data):
        '''This does the Random forest classification for given train and test set
        '''
        # Create model and make prediction
        model = RandomForest.trainClassifier(train_data, numClasses=9, categoricalFeaturesInfo={},\
                                            numTrees=100, featureSubsetStrategy="auto",\
                                         impurity='gini', maxDepth=4, maxBins=32)
        predictions = model.predict(test_data.map(lambda x: x.features))
        print('prediction is done')
        resTrain = predictions.collect()
        print('==============================================================')
        for i in range(len(resTrain)):
            print(str(int(resTrain[i]+1)))
        print('==============================================================')
        if(args.evaluate):
            SparkRDDMl(sc).score(predictions,test_data, model, args.mlModel)
        SparkRDDMl(sc).write_output(predictions, args.output)

    def naive_bayes_mllib(self, sc, args, train_data, test_data):
        '''This does the Naive Bayes Classification for given train and test set
        '''
        # Create model and make predictions
        model = NaiveBayes.fit(train_data, 1.0)
        predictions = model.predict(test_data.map(lambda x: x.features))
        print('prediction is done')
        resTrain = predictions.collect()
        print('==============================================================')
        for i in range(len(resTrain)):
            print(str(int(resTrain[i]+1)))
        print('==============================================================')
        if(args.evaluate):
            score(predictions,test_data, model, args.mlModel)
        write_output(predictions, args.output)

    def logistic_regression_classification(self, sc, args, train_data, test_data):
    	'''This does the logistic regression classification for given train and test set
    	'''
    	# create model and make prediction
    	model = LogisticRegressionWithLBFGS.train(train_data, iterations=100, numClasses=9)
    	predictions = model.predict(test_data.map(lambda x: x.features))
    	if(args.evaluate):
    		score(predictions,test_data, model, args.mlModel)
    	write_output(predictions, args.output)

    def svm_classification(self, sc, args, train_data, test_data):
    	'''This does the svm classification for given train and test set
    	'''
    	# create model and make prediction
    	model = SVMWithSGD.train(train_data, iterations=100, numClasses=9)
    	predictions = model.predict(test_data.map(lambda x: x.features))
    	if(args.evaluate):
    		score(predictions,test_data, model, args.mlModel)
    	write_output(predictions, args.output)

    def write_output(self, predictions, path):
    	'''Write the output to file given in the args output path
    	'''
    	resTrain = predictions.collect()
    	f = open(path, 'w')
    	for i in range(len(resTrain)):
    		f.write(str(int(resTrain[i]+1)) + '\n')
    	f.close()
    	return path

    def score(self, predictions,test_data, model, mlModel):
    	'''Prints out the accuracy accuracy
    	'''
    	print('score called')
    	labelsAndPredictions = test_data.map(lambda x: x.label).zip(predictions)
    	testErr = labelsAndPredictions.filter(lambda lp: lp[0] == lp[1]).count() / float(test_data.count())
    	print('For '+mlModel+' Accuracy = ' + str(testErr))

def main(args):
	# Refer https://spark.apache.org/docs/latest/configuration.html for additional changes to config
	config = SparkConf().setAppName("team-marianne-p2")\
				.set("spark.hadoop.validateOutputSpecs", "false")\

	sc = SparkContext.getOrCreate(config)
	sql_context = SQLContext(sc)
	# Preprocess training data
	train_data, train_labels = ByteFeatures(sc).load_data(args.dataset, args.labels)	# converts training byte data and labels to RDDs
	# training: updates data and labels rdds. labels will be updated to (hash, label) and data will be updated to (hash, bigram_dict)
	train_data, train_labels = ByteFeatures(sc).transform_data(train_data, args.bytestrain, train_labels)
	# Preprocdess testing data
	test_data, test_labels = ByteFeatures(sc).load_data(args.testset, args.testlabels)	# converts testing byte data and labels to RDDs
	# testing: updates data and labels rdds. labels will be updated to (hash, label) and data will be updated to (hash, bigram_dict)
	test_data, test_labels = ByteFeatures(sc).transform_data(test_data, args.bytestest, test_labels)

	# Convert the RDDs to dataframes while combining features and labels for model creation
	train_df = ByteFeatures(sc).convert_to_dataframe(sql_context, train_data, train_labels)
	test_df = ByteFeatures(sc).convert_to_dataframe(sql_context, test_data, test_labels)
	# Execute the Naive Bayes algorithm for dataframes
	SparkDFMl(sc).naive_bayes(train_df, test_df)

	#train_data, test_data = ByteFeatures(sc).convert_svmlib(sc, args, train_data, train_labels, test_data, test_labels)

	if args.mlModel is 'rf':
		print('Random Forest called')
		SparkRDDMl(sc).random_forest_classification(sc, args, train_data, test_data)
	elif args.mlModel is 'nbs':
		print('Naive Bayes called')
		SparkRDDMl(sc).naive_bayes_mllib(sc, args, train_data, test_data)
	elif args.mlModel is 'lr':
		print('Logistic Regression called')
		SparkRDDMl(sc).logistic_regression_classification(sc, args, train_data, test_data)
	elif args.mlModel is 'svm':
		print('svm called')
		SparkRDDMl(sc).logistic_regression_classification(sc, args, train_data, test_data)
parser = argparse.ArgumentParser(description='Team Marianne solution for Malware Classification')

# All args are optional. Default values are set for each argument
parser.add_argument ("-d", "--dataset", default="data/sample/X_vs_train.txt",
    help = "Path to text file containing hash of documents in training set")

parser.add_argument ("-l", "--labels", default="data/sample/y_vs_train.txt",
    help = "Path to text file containing labels of documents in training set")

parser.add_argument ("-t", "--testset", default="data/sample/X_vs_test.txt",
    help = "Path to text file containing hash of documents in testing set")

parser.add_argument ("-e", "--evaluate", action="store_true",
    help = "Set this to evaluate accuracy on the test set")

parser.add_argument ("-m", "--testlabels",
    help = "Path to text file containing labels of documents in testing set."
            "If evaluate is set true, this file is compared with classifier output")

parser.add_argument ("-a", "--asmtrain", default="data/sample/TrainAsm/",
    help = "Path to directory that contains asm documemts of training set")

parser.add_argument ("-at", "--asmtest", default="data/sample/TestAsm/",
    help = "Path to directory that contains asm documemts of testing set")

parser.add_argument ("-b", "--bytestrain", default="data/sample/TrainBytes/",
    help = "Path to directory that contains bytes documemts of training set")

parser.add_argument ("-bt", "--bytestest", default="data/sample/TestBytes/",
    help = "Path to directory that contains bytes documemts of testing set")

parser.add_argument ("-A", "--asmrdd", default="data/sample/asm_rdd.txt",
    help = "Path to text file in which RDD from asm file is stored after preprocessing")

parser.add_argument ("-B", "--bytesrdd", default="data/sample/bytes_rdd.txt",
    help = "Path to text file in which RDD from bytes file is stored after preprocessing")

parser.add_argument ("-C", "--bytesrddTest", default="data/sample/bytes_rdd_test.txt",
    help = "Path to text file in which RDD from bytes file is stored after preprocessing for test")

parser.add_argument ("-o", "--output", default="data/sample/output.txt",
    help = "Path to the directory where output will be written")

parser.add_argument ("-model", "--mlModel", default="xyz",
    help = "Specifies which ML model is to be used")

args = parser.parse_args()

main(args)
