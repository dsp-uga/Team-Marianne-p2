
from pyspark import SparkContext, SparkConf
from src.preprocess.BytePreProcessing import ByteFeatures
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors, SparseVector, _convert_to_vector
import numpy as np
import re

def loadLibSVMFile(sc, data, numFeatures=-1, minPartitions=None, multiclass=None):
	from pyspark.mllib.regression import LabeledPoint
	if multiclass is not None:
		warnings.warn("deprecated", DeprecationWarning)

	lines = data
	def _parse_libsvm_line(line, multiclass=None):
		"""
		Parses a line in LIBSVM format into (label, indices, values).
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

def random_forest_classification(sc, args, train_data, train_labels, test_data, test_labels):
	# Converts the training data and labels into svm format
	train_data = train_data.join(train_labels)
	a = train_data.collect()
	x = train_data.map(lambda x: (x[1][1], sorted(x[1][0].items(), key = lambda d:d[0])))
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
	y = x.mapValues(lambda x:convertToSVMFormat(str(x)))
	y = y.map(lambda x:(x[0], x[0]+' '+x[1]))
	# Writes the train_data rdd to path provided in args.bytesrdd
	#path_train = ByteFeatures(sc).write_to_file(y, args.bytesrdd)
	#train_data = MLUtils.loadLibSVMFile(sc, path_train)
	train_data = loadLibSVMFile(sc, y.values())

	# Converts the testing data and labels into svm format
	test_data = test_data.join(test_labels)\
	     .map(lambda x: (x[1][1], sorted(x[1][0].items(), key = lambda d:d[0])))\
         .mapValues(lambda x: convertToSVMFormat(str(x)))\
	# Writes the test_data rdd to path provided in args.bytesrddTest
	#path_test = ByteFeatures(sc).write_to_file(test_data, args.bytesrddTest)
	#test_data = MLUtils.loadLibSVMFile(sc, path_test)
	test_data = test_data.map(lambda x: (x[0], x[0]+' '+x[1]))
	test_data = loadLibSVMFile(sc, test_data.values())

	model = RandomForest.trainClassifier(train_data, numClasses=9, categoricalFeaturesInfo={},\
                                     numTrees=9, featureSubsetStrategy="auto",\
                                     impurity='gini', maxDepth=4, maxBins=32)\

	predictions = model.predict(test_data.map(lambda x: x.features))
	if(args.evaluate):
		score(predictions,test_data)
	write_output(predictions, args.output)

def write_output(predictions, path):
	resTrain = predictions.collect()
	f = open(path, 'w')
	for i in range(len(resTrain)):
		f.write(str(resTrain[i]+1) + '\n')
	f.close()
	return path

def score(predictions,test_data):
	#print('predictions are----------',predictions.collect())
	labelsAndPredictions = test_data.map(lambda x: x.label).zip(predictions)
	#print('labelsAndPredictions are----------',labelsAndPredictions.collect())
	testErr = labelsAndPredictions.filter(lambda lp: lp[0] == lp[1]).count() / float(test_data.count())
	print('Accuracy = ' + str(testErr))
	print('Learned classification forest model:')
	print(model.toDebugString())

def main(args):

	# Refer https://spark.apache.org/docs/latest/configuration.html for additional changes to config
	config = SparkConf().setAppName("team-marianne-p2")\
				.set("spark.hadoop.validateOutputSpecs", "false")\
				.set('spark.driver.memory','12G')\
				.set('spark.executor.memory','2G')
	sc = SparkContext.getOrCreate(config)

	train_data, train_labels = ByteFeatures(sc).load_data(args.dataset, args.labels)	# converts training byte data and labels to RDDs
	# training: updates data and labels rdds. labels will be updated to (hash, label) and data will be updated to (hash, bigram_dict)
	train_data, train_labels = ByteFeatures(sc).transform_data(train_data, args.bytestrain, train_labels)

	test_data, test_labels = ByteFeatures(sc).load_data(args.testset, args.testlabels)	# converts testing byte data and labels to RDDs
	# testing: updates data and labels rdds. labels will be updated to (hash, label) and data will be updated to (hash, bigram_dict)
	test_data, test_labels = ByteFeatures(sc).transform_data(test_data, args.bytestest, test_labels)

	if args.mlModel is 'lr':
		random_forest_classification(sc, args, train_data, train_labels, test_data, test_labels)
