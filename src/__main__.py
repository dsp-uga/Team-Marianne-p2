
from pyspark import SparkContext, SparkConf
from src.preprocess.BytePreProcessing import ByteFeatures
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
import re

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
		return Str
	y = x.mapValues(lambda x: convertToSVMFormat(str(x)))
	# Writes the train_data rdd to path provided in args.bytesrdd
	path_train = ByteFeatures(sc).write_to_file(y, args.bytesrdd)
	train_data = MLUtils.loadLibSVMFile(sc, path_train)
	# Converts the training data and labels into svm format
	test_data = test_data.join(test_labels)\
	     .map(lambda x: (x[1][1], sorted(x[1][0].items(), key = lambda d:d[0])))\
         .mapValues(lambda x: convertToSVMFormat(str(x)))\

	# Writes the test_data rdd to path provided in args.bytesrddTest
	path_test = ByteFeatures(sc).write_to_file(test_data, args.bytesrddTest)
	test_data = MLUtils.loadLibSVMFile(sc, path_test)

	model = RandomForest.trainClassifier(train_data, numClasses=9, categoricalFeaturesInfo={},\
                                     numTrees=3, featureSubsetStrategy="auto",\
                                     impurity='gini', maxDepth=4, maxBins=32)\

	predictions = model.predict(test_data.map(lambda x: x.features))
	print('predictions are----------',predictions.collect())
	labelsAndPredictions = test_data.map(lambda x: x.label).zip(predictions)
	print('labelsAndPredictions are----------',labelsAndPredictions.collect())
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
