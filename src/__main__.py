
from pyspark import SparkContext, SparkConf
from src.preprocess.BytePreProcessing import ByteFeatures
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel, LogisticRegressionWithLBFGS, LogisticRegressionModel, SVMWithSGD, SVMModel
import re

def random_forest_classification(sc, args, train_data, test_data):
	'''This does the Random forest classification for given train and test set
	'''
	# Create model and make prediction
	model = RandomForest.trainClassifier(train_data, numClasses=9, categoricalFeaturesInfo={},\
                                     numTrees=9, featureSubsetStrategy="auto",\
                                     impurity='gini', maxDepth=4, maxBins=32)
	predictions = model.predict(test_data.map(lambda x: x.features))
	if(args.evaluate):
		score(predictions,test_data, model, args.mlModel)
	write_output(predictions, args.output)

def naive_bayes_mllib(sc, args, train_data, test_data):
	'''This does the Naive Bayes Classification for given train and test set
	'''
	# Create model and make predictions
	model = NaiveBayes.train(train_data, 1.0)
	predictions = model.predict(test_data.map(lambda x: x.features))
	if(args.evaluate):
		score(predictions,test_data, model, args.mlModel)
	write_output(predictions, args.output)

def logistic_regression_classification(sc, args, train_data, test_data):
	'''This does the logistic regression classification for given train and test set
	'''
	# create model and make prediction
	model = LogisticRegressionWithLBFGS.train(train_data, iterations=100, numClasses=9)
	predictions = model.predict(test_data.map(lambda x: x.features))
	if(args.evaluate):
		score(predictions,test_data, model, args.mlModel)
	write_output(predictions, args.output)

def svm_classification(sc, args, train_data, test_data):
	'''This does the svm classification for given train and test set
	'''
	# create model and make prediction
	model = SVMWithSGD.train(train_data, iterations=100, numClasses=9)
	predictions = model.predict(test_data.map(lambda x: x.features))
	if(args.evaluate):
		score(predictions,test_data, model, args.mlModel)
	write_output(predictions, args.output)

def write_output(predictions, path):
	'''Write the output to file given in the args output path
	'''
	resTrain = predictions.collect()
	f = open(path, 'w')
	for i in range(len(resTrain)):
		f.write(str(int(resTrain[i]+1)) + '\n')
	f.close()
	return path

def score(predictions,test_data, model, mlModel):
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
				.set('spark.driver.memory','12G')\
				.set('spark.executor.memory','6G')\
				.set('spark.executor.cores','4')\
				.set('spark.python.worker.memory','5G')\
				.set('spark.driver.cores','4')\

	sc = SparkContext.getOrCreate(config)
	# Preprocess training data
	train_data, train_labels = ByteFeatures(sc).load_data(args.dataset, args.labels)	# converts training byte data and labels to RDDs
	# training: updates data and labels rdds. labels will be updated to (hash, label) and data will be updated to (hash, bigram_dict)
	train_data, train_labels = ByteFeatures(sc).transform_data(train_data, args.bytestrain, train_labels)
	# Preprocdess testing data
	test_data, test_labels = ByteFeatures(sc).load_data(args.testset, args.testlabels)	# converts testing byte data and labels to RDDs
	# testing: updates data and labels rdds. labels will be updated to (hash, label) and data will be updated to (hash, bigram_dict)
	test_data, test_labels = ByteFeatures(sc).transform_data(test_data, args.bytestest, test_labels)
	train_data, test_data = ByteFeatures(sc).convert_svmlib(sc, args, train_data, train_labels, test_data, test_labels)
	if args.mlModel is 'rf':
		print('Random Forest called')
		random_forest_classification(sc, args, train_data, test_data)
	elif args.mlModel is 'nbs':
		print('Naive Bayes called')
		naive_bayes_mllib(sc, args, train_data, test_data)
	elif args.mlModel is 'lr':
		print('Logistic Regression called')
		logistic_regression_classification(sc, args, train_data, test_data)
	elif args.mlModel is 'svm':
		print('svm called')
		logistic_regression_classification(sc, args, train_data, test_data)
