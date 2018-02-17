
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from src.preprocess.BytePreProcessing import ByteFeatures
from src.spark_mllib.spark_mllib_rdd import SparkRDDMl
from src.spark_mllib.spark_mllib_df import SparkDFMl
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel, LogisticRegressionWithLBFGS, LogisticRegressionModel, SVMWithSGD, SVMModel
import re

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

	if args.mlModel is 'rf':
		print('Random Forest called')
		train_data, test_data = ByteFeatures(sc).convert_svmlib(sc, args, train_data, train_labels, test_data, test_labels)
		SparkRDDMl(sc).random_forest_classification(sc, args, train_data, test_data)
	elif args.mlModel is 'nbs':
		print('Naive Bayes called')
		train_data, test_data = ByteFeatures(sc).convert_svmlib(sc, args, train_data, train_labels, test_data, test_labels)
		SparkRDDMl(sc).naive_bayes_mllib(sc, args, train_data, test_data)
	elif args.mlModel is 'lr':
		print('Logistic Regression called')
		train_data, test_data = ByteFeatures(sc).convert_svmlib(sc, args, train_data, train_labels, test_data, test_labels)
		SparkRDDMl(sc).logistic_regression_classification(sc, args, train_data, test_data)
	elif args.mlModel is 'svm':
		print('svm called')
		train_data, test_data = ByteFeatures(sc).convert_svmlib(sc, args, train_data, train_labels, test_data, test_labels)
		SparkRDDMl(sc).logistic_regression_classification(sc, args, train_data, test_data)
	elif args.mlModel is 'nbs_df':
		print('Naive Bayes (Dataframe based implementation) called')
		# Convert the RDDs to dataframes while combining features and labels for model creation
		train_df = ByteFeatures(sc).convert_to_dataframe(sql_context, train_data, train_labels)
		test_df = ByteFeatures(sc).convert_to_dataframe(sql_context, test_data, test_labels)
		# Execute the Naive Bayes algorithm for dataframes
		SparkDFMl(sc).naive_bayes(train_df, test_df)
