
from pyspark import SparkContext, SparkConf

from src.preprocess.BytePreProcessing import ByteFeatures

def main(args):

	# Refer https://spark.apache.org/docs/latest/configuration.html for additional changes to config
	config = SparkConf().setAppName("team-marianne-p2")\
				.set("spark.hadoop.validateOutputSpecs", "false")\
				.set('spark.driver.memory','12G')\
				.set('spark.executor.memory','2G')
	sc = SparkContext.getOrCreate(config)

	# byteRDD contains hexadecimals from byte files in a form of list
	byteRDD = ByteFeatures(sc).get_bytesrdd(args.dataset, args.labels, args.bytestrain)
	ByteFeatures(sc).write_to_file(byteRDD, args.bytesrdd)
	byteRDD.cache()


	# bytes trained model will be the naive bayes trained application on bytes data. later used in classification
	bytes_trained_model = NaiveBayesClassifier(sc).bytes_train(byteRDD)
	ByteFeatures(sc).write_to_file(bytes_trained_model, "train.txt")	# just to check the values
