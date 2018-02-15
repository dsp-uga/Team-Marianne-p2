
from pyspark import SparkContext, SparkConf

import os

from src.preprocess.BytePreProcessing import ByteFeatures
from src.preprocess.AsmPreProcessing import AsmFeatures
from src.classifiers.Bytes_NaiveBayes import Bytes_NaiveBayesTrainer
from src.classifiers.Asm_NaiveBayes import Asm_NaiveBayesTrainer
from src.classifiers.NaiveBayesClassifier import NaiveBayesClassifier

def main(args):
	
	# Refer https://spark.apache.org/docs/latest/configuration.html for additional changes to config 

	config = SparkConf().setAppName("team-marianne-p2")\
				.set("spark.hadoop.validateOutputSpecs", "false")\
				.set('spark.driver.memory','12G')\
				.set('spark.executor.memory','6G')\
				.set('spark.executor.cores','4')\
				.set('spark.python.worker.memory','5G')\
				.set('spark.driver.cores','4')
	
	sc = SparkContext.getOrCreate(config)

	 
	# BYTES PART - LEAVE ASIDE FOR SOMETIME
	# byteRDD contains hexadecimals from byte files in form of a list
	byteRDD = ByteFeatures(sc).get_bytesrdd(args.dataset, args.labels, args.bytestrain)
	# ByteFeatures(sc).write_to_file(byteRDD, args.bytesrdd)
	byteRDD.cache()


	# bytes trained model will be the naive bayes trained application on bytes data. later used in classification
	bytes_trained_model = Bytes_NaiveBayesTrainer(sc).bytes_train(byteRDD)
	# ByteFeatures(sc).write_to_file(bytes_trained_model, "train.txt")	# just to check the values
	

	# asmRDD contains opcodes in the asm files in form of a list
	asmRDD = AsmFeatures(sc).get_opcodes(args.dataset, args.labels, args.asmtrain)
	# AsmFeatures(sc).write_to_file(asmRDD, args.asmrdd)
	asmRDD.cache()

	# asm training model will be the naive bayes trained application on bytes data. later used in classification
	asm_trained_model = Asm_NaiveBayesTrainer(sc).opcode_train(asmRDD)
	# AsmFeatures(sc).write_to_file(asm_trained_model, "asm_train.txt")

	# TODO segments in asm file


	# classifcation on test data
	# passing training labels because it shouldn't matter
	bytes_test_rdd = ByteFeatures(sc).get_bytesrdd(args.testset, args.labels, args.bytestest)
	asm_test_rdd = AsmFeatures(sc).get_opcodes(args.testset, args.labels, args.asmtest)
	predicted_labels = NaiveBayesClassifier(sc).classify(bytes_trained_model, asm_trained_model, args.labels, bytes_test_rdd, asm_test_rdd)

	if not os.path.dir(args.output):
		os.makedirs(args.output)

	output_path = args.output
	if output_path[len(output_path) - 1] != '/':
		output_path += '/'

	f = open(output_path + "output.txt", 'w')
	for i in predicted_labels:
		f.write(str(i)+"\n")
