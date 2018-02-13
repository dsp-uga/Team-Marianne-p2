
from pyspark import SparkContext, SparkConf

from src.preprocess.BytePreProcessing import ByteFeatures

def main(args):

	# Refer https://spark.apache.org/docs/latest/configuration.html for additional changes to config
	config = SparkConf().setAppName("team-marianne-p2")\
				.set("spark.hadoop.validateOutputSpecs", "false")

	sc = SparkContext.getOrCreate(config)

	data, labels = ByteFeatures(sc).load_data(args.dataset, args.labels)	# converts byte data and labels to RDDs
	# updates data and labels rdds. labels will be updated to (hash, label) and data will be updated to (hash, bigram_dict)
	data, labels = ByteFeatures(sc).transform_data(data, args.bytestrain, labels)
	# write data rdd to file -> makes it easier to retrieve processed data
	ByteFeatures(sc).write_to_file(data, args.bytesrdd)
