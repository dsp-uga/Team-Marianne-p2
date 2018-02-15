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

class AsmFeatures:
	"""
	preprocess asm files
	extracts opcodes from asm files
	"""

	def __init__(self, ctx):
		self.sc = ctx

	def extract_data(self, hash_list, asm_files_path):
		'''
		Extracts byte files of hash number from the path provided and adds
		all up in one single rdd
		'''
		if asm_files_path[len(asm_files_path)-1] != '/':	# making sure path ends with a '/'
			asm_files_path += '/'

		asm_files = self.sc.emptyRDD()
		for hash in hash_list:
			temp_rdd = self.sc.wholeTextFiles(str(asm_files_path)+str(hash)+'.asm')
			temp_rdd = temp_rdd.mapValues(preprocess.asm_opcodes)	# opcodes from this file
			asm_files += temp_rdd	# union of all rdds
		return asm_files # (hash, list of opcode_lists)

	def zip_rdds(self, rdd1, rdd2):
		"""
		this function joins two rdds
		"""
		r1 = rdd1.zipWithIndex().map(lambda x: (x[1], x[0]))   # (index, data)
		r2 = rdd2.zipWithIndex().map(lambda x: (x[1], x[0]))   # (index, data)
		rdd_join = r1.join(r2).map(lambda x: x[1])  # excluding indexes
		return rdd_join

	def get_opcodes(self, hash_list, labels_list, asm_data_path):
		"""
		this function returns the opcodes in each document mappend with corresponding class labels
		returns rdd of ("opcode_list", label)
		"""
		hashes = self.sc.textFile(hash_list)    # hashes rdd
		labels = self.sc.textFile(labels_list)  # labels rdd
		hashlist = hashes.collect()    # list of hashes
		# returns data in each file in form of (filepath, string) RDD[("file1", "file2", "file3", ... ]
		asm_data = self.extract_data(hashlist, asm_data_path)
		asm_data = asm_data.values()  # has only the opcodes_lists
		labeled_documents = self.zip_rdds(asm_data, labels)    # RDD[("file1", label), ("file2", label), ... ]
		# return labeled_documents2
		return labeled_documents

	def write_to_file(self, data, path):
		"""
		writes data RDD to file at location `path`
		"""
		resTrain = data.collect()
		f = open(path, 'w')
		numDocs = data.count()
		for i in range(numDocs):
		    f.write(str(resTrain[i]) + '\n')
		f.close()