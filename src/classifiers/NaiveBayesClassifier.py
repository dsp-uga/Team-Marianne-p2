
from pyspark.mllib.feature import HashingTF, IDF
import pyspark
import numpy as np

from src.preprocess.AsmPreProcessing import AsmFeatures
import src.preprocess.preprocess as preprocess

class NaiveBayesClassifier(object):
	"""
	this is the classifying class
	"""
	def __init__(self, spark_context):
		super(NaiveBayesClassifier, self).__init__()
		self.sc = spark_context

	def classify_document (self, bytes_trained_model, opcodes_trained_model, train_labels, bytes_test_rdd, asm_test_rdd):
		"""
		this function takes the trained models and test document and predicts a class to this document
		"""
		labels = self.sc.textFile(train_labels)  # labels rdd

		total_train_docs = labels.count()
		total_test_docs = bytes_test_rdd.count()
		
		# distribution of classes among training data set
		labels = labels.map(lambda x: (x,1))
		labels_count = labels.reduceByKey(lambda a, b: a + b)
		class_probability = labels_count.mapValues(lambda x: x / total_train_docs.value)
		class_prob = dict(class_probability.collect())
		marginals = [class_prob.value[i] for i in range(10)]	# dataset has 10 classes

		prediction = np.zeros(test_docs)
		a_prediction = np.zeros(test_docs)
		b_prediction = np.zeros(test_docs)

		asm_data = asm_test_rdd.keys().collect()	# list of [opcodes]
		viewed_opcodes = opcodes_trained_model.keys().collect()
		dict_otm = dict(opcodes_trained_model.collect())

		cnt = 0
		for a in asm_data:	# a is list of opcodes for each document
			p_vector = np.zeros(10)
			p_vector += np.log(marginals)
			for i in a:		# i is opcode
				if i in viewed_opcodes:
					prob = dict_otm[i]
					p_vector += np.log(prob)

			class_index = np.argmax(p_vector)
			a_prediction[cnt] = class_index
			cnt += 1


		words = bytes_test_rdd.map(lambda tuple: (tuple[1], tuple[0]))
		word_tuples = words.MapValues(preprocess.byte_tokenize)
		bytes_data = word_tuples.values().collect()
		viewed_hex = bytes_trained_model.keys().collect()
		dict_btm = dict(bytes_trained_model.collect())

		cnt = 0
		for b in bytes_data:
			p_vector = np.zeros(10)
			p_vector += np.log(marginals)
			for i in b:		# i is hex value
				if i in viewed_hex:
					prob = dict_btm[i]
					p_vector += np.log(prob)

			class_index = np.argmax(p_vector)
			b_prediction[cnt] = class_index
			cnt += 1

		prediction = (a_prediction * 0.5) + (b_prediction * 0.5)
		return prediction
