
from pyspark.mllib.feature import HashingTF, IDF
import pyspark
import numpy as np

from src.preprocess.AsmPreProcessing import AsmFeatures
import src.preprocess.preprocess as preprocess

class Asm_NaiveBayesTrainer(object):
	"""
	This classifier trains the documents using the Naive Bayes algorithm
	"""
	def __init__(self, spark_context):
		super(Asm_NaiveBayesTrainer, self).__init__()
		self.sc = spark_context

	def opcode_train (self, opcode_data):
		"""
		Trains the classifier using Naive Bayes

		:param data: RDD containing (--- structure of RDD here ---) ([opcodes], label)
		:param labels: RDD containing (--- structure of RDD here ---)
		:return: None
		"""
		total_docs = self.sc.broadcast(opcode_data.count())
		documents = opcode_data.keys()
		labels = opcode_data.values()
		words = opcode_data.map(lambda tuple: (tuple[1], tuple[0]))	# (label, opcode)
#		words = words.mapValues(preprocess.process_opcode_data)		# (label, processed-file) -> processed file has only hex values
		word_tuples = words.flatMapValues(preprocess.asm_tokenize)			# (label, ngram)
		word_tuples.cache()
#		return word_tuples
		
		def wordtuple_to_classvector(x):
			"""
			this function takes (class, word) tuple and convert to class vector
			this function supports 10 classes
			"""
			label, word = x
			class_vector = np.zeros(10)
			class_vector[int(label)] = 1
			return (word, class_vector)

		term_frequencies = word_tuples.map(wordtuple_to_classvector)
		term_frequencies = term_frequencies.reduceByKey(lambda a, b: a + b)

		distinct_words_size = self.sc.broadcast(term_frequencies.count())

		def tficf(x):
			"""
			this function takes calculates tficf scores of each word
			"""
			term = x[0]
			frequencies = x[1]
			tf = np.log(frequencies+1)
			N = 10	# number of classes
			nt = np.count_nonzero(frequencies)
			icf = N / nt
			tficf = tf * icf
			return (term, tficf)

		tficf_scores = term_frequencies.map(tficf)

		def filter_by_std_deviation(arr):
			"""
			this function eliminate terms with close standard deviation across various documents
			"""
			sd = 0	# magic number --- change based on accuracy
			return np.std(arr) < sd

#		tficf_scores = tficf_scores.filter(lambda x: filter_by_std_deviation(x[1]))

		# total words we consider for bayes training
		total_words = term_frequencies.map(lambda x: x[1]).reduce(lambda a, b: a + b)

		total_words = self.sc.broadcast(total_words)

		def training_model(x):
			"""
			this function calculates conditional probability vector for each word based on 
			distribution accross different classes
			"""
			smoothing_parameter = 1	# to solve zero-frequency problem
			word, term_freq = x
			term_freq = term_freq + smoothing_parameter
			totalwords = total_words.value + smoothing_parameter * distinct_words_size.value
			probability_vector = term_freq / totalwords
			return (word, probability_vector)

		probability_vector = tficf_scores.map(training_model)
		# trained_model = dict(probability_vector.collect())

		return probability_vector
		
