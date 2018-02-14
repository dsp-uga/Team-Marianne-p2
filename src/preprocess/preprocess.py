"""
This file contains preprocess functions
"""

import re
import numpy as np
import nltk
import string
from nltk.util import ngrams

def process_byte_data(byte_data):
    """
    this function excludes all the irrelevant information from a byte file
    returns only the hexadecimals seperated with a whitespace
    """
    single_string = re.sub(r'\b\w{3,}\b', '', byte_data)
    single_string = re.sub('\n', ' ', single_string)
    return single_string

def byte_tokenize(hexstring):
	"""
	this function takes a hexstring and returns a list of ngram tuples
	ngram parameter is a "magic number" here
	can be changed and checked for accuracy
	return list contains tuples of unigram, bigram, trigram, ... n-gram
	"""
	hexlist = nltk.word_tokenize(hexstring)
	n = 2	# may need to change depending on accuracy
	grams = [i+1 for i in range(n)]
	retlist = []
	for i in grams:
		retlist +=  list(ngrams(hexlist, i))
	return retlist
