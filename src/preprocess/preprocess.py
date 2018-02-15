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

    # substitues all strings with length >= 3 to ''. basically, eliminating pointers in point files
    single_string = re.sub(r'\b\w{3,}\b', '', byte_data)
    # substitutes '\n' with space
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

def asm_opcodes(asm_data):
	"""
	this function takes a asm file data as string, extracts the opcodes
	returns: list of opcodes in that order
	"""

	# gives all opcodes. assuming opcodes are followed by seven spaces in asm files
	opcodes = re.findall('\s{7}(\w+)', asm_data)
	opcodes = [re.sub('_.*', '', i) for i in opcodes]
	return opcodes

def asm_tokenize(opcodes):
	"""
	this function takes a opcodes list and returns a list of ngram tuples
	ngram parameter is a "magic number" here
	can be changed and checked for accuracy
	return list contains tuples of unigram, bigram, trigram, ... n-gram
	"""
	n = 2	# may need to change depending on accuracy
	grams = [i+1 for i in range(n)]
	retlist = []
	for i in grams:
		retlist +=  list(ngrams(opcodes, i))
	return retlist
