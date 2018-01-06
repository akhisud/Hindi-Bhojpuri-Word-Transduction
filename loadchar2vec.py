from nltk import FreqDist
import numpy as np
import os
import datetime
import sys
import gc 
import csv
import random

def load_char2vec(file):
	maxlen = 9999
	f = open(file, 'r', encoding="utf-8").readlines()
	embeddings_index = {}

	#f.sort(key = lambda x:x[0])
	for line in f:
		values = line.split()
		char = values[0]
		coeffs = np.asarray(values[1:300], dtype='float32')
		embeddings_index[char] = coeffs
		maxlen = min(maxlen, coeffs.size)

	#f.close()
	return (embeddings_index, maxlen)

'''
embed_index, maxlen = load_char2vec("char2vec.txt")

#print(list(embed_index.keys())[0])
#print(list(embed_index.values())[0])
print(len(embed_index)) # 69
print(maxlen)
for key, value in embed_index.items():
	print(key)
'''

