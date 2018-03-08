from nltk import FreqDist
import numpy as np
import os
import datetime
import sys
import gc

def generate_files(data_file):
	text = open(data_file, 'r', encoding="utf-8").readlines()

	word_list = []

	for line in text:
		line = line.strip()
		stripped_line = line.replace('\u200b','')
		stripped_line = line.replace('\u200d','').split(',')
		word_list.append(stripped_line)
		#print(word_list)
		
	X = [c[0] for c in word_list]
	y = [c[1] for c in word_list]

	#print(X)
	return X,y


file1 = 'train_data.txt'
file2 = 'test_data.txt' 

X, y = generate_files(file1)

filename = 'Hindi_train.txt'
with open(filename, 'w', encoding='utf-8') as f:
	for a in X:
		f.write(str(a) + '\n')

filename = 'Bhojpuri_train.txt'
with open(filename, 'w', encoding='utf-8') as f:
	for a in y:
		f.write(str(a) + '\n')

X_test, y_test = generate_files(file2)

filename = 'Hindi_test.txt'
with open(filename, 'w', encoding='utf-8') as f:
	for a in X_test:
		f.write(str(a) + '\n')

filename = 'Bhojpuri_test.txt'
with open(filename, 'w', encoding='utf-8') as f:
	for a in y_test:
		f.write(str(a) + '\n')