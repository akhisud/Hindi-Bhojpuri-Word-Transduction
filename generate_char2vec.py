from nltk import FreqDist
import numpy as np
import os
import datetime
import sys
import gc 
import csv
import random
import timeit
import json

def load_data(file,char2vec):
	f = open(file, 'r', encoding="utf-8").readlines()[2:] 
	vectors = []
	
	for line in f:
		#line = line.strip()
		vectors.append([x for x in line.strip().split()])
	# print(len(vectors))
	print(vectors[0])
	vec_len = len(vectors[0])-1

	f = open("new_data.txt", 'r', encoding="utf-8").readlines()
	hin_wrds=[]
	for line in f:
		wrds = line.strip().split(',')
		hin_wrds.append(wrds[0])
		hin_wrds.append(wrds[1])
	unique_hin =set()
	for wrd in hin_wrds:
		for char in wrd:
			# print(char)
			unique_hin.add(char)
		# x=input("pause")
	unique_hin = sorted(unique_hin)
	
	for char in unique_hin:
		print(char)
	#x=input("pause")	
	# find average of the vectors for each unique character
	badvecs =0
	for char in unique_hin:
		total = 0
		this_char_vec = [0.0]*vec_len
		for vec in vectors:
			word = vec[0] # the word 
			#print(vec)
			try:
				vector=[float(x) for x in vec[1:]]
			except ValueError:
				badvecs+=1
				continue			
			chars = [x for x in word] # list of all chars in this word
			cnt = chars.count(char) # no of times the character appears in this word
			# if char doesn't appear
			
			if cnt == 0:
				continue
			# print(char,word,cnt)
			# x=input("pause")
			total+=cnt
			for i in range(cnt):
				this_char_vec= [sum(x) for x in zip(this_char_vec,vector)]
		''' Error: some vectors still have words appearing in them ??!!!'''
		#TAKE AVERAGE OVER ALL ADDITIONS OF VECTORS
		if total != 0:
			this_char_vec = [float(x)/float(total) for x in this_char_vec]
		else:
			#randomly initialize character vector
			this_char_vec =  [float(random.random()) for i in range(vec_len)]
			#x = vec_len
			#while (x > 0):
			#	num = float(random.random())
			#	if num > 0:
			#		this_char_vec.append(num)
			#		x = x - 1	
			
		#print(char)
		#print(this_char_vec)

		char2vec[char]= this_char_vec
		#x=input("pause")


	print("Number of bad vectors=",badvecs)

file = "wiki.hi.vec"
char2vec = {} # should store the char2vector values
load_data(file,char2vec)
print(list(char2vec.keys())[0])
print(list(char2vec.values())[0])

with open('char2vec.txt', 'w', encoding="utf-8") as file:
	for key, value in char2vec.items():
		file.write('%s' % key)
		for x in value:
			file.write(' %s' % x)
		file.write('\n')