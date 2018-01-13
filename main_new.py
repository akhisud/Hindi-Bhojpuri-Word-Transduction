import keras.backend as K
from keras import initializers, regularizers, constraints
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model
from keras.layers import dot, Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding, Input, merge
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Layer
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, constraints
from attention_decoder import AttentionDecoder 
from seq2seq.models import SimpleSeq2Seq, Seq2Seq
import seq2seq

from nltk import FreqDist
import numpy as np
import os
import datetime
import sys
import gc 

MAX_LEN = 20
VOCAB_SIZE = 65
BATCH_SIZE = 10
LAYER_NUM = 5
HIDDEN_DIM = 1000
EPOCHS = 500
TIME_STEPS = 20
SINGLE_ATTENTION_VECTOR = False
EMBEDDING_DIM = 299

# Take this MODE as command line arg
MODE = 'train' 
#MODE = 'test'

def load_char2vec(file):
	f = open(file, 'r', encoding="utf-8").readlines()
	embeddings_index = {}

	for line in f:
		values = line.split()
		char = values[0]
		coeffs = np.asarray(values[1:300], dtype='float32')
		embeddings_index[char] = coeffs

	return embeddings_index

def load_data(data_file):
	text = open(data_file, 'r', encoding="utf-8").readlines()

	word_list = []

	for line in text:
		line = line.strip()
		stripped_line = line.replace('\u200b','')
		stripped_line = line.replace('\u200d','').split(',')
		word_list.append(stripped_line)
		
		
	X = [c[0] for c in word_list]
	y = [c[1] for c in word_list]
	#print(len(text))
	#print(len(X))
	#print(len(y))

	X = [list(x)[::-1] for x, w in zip(X, y) if len(x) > 0 and len(w) > 0] # list of lists
	y = [list(w) for x, w in zip(X,y) if len(x) > 0 and len(w) > 0]
	
	# creating vocabulary with all words
	dist = FreqDist(np.hstack(X))
	X_vocab = dist.most_common(69)
	#dist = FreqDist(np.hstack(y))
	#y_vocab = dist.most_common(60)

	#print(len(y_vocab))
	#print(y_vocab)

	# taking whole of X as vocab set to create index_to_word dictionary
	X_idx2word = [letter[0] for letter in X_vocab]
	# Adding the letter 'O' to the beginning of array
	X_idx2word.insert(0, 'Z')
	X_idx2word.append('U') # for Unown letters

	# creating letter-to-index mapping
	X_word2idx = {letter:idx for idx, letter in enumerate(X_idx2word)}
	print(X_word2idx)

	a = input("pause")

	# converting each word to its index value
	for i, sentence in enumerate(X):
		for j, letter in enumerate(sentence):
			if letter in X_word2idx:
				X[i][j] = X_word2idx[letter]
			else:
				X[i][j] = X_word2idx['U']


	y_idx2word = [letter[0] for letter in X_vocab]
	y_idx2word.insert(0, 'Z')
	y_idx2word.append('U')
	y_word2idx = {letter:idx for idx,letter in enumerate(y_idx2word)}
	
	for i, sentence in enumerate(y):
		for j , letter in enumerate(sentence):
			if letter in y_word2idx:
				y[i][j] = y_word2idx[letter]
			else:
				y[i][j] = y_word2idx['U']

	return(X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, y_word2idx, y_idx2word)

def load_test_data(data_file, X_word2idx):
	text = open(data_file, 'r', encoding="utf-8").readlines()

	word_list = []
	all_x = []

	for line in text:
		line = line.strip()
		stripped_line = line.replace('\u200d','')
		stripped_line = line.replace('\u200b','').split(',')
		word_list.append(stripped_line)
		#print(word_list)
		
	X_te = [c[0] for c in word_list]
	y = [c[1] for c in word_list]

	X = [list(x)[::-1] for x in X_te if len(X_te) > 0]

	for i,word in enumerate(X):
		for j,letter in enumerate(word):
			if letter in X_word2idx:
				X[i][j] = X_word2idx[letter]
			else:
				X[i][j] = X_word2idx['U']

	return (y, X_te, X)


####################### End of Attention class ##########################

def create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, hidden_size, num_layers, embedding_matrix, X_word_to_ix, y_word_to_ix):

	# this embedding encodes input sequence into a sequence of 
	# dense X_vocab_len-dimensional vectors.
	emb_layer = Embedding(X_vocab_len, EMBEDDING_DIM, 
				weights = [embedding_matrix], input_length=X_max_len,
				mask_zero=True, trainable=False) 
	
	model = Sequential()
	seq2seq = Seq2Seq(input_dim=len(X_word_to_ix),
		input_length=X_max_len,
    	hidden_dim=512,
    	output_dim=len(y_word_to_ix),
    	output_length=y_max_len, depth=(4,5),
    	broadcast_state=False)
	model.add(seq2seq)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

def find_checkpoint_file(folder):
	checkpoint_file = [f for f in os.listdir(folder) if 'checkpoint' in f]
	if len(checkpoint_file) == 0:
		return []
	modified_time = [os.path.getmtime(f) for f in checkpoint_file]
	return checkpoint_file[np.argmax(modified_time)]

# one-hot encoding
def process_data(word_sentences, max_len, word_to_ix):
	# Vectorizing each element in each sequence
	sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
	for i, sentence in enumerate(word_sentences):
		for j, word in enumerate(sentence):
			sequences[i, j, word] = 1
	return sequences


data_file = "train_data.txt"
X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word= load_data(data_file)


print(X_vocab_len)
print(len(X_word_to_ix))
print(len(X_ix_to_word))
print(len(y_word_to_ix))
print(len(y_ix_to_word))


X_max = max([len(word) for word in X])
y_max = max([len(word) for word in y])
X_max_len = max(X_max,y_max)
y_max_len = max(X_max,y_max)

############### char2vec here ##################
char2vec_file = "char2vec.txt"
embedding_index = load_char2vec(char2vec_file)

embedding_matrix = np.zeros((X_vocab_len, EMBEDDING_DIM))
for char, i in X_word_to_ix.items():
	embedding_vector = embedding_index.get(char)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

############# end of char2vec embeddings #########
print("maxlen: ")
print(X_max_len)
print(y_max_len)
a = input("pause")

print('Zero padding.. ')
X = pad_sequences(X, maxlen = X_max_len, dtype='int32')
y = pad_sequences(y, maxlen = y_max_len, dtype='int32')

print("Model compiling ..")
model = create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, HIDDEN_DIM, LAYER_NUM, embedding_matrix, X_word_to_ix, y_word_to_ix)

saved_weights = "best_checkpoint_seq2seq.hdf5"

if MODE == 'train':

	print("Training")
	y_sequences = process_data(y, y_max_len, y_word_to_ix)
	X = process_data(X, X_max_len, X_word_to_ix)
	early_stop = EarlyStopping(patience = 5)

	model.fit(X, y_sequences, batch_size=BATCH_SIZE, 
		epochs = 5, 
		verbose=1, validation_split= 0.1,
		callbacks=[early_stop, 
		ModelCheckpoint('best_checkpoint_seq2seq.hdf5',
		monitor='val_loss',save_best_only=True,verbose=1)])
	
									

# performing test by loading saved training weights if we choose test mode
else:
	if len(saved_weights) == 0:
		print("network hasn't been trained!")
		sys.exit()
	else:
		test_sample_num =0 
		bhojpuri, hindi, X_test = load_test_data('test_data.txt', X_word_to_ix)
		# for i in X_test:
		# 	print(i)
		# x=input("pause")

		X_test = pad_sequences(X_test, maxlen = X_max_len, dtype='int32')
		# for i in X_test:
		# 	print(len(i))
		# x=input("pause")
		X_test = process_data(X_test, X_max_len, X_word_to_ix)
		print(X_test[0])
		model.load_weights(saved_weights)
		# print("saved_weights")
		# print(model.weights)
		# x=input("pause")

		print("model.predict")
		print(model.predict(X_test))
		x=input("pause")
		# print(model.predict(X_test))
		predictions = np.argmax(model.predict(X_test), axis=2)
		print(predictions)
		x=input("pause")
		# print(predictions[0])
		sequences = []

		for prediction in predictions:
			test_sample_num=test_sample_num+1
			# print(prediction)
			char_list = []
			for index in prediction:
				# print("index:",index)
				if index>0:
					char_list.append(y_ix_to_word[index])

					
			# print(char_list)
			# x=input("pause")
			sequence = ''.join(char_list)
			#sequence = sequence.decode('us-ascii')
			print(test_sample_num,":",sequence)
			sequences.append(sequence)
		# np.savetxt('test_result.txt', sequences, fmt='%s')
		filename = 'seq2seq'+str(LAYER_NUM)+'layers_'+str(BATCH_SIZE)+'batches.txt'
		with open(filename, 'a', encoding='utf-8') as f:
			f.write("Hindi words" + '\t' + "Machine Generated" + '\t'+ "Gold standard" + '\n')
			for a,b,c in zip(hindi, bhojpuri, sequences):
				f.write(str(a) + '\t' + str(b) + '\t'+ str(c) + '\n')

