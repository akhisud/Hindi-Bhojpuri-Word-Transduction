import keras.backend as K
from keras import initializers, regularizers, constraints
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model
from keras.layers import dot, Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding, Input, merge
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Layer
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, constraints
from attention_decoder import AttentionDecoder 

from nltk import FreqDist
import numpy as np
import os
import datetime
import sys
import gc 

MAX_LEN = 20
VOCAB_SIZE = 65
BATCH_SIZE = 8
LAYER_NUM = 2
HIDDEN_DIM = 40
EPOCHS = 500
dropout= 0.2
TIME_STEPS = 20
SINGLE_ATTENTION_VECTOR = False
EMBEDDING_DIM = 150

# Take this MODE as command line arg
#MODE = 'train' 
MODE = 'test'

def load_char2vec(file):
	f = open(file, 'r', encoding="utf-8").readlines()
	embeddings_index = {}

	for line in f:
		values = line.split()
		char = values[0]
		coeffs = np.asarray(values[1:151], dtype='float32')
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
		#print(word_list)
		
	X = [c[0] for c in word_list]
	y = [c[1] for c in word_list]
	#print(len(text))
	#print(len(X))
	#print(len(y))

	X = [list(x) for x, w in zip(X, y) if len(x) > 0 and len(w) > 0] # list of lists
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

	X = [list(x) for x in X_te if len(X_te) > 0]

	for i,word in enumerate(X):
		for j,letter in enumerate(word):
			if letter in X_word2idx:
				X[i][j] = X_word2idx[letter]
			else:
				X[i][j] = X_word2idx['U']

	return (y, X_te, X)

################### Attention with context #####################
class AttentionWithContext(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self, init='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)

        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.built = True

    def compute_mask(self, input, mask):
        return None

    def call(self, x, mask=None):
        # (x, 40, 300) x (300, 1)
        multData =  K.dot(x, self.kernel) # (x, 40, 1)
        multData = K.squeeze(multData, -1) # (x, 40)
        multData = multData + self.b # (x, 40) + (40,)

        multData = K.tanh(multData) # (x, 40)

        multData = multData * self.u # (x, 40) * (40, 1) => (x, 1)
        multData = K.exp(multData) # (X, 1)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(x, 40)
            multData = mask*multData #(x, 40) * (x, 40, )

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        multData /= K.cast(K.sum(multData, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        multData = K.expand_dims(multData)
        weighted_input = x * multData
        return K.sum(weighted_input, axis=1)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1],)

####################### End of Attention class ##########################

def create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, hidden_size, num_layers, embedding_matrix):

	def smart_merge(vectors, **kwargs):
			return vectors[0] if len(vectors)==1 else merge(vectors, **kwargs)		
	
	root_word_in = Input(shape=(X_max_len,), dtype='int32')
	
	emb_layer = Embedding(X_vocab_len, EMBEDDING_DIM, 
				weights = [embedding_matrix], input_length=X_max_len,
				mask_zero=True) 
	
	hindi_word_embedding = emb_layer(root_word_in) # POSITION of layer

	BidireLSTM_vector= Bidirectional(LSTM(40, dropout=dropout, return_sequences=False, kernel_regularizer=regularizers.l2(0.1)))(hindi_word_embedding)
	#BidireLSTM_vector = Bidirectional(LSTM(512, return_sequences=True, dropout=0.5, recurrent_dropout=0.2))(BidireLSTM_vector)
	#BidireLSTM_vector = Bidirectional(LSTM(512, return_sequences=True, dropout=0.5))(BidireLSTM_vector)
	att = AttentionWithContext()(BidireLSTM_vector)
	#print(att.shape)
	RepLayer= RepeatVector(y_max_len)
	RepVec= RepLayer(att)
	Emb_plus_repeat=[hindi_word_embedding]
	Emb_plus_repeat.append(RepVec)
	Emb_plus_repeat = smart_merge(Emb_plus_repeat, mode='concat')
	
	
	for _ in range(num_layers):
		LtoR_LSTM = LSTM(40, dropout=dropout, return_sequences=True, kernel_regularizer=regularizers.l2(0.1))
		temp = LtoR_LSTM(Emb_plus_repeat)
	
	# for each time step in the input, we intend to output |y_vocab_len| time steps
	time_dist_layer = TimeDistributed(Dense(y_vocab_len))(temp)
	outputs = Activation('softmax')(time_dist_layer)

	all_inputs = [root_word_in]
	model = Model(input=all_inputs, output=outputs)
	#opt = Adagrad(lr=0.01)
	opt = Adam()
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	
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
X = pad_sequences(X, maxlen = X_max_len, dtype='int32', padding='post')
y = pad_sequences(y, maxlen = y_max_len, dtype='int32', padding='post')


print("Model compiling ..")
model = create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, HIDDEN_DIM, LAYER_NUM, embedding_matrix)

saved_weights = "AttentionEncoder_sgd.hdf5"

if MODE == 'train':
	print("Training")
	y_sequences = process_data(y, y_max_len, y_word_to_ix)
	history = model.fit(X, y_sequences, validation_split= 0.1, batch_size=BATCH_SIZE, epochs = EPOCHS, 
		verbose=1,callbacks=[EarlyStopping(patience=20,verbose=1), 
		ModelCheckpoint('AttentionEncoder_sgd.hdf5',
		save_best_only=True, verbose=1)])
	print(history.history.keys())
	print(history)
	# model.save_weights('checkpoint_epoch_{}.hdf5'.format(k))
	# gc.collect()

# performing test by loading saved training weights if we choose test mode
else:
	if len(saved_weights) == 0:
		print("network hasn't been trained!")
		sys.exit()
	else:
		test_sample_num =0 
		bhojpuri, hindi, X_test = load_test_data('test_data.txt', X_word_to_ix)


		X_test = pad_sequences(X_test, maxlen = X_max_len, dtype='int32', padding='post')

		print(X_test[0])
		model.load_weights(saved_weights)

		plot_model(model, to_file="Attention_after_encoder.png", show_shapes=True, show_layer_names=True)
		
		print("model.predict")
		print(model.predict(X_test))
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
		filename = 'Attention_after_encoder'+str(dropout)+'dropout.txt'
		with open(filename, 'w', encoding='utf-8') as f:
			f.write("Hindi words" + '\t' + "Gold standard" + '\t'+ "Machine Generated" + '\n')
			for a,b,c in zip(hindi, bhojpuri, sequences):
				f.write(str(a) + '\t\t\t' + str(b) + '\t\t\t'+ str(c) + '\n')

