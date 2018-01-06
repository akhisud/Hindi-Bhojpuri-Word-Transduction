import numpy as np
import pandas as pd
import re

import FMeasure
import wordClassFeatures
import genderPreferentialFeatures
#import minePOSPats
import baseFeatures
import genderDifferencesFeatures
import get_CBOW_features
import elm 

from collections import defaultdict
from bs4 import BeautifulSoup

import sys
import os
import gc
#os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding, Recurrent, TimeDistributed, Activation, BatchNormalization
from keras.layers import Dense, Input, Flatten, recurrent, wrappers, InputLayer
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM
from keras.layers import concatenate, GRU, Bidirectional
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint,  LearningRateScheduler
from keras import backend as K
from keras.utils import plot_model
from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.regularizers import l2

from sklearn.svm import NuSVC, SVC
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


mode = 'TRAIN'
#mode = 'TEST'

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 30000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1

np.random.seed(0)

cachedStopWords = stopwords.words("english")

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = string.decode('utf-8')
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)
    string = re.sub(r"\~", "", string)
    string = ' '.join([word for word in string.split() if word not in cachedStopWords])    
    return string.strip().lower()

def findFeature(text):
	#fMeasureFeature = FMeasure.FMeasure(text)
	genderPreferentialFeature = genderPreferentialFeatures.genderPreferentialFeatures(text)
	#posFeature = minePOSPats.POSFeatures(text)
	#baseFeature = baseFeatures.baseFeatures(text)
	#genderDifferencesFeature = genderDifferencesFeatures.genderDifferencesFeatures(text)
	#wordClassFeature = wordClassFeatures.wordClassFeatures(text)

	fMeasureToTuple = []
	featureVector = []
	#fMeasureToTuple.append(fMeasureFeature)
	'''
	features = tuple(fMeasureToTuple) + genderDifferencesFeature + genderPreferentialFeature +\
				wordClassFeature + baseFeature
	'''
	features = genderPreferentialFeature
	features = list(features)
	
	return features

data1 = pd.read_csv('original_blogs.csv', sep='\t')
print(data1.shape)

#a = input("pause")

texts = []
labels = []

#print(texts)

for idx in range(data1.Blog.shape[0]):
    text = BeautifulSoup(data1.Blog[idx], "html.parser")
    texts.append(clean_str(text.get_text().encode('ascii','ignore')))
    labels.append(data1.Gender[idx])

print(len(texts))
print(len(labels))

features = [findFeature(i) for i in texts]
#print(len(features[1]))
features = np.asarray(features, dtype=np.float32)
#print(features.shape)

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
features = features[indices]

'''
######################## PCA visualization #################
plt.cla()

#print(len(features[1]))
features = scale(features)

pca = PCA(n_components= 93)
features = pca.fit_transform(features)

print(len(features[1]))
'''
'''
variance = pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(var1)


plt.plot(var1, color='red', linestyle='solid', linewidth=2.0)

plt.xlabel('No. of Principal Components')
plt.ylabel('Explained variance in percent')
plt.savefig('pca_features.png', dpi=150)
plt.show()
'''

###############  End of PCA visualization  ##########################
#b = input("pause")


nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-2*nb_validation_samples]
y_train = labels[:-2*nb_validation_samples]
train_features = features[:-2*nb_validation_samples]
#print(train_features.shape)

x_val = data[-2*nb_validation_samples:-nb_validation_samples]
y_val = labels[-2*nb_validation_samples:-nb_validation_samples]
val_features = features[-2*nb_validation_samples:-nb_validation_samples]

x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]
test_features = features[-nb_validation_samples:]

print('Traing and validation set number of positive and negative reviews:')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

filename = 'glove.6B.100d.txt'
#filename = 'wiki.simple.vec'
embeddings_index = {}
f = open(filename, 'r').readlines()
for line in f:
	values = line.strip().split()
	try:
		coefs = np.asarray(values[1:101], dtype='float32')
		word = values[0]
	except:
		continue
	embeddings_index[word] = coefs


print('Total %s word vectors.' % len(embeddings_index))

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

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
'''
######################## First model ###########################
def create_model():
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)

	l_lstm = Bidirectional(GRU(300, name='Bidirectional', return_sequences=True))(embedded_sequences)

	num_layers = 0
	for _ in range(num_layers):
		l_lstm = Bidirectional(GRU(100, return_sequences=True))(l_lstm)
	

	l_lstm = Bidirectional(GRU(300, return_sequences=True))(l_lstm)
	l_lstm = AttentionWithContext()(l_lstm)

	time_dist_layer = Dense(2, name='Dense')(l_lstm)

	#temp = K.expand_dims(time_dist_layer, axis=-1)
	#flat1 = Flatten()(temp)
	#batch_norm_layer = BatchNormalization()(time_dist_layer)
	outputs = Activation('softmax', name='Softmax_Activation')(time_dist_layer)

	model = Model(sequence_input, outputs)

	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
	return model

model = create_model()

print("model fitting - Bidirectional LSTM")
model.summary()


if mode == 'TRAIN':
	print("model fitting - attention GRU network")
	#plot_model(model, to_file='Simplest_LSTM_SVM_model.png')
	early_stop = EarlyStopping(patience=7)
	model.fit(x_train, y_train, validation_data=(x_val, y_val),
        epochs=30, batch_size=10,
		callbacks=[early_stop, ModelCheckpoint('best_checkpoint_simplest_GRU_model_att.hdf5',
		save_best_only=True,verbose=1)])
	gc.collect()
	#model.save_weights('best_checkpoint.hdf5')


else:
	saved_weights = 'best_checkpoint_simplest_GRU_model_att.hdf5'
	model.load_weights(saved_weights)
	scores = model.evaluate(x_test, y_test, verbose=0)
	print("Testing Accuracy: %.2f%%" % (scores[1]*100))
	print("############################")

####################### End of simple model #######################

'''


####################### Second Model ############################
def create_model():
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='sequence_input')
	embedded_sequences = embedding_layer(sequence_input)
	l_gru = Bidirectional(GRU(300, return_sequences=True))(embedded_sequences)
	l_gru = Bidirectional(GRU(300, return_sequences=True))(l_gru)

	l_gru = AttentionWithContext()(l_gru)

	auxiliary_output = Dense(2, activation='softmax', name='aux_output')(l_gru)

	auxiliary_input = Input(shape=(len(features[1]),1), dtype='float32', name='aux_input')
	gru2 = Bidirectional(GRU(50))(auxiliary_input)

	combined = concatenate([l_gru, gru2])

	d1 = Dense(64, activation='relu')(combined)
	#d2 = Dense(64, activation='relu')(d1)
	#d3 = Dense(64, activation='relu')(d2)	
	
	#flat1 = Flatten()(d3)

	preds = Dense(2, W_regularizer=l2(0.01)	)(d1)
	#preds = Dense(2)(d3)
	#flat2 = Flatten()(preds)
	#batch_norm_layer = BatchNormalization()(preds)
	main_output = Activation('softmax', name='main_output')(preds)

	model = Model(inputs=[sequence_input, auxiliary_input], outputs=[main_output,auxiliary_output])
	
	model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'],
              loss_weights=[1., 0.2])
	
	#model.compile(loss='hinge', optimizer='adadelta', metrics=['accuracy'])
	
	return model

model= create_model()

train_features = np.reshape(train_features, (train_features.shape[0], train_features.shape[1], 1))
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
val_features = np.reshape(val_features, (val_features.shape[0], val_features.shape[1], 1))

if mode == 'TRAIN':
	print("model fitting - attention GRU network")
	model.summary()
	
	early_stop = EarlyStopping(patience=10)
	model.fit([x_train, train_features], [y_train,y_train], 
		validation_data=([x_val, val_features], [y_val,y_val]),
		epochs=50, batch_size=10, 
		callbacks=[early_stop, 
		ModelCheckpoint('best_checkpoint_features_all_GRU_genPref.hdf5',
		save_best_only=True,verbose=1)])

else:
	saved_weights = 'best_checkpoint_features_all_GRU_genPref.hdf5'
	model.load_weights(saved_weights)
	plot_model(model, to_file='multi_input_model.png')
	scores = model.evaluate([x_test, test_features], [y_test, y_test], verbose=0)
	print("Testing Accuracy: %.2f%%" % (scores[1]*100))
	print("############################")


###########################################################
'''
####################### Third Model #########################

def create_model():
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='sequence_input')
	auxiliary_input = Input(shape=(len(features[1]),), name='aux_input')

	embedded_sequences = embedding_layer(sequence_input)
	
	gru1 = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
	gru1 = AttentionWithContext()(gru1)

	gru2 = Bidirectional(GRU(100))(auxiliary_input)

	combined = concatenate([gru1, gru2])

	d1 = Dense(50, activation='relu')(combined)
	d2 = Dense(2)(d1)

	main_output = Activation('softmax', name='main_output')(preds)

	model = Model(inputs=[sequence_input, auxiliary_input], outputs=main_output)
	
	model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
	return model

model= create_model()

if mode == 'TRAIN':
	print("model fitting - attention GRU network")
	model.summary()
	
	early_stop = EarlyStopping(patience=10)
	model.fit([x_train, train_features], y_train, 
		validation_data=([x_val, val_features], y_val),
		epochs=50, batch_size=20, 
		callbacks=[early_stop, 
		ModelCheckpoint('best_checkpoint_model3_rmsprop.hdf5',
		save_best_only=True,verbose=1)])
	#model.save_weights('best_checkpoint.hdf5')


else:
	saved_weights = 'best_checkpoint_model3_rmsprop.hdf5'
	model.load_weights(saved_weights)
	plot_model(model, to_file='model3.png')
	scores = model.evaluate([x_test, test_features], y_test, verbose=0)
	print("Testing Accuracy: %.2f%%" % (scores[1]*100))
	print("############################")
'''