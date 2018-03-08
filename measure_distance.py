from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk
from random import shuffle
from libindic.soundex import Soundex

#print(fuzz.ratio("this is a test", "this is a pre-test!"))

def get_string_similarity(i, j):
	return fuzz.ratio(i, j)

def load_transformer(data_file):
	text = open(data_file, 'r', encoding="utf-8").readlines()[1:]

	word_list = []

	for line in text:
		line = line.strip()
		stripped_line = line.replace('\u200b','')
		stripped_line = line.replace('\u200d','').split('\t\t\t')
		word_list.append(stripped_line)
		#print(word_list)
		
	X = [c[0] for c in word_list]
	y = [c[1] for c in word_list]
	


	#for i,j in zip(X,y):
	#	print(i + '\t' + j)

	#X = [list(x) for x, w in zip(X, y) if len(x) > 0 and len(w) > 0] # list of lists
	#y = [list(w) for x, w in zip(X,y) if len(x) > 0 and len(w) > 0]

	
	return (X, y)
	
def load_data(data_file):
	text = open(data_file, 'r', encoding="utf-8").readlines()[1:]

	word_list = []

	for line in text:
		line = line.strip()
		stripped_line = line.replace('\u200b','')
		stripped_line = line.replace('\u200d','').split('\t\t\t')
		word_list.append(stripped_line)
		#print(word_list)
		
	X = [c[0] for c in word_list]
	y = [c[1] for c in word_list]
	z = [c[2] for c in word_list]


	#for i,j in zip(X,y):
	#	print(i + '\t' + j)

	#X = [list(x) for x, w in zip(X, y) if len(x) > 0 and len(w) > 0] # list of lists
	#y = [list(w) for x, w in zip(X,y) if len(x) > 0 and len(w) > 0]

	
	return (X, y, z)
	

data_file = "Attention_after_encoder0.2dropout.txt"
#data_file = "AttentionDecoder_with_dropout8batches0.2dropout.txt"
data_file2 = "Transformer_output.txt"
# X and y being list of lists, each list contains characters of words
X, y, z = load_data(data_file)
a, b = load_transformer(data_file2)
print(data_file)
instance = Soundex()
lis1 = []
lis2 = []
lis3 = []
'''
for i,j,k in zip(X,y,z):
	lis1.append(instance.compare(i,j))
	lis2.append(instance.compare(i,k))
	lis3.append(instance.compare(j,k))
'''


transformer = []
for i in a:
	for j,k in zip(X,y):
		if i == k:
			#print(i + '\t' + k)
			transformer.append(j)

for i,j,k in zip(transformer,a,b):
	lis1.append(instance.compare(i,j))
	lis2.append(instance.compare(i,k))
	lis3.append(instance.compare(j,k))

filename = 'Transformer_soundex.txt'
with open(filename, 'w', encoding='utf-8') as f:
	f.write("Hindi-Bhojpuri" + '\t\t' + "Hindi-predicted" + '\t\t'+ "Bhojpuri-predicted" + '\n')
	for a,b,c in zip(lis1, lis2,lis3):
		f.write(str(a) + '\t\t\t' + str(b) + '\t\t\t'+ str(c) + '\n')


'''
for i in range(0, 101, 10):
	if i == 0: 
		continue
	newX = X[: int(len(X) * (i/100))]
	newy = y[: int(len(y) * (i/100))]
	newz = z[: int(len(z) * (i/100))]
	l1 = []
	l2 = []

	for i,j in zip(newX, newy):
		l1.append(instance.compare(i,j))

	for i,j in zip(newX, newz):
		l2.append(instance.compare(i,j))

	cnt = 0
	for i,j in zip(l1, l2):
		if i == j:
			cnt += 1
	print(cnt/len(newX) * 100)


for i in range(0, 101, 10):
	if i == 0: 
		continue
	newX = transformer[: int(len(X) * (i/100))]
	newy = a[: int(len(y) * (i/100))]
	newz = b[: int(len(z) * (i/100))]
	l1 = []
	l2 = []

	for i,j in zip(newX, newy):
		l1.append(instance.compare(i,j))

	for i,j in zip(newX, newz):
		l2.append(instance.compare(i,j))

	cnt = 0
	for i,j in zip(l1, l2):
		if i == j:
			cnt += 1
	print(cnt/len(newX) * 100)

'''