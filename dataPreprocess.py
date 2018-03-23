import os
import glob
import re
import numpy as np
data_path_1 = '/home/wilburzhai/mlpractical/data/aclImdb/train/neg/*.txt'
data_path_2 = '/home/wilburzhai/mlpractical/data/aclImdb/train/pos/*.txt'
data_path_3 = '/home/wilburzhai/mlpractical/data/aclImdb/train/unsup/*.txt'
data_path_4 = '/home/wilburzhai/mlpractical/data/aclImdb/test/neg/*.txt'
data_path_5 = '/home/wilburzhai/mlpractical/data/aclImdb/test/pos/*.txt'
# print(glob.glob(data_path))

V = []
i=0
maxLength = 2505
# read pos reviews to vocabulary
for filename in glob.glob(data_path_1):
	with open(filename, 'r') as f:
		text = f.readlines()
		text[0] = text[0].replace('<br />', ' ')
		words = re.sub('[^\w]', ' ', text[0]).split()
		# print(words)
		# if len(words)>maxLength:
		# 	maxLength = len(words)
		if i>=10000:
			break
		for word in words:
			if word != '' and word not in V:
				V.append(word)
	i = i+1
	if i%100 == 0:
		print(i//100)
	# print(filename)
# read neg reviews to vocabulary		
for filename in glob.glob(data_path_2):
	with open(filename, 'r') as f:
		text = f.readlines()
		text[0] = text[0].replace('<br />', ' ')
		words = re.sub('[^\w]', ' ', text[0]).split()
		# print(words)
		# if len(words)>maxLength:
		# 	maxLength = len(words)
		if i>=22500:
			break
		for word in words:
			if word != '' and word not in V:
				V.append(word)
	i = i+1
	if i%100 == 0:
		print(i//100)
	# print(filename)
# read unsup reviews to vocabulary
# for filename in glob.glob(data_path_3):
# 	with open(filename, 'r') as f:
# 		text = f.readlines()
# 		text[0] = text[0].replace('<br />', ' ')
# 		words = re.sub('[^\w]', ' ',  text[0]).split()
# 		# print(words)
# 		for word in words:
# 			if(word != '' and word not in V):
# 				V.append(word)
# 	i = i+1
# 	if i%100 == 0:
# 		print(i//100)
# 	# print(filename)
V.sort()
print(len(V))
# i = 1
# V_dict = dict()
with open('Vac.txt', 'w') as fout:
	for w in V:
# 		V_dict[w] = i
		fout.write(w + '\n')
# 		i = i+1

# print(maxLength)
i = 0
V_dict = dict()
with open('Vac.txt', 'r') as f:
	V = f.readlines()
	# print(V)
	for w in V:
		V_dict[w.strip()] = i
		# f.write(w + '\n')
		i = i+1


# train_X = np.zeros((20000,None))
train_X = []
train_y = np.zeros(20000,dtype=np.int)
# valid_X = np.zeros((5000,None))
valid_X = []
valid_y = np.zeros(5000,dtype=np.int)
i = 0
counter_train=0
counter_valid=0
for filename in glob.glob(data_path_1):
	with open(filename, 'r') as f:
		text = f.readlines()
		text[0] = text[0].replace('<br />', ' ')
		words = re.sub('[^\w]', ' ', text[0]).split()
		# print(words)
		word_List = []
		for c,word in enumerate(words):
			# if c<maxLength:
			if word in V_dict:
				word_List.append(V_dict[word])
		word_List = np.array(word_List)
		if i<10000:
			# train_X[counter_train] = word_List
			train_X.append(word_List)
			train_y[counter_train] = 0 # target label 0 means neg
			counter_train = counter_train + 1
		else:
			# valid_X[counter_valid] = word_List
			valid_X.append(word_List)
			valid_y[counter_valid] = 0 # target label 0 means neg
			counter_valid = counter_valid + 1
		i = i+1
print('neg finished')
i = 0
for filename in glob.glob(data_path_2):
	with open(filename, 'r') as f:
		text = f.readlines()
		text[0] = text[0].replace('<br />', ' ')
		words = re.sub('[^\w]', ' ', text[0]).split()
		# print(words)
		word_List = []
		for c,word in enumerate(words):
			# if c<maxLength:
			if word in V_dict:
				word_List.append(V_dict[word])
		word_List = np.array(word_List)
		if i<10000:
			# train_X[counter_train] = word_List
			train_X.append(word_List)
			train_y[counter_train] = 1 # target label 1 means pos
			counter_train = counter_train + 1
		else:
			# valid_X[counter_valid] = word_List
			valid_X.append(word_List)
			valid_y[counter_valid] = 1 # target label 1 means pos
			counter_valid = counter_valid + 1
		i = i+1
print('pos finished')

# test_X = np.zeros((25000,None))
test_X = []
test_y = np.zeros(25000,dtype=np.int)
counter_test = 0
# unknow_word = 0
for filename in glob.glob(data_path_4):
	with open(filename, 'r') as f:
		text = f.readlines()
		text[0] = text[0].replace('<br />', ' ')
		words = re.sub('[^\w]', ' ', text[0]).split()
		word_List = np.zeros(len(words))
		for c,word in enumerate(words):
			if word in V_dict:
				word_List[c] = V_dict[word] 
		# test_X[counter_test] = word_List
		test_X.append(word_List)
		test_y[counter_test] = 0 # target label 0 means neg
		counter_test = counter_test + 1
print('neg_test finished')

for filename in glob.glob(data_path_5):
	with open(filename, 'r') as f:
		text = f.readlines()
		text[0] = text[0].replace('<br />', ' ')
		words = re.sub('[^\w]', ' ', text[0]).split()
		word_List = np.zeros(len(words))
		for c,word in enumerate(words):
			if word in V_dict:
				word_List[c] = V_dict[word] 
		# test_X[counter_test] = word_List
		test_X.append(word_List)
		test_y[counter_test] = 1 # target label 1 means pos
		counter_test = counter_test + 1
print('pos_test finished')

# train_X_np = train_X.reshape((10000,maxLength))
# train_y_np = train_y.reshape((10000,1))
# valid_X_np = valid_X.reshape((10000,maxLength))
# valid_y_np = valid_y.reshape((10000,1))
# test_X_np = np.array(test_X)
# test_y_np = np.array(test_y)
print(len(train_X))
print(train_y.shape)
print(len(valid_X))
print(valid_y.shape)
print(len(test_X))
print(test_y.shape)
# print(unknow_word,'unknow_word')
np.savez('aclimdb-train',inputs = np.array(train_X), targets = train_y)
np.savez('aclimdb-valid',inputs = np.array(valid_X), targets = valid_y)
np.savez('aclimdb-test',inputs = np.array(test_X), targets = test_y)

# loaded = np.load('aclimdb-train.npz')
# print(loaded)
# print(len(loaded['inputs']))
