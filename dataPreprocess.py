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

# V = []
# i=0
# # read pos reviews to vocabulary
# for filename in glob.glob(data_path_1):
# 	with open(filename, 'r') as f:
# 		text = f.readlines()
# 		text[0] = text[0].replace('<br />', ' ')
# 		words = re.sub('[^\w]', ' ', text[0]).split()
# 		# print(words)
# 		for word in words:
# 			if(word != '' and word not in V):
# 				V.append(word)
# 	i = i+1
# 	if i%100 == 0:
# 		print(i//100)
# 	# print(filename)
# # read neg reviews to vocabulary		
# for filename in glob.glob(data_path_2):
# 	with open(filename, 'r') as f:
# 		text = f.readlines()
# 		text[0] = text[0].replace('<br />', ' ')
# 		words = re.sub('[^\w]', ' ', text[0]).split()
# 		# print(words)
# 		for word in words:
# 			if(word != '' and word not in V):
# 				V.append(word)
# 	i = i+1
# 	if i%100 == 0:
# 		print(i//100)
# 	# print(filename)
# # read unsup reviews to vocabulary
# # for filename in glob.glob(data_path_3):
# # 	with open(filename, 'r') as f:
# # 		text = f.readlines()
# # 		text[0] = text[0].replace('<br />', ' ')
# # 		words = re.sub('[^\w]', ' ',  text[0]).split()
# # 		# print(words)
# # 		for word in words:
# # 			if(word != '' and word not in V):
# # 				V.append(word)
# # 	i = i+1
# # 	if i%100 == 0:
# # 		print(i//100)
# # 	# print(filename)
# V.sort()
# # i = 1
# # V_dict = dict()
# with open('Vac.txt', 'w') as fout:
# 	for w in V:
# # 		V_dict[w] = i
# 		fout.write(w + '\n')
# # 		i = i+1

i = 1
V_dict = dict()
with open('Vac.txt', 'r') as f:
	V = f.readlines()
	# print(V)
	for w in V:
		V_dict[w.strip()] = i
		# f.write(w + '\n')
		i = i+1


train_X = []
train_y = []
valid_X = []
valid_y = []
i = 1
for filename in glob.glob(data_path_1):
	with open(filename, 'r') as f:
		text = f.readlines()
		text[0] = text[0].replace('<br />', ' ')
		words = re.sub('[^\w]', ' ', text[0]).split()
		# print(words)
		word_List = []
		for word in words:
			word_List.append(V_dict[word])
		if(i<=10000):
			train_X.append(word_List)
			train_y.append(0) # target label 0 means neg
		else:
			valid_X.append(word_List)
			valid_y.append(0) # target label 0 means neg
		i = i+1
print('neg finished')
i = 1
for filename in glob.glob(data_path_2):
	with open(filename, 'r') as f:
		text = f.readlines()
		text[0] = text[0].replace('<br />', ' ')
		words = re.sub('[^\w]', ' ', text[0]).split()
		# print(words)
		word_List = []
		for word in words:
			word_List.append(V_dict[word])
		if(i<=10000):
			train_X.append(word_List)
			train_y.append(1) # target label 1 means pos
		else:
			valid_X.append(word_List)
			valid_y.append(1) # target label 1 means pos
		i = i+1
print('pos finished')

test_X = []
test_y = []
unknow_word = 0
for filename in glob.glob(data_path_4):
	with open(filename, 'r') as f:
		text = f.readlines()
		text[0] = text[0].replace('<br />', ' ')
		words = re.sub('[^\w]', ' ', text[0]).split()
		# print(words)
		word_List = []
		for word in words:
			if(word in V_dict):
				word_List.append(V_dict[word])
			else:
				unknow_word = unknow_word + 1
		test_X.append(word_List)
		test_y.append(0)
print('neg_test finished')

for filename in glob.glob(data_path_5):
	with open(filename, 'r') as f:
		text = f.readlines()
		text[0] = text[0].replace('<br />', ' ')
		words = re.sub('[^\w]', ' ', text[0]).split()
		# print(words)
		word_List = []
		for word in words:
			if(word in V_dict):
				word_List.append(V_dict[word])
			else:
				unknow_word = unknow_word + 1
		test_X.append(word_List)
		test_y.append(1)
print('pos_test finished')

train_X_np = np.array(train_X)
train_y_np = np.array(train_y)
valid_X_np = np.array(valid_X)
valid_y_np = np.array(valid_y)
test_X_np = np.array(test_X)
test_y_np = np.array(test_y)
print(len(train_X_np))
print(len(train_y_np))
print(len(valid_X_np))
print(len(valid_y_np))
print(len(test_X_np))
print(len(test_y_np))
print(unknow_word,'unknow_word')
np.savez('aclimdb-train',inputs = train_X_np, targets = train_y_np)
np.savez('aclimdb-valid',inputs = valid_X_np, targets = valid_y_np)
np.savez('aclimdb-test',inputs = test_X_np, targets = test_y_np)

# loaded = np.load('aclimdb-train.npz')
# print(loaded)
# print(len(loaded['inputs']))
