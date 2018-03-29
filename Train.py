import tensorflow as tf
import data_providers as data_providers
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-hs',default=64,help="Hidden size of LSTM cell, default is 64.")
parser.add_argument('-fs',default=100,help="Size of output fully connected network, default is 100.")
parser.add_argument('-lr',default=0.0001,help="Learning Rate for Adam optimizer, default is 0.0001.")
parser.add_argument('-v',default=100,help="Word vector Length, default is 100.")
parser.add_argument('-bi',default=True,help="Bidirectional network or not, default is True.")
parser.add_argument('-d',default=0.75,help="Dropout for LSTM, default is 0.75.")
parser.add_argument('-att',default=True,help="Use attention or not, default is False.")
parser.add_argument('-bs',default=100,help="Batch size of model input, default is 100.")
parser.add_argument('-rl',default=2,help="RNN layer number, default is 2.")
parser.add_argument('-tflag',default=True,help="Train flag true for training, false for testing, default is True")
parser.add_argument('-model',default="tensorboard/baseline/",help="Model name, default is 'tensorboard/baseline/'.")
args = parser.parse_args()
arg_dict = vars(args)
# print(arg_dict)	

batchSize = int(arg_dict['bs'])

train_data = data_providers.ACLIMDBDataProvider('train', batch_size=batchSize)
valid_data = data_providers.ACLIMDBDataProvider('valid', batch_size=batchSize)
test_data = data_providers.ACLIMDBDataProvider('test', batch_size=batchSize)
def processBatch(nextBatch):
	max_len = 0
	nextBatchLength = np.zeros(batchSize)
	for j in range(batchSize):
		if max_len < len(nextBatch[j]):
			 max_len = len(nextBatch[j])
		nextBatchLength[j] = len(nextBatch[j])
	new_nextBatch = np.zeros((batchSize,max_len))
	for j in range(batchSize):
		new_nextBatch[j] = np.pad(nextBatch[j],(0,max_len-len(nextBatch[j])),'constant')
	return new_nextBatch, nextBatchLength
def getTrainBatch():
	return train_data.next()
def getValidBatch():
	return valid_data.next()
def getTestBatch():
	return test_data.next()

lstmUnits = int(arg_dict['hs'])
numClasses = 2
epoches = 5
iterations = 20000//batchSize * epoches
numDimensions = int(arg_dict['v']) #Dimensions for each word vector
vocSize = 89475
dropoutRatio = float(arg_dict['d'])
useBidirectional = False if arg_dict['bi']=="False" else True
rnnLayers = int(arg_dict['rl'])
useAttention = False if arg_dict['att']=="False" else True
train_flag = False if arg_dict['tflag']=="False" else True
model_name = arg_dict['model']
print(arg_dict)


V_dict = dict()
i = 1
with open('Vac.txt', 'r') as f:
	V = f.readlines()
	# print(V)
	for w in V:
		V_dict[w.strip()] = i
		# f.write(w + '\n')
		i = i+1

def Train():
	tf.reset_default_graph()
	with tf.name_scope("Label"):
		# labels
		labels = tf.placeholder(tf.int32, [batchSize, numClasses], name="labels")

	with tf.name_scope("Input"):
		# inputs
		inputs = tf.placeholder(tf.int32, [batchSize,None]) # batchSize, max_len

	with tf.name_scope("Length"):
		# inputs
		sequence_length = tf.placeholder(tf.int32, [batchSize]) # batchSize, max_len

	with tf.name_scope("RNNDropout"):
		# inputs
		dropout_ratio = tf.placeholder(tf.float32) # constant

	# with tf.name_scope("One_hot"):
	# 	one_hots = tf.one_hot(inputs,vocSize,axis=-1) # batchSize, max_len, voc
	# 	# one_hots_r = tf.one_hot(inputs_r,vocSize,axis=-1)

	with tf.name_scope("Word2vector"):
		# word embedding table
		embedding = tf.get_variable(
			"embedding", [vocSize, numDimensions], dtype=tf.float32)
		# data ready for rnn input
		datas = tf.nn.embedding_lookup(embedding, inputs)

	with tf.name_scope("RNN"):
		# define RNN network
		# b, sequence_length, 300
		rnn_layers = []
		for i in range(rnnLayers):
			# do not use same variable name represent diffent cells
			rnn_layers.append(
				tf.contrib.rnn.DropoutWrapper(
					cell=tf.contrib.rnn.BasicLSTMCell(lstmUnits), output_keep_prob=dropout_ratio))
		lstmCell_m = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
		if useBidirectional:
			rnn_layers_r = []
			for i in range(rnnLayers):
				# do not use same variable name represent diffent cells
				rnn_layers_r.append(
					tf.contrib.rnn.DropoutWrapper(
						cell=tf.contrib.rnn.BasicLSTMCell(lstmUnits), output_keep_prob=dropout_ratio))
			lstmCell_m_r = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_r)
			(outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
				lstmCell_m,lstmCell_m_r, datas,
				sequence_length = sequence_length, dtype=tf.float32) #values: batchSize, max_len, LSTM_size
			final_state = tf.concat([final_state_fw[-1].h, final_state_bw[-1].h],1)
			outputs = tf.concat([outputs_fw, outputs_bw],2)
		else:
			outputs, final_states = tf.nn.dynamic_rnn(lstmCell_m, datas,
				sequence_length = sequence_length, dtype=tf.float32)
			final_state = final_states[-1].h

	with tf.name_scope("Attention"):
		if useAttention:
			if useBidirectional:
				a = tf.reshape(tf.matmul(outputs,tf.reshape(final_state,[batchSize,2*lstmUnits,1])),
					[batchSize,-1,1,1])
				alpha = tf.nn.softmax(processPadding(a),dim=1)
				context_vector = tf.reshape(tf.matmul(
					alpha,
					tf.expand_dims(outputs,2)
					),[batchSize,-1,2*lstmUnits])
				context_vector = tf.reduce_mean(context_vector,axis=1)
			else:
				a = tf.reshape(tf.matmul(outputs,tf.reshape(final_state,[batchSize,lstmUnits,1])),
					[batchSize,-1,1,1])
				alpha = tf.nn.softmax(processPadding(a),dim=1)
				context_vector = tf.reshape(tf.matmul(
					alpha,
					tf.expand_dims(outputs,2)
					),[batchSize,-1,lstmUnits])
				context_vector = tf.reduce_mean(context_vector,axis=1)
				# context_vector
		#b, lstm_units # output size
		#b, sequence_length, lstm_units # hidden vector size


	with tf.name_scope("Prediction"):
		# weight and bias after RNN
		if useAttention:
			if useBidirectional:
				weight1 = tf.Variable(np.random.rand(lstmUnits*4, numClasses), dtype=tf.float32)
			else:
				weight1 = tf.Variable(np.random.rand(lstmUnits*2, numClasses), dtype=tf.float32)
		else:
			if useBidirectional:
				weight1 = tf.Variable(np.random.rand(lstmUnits*2, numClasses), dtype=tf.float32)
			else:
				weight1 = tf.Variable(np.random.rand(lstmUnits, numClasses), dtype=tf.float32)
		bias1 = tf.Variable(tf.constant(0.1, shape=[numClasses]))
		# get he prediction
		if useAttention:
			prediction = tf.matmul(tf.concat([context_vector,final_state],1),weight1) + bias1
		else:
			prediction = tf.matmul(final_state, weight1) + bias1

	with tf.name_scope("Acc"):
		# calculate the accuracy
		predict_test = tf.argmax(prediction,1)
		correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
		accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

	with tf.name_scope("Err"):
		# define the loss
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

	# with tf.name_scope("ValidAcc"):
	# 	# calculate the accuracy
	# 	valid_correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
	# 	valid_accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

	# with tf.name_scope("ValidErr"):
	# 	# define the loss
	# 	valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

	# set learning method
	optimizer = tf.train.AdamOptimizer().minimize(loss)

	import datetime
	# define tensorboard related
	tf.summary.scalar('Loss', loss, collections=['train'])
	tf.summary.scalar('Accuracy', accuracy, collections=['train'])
	merged = tf.summary.merge_all('train')
	logdir = model_name

	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	
	if train_flag:
		# define output files
		writer_train = tf.summary.FileWriter(logdir + 'train', sess.graph)
		writer_valid = tf.summary.FileWriter(logdir + 'valid', sess.graph)
		sess.run(tf.global_variables_initializer())
		# start train
		best_val_acc = 0.0
		for i in range(iterations):
			start_time = time.time()
			#Next Batch of reviews
			nextBatch, nextBatchLabels = getTrainBatch()
			dict_feed = {}
			dict_feed[inputs], dict_feed[sequence_length] = processBatch(nextBatch)
			dict_feed[labels] = nextBatchLabels
			dict_feed[dropout_ratio] = dropoutRatio
			# print(nextBatch[j].shape)
			# print(sess.run(datas[0], dict_feed))
			# print(sess.run(predict_test, dict_feed))
			# print(sess.run(final_state_fw, dict_feed))
			# print(sess.run(outputs_fw, dict_feed))
			# print(sess.run(a, dict_feed))
			# print(sess.run(final_states, dict_feed))
			sess.run(optimizer, dict_feed)
			print("%f min left for complete" % ((time.time() - start_time)*(iterations-i)/60))
			#Write summary to Tensorboard
			if (i % 50 == 0):
				# Train summary
				summary = sess.run(merged, dict_feed)
				writer_train.add_summary(summary, i)
				# Validation summary
				valid_acc = 0.0
				valid_loss = 0.0
				for j in range(5000//batchSize):
					valid_nextBatch, valid_nextBatchLabels = getValidBatch()
					valid_dict = {}
					valid_dict[inputs], valid_dict[sequence_length] = processBatch(valid_nextBatch)
					valid_dict[labels] = valid_nextBatchLabels
					valid_dict[dropout_ratio] = 1
					# if useAttention:
					# 	alpha = sess.run(alpha, valid_dict)
					# 	pred = sess.run(tf.argmax(prediction,1)[0], valid_dict)
					# 	plot_attention(alpha[0],index2word(valid_dict[inputs][0],valid_dict[sequence_length][0]),pred,
					# 		valid_dict[sequence_length][0],plot_name="test")
					# 	# break
					summary = sess.run([accuracy,loss], valid_dict)
					valid_acc += summary[0]
					valid_loss += summary[1]

				valid_summary = tf.Summary(value=[
					tf.Summary.Value(tag="Accuracy", simple_value=valid_acc/(5000//batchSize)), 
					tf.Summary.Value(tag="Loss", simple_value=valid_loss/(5000//batchSize)), 
					])
				writer_valid.add_summary(valid_summary, i)
				#check the best validation accuracy and update best model
				if valid_acc/(5000//batchSize) > best_val_acc:
					best_val_acc = valid_acc/(5000//batchSize)
					if not os.path.exists("models/" + logdir):
						os.makedirs("models/" + logdir)
					save_path = saver.save(sess, "models/" + logdir + "pretrained_lstm.ckpt")
					print("saved to %s" % save_path)
		writer_train.close()
		writer_valid.close()		
	else:
		# test section
		saver.restore(sess, "models/" + model_name +"pretrained_lstm.ckpt")
		test_acc = 0.0
		for j in range(25000//batchSize):
			test_nextBatch, test_nextBatchLabels = getTestBatch()
			test_dict = {}
			test_dict[inputs], test_dict[sequence_length] = processBatch(test_nextBatch)
			test_dict[labels] = test_nextBatchLabels
			test_dict[dropout_ratio] = 1
			if useAttention:
				alpha = sess.run(alpha, test_dict)
				pred = sess.run(tf.argmax(prediction,1), test_dict)
				label_att = test_dict[labels][:,1]
				# for x in range(len(pred)):
				# 	print(x,pred[x],label_att[x])
				index_ = 0
				plot_attention(alpha[index_],index2word(test_dict[inputs][index_],test_dict[sequence_length][index_]),
					pred[index_], int(label_att[index_]),
					test_dict[sequence_length][index_],plot_name="test")
				break
			test_acc += sess.run(accuracy, test_dict)
		print("Test accuracy of '" + model_name + "' is " + str(test_acc/(25000//batchSize)))

	# writer_train.close()
	# writer_valid.close()

def processPadding(a):
	# process the padded zeros to -Inf.
	return a+tf.log(tf.abs(tf.sign(a)))

def plot_attention(alpha_arr, inputs, pred, label_att, sequence_length, plot_name=None):
	'''
	Support function to plot attention vectors
	# '''
	# if gpuid >= 0:
	#     alpha_arr = cuda.to_cpu(alpha_arr).astype(np.float32)
	# print(alpha_arr.shape,pred,sequence_length)
	sequence_length = int(sequence_length)
	fig = plt.figure()
	fig.set_size_inches(1*20+4,int(0.6*(sequence_length//20+1)+0.5))

	gs = gridspec.GridSpec(2, 2, width_ratios=[20,1],height_ratios=[20,1])

	ax = plt.subplot(gs[0])
	ax_c = plt.subplot(gs[1])

	cmap = sns.light_palette((200, 75, 60), input="husl", as_cmap=True)
	# prop = FontProperties(fname='fonts/IPAfont00303/ipam.ttf', size=12)
	# inputs = inputs[:int(sequence_length)]
	padding = (sequence_length//20+1)*20-sequence_length
	data_ = np.append(alpha_arr[:sequence_length],np.zeros(padding)).reshape((sequence_length//20+1,20))
	annot = np.append(np.array(inputs),np.array(['' for i in range(padding)])).reshape((sequence_length//20+1,20))

	ax = sns.heatmap(data_, annot=annot,fmt = '', cmap=cmap, ax=ax, cbar_ax=ax_c)

	ax.xaxis.tick_bottom()
	ax.yaxis.tick_left()

	# ax.set_xticklabels(inputs, minor=True, rotation=60, size=12)
	for label in ax.get_xticklabels(minor=False):
		label.set_fontsize(12)
		# label.set_font_properties(prop)

	for label in ax.get_yticklabels(minor=False):
		label.set_fontsize(12)
		label.set_rotation(-90)
		# label.set_horizontalalignment('left')
	text = ["A ", " Sample, Predicted as "]
	neg_pos = ['Negative','Positive']
	ax.set_title(text[0] + neg_pos[label_att] + text[1] + neg_pos[pred],size=20)

	if plot_name:
		fig.savefig(plot_name + str(label_att) + str(pred)+'.pdf', format="pdf")

def index2word(inputs,seq_len):
	words = []
	# print(inputs)
	for x in range(int(seq_len)):
		for word, index in V_dict.items():
			if index-1 == inputs[x]:
				words.append(word)
	return words

if __name__ == "__main__":
	Train()