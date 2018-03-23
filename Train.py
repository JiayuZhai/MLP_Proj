import tensorflow as tf
import data_providers as data_providers
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

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
args = parser.parse_args()
arg_dict = vars(args)
# print(arg_dict)	

batchSize = arg_dict['bs']

train_data = data_providers.ACLIMDBDataProvider('train', batch_size=batchSize)
valid_data = data_providers.ACLIMDBDataProvider('valid', batch_size=batchSize)
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

lstmUnits = arg_dict['hs']
numClasses = 2
epoches = 5
iterations = 20000//batchSize * epoches
numDimensions = arg_dict['v'] #Dimensions for each word vector
vocSize = 93929
dropoutRatio = arg_dict['d']
useBidirectional = True
rnnLayers = arg_dict['rl']
useAttention = arg_dict['att']

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
		# word embedding weights and bias
		# W1 = tf.Variable(np.random.rand(1, middleDimension), dtype=tf.float32)
		# # one_hot version for weight
		# # W1 = tf.Variable(np.random.rand(vacSize, numDimensions), dtype=tf.float32, name="inputs_")
		# b1 = tf.Variable(np.zeros((1,middleDimension)), dtype=tf.float32)
		# W2 = tf.Variable(np.random.rand(middleDimension, middleDimension), dtype=tf.float32)
		# b2 = tf.Variable(np.zeros((1,middleDimension)), dtype=tf.float32)
		embedding = embedding = tf.get_variable(
			"embedding", [vocSize, numDimensions], dtype=tf.float32)
		
		# W3 = tf.Variable(np.random.rand(1, numDimensions), dtype=tf.float32)
		# b3 = tf.Variable(np.zeros((1,numDimensions)), dtype=tf.float32)

		# data ready for rnn input
		datas = tf.nn.embedding_lookup(embedding, inputs)

	with tf.name_scope("RNN"):
		# define RNN network
		# b, sequence_length, 300
		rnn_layers = []
		# rnn_layers = [tf.contrib.rnn.BasicLSTMCell(size) for size in [lstmUnits for i in range(rnnLayers)]]
		for i in range(rnnLayers):
			# lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
			# lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropout_ratio)
			rnn_layers.append(tf.contrib.rnn.DropoutWrapper(cell=tf.contrib.rnn.BasicLSTMCell(lstmUnits), output_keep_prob=dropout_ratio))
		lstmCell_m = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
		if useBidirectional:
			rnn_layers_r = []
			for i in range(rnnLayers):
				# lstmCell_r = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
				# lstmCell_r = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropout_ratio)
				rnn_layers_r.append(tf.contrib.rnn.DropoutWrapper(cell=tf.contrib.rnn.BasicLSTMCell(lstmUnits), output_keep_prob=dropout_ratio))
			lstmCell_m_r = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_r)
			(outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(lstmCell_m,lstmCell_m_r, datas,
				sequence_length = sequence_length, dtype=tf.float32) #values: batchSize, max_len, LSTM_size
			final_state = tf.concat([final_state_fw[-1].h, final_state_bw[-1].h],1)
			outputs = tf.concat([outputs_fw, outputs_bw],2)
		else:
			outputs, final_state = tf.nn.dynamic_rnn(lstmCell_m, datas,
				sequence_length = sequence_length, dtype=tf.float32)

	with tf.name_scope("Attention"):
		if useAttention:
			if useBidirectional:
				a = tf.reshape(tf.matmul(outputs,tf.reshape(final_state,[batchSize,2*lstmUnits,1])),[batchSize,-1,1,1])
				alpha = tf.nn.softmax(processPadding(a),dim=1)
				context_vector = tf.reshape(tf.matmul(
					alpha,
					tf.expand_dims(outputs,2)
					),[batchSize,-1,2*lstmUnits])
				context_vector = tf.reduce_mean(context_vector,axis=1)
			else:
				a = tf.reshape(tf.matmul(outputs,tf.reshape(final_state,[batchSize,2*lstmUnits,1])),[batchSize,-1,1,1])
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
		# weight2 = tf.Variable(np.random.rand(middleDimension, middleDimension), dtype=tf.float32)
		# bias2 = tf.Variable(tf.constant(0.1, shape=[middleDimension]))
		# weight3 = tf.Variable(np.random.rand(middleDimension, numClasses), dtype=tf.float32)
		# bias3 = tf.Variable(tf.constant(0.1, shape=[numClasses]))
		# get he prediction
		if useAttention:
			prediction = tf.matmul(tf.concat([context_vector,final_state],1),weight1) + bias1
		else:
			prediction = tf.matmul(final_state, weight1) + bias1

	with tf.name_scope("Acc"):
		# calculate the accuracy
		predict_test = tf.argmax(prediction,1)
		# labels_test = tf.argmax(labels,1)
		correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
		accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

	with tf.name_scope("Err"):
		# define the loss
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

	with tf.name_scope("ValidAcc"):
		# calculate the accuracy
		valid_correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
		valid_accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

	with tf.name_scope("ValidErr"):
		# define the loss
		valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

	# set learning method
	optimizer = tf.train.AdamOptimizer().minimize(loss)

	import datetime
	# define tensorboard related
	tf.summary.scalar('Loss', loss, collections=['train'])
	tf.summary.scalar('Accuracy', accuracy, collections=['train'])
	merged = tf.summary.merge_all('train')
	# valid_merged = tf.summary.merge([tf.summary.scalar('ValidLoss', valid_loss),
	# 	tf.summary.scalar('ValidAccuracy', valid_accuracy)])
	logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

	# define output files
	sess = tf.InteractiveSession()
	writer_train = tf.summary.FileWriter(logdir + 'train', sess.graph)
	writer_valid = tf.summary.FileWriter(logdir + 'valid', sess.graph)
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	# start train
	best_val_acc = 0.0
	for i in range(iterations):
		#Next Batch of reviews
		start_time = time.time()
		nextBatch, nextBatchLabels = getTrainBatch()
		dict_feed = {}
		# for j in range(batchSize):
		# 	dict_feed[inputs[j]] = nextBatch[j].reshape((len(nextBatch[j]),1))/9392.8
		dict_feed[inputs], dict_feed[sequence_length] = processBatch(nextBatch)
		dict_feed[labels] = nextBatchLabels
		dict_feed[dropout_ratio] = dropoutRatio
		# print(nextBatch[j].shape)
		# print(sess.run(datas[0], dict_feed))
		# print(sess.run(predict_test, dict_feed))
		# print(sess.run(final_state_fw, dict_feed))
		# print(sess.run(outputs_fw, dict_feed))
		# print(sess.run(a, dict_feed))
		sess.run(optimizer, dict_feed)
		print("%f min left for complete" % ((time.time() - start_time)*(iterations-i)/60))
		#Write summary to Tensorboard
		if (i % 50 == 0):
			# Train summary
			summary = sess.run(merged, dict_feed)
			writer_train.add_summary(summary, i)
			# Validation summary
			valid_acc = 0.0
			for j in range(5000//batchSize):
				valid_nextBatch, valid_nextBatchLabels = getValidBatch()
				valid_dict = {}
				valid_dict[inputs], valid_dict[sequence_length] = processBatch(valid_nextBatch)
				valid_dict[labels] = valid_nextBatchLabels
				valid_dict[dropout_ratio] = 1
				# if useAttention:
				# 	alpha = sess.run(alpha, valid_dict)
				# 	pred = sess.run(tf.argmax(prediction,1)[0], valid_dict)
				# 	plot_attention(alpha[0],index2word(valid_dict[inputs][0]),pred,valid_dict[sequence_length][0],plot_name="test")
				# 	# break
				valid_acc += sess.run(accuracy, valid_dict)
			valid_summary = tf.Summary(value=[
				tf.Summary.Value(tag="Accuracy", simple_value=valid_acc/(5000//batchSize)), 
				])
			writer_valid.add_summary(valid_summary, i)
			#check the best validation accuracy and update best model
			if valid_acc/(5000//batchSize)>best_val_acc:
				best_val_acc = valid_acc/(5000//batchSize)
				save_path = saver.save(sess, "models/" + logdir + "pretrained_lstm.ckpt", global_step=i)
				print("saved to %s" % save_path)
		#Save the network every 10,000 training iterations
		# if (i % 10000 == 0 and i != 0):
		# 	save_path = saver.save(sess, "models/" + logdir + "pretrained_lstm.ckpt", global_step=i)
		# 	print("saved to %s" % save_path)

	# writer_train.close()
	# writer_valid.close()

def processPadding(a):
	# process the padded zeros to -Inf.
	return a+tf.log(tf.abs(tf.sign(a)))

def plot_attention(alpha_arr, inputs, pred, sequence_length, plot_name=None):
	'''
	Support function to plot attention vectors
	# '''
	# if gpuid >= 0:
	#     alpha_arr = cuda.to_cpu(alpha_arr).astype(np.float32)
	# print(alpha_arr.shape,pred,sequence_length)
	fig = plt.figure()
	fig.set_size_inches(1*sequence_length+4,3)

	gs = gridspec.GridSpec(2, 2, width_ratios=[12,1],height_ratios=[12,1])

	ax = plt.subplot(gs[0])
	ax_c = plt.subplot(gs[1])

	cmap = sns.light_palette((200, 75, 60), input="husl", as_cmap=True)
	# prop = FontProperties(fname='fonts/IPAfont00303/ipam.ttf', size=12)
	# inputs = inputs[:int(sequence_length)]
	ax = sns.heatmap(alpha_arr[:int(sequence_length)].reshape((1,int(sequence_length))), xticklabels=inputs[:int(sequence_length)], yticklabels=[pred], ax=ax, cmap=cmap, cbar_ax=ax_c)

	ax.xaxis.tick_top()
	ax.yaxis.tick_right()

	ax.set_xticklabels(inputs, minor=True, rotation=60, size=12)
	for label in ax.get_xticklabels(minor=False):
		label.set_fontsize(12)
		# label.set_font_properties(prop)

	for label in ax.get_yticklabels(minor=False):
		label.set_fontsize(12)
		label.set_rotation(-90)
		label.set_horizontalalignment('left')

	ax.set_xlabel("Sentence", size=10)
	# ax.set_ylabel("Hypothesis", size=20)

	if plot_name:
		fig.savefig(plot_name, format="pdf")

def index2word(inputs,seq_len):
	words = []
	# print()
	for x in range(seq_len):
		for word, index in V_dict.items():
			if index == inputs[x][0]:
				words.append(word)
	return words

if __name__ == "__main__":
	Train()