import tensorflow as tf
import data_providers as data_providers
import numpy as np
import time

batchSize = 20
train_data = data_providers.ACLIMDBDataProvider('train', batch_size=batchSize)
valid_data = data_providers.ACLIMDBDataProvider('valid', batch_size=batchSize)
# print(train_data.next())
def getTrainBatch():
	# inputs, targets = train_data.next()
	return train_data.next()
	# try:
	# 	return train_data.next()
	# except:
	# 	train_data = data_providers.ACLIMDBDataProvider('train', batch_size=batchSize)
	# 	return train_data.next()
    # return train_data.next()
def getValidBatch():
	# try:
	return valid_data.next()
	# except:
	# 	valid_data = data_providers.ACLIMDBDataProvider('valid', batch_size=batchSize)
	# 	return valid_data.next()

lstmUnits = 64
numClasses = 2
epoches = 10
iterations = 20000//batchSize * epoches
# print(iterations)
# iterations = 10
# maxSeqLength = 2505
middleDimension = 100
numDimensions = 100 #Dimensions for each word vector
vacSize = 93928


tf.reset_default_graph()
with tf.name_scope("Label"):
	# labels
	labels = tf.placeholder(tf.int32, [batchSize, numClasses], name="labels")

with tf.name_scope("Input"):
	# inputs
	inputs = []
	for i in range(batchSize):
		inputs.append(tf.placeholder(tf.float32, [None,1]))

# with tf.name_scope("One_hot"):
# 	one_hots = []
# 	for i in range(batchSize):
# 		one_hots.append(tf.one_hot(inputs[i],vacSize,axis=-1))

with tf.name_scope("Word2vector"):
	# word embedding weights and bias
	# W1 = tf.Variable(np.random.rand(1, middleDimension), dtype=tf.float32)
	# # one_hot version for weight
	# # W1 = tf.Variable(np.random.rand(vacSize, numDimensions), dtype=tf.float32, name="inputs_")
	# b1 = tf.Variable(np.zeros((1,middleDimension)), dtype=tf.float32)
	# W2 = tf.Variable(np.random.rand(middleDimension, middleDimension), dtype=tf.float32)
	# b2 = tf.Variable(np.zeros((1,middleDimension)), dtype=tf.float32)
	W3 = tf.Variable(np.random.rand(1, numDimensions), dtype=tf.float32)
	b3 = tf.Variable(np.zeros((1,numDimensions)), dtype=tf.float32)
	# layer1 = []
	# layer2 = []
	# data ready for rnn input
	datas = []
	for i in range(batchSize):
		# layer1.append(tf.matmul(	#First Layer 100 units
		# 			tf.transpose(inputs[i],[1,0]),
		# 			W1) + # 1x100 weights
		# 			b1) #1x100 bias
		# layer2.append(tf.matmul(	#Second Layer 100 units
		# 			layer1[i],
		# 			W2) + # 100x100 weights
		# 			b2) #1x100 bias
		datas.append(tf.matmul(	# Third Layer 300 units as output
					inputs[i], 
					W3) + #100x300 weights
					b3) #1x300 bias
		# one_hot version for w2v
		# datas.append(tf.matmul(one_hots[i],tf.expand_dims(W3,0)) + b3)

with tf.name_scope("RNN"):
	# define RNN network
	# b, sequence_length, 300
	lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
	lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
	values = []
	final_states = []
	for i in range(batchSize):
		value, final_state = tf.nn.dynamic_rnn(lstmCell, tf.expand_dims(datas[i],0), dtype=tf.float32)
		values.append(value)
		final_states.append(final_state)
	#b, lstm_units # output size
	#b, sequence_length, lstm_units # hidden vector size
with tf.name_scope("Max"):
	# get last value from RNN output
	for i in range(batchSize):
		values[i] = tf.transpose(values[i], [1, 0, 2])#sequence_length, b, lstm_units
	maxs = []
	for i in range(batchSize):
		maxs.append(tf.reduce_max(values[i], axis=0))#b, lstm_units

with tf.name_scope("Concat"):
	max_batch = tf.concat([max_item for max_item in maxs], 0)

with tf.name_scope("Prefiction"):
	# weight and bias after RNN
	weight1 = tf.Variable(np.random.rand(lstmUnits, numClasses), dtype=tf.float32)
	bias1 = tf.Variable(tf.constant(0.1, shape=[numClasses]))
	# weight2 = tf.Variable(np.random.rand(middleDimension, middleDimension), dtype=tf.float32)
	# bias2 = tf.Variable(tf.constant(0.1, shape=[middleDimension]))
	# weight3 = tf.Variable(np.random.rand(middleDimension, numClasses), dtype=tf.float32)
	# bias3 = tf.Variable(tf.constant(0.1, shape=[numClasses]))
	# get he prediction
	prediction = tf.matmul(
		# (tf.matmul(
			# (tf.matmul(
				max_batch, 
				weight1) + bias1
			# ),weight2)+bias2
		# ),weight3)+bias3

with tf.name_scope("Acc"):
	# calculate the accuracy
	# predict_test = tf.argmax(prediction,1)
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
merged = tf.summary.merge([tf.summary.scalar('TrainLoss', loss),
	tf.summary.scalar('TrainAccuracy', accuracy)])
valid_merged = tf.summary.merge([tf.summary.scalar('ValidLoss', valid_loss),
	tf.summary.scalar('ValidAccuracy', valid_accuracy)])
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

# define output files
sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(logdir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
# start train
for i in range(iterations):
    #Next Batch of reviews
    start_time = time.time()
    nextBatch, nextBatchLabels = getTrainBatch()
    dict_feed = {}
    for j in range(batchSize):
    	dict_feed[inputs[j]] = nextBatch[j].reshape((len(nextBatch[j]),1))/9392.8
    dict_feed[labels] = nextBatchLabels
    # print(nextBatch[j].shape)
    # print(sess.run(datas[0], dict_feed))
    # print(sess.run(predict_test, dict_feed))
    # print(sess.run(labels_test, dict_feed))
    sess.run(optimizer, dict_feed)
    print("%f min left for complete" % ((time.time() - start_time)*(iterations-i)/60))
    #Write summary to Tensorboard
    if (i % 50 == 0):
    	# Train summary
        summary = sess.run(merged, dict_feed)
        writer.add_summary(summary, i)
        # Validation summary
        valid_nextBatch, valid_nextBatchLabels = getValidBatch()
        valid_dict = {}
        for j in range(batchSize):
            valid_dict[inputs[j]] = valid_nextBatch[j].reshape((len(valid_nextBatch[j]),1))/9392.8
        valid_dict[labels] = valid_nextBatchLabels
        valid_summary = sess.run(valid_merged, valid_dict)
        writer.add_summary(valid_summary, i)

    #Save the network every 10,000 training iterations
    if (i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)

writer.close()

