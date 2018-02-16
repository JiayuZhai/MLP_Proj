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
	try:
		return train_data.next()
	except:
		train_data = data_providers.ACLIMDBDataProvider('train', batch_size=batchSize)
		return train_data.next()
    # return train_data.next()
def getTestBatch():
    return valid_data.next()
lstmUnits = 64
numClasses = 2
epoches = 50
iterations = 20000//batchSize * epoches
# iterations = 10
maxSeqLength = 2505
numDimensions = 300 #Dimensions for each word vector
vacSize = 93928


tf.reset_default_graph()
with tf.name_scope("Label"):
	# labels
	labels = tf.placeholder(tf.int32, [batchSize, numClasses], name="labels")

with tf.name_scope("Input"):
	# inputs
	inputs = []
	for i in range(batchSize):
		inputs.append(tf.placeholder(tf.float32, [1,None]))

# with tf.name_scope("One_hot"):
# 	one_hots = []
# 	for i in range(batchSize):
# 		one_hots.append(tf.one_hot(inputs[i],vacSize,axis=-1))

with tf.name_scope("Word2vector"):
	# word embedding weights and bias
	W1 = tf.Variable(np.random.rand(1, numDimensions), dtype=tf.float32, name="inputs_")
	# one_hot version for weight
	# W1 = tf.Variable(np.random.rand(vacSize, numDimensions), dtype=tf.float32, name="inputs_")
	b1 = tf.Variable(np.zeros((1,numDimensions)), dtype=tf.float32, name="inputs_")
	# data ready for rnn input
	datas = []
	for i in range(batchSize):
		datas.append(tf.multiply(tf.expand_dims(inputs[i], 2),tf.expand_dims(W1, 1)) + tf.expand_dims(b1, 1))
		# one_hot version for w2v
		# datas.append(tf.matmul(one_hots[i],tf.expand_dims(W1,0)) + b1)

with tf.name_scope("RNN"):
	# define RNN network
	lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
	lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
	values = []
	for i in range(batchSize):
		value, _ = tf.nn.dynamic_rnn(lstmCell, datas[i], dtype=tf.float32)
		values.append(value)

with tf.name_scope("Mean"):
	# get last value from RNN output
	for i in range(batchSize):
		values[i] = tf.transpose(values[i], [1, 0, 2])
	means = []
	for i in range(batchSize):
		means.append(tf.reduce_mean(values[i], axis=0))

with tf.name_scope("Concat"):
	mean_batch = tf.concat([mean for mean in means], 0)

with tf.name_scope("Prefiction"):
	# weight and bias after RNN
	weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]), name="inputs_")
	bias = tf.Variable(tf.constant(0.1, shape=[numClasses]), name="inputs_")
	# get he prediction
	prediction = (tf.matmul(mean_batch, weight) + bias)

with tf.name_scope("Acc"):
	# calculate the accuracy
	correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
	accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

with tf.name_scope("Err"):
	# define the loss
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
# set learning method
optimizer = tf.train.AdamOptimizer().minimize(loss)

import datetime
# define tensorboard related
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
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
    	dict_feed[inputs[j]] = nextBatch[j].reshape((1,len(nextBatch[j])))
    dict_feed[labels] = nextBatchLabels
    sess.run(optimizer, dict_feed)
    print("%f min left for complete" % ((time.time() - start_time)*(iterations-i)/60))
    #Write summary to Tensorboard
    if (i % 50 == 0):
        summary = sess.run(merged, dict_feed)
        writer.add_summary(summary, i)

    #Save the network every 10,000 training iterations
    if (i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)

writer.close()

