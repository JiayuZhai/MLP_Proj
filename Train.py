import tensorflow as tf
import data_providers as data_providers
import numpy as np
import time

batchSize = 1
train_data = data_providers.ACLIMDBDataProvider('train', batch_size=batchSize)
valid_data = data_providers.ACLIMDBDataProvider('valid', batch_size=batchSize)
# print(train_data.next())
def getTrainBatch():
	# inputs, targets = train_data.next()
    return train_data.next()
def getTestBatch():
    return valid_data.next()
lstmUnits = 64
numClasses = 2
iterations = 100000
# iterations = 10
maxSeqLength = 2505
numDimensions = 300 #Dimensions for each word vector


tf.reset_default_graph()
# labels
labels = tf.placeholder(tf.int32, [batchSize, numClasses], name="labels")
# inputs
input_data = tf.placeholder(tf.float32, [batchSize,None], name="inputs")
# input_data = tf.transpose(input_data)
# word embedding weights and bias
W1 = tf.Variable(np.random.rand(batchSize, numDimensions), dtype=tf.float32)
b1 = tf.Variable(np.zeros((batchSize,numDimensions)), dtype=tf.float32)
# data ready for rnn input
data = tf.multiply(tf.expand_dims(input_data, 2),tf.expand_dims(W1, 1)) + tf.expand_dims(b1, 1)

# define RNN network
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

# weight and bias after RNN
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))

# get last value from RNN output
value = tf.transpose(value, [1, 0, 2])
# last = tf.gather(value, int(value.get_shape()[0]) - 1)
mean = tf.reduce_mean(value, axis=0)
# get he prediction
prediction = (tf.matmul(mean, weight) + bias)

# calculate the accuracy
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

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
    # print(nextBatch,nextBatchLabels)
#     print(nextBatchLabels,"this is next labels") [0].reshape((1,len(nextBatch[0])))
    sess.run(optimizer, {input_data: nextBatch[0].reshape((1,len(nextBatch[0]))), labels: nextBatchLabels})
    # print(outs)
    print("%f min left for complete" % ((time.time() - start_time)*(iterations-i)/60))
    #Write summary to Tensorboard
    if (i % 50 == 0):
        summary = sess.run(merged, {input_data: nextBatch[0].reshape((1,len(nextBatch[0]))), labels: nextBatchLabels})
        writer.add_summary(summary, i)

    #Save the network every 10,000 training iterations
    if (i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)

writer.close()

