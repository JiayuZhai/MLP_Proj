import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
parser.add_argument('-hs',default=64,help="Hidden size of LSTM cell, default is 64.")
parser.add_argument('-fs',default=100,help="Size of output fully connected network, default is 64.")
parser.add_argument('-lr',default=0.0001,help="Learning Rate for Adam optimizer, default is 64.")
parser.add_argument('-v',default=100,help="Word vector Length, default is 64.")
parser.add_argument('-bi',default=True,help="Bidirectional network or not, default is 64.")
parser.add_argument('-d',default=0.5,help="Dropout for LSTM, default is 64.")
parser.add_argument('-att',default=False,help="Use attention or not, default is False.")
args = parser.parse_args()
arg_dict = vars(args)
print(arg_dict)	

# print(arg_dict['b'])


def attention(inputs, size, scope):
    with tf.variable_scope(scope or 'attention') as scope:
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                             shape=[size],
                                             regularizer=layers.l2_regularizer(scale=L2_REG),
                                             dtype=tf.float32)
        input_projection = layers.fully_connected(inputs, size,
                                            activation_fn=tf.tanh,
                                            weights_regularizer=layers.l2_regularizer(scale=L2_REG))
        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
        attention_weights = tf.nn.softmax(vector_attn, dim=1)
        weighted_projection = tf.multiply(inputs, attention_weights)
        outputs = tf.reduce_sum(weighted_projection, axis=1)

return outputs



# Batch size
B = 4
# (Maximum) number of time steps in this batch
T = 8
RNN_DIM = 128
NUM_CLASSES = 10
 
# The *acutal* length of the examples
example_len = [1, 2, 3, 8]
 
# The classes of the examples at each step (between 1 and 9, 0 means padding)
y = np.random.randint(1, 10, [B, T])
for i, length in enumerate(example_len):
    y[i, length:] = 0  
     
# The RNN outputs
rnn_outputs = tf.convert_to_tensor(np.random.randn(B, T, RNN_DIM), dtype=tf.float32)
 
# Output layer weights
W = tf.get_variable(
    name="W",
    initializer=tf.random_normal_initializer(),
    shape=[RNN_DIM, NUM_CLASSES])
 
# Calculate logits and probs
# Reshape so we can calculate them all at once
rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, RNN_DIM])
logits_flat = tf.batch_matmul(rnn_outputs_flat, W)
probs_flat = tf.nn.softmax(logits_flat)
 
# Calculate the losses 
y_flat =  tf.reshape(y, [-1])
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_flat, y_flat)
 
# Mask the losses
mask = tf.sign(tf.to_float(y_flat))
masked_losses = mask * losses
 
# Bring back to [B, T] shape
masked_losses = tf.reshape(masked_losses,  tf.shape(y))
 
# Calculate mean loss
mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / example_len
mean_loss = tf.reduce_mean(mean_loss_by_example)