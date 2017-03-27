import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#our formula = (input data * weight) + biases

### Building The Model
n_nodes_hl1 = 15
n_nodes_hl2 = 10
n_nodes_hl3 = 5
n_classes = 10
batch_size = 96
all_epochs = 10 #epoch - cycles of feed forward & back propagation



X = tf.placeholder('float', [None, 784])  #None - batch size is not having constraints
y = tf.placeholder('float')

# creating computational graph
def neural_network_model(data):
											#tf.random_normal outputs random values for the shape we want
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					'biases':tf.Variable(tf.random_normal([n_classes]))}


	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) #output of l1

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2) #output of l2

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3) #output of l3

	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

	return output


# training computational graph
def train_neural_network(X):
	prediction = neural_network_model(X)
	#for more information about softmax_cross_entropy_with_logits https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
	loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer(0.001).minimize(loss_function) #AdamOptimizer gives 93%



	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(all_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)): #take only 60000/100 examples for running the training step
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
				# We run train_step feeding in the batches data to replace the placeholder
				_, ans = sess.run([optimizer, loss_function], feed_dict={X: batch_xs, y: batch_ys})
				epoch_loss += ans
			print('Epoch', epoch, 'completed out of %i loss: %i' % (all_epochs, epoch_loss))

		#tf.argmax gives the index of the highest entry in a tensor along some axis.
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

		#accuracy on the testing set.
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:',accuracy.eval({X: mnist.test.images, y: mnist.test.labels}))

train_neural_network(X)
