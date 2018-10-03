
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# In[2]:


mnist = input_data.read_data_sets(".", one_hot=True)
print(len(mnist.test.images))
print(len(mnist.test.labels))
print(len(mnist.train.images))
print(len(mnist.train.labels))
print(mnist.train.images.shape) # An array of flatten images
print(mnist.train.labels.shape) # An array of one-hot lables


# In[25]:


class Const():
    def __init__(self):
        self.neurons_1 = 10
        self.neurons_2 = 5
        self.number_classes= 10
        self.input_dimension = 28*28
        self.learning_rate = 0.01
        self.epochs = 100
        self.batch_size = 2**9
        self.iterations = int(np.ceil(mnist.train.num_examples / self.batch_size))
        
const = Const()


# In[26]:


x = tf.placeholder(dtype=tf.float32, shape= [None, 28*28])
y = tf.placeholder(dtype=tf.float32, shape= [None, 10])


# In[35]:


'''
    * some tools
'''
class fully_connected_layer():
    '''Holds a layer weights and biases'''
    def __init__(self, neurons, input_size):
        '''shape: neurons x input_size'''
        self.weights = tf.Variable(tf.random_normal([input_size, neurons]))
        self.biases  = tf.Variable(tf.random_normal([neurons]))


def activation(vector):
    return tf.nn.relu(vector)


def feed_forward_neural_network(X):
    '''Defining the grpah
        * layer_m: holds the weights and biases of layer m
    '''
    layer_1      = fully_connected_layer(input_size=const.input_dimension, neurons=const.neurons_1)
    layer_2      = fully_connected_layer(input_size=const.neurons_1, neurons=const.neurons_2)
    layer_output = fully_connected_layer(input_size=const.neurons_2, neurons=const.number_classes)
    
    '''
    print('W.shape = ', layer_1.weights.shape)
    print('X', X.shape)
    print(layer_1.weights.shape,  np.transpose(X).shape)
    print(layer_1.biases.shape)
    print(layer_output.weights.shape)
    '''
    ############################
    a_1    = activation(tf.matmul(X, layer_1.weights) + layer_1.biases)
    a_2    = activation(tf.matmul(a_1, layer_2.weights) + layer_2.biases)
    logits = tf.matmul(a_2, layer_output.weights) + layer_output.biases
    return logits

def train(X_train, labels_train):
    # Feeding the network
    logits_ = feed_forward_neural_network(X_train) # computing the predictions
    # Computing the cost
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels_train, logits=logits_)
    cost = tf.reduce_mean(losses) 
    # Minimizing the cost
    optmizer = tf.train.AdamOptimizer(learning_rate=const.learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(const.epochs):
            epoch_loss = 0
            for m in range(const.iterations):
                batch_x, batch_y = mnist.train.next_batch(const.batch_size)
                _, c = sess.run([optmizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
            print('loss = ', epoch_loss, ' -> Epoch: ', i)

        # NOTE: تهبيدة
        correct = tf.equal(tf.argmax(logits_, 1),
                           tf.argmax(labels_train, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        # NOTE: تهبيدة
        print('acc = ', accuracy.eval({x:mnist.test.images, y: mnist.test.labels}))

train(mnist.train.images[0:10000], mnist.train.labels[0:10000])
print("DONE")

