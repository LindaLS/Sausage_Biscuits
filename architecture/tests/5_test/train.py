# Example implementing 5 layer encoder
# Original code taken from
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
# The model trained here is restored in load.py

from __future__ import division, print_function, absolute_import

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# data_set = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as sio
sys.path.insert(0, '../..') # Add path to where TF_Model.py is, if not in the same dir
from TF_Model import *
from utils import *
from load_all_data import *

data_path = ("../../")

np.set_printoptions(threshold=np.nan)



# Create TF_Model, a wrapper for models created using tensorflow
# Note that the configuration file 'config.txt' must be present in the directory
model = TF_Model('model')


# Parameters
learning_rate = 1e-4
training_epochs = 6000
batch_size = 50
display_step = 10
examples_to_show = 10
# total_batch = int(data_set.train.num_examples/batch_size)
dropout = tf.placeholder(tf.float32)

batches_y, batches_x, test_batch_y, test_batch_x = get_data(data_path, batch_size)

# Create variables for inputs, outputs and predictions
x = tf.placeholder(tf.float32, [None, 1000])
y = tf.placeholder(tf.float32, [None, 5])
model_output = model.predict(x, dropout=0.3)

cost = tf.reduce_mean(tf.pow(y - model_output, 2))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(model_output), reduction_indices=[1]))
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(model_output,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


print (len(batches_x))
# Train
for epoch in range(training_epochs):
    for i in range(len(batches_x)):
        _, c = sess.run([train_step, cost], feed_dict={x: batches_x[i], y:batches_y[i], dropout:0.3})

    # Display logs per epoch step
    if epoch % display_step == 0:
        # print("Epoch:", '%04d' % (epoch+1))
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
        # print(sess.run(model_output, feed_dict={x: batches_x[0], y: batches_y[0]}))
        print(sess.run(accuracy, feed_dict={x: batches_x[0], y: batches_y[0]}))
        print(sess.run(accuracy, feed_dict={x: test_batch_x, y: test_batch_y}))


# Save
model.save(sess, 'example_3')
