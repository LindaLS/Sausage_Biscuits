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

# 0001 no action
# 0010 index up
# 0100 thumb up
# 1000 pinky up

# index up data
mat_contents_i0 = sio.loadmat('/home/linda/school/capstone/data/set2/Fred_index0.mat')
mat_contents_i1 = sio.loadmat('/home/linda/school/capstone/data/set2/Fred_index1.mat')
mat_contents_i2 = sio.loadmat('/home/linda/school/capstone/data/set2/Fred_index2.mat')
mat_contents_i3 = sio.loadmat('/home/linda/school/capstone/data/set2/Fred_index3.mat')

action_map = {}
action_map[0] = [0,0,0,1]
action_map[1] = [0,0,1,0]

data_i0 = mat_contents_i0['EMGdata']
data_i1 = mat_contents_i1['EMGdata']
data_i2 = mat_contents_i2['EMGdata']
data_i3 = mat_contents_i3['EMGdata']

batch_y_i0, batch_x_i0 = get_batch_from_raw_data(data_i0, action_map, [0])
batch_y_i1, batch_x_i1 = get_batch_from_raw_data(data_i1, action_map, [0])
batch_y_i2, batch_x_i2 = get_batch_from_raw_data(data_i2, action_map, [0])
batch_y_i3, batch_x_i3 = get_batch_from_raw_data(data_i3, action_map, [0])

# thumb up
mat_contents_t0 = sio.loadmat('/home/linda/school/capstone/data/set2/Fred_thumb4.mat')
mat_contents_t1 = sio.loadmat('/home/linda/school/capstone/data/set2/Fred_thumb5.mat')

action_map[0] = [0,0,0,1]
action_map[1] = [0,1,0,0]

data_t0 = mat_contents_t0['EMGdata']
data_t1 = mat_contents_t1['EMGdata']

batch_y_t0, batch_x_t0 = get_batch_from_raw_data(data_t0, action_map, [0])
batch_y_t1, batch_x_t1 = get_batch_from_raw_data(data_t1, action_map, [0])

# pinky up
mat_contents_p0 = sio.loadmat('/home/linda/school/capstone/data/set2/Fred_pinky4.mat')
mat_contents_p1 = sio.loadmat('/home/linda/school/capstone/data/set2/Fred_pinky5.mat')
mat_contents_p2 = sio.loadmat('/home/linda/school/capstone/data/set2/Fred_pinky6.mat')

action_map[0] = [0,0,0,1]
action_map[1] = [1,0,0,0]

data_p0 = mat_contents_p0['EMGdata']
data_p1 = mat_contents_p1['EMGdata']
data_p2 = mat_contents_p2['EMGdata']

batch_y_p0, batch_x_p0 = get_batch_from_raw_data(data_p0, action_map, [0])
batch_y_p1, batch_x_p1 = get_batch_from_raw_data(data_p1, action_map, [0])
batch_y_p2, batch_x_p2 = get_batch_from_raw_data(data_p2, action_map, [0])

# test set
mat_contents_test0 = sio.loadmat('/home/linda/school/capstone/data/set2/Fred_thumb_index_pinky0.mat')
mat_contents_test1 = sio.loadmat('/home/linda/school/capstone/data/set2/Fred_thumb_index_pinky1.mat')
action_map[0] = [0,0,0,1]
action_map[1] = [0,1,0,0]
action_map[2] = [0,0,0,1]
action_map[3] = [0,0,1,0]
action_map[4] = [0,0,0,1]
action_map[5] = [1,0,0,0]

data_test0 = mat_contents_test0['EMGdata']
data_test1 = mat_contents_test1['EMGdata']
batch_y_test0, batch_x_test0 = get_batch_from_raw_data(data_test0, action_map, [0])
batch_y_test1, batch_x_test1 = get_batch_from_raw_data(data_test1, action_map, [0])

# Create TF_Model, a wrapper for models created using tensorflow
# Note that the configuration file 'config.txt' must be present in the directory
model = TF_Model('model')


# Parameters
learning_rate = 0.01
training_epochs = 2000
batch_size = 256
display_step = 1
examples_to_show = 10
# total_batch = int(data_set.train.num_examples/batch_size)
dropout = tf.placeholder(tf.float32)

# Create variables for inputs, outputs and predictions
x = tf.placeholder(tf.float32, [None, 1000])
y = tf.placeholder(tf.float32, [None, 4])
y_true = y
y_pred = model.predict(x)

# Cost function
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

model_output = model.predict(x)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(model_output), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(model_output,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Train
for epoch in range(training_epochs):
    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x_i0, y: batch_y_i0})
    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x_i1, y: batch_y_i1})
    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x_i2, y: batch_y_i2})
    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x_i3, y: batch_y_i3})
    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x_t0, y: batch_y_t0})
    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x_t1, y: batch_y_t1})
    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x_p0, y: batch_y_p0})
    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x_p1, y: batch_y_p1})
    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x_p2, y: batch_y_p2})

    # Display logs per epoch step
    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
    print(sess.run(cost, feed_dict={x: batch_x_test0, y: batch_y_test0}))
    print(sess.run(cost, feed_dict={x: batch_x_test1, y: batch_y_test1}))
print("===final===")
print(sess.run(cost, feed_dict={x: batch_x_test0, y: batch_y_test0}))
print(sess.run(cost, feed_dict={x: batch_x_test1, y: batch_y_test1}))
# Save
model.save(sess, 'example_3')
