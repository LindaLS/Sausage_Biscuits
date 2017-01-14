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


action_map = {}
action_map[0] = [0,0,0,1]
action_map[1] = [0,0,1,0]
action_map[2] = [0,1,0,0]
action_map[3] = [1,0,0,0]

mat_contents_0 = sio.loadmat('/home/linda/school/capstone/data/jan12/Jan_12/Fred_index_jan12_0.mat')
mat_contents_1 = sio.loadmat('/home/linda/school/capstone/data/jan12/Jan_12/Fred_index_jan12_1.mat')
mat_contents_2 = sio.loadmat('/home/linda/school/capstone/data/jan12/Jan_12/Fred_index_jan12_2.mat')
mat_contents_3 = sio.loadmat('/home/linda/school/capstone/data/jan12/Jan_12/Fred_index_jan12_3.mat')

mat_contents_4 = sio.loadmat('/home/linda/school/capstone/data/jan12/Jan_12/Fred_pinky_jan12_0.mat')
mat_contents_5 = sio.loadmat('/home/linda/school/capstone/data/jan12/Jan_12/Fred_pinky_jan12_2.mat')

mat_contents_6 = sio.loadmat('/home/linda/school/capstone/data/jan12/Jan_12/Fred_thum_jan12_0.mat')
mat_contents_7 = sio.loadmat('/home/linda/school/capstone/data/jan12/Jan_12/Fred_thumb_jan12_1.mat')
mat_contents_8 = sio.loadmat('/home/linda/school/capstone/data/jan12/Jan_12/Fred_thumb_jan12_3.mat')
mat_contents_9 = sio.loadmat('/home/linda/school/capstone/data/jan12/Jan_12/Fred_tumb_jan12_2.mat')

mat_contents_test_0 = sio.loadmat('/home/linda/school/capstone/data/jan12/Jan_12/Fred_index_jan12_4.mat')
mat_contents_test_1 = sio.loadmat('/home/linda/school/capstone/data/jan12/Jan_12/Fred_tumb_jan12_4.mat')
mat_contents_test_2 = sio.loadmat('/home/linda/school/capstone/data/jan12/Jan_12/Fren_pinky_jan12_1.mat')

data_0 = mat_contents_0['EMGdata']
data_1 = mat_contents_1['EMGdata']
data_2 = mat_contents_2['EMGdata']
data_3 = mat_contents_3['EMGdata']
data_4 = mat_contents_4['EMGdata']
data_5 = mat_contents_5['EMGdata']
data_6 = mat_contents_6['EMGdata']
data_7 = mat_contents_7['EMGdata']
data_8 = mat_contents_8['EMGdata']
data_9 = mat_contents_9['EMGdata']

data_test0 = mat_contents_test_0['EMGdata']
data_test1 = mat_contents_test_1['EMGdata']
data_test2 = mat_contents_test_2['EMGdata']

sets_y = [[] for i in range(10)]
sets_x = [[] for i in range(10)]
sets_y[0], sets_x[0] = get_batch_from_raw_data_new_format(data_0, action_map, [])
sets_y[1], sets_x[1] = get_batch_from_raw_data_new_format(data_1, action_map, [])
sets_y[2], sets_x[2] = get_batch_from_raw_data_new_format(data_2, action_map, [])
sets_y[3], sets_x[3] = get_batch_from_raw_data_new_format(data_3, action_map, [])
sets_y[4], sets_x[4] = get_batch_from_raw_data_new_format(data_4, action_map, [])
sets_y[5], sets_x[5] = get_batch_from_raw_data_new_format(data_5, action_map, [])
sets_y[6], sets_x[6] = get_batch_from_raw_data_new_format(data_6, action_map, [])
sets_y[7], sets_x[7] = get_batch_from_raw_data_new_format(data_7, action_map, [])
sets_y[8], sets_x[8] = get_batch_from_raw_data_new_format(data_8, action_map, [])
sets_y[9], sets_x[9] = get_batch_from_raw_data_new_format(data_9, action_map, [])

batch_y_test0, batch_x_test0 = get_batch_from_raw_data_new_format(data_test0, action_map, [])
batch_y_test1, batch_x_test1 = get_batch_from_raw_data_new_format(data_test1, action_map, [])
batch_y_test2, batch_x_test2 = get_batch_from_raw_data_new_format(data_test2, action_map, [])


print("done reading data")
# Create TF_Model, a wrapper for models created using tensorflow
# Note that the configuration file 'config.txt' must be present in the directory
model = TF_Model('model')


# Parameters
learning_rate = 0.01
training_epochs = 200
batch_size = 100
display_step = 1
examples_to_show = 10
# total_batch = int(data_set.train.num_examples/batch_size)
dropout = tf.placeholder(tf.float32)

batches_x, batches_y = create_batches(sets_x, sets_y, batch_size)

# Create variables for inputs, outputs and predictions
x = tf.placeholder(tf.float32, [None, 1000])
y = tf.placeholder(tf.float32, [None, 4])
model_output = model.predict(x)

cost = tf.reduce_mean(tf.pow(y - model_output, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(model_output), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(model_output,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)




# Train
for epoch in range(training_epochs):
    for i in range(len(batches_x)):
        sess.run(train_step, feed_dict={x: batches_x[i], y: batches_y[i]})
        # _, c = sess.run([optimizer, cost], feed_dict={x: batches_x[i], y:batches_y[i], dropout:0.75})

    # Display logs per epoch step
    print("Epoch:", '%04d' % (epoch+1))
    print(sess.run(accuracy, feed_dict={x: batch_x_test0, y: batch_y_test0}))
    print (model_output.eval(feed_dict={x: batch_x_test0}, session=sess))
    print(sess.run(accuracy, feed_dict={x: batch_x_test1, y: batch_y_test1}))
    print(sess.run(accuracy, feed_dict={x: batch_x_test2, y: batch_y_test2}))

print("===final===")
print(sess.run(accuracy, feed_dict={x: batch_x_test0, y: batch_y_test0}))
print(sess.run(accuracy, feed_dict={x: batch_x_test1, y: batch_y_test1}))
print(sess.run(accuracy, feed_dict={x: batch_x_test2, y: batch_y_test2}))

# Save
model.save(sess, 'example_3')
