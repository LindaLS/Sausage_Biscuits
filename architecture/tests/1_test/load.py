# Example implementing 5 layer encoder
# Original code taken from
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
# First train a model using train.py

from __future__ import division, print_function, absolute_import

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../..') # Add path to where TF_Model.py is, if not in the same dir
from TF_Model import *

# Create TF_Model, a wrapper for models created using tensorflow
# Note that the configuration file 'config.txt' must be present in the directory
model = TF_Model('model')

# Parameters
examples_to_show = 10

# Create variables for inputs, outputs and predictions
X = tf.placeholder("float", [None, 784])
y_true = X
y_pred = model.predict(X)

# Restore
sess = tf.Session()
model.restore(sess, 'example_2')

# Applying encode and decode over test set
encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
# Compare original images with their reconstructions
f, a = plt.subplots(2, examples_to_show, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
f.show()
plt.draw()
plt.waitforbuttonpress()
