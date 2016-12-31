# This example implements https://www.tensorflow.org/tutorials/mnist/beginners/
# using the configuration based infra
# First train a model using train.py

# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Import libraries
import tensorflow as tf
import sys
sys.path.insert(0, '../..') # Add path to where TF_Model.py is, if not in the same dir
from TF_Model import *

# Create TF_Model, a wrapper for models created using tensorflow
model = TF_Model('model')

# Create variables for inputs, outputs and predictions
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
model_output = model.predict(x)

# Restore
sess = tf.Session()
model.restore(sess, 'example_1')

# Evaluate accuracy
correct_prediction = tf.equal(tf.argmax(model_output,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

