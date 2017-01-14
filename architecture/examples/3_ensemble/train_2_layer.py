# This example implements https://www.tensorflow.org/tutorials/mnist/beginners/
# using the configuration based infra
# The model trained here is restored in load.py

# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Import libraries
import tensorflow as tf
import sys
sys.path.insert(0, '../..') # Add path to where TF_Model.py is, if not in the same dir
from TF_Model import *

# Create TF_Model, a wrapper for models created using tensorflow
# Note that the configuration file 'config.txt' must be present in the directory
model = TF_Model('2_layer_model')

# Create variables for inputs, outputs and predictions
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
model_output = model.predict(x)

for arch_id in range(0,5):
    # Start session
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    arch_name = 'arch_' + str(arch_id+1)

    # Train
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(model_output), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(model_output,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for i in range(2000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

    # Evaluate accuracy
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print(arch_name + ': ' + str(acc))

    # Save
    model.save(sess, arch_name)
