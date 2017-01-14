# This example implements https://www.tensorflow.org/tutorials/mnist/beginners/
# using the configuration based infra
# First train a model using train.py

# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Import libraries
import numpy as np
import sys
sys.path.insert(0, '../..') # Add path to where TF_Model.py is, if not in the same dir
from Ensemble import *
from scipy import stats

# Create a list of model directories for models you want to use
model_list = []
model_list.append('simple_model')
model_list.append('2_layer_model')
model_list.append('3_layer_model')

# Get predictions
# Note that any tensors created before ensemble.model_predictions() will be reset!!!
x = mnist.test.images
y = mnist.test.labels
x_validation = x[5000:]
y_validation = y[5000:]
x_test = x[:5000]
y_test = y[:5000]
model_outputs_validation, models, architectures = model_predictions(model_list, x_validation)
model_outputs_test, models, architectures = model_predictions(model_list, x_test)

# Evaluate accuracy
# Each architecture
for i in range(len(model_outputs_test)):
    correct_prediction = np.equal(np.argmax(model_outputs_test[i],1), np.argmax(y_test,1)).astype(float)
    accuracy = np.average(correct_prediction)
    print(models[i] + ' ' + architectures[i] + ' ' + str(accuracy*100) + '%')

# Collective voting
vote = get_voted_prediction(model_outputs_test)
correct_prediction = np.equal(vote, np.argmax(y_test,1)).astype(float)
accuracy = np.average(correct_prediction)
print( 'Voting: ' + str(accuracy*100) + '%')

# Average output
average = get_average_prediction(model_outputs_test)
correct_prediction = np.equal(np.argmax(average,1), np.argmax(y_test,1)).astype(float)
accuracy = np.average(correct_prediction)
print( 'Average: ' + str(accuracy*100) + '%')

# Weighted average
# Generate weights based on validation set performance
weights = []
for i in range(len(model_outputs_validation)):
    correct_prediction = np.equal(np.argmax(model_outputs_validation[i],1), np.argmax(y_validation,1)).astype(float)
    accuracy = np.average(correct_prediction)
    weights.append(np.log(accuracy / (1-accuracy)))
weights = weights / np.sum(weights)

weighted_prediction = get_weighted_prediction(model_outputs_test, weights)
correct_prediction = np.equal(np.argmax(weighted_prediction,1), np.argmax(y_test,1)).astype(float)
accuracy = np.average(correct_prediction)
print( 'Weighted: ' + str(accuracy*100) + '%')
