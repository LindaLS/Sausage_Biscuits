from TF_Model import *
import tensorflow as tf
import numpy as np
from scipy import stats
import os
import math

def model_predictions(model_directory_list, x):
    predictions = []
    models = []
    architectures = []

    # For each model
    for model_dir in model_directory_list:
        #Create model
        tf.reset_default_graph()
        model = TF_Model(model_dir)
                
        X = tf.placeholder(tf.float32, [None, x.shape[1]])

        # For each architecture
        for file in os.listdir(model_dir):
            base_name = os.path.basename(file)
            architecture_name, file_extension = os.path.splitext(base_name)
            if file_extension == '.ckpt':

                _ = model.predict(X)

                # Restore weights
                sess = tf.Session()
                model.restore(sess, architecture_name)
                
                # Get preditcion using restored model and restored weights
                model_prediction = sess.run(_, feed_dict={X:x})

                sess.close()

                predictions.append(model_prediction)
                models.append(model_dir)
                architectures.append(architecture_name)
    return predictions, models, architectures

def get_voted_prediction(model_predictions):
    predicted_classes = []
    # Each architecture
    for i in range(len(model_predictions)):
        predicted_class = np.argmax(model_predictions[i],1)
        predicted_classes.append(predicted_class)
    vote, _ = stats.mode(predicted_classes)
    return vote

def get_average_prediction(model_predictions):
    average = np.empty_like(model_predictions[0])
    for i in range(len(model_predictions)):
        average = average + model_predictions[i]
    average = average / len(model_predictions)
    return average

def get_weighted_prediction(model_predictions, weights):
    weighted_prediction = np.empty_like(model_predictions[0])
    for i in range(len(model_predictions)):
        current = model_predictions[i] * weights[i]
        weighted_prediction = weighted_prediction + current
    return weighted_prediction
