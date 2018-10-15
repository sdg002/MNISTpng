# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(2016)

import os
import glob
import cv2
import math
import pickle
import datetime
#import pandas as pd
from keras.optimizers import SGD

#from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
import time
from MnistImageLoader import load_images,ReShapeData


#
#The following variables should be set to the folder where MNIST images have been extracted 
#
cat_dog_train_path_50="C:\\Users\\saurabhd\\MyTrials\\MachineLearnings\\MNIST101\\mnist_png_50\\*\\*.png"
cat_dog_train_path_full="C:\\Users\\saurabhd\\MyTrials\\MachineLearnings\\MNIST101\\mnist_png\\training\\*\\*.png"
cat_dog_test_path="C:\\Users\\saurabhd\\MyTrials\\MachineLearnings\\MNIST101\\mnist_png\\testing\\*\\*.png"
nb_classes = 10 #we have these many digits in our training
#
#Load training images
#
print("Loading training images")
(train_data, train_target)=load_images(cat_dog_train_path_full)
(train_data1,train_target1)=ReShapeData(train_data,train_target,nb_classes)
print('Shape:', train_data1.shape)
print(train_data1.shape[0], ' train images were loaded')
#
#Load test images
#
print("Loading testing images")
(test_data, test_target)=load_images(cat_dog_test_path)
(test_data1,test_target1)=ReShapeData(test_data,test_target,nb_classes)
print('Shape:', test_data1.shape)
print(test_data1.shape[0], ' test images were loaded')
print("Load complete")
# 
# Create a sequential model
#
model = Sequential()
# Add the first convolution layer
model.add(Convolution2D(
    name="conv1",
    filters = 20,
    kernel_size = (5, 5),
    padding = "same",
    input_shape = (28, 28, 1)))
# Add a ReLU activation function
model.add(Activation(
    activation = "relu"))
# Add a pooling layer
model.add(MaxPooling2D(
    name="maxpool1",
    pool_size = (2, 2),
    strides =  (2, 2)))
# Add the second convolution layer
model.add(Convolution2D(
    name="conv2",
    filters = 50,
    kernel_size = (5, 5),
    padding = "same"))
# Add a ReLU activation function
model.add(Activation(
    activation = "relu"))
# Add a second pooling layer
model.add(MaxPooling2D(
    name="maxpool2",
    pool_size = (2, 2),
    strides = (2, 2)))
# Flatten the network
model.add(Flatten())
# Add a fully-connected hidden layer
model.add(Dense(500))
# Add a ReLU activation function
model.add(Activation(activation = "relu"))
# Add a fully-connected output layer - the output layer nodes should match the count of image classes
model.add(Dense(nb_classes,name="outputlayer")) 
# Add a softmax activation function
model.add(Activation("softmax"))
#
#Display Summary
#
model.summary()

# Compile the network
model.compile(
    loss = "categorical_crossentropy", 
    optimizer = SGD(lr = 0.01),
    metrics = ["accuracy"])
print("Compilation complete");
print("Train begin");
# Train the model 

start = time.time()
model.fit(
    train_data1, 
    train_target1, 
    batch_size = 128, 
    epochs = 20,
	  verbose = 1)
print("Train complete");
#
#Test the model
#
print("Testing on test data")
(loss, accuracy) = model.evaluate(
    test_data1, 
    test_target1,
    batch_size = 128, 
    verbose = 1)

# Print the model's accuracy
print(accuracy)
#
#Save the model
#
model_json = model.to_json()
filenameModel="TrainedMnistModel.json"
with open(filenameModel,"w") as modelf:
	modelf.write(model_json)
print("Model written to file:" + filenameModel);

# serialize weights to HDF5
filenameWeights="TrainedMnistModelWts.h5"
model.save_weights(filenameWeights)
print("Weights were saved to file:" + filenameWeights)

