#!/usr/bin/env python3
"""
Created on Sun Mar 11 16:56:24 2018
@author: Ketul Patel
A second project to identify the hand written digits using simple Neural Network 
in Deep Learning . The purpose of this is to learn multiclass handwritten numeric digit classification.
"""
# Import libraries
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras import backend as K

import numpy as np
import cv2

# fix random seed for reproducibility
seed = 100
np.random.seed(seed)

K.set_image_dim_ordering('th')

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255


# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# Build the model
model = Sequential()

# Convulation 
model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# Convulation 
model.add(Conv2D(15, (3, 3), input_shape=(1, 28, 28), activation='relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#Dropout
model.add(Dropout(0.2))
# Flattering 
model.add(Flatten())
# Densing
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax', name='predict'))

#Comile the model with metric- Accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)

# Display result - Accuracy %
scores = model.evaluate(x_test, y_test, verbose=0)
print("\nacc: %.2f%%" % (scores[1]*100))

# Load Image
img_input = "number-four.png"
img_pred = cv2.imread(img_input, 0)

# Display input image
plt.imshow(img_pred, cmap='gray')

# Resizing and adjusing dimension
if img_pred.shape != [28,28]:
    img2 = cv2.resize(img_pred, (28, 28))
    img_pred = img2.reshape(28, 28, -1)
else:
    img_pred = img2.reshape(28, 28, -1)
    
# Adjust Rows and Column to 28,28 respectively
img_pred = img_pred.reshape(1, 1, 28, 28)

# Apply model and identify
pred = model.predict_classes(img_pred)

# Find probability 
pred_proba = model.predict_proba(img_pred)

pred_proba = "%.2f%%" % (pred_proba[0][pred]*100)

# Display predicted value
print(pred[0], "Number", pred_proba)

#######################################################################################
