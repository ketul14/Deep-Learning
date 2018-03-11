#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:00:34 2018

@author: Ketul
A first project to classify Dog and Cat using CNN:
Thanks to Venkatesh Tata, Author of an Article 
"Simple Image Classification using Convolutional Neural Network — Deep Learning in python."
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

#Create a class Classifier from Sequential
classifier = Sequential()

#Apply Convulation using Keras
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Apply Max Pooling to the results
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Convert results to Flattering - Continous Vector
classifier.add(Flatten())

#Apply dense to it - It is a kind of hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))

#Initialize output layer - Sigmoid as binary classifications for this example
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Apply Optimazation and Loss along with performance measurement metrics
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Most Important - Fitting the CNN model to Images
# Data folder is the target folder in which train and test folders are created.
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('Data/train',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory('Data/test',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

# Fit the model
classifier.fit_generator(training_set,
steps_per_epoch = 8000,
epochs = 20,
validation_data = test_set,
validation_steps = 2000)


### Predict a Dot or Cat from the input image using developed model
import numpy as np
from keras.preprocessing import image

in_image = 'Cat.jpg'
test_image = image.load_img(in_image, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices


# Display Result - pls check indentation - It's 4 spaces not a tab
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print (prediction)
