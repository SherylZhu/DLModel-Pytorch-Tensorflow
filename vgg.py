import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from tensorflow.keras.layers import Cropping2D, Reshape, Input, Lambda, Conv2D, Dense, MaxPool2D, Flatten, Dropout, AveragePooling2D
from tensorflow.keras import models
import os
from tensorflow.keras import Sequential
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
from tensorflow.keras import applications

num_classes = 10

base_model = applications.VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
base_model.summary()
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9), loss='categorical_crossentropy',
              metrics=["accuracy"])

model.summary()