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

epochs = 10
batch_size = 100
tf.random.set_seed(123)
# data preprocessing
cifar = datasets.cifar10
num_classes = 10
(train_images, train_labels), (test_images, test_labels) = cifar.load_data()
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)


def process_images(image):
    # Resize images from 32x32 to 256x256
    image = tf.image.resize(image, (256, 256))
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.image.per_image_standardization(image)
    return image



train_images = process_images(train_images)
test_images = process_images(test_images)



#load the model
base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
#base_model.summary()
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

start = time.time()
history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs)
end = time.time()
model.save('vgg16.h5')
start1 = time.time()
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
end1= time.time()
print(f"Test accuracy: {test_acc*100 :>0.1f}%")
print("The training time is: {}".format(end-start))
print("The test time is: {}".format(end1-start1))
