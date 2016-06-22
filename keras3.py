import gym
import numpy as np
from skimage import color
from skimage.transform import resize
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import matplotlib.pyplot as plt	

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 1, 28, 28)
X_test = X_test.reshape(10000, 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


nb_classes = 10
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

nb_filters = 32;
nb_conv = 3;
nb_pool =2;

model = Sequential()
model.add(Convolution2D(32, 8, 8, 		border_mode='valid',input_shape=(1,84,84)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(64, 4, 4, border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(6))
model.add(Activation('softmax')) #aqui o certo é um htan

sgd = SGD()
model.compile(loss = 'categorical_crossentropy', optimizer = sgd)

env = gym.make('Pong-v0')
obs = env.reset()
xyz = color.rgb2xyz(obs)
y = xyz[:,:,1]	
small = resize(y,(84,84))
in_obs = small.reshape(1, 1, 84, 84)

pred = model.predict(in_obs,batch_size=1,verbose=0)

print(pred)

#h = model.fit(X_train, Y_train, batch_size = 128, nb_epoch=3, validation_data =(X_test, Y_test), verbose=1) 


