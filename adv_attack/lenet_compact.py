# Run in python3
import keras
import pickle
import numpy as np
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Subtract
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import optimizers
from keras.models import load_model
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# load parameters
fr = open('pca_params_compact.pkl', 'rb')
pca_params = pickle.load(fr,encoding='latin1')
fr.close()

fr = open('llsr_weights_compact_v2.pkl', 'rb')
weights = pickle.load(fr,encoding='latin1')
fr.close()

fr = open('llsr_bias_compact_v2.pkl', 'rb')
biases = pickle.load(fr,encoding='latin1')
fr.close()

# read data
print (pca_params['Layer_0/kernel'].shape)
print (pca_params['Layer_1/kernel'].shape)

def bias_init_conv_1(shape, dtype=None):
#     print 'bias shape: {}'.format(shape)
    tmp_bias=(-1* np.matmul(pca_params['Layer_0/feature_expectation'],np.transpose(pca_params['Layer_0/kernel']))).reshape(shape)
    return tmp_bias

def weight_init_conv_1(shape, dtype=None):
    print('shape: ',shape)
    # print 'init conv1 function: {}'.format( pca_params['Layer_0/kernel'])
    weight=pca_params['Layer_0/kernel'].astype(np.float32)
    weight=np.moveaxis(weight, 1, 0)
    return weight.reshape(shape)


def weight_init_conv_2(shape, dtype=None):
    print('shape: ', shape)
    weight=pca_params['Layer_1/kernel'].astype(np.float32)
    weight=np.moveaxis(weight, 1,0)
    weight=weight.reshape(6,5,5, 16)
    weight=np.moveaxis(weight, 0, 2)
    return weight.reshape(shape)


def bias_init_conv_2(shape, dtype=None):
    # print('bias_init_conv_2',shape)
    weight=np.transpose(pca_params['Layer_1/kernel'].astype(np.float32))
    # print 'weight shape: {}'.format(weight.shape)
    tmp_bias = (-1 * np.matmul(pca_params['Layer_1/feature_expectation'].astype(np.float32), weight)).reshape(shape)
    # print 'conv2 tmp_bias: {}'.format(tmp_bias)

    bias=np.zeros(150)
    bias=bias+1/np.sqrt(150)*pca_params['Layer_1/bias'].astype(np.float32)

    bias1=np.matmul(bias, weight)
    # print 'conv2 bias1: {}'.format(bias1)

    bias2 = np.zeros(shape, dtype=np.float32)
    bias2[0] = -1
    bias2 = bias2 * pca_params['Layer_1/bias'].astype(np.float32)

    bias_final=tmp_bias + bias1+ bias2
    # print 'bias_final.shape: {}'.format(bias_final.shape)
    # print 'required shape: {}'.format(shape)
    return bias_final.reshape(shape)


def weight_init_fc_1(shape, dtype=None):
    return weights['0 LLSR weight'].reshape(shape)


def bias_init_fc_1(shape, dtype=None):
    return biases['0 LLSR bias'].reshape(shape)


def weight_init_fc_2(shape, dtype=None):
    return weights['1 LLSR weight'].reshape(shape)


def bias_init_fc_2(shape, dtype=None):
    return biases['1 LLSR bias'].reshape(shape)


def weight_init_fc_3(shape, dtype=None):
    return weights['2 LLSR weight'].reshape(shape)


def bias_init_fc_3(shape, dtype=None):
    return biases['2 LLSR bias'].reshape(shape)



(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)
train_images = train_images / 255.
test_images = test_images / 255.
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)
train_images=np.pad(train_images, ((0,0),(2,2),(2,2),(0,0)), mode='constant')
test_images=np.pad(test_images,  ((0,0),(2,2),(2,2),(0,0)), mode='constant')
print('Training image size:', train_images.shape)
print('Testing_image size:', test_images.shape)
test_images=np.float32(test_images)
test_images=test_images.reshape(-1,32,32,1)


from keras.layers import Lambda
from keras.layers import Reshape


lenet = Sequential()
lenet.add(Conv2D(6, activation=None, kernel_size=5, strides=1, padding='valid', data_format='channels_last',
                 kernel_initializer=weight_init_conv_1, input_shape=(32, 32, 1),  bias_initializer=bias_init_conv_1) ) # bias?
lenet.add(MaxPool2D(pool_size=2, strides=2))
lenet.add(Conv2D(16, activation=None, kernel_size=5, strides=1, padding='valid', data_format='channels_last',
                 kernel_initializer=weight_init_conv_2, bias_initializer=bias_init_conv_2))
lenet.add(MaxPool2D(pool_size=2, strides=2))
lenet.add(Flatten())
lenet.add(Dense(120, activation='relu', kernel_initializer=weight_init_fc_1, bias_initializer=bias_init_fc_1))
lenet.add(Dense(84, activation='relu', kernel_initializer=weight_init_fc_2, bias_initializer=bias_init_fc_2))
lenet.add(Dense(10, activation='softmax' , kernel_initializer=weight_init_fc_3, bias_initializer=bias_init_fc_3))
lenet.summary()

lenet.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

# output=lenet.layers[7].output
# print lenet.layers[7].__class__.__name__
# f=K.function([lenet.input],[output])
# y=f([test_images[:1]])[0]
# print y.shape, y


score, acc = lenet.evaluate(test_images, test_labels, 10, 1)
print ('Test accuracy: ', acc)

import tensorflow as tf
saver = tf.train.Saver()
sess = keras.backend.get_session()
save_path = saver.save(sess, "mnist_ff_model_v2/FF_init_model_v2.ckpt")
print('model saved!')







