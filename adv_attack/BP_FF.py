# Run in python3
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Lambda
import os
import pickle
import numpy as np
from keras import backend as K


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

# load parameters
# data_dir='final_ff_weight/'
data_dir=''

# with open(data_dir+'pca_params.pkl','rb') as fr:
with open(data_dir+'pca_params_v2.pkl','rb') as fr:
    pca_params=pickle.load(fr, encoding='latin1')
# fr=open(data_dir+'llsr_weights.pkl','rb')
fr=open(data_dir+'llsr_weights_v2.pkl','rb')
weights=pickle.load(fr, encoding='latin1')
fr.close()
# fr=open(data_dir+'llsr_bias.pkl','rb')
fr=open(data_dir+'llsr_bias_v2.pkl','rb')
biases=pickle.load(fr,encoding='latin1')
fr.close()

with open('std_var.pkl','rb') as fr:
    std_var=pickle.load(fr)

def weight_init_conv_1(shape, dtype=None):
#     print('shape: ', shape)
    weight=pca_params['Layer_0/kernel'].astype(np.float32)
    weight=np.moveaxis(weight, 1,0)
    return weight.reshape(shape)


def weight_init_conv_2(shape, dtype=None):
    weight=pca_params['Layer_1/kernel'].astype(np.float32)
    weight=np.moveaxis(weight, 1,0)
    return weight.reshape(shape)


def bias_init_conv_2(shape, dtype=None):
    weight=np.transpose(pca_params['Layer_1/kernel'].astype(np.float32))
    bias=np.zeros(800)
    bias=bias+1/np.sqrt(800)*pca_params['Layer_1/bias'].astype(np.float32)

    bias1=np.matmul(bias, weight)

    bias2 = np.zeros(shape, dtype=np.float32)
    bias2[0] = -1
    bias2 = bias2 * pca_params['Layer_1/bias'].astype(np.float32)

    bias_final= bias1+ bias2
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


batch_size = 500
num_classes = 10
epochs=10

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
print(x_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# model construction
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation=None, data_format='channels_last',  input_shape=x_train.shape[1:], kernel_initializer=weight_init_conv_1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation=None, data_format='channels_last',  kernel_initializer=weight_init_conv_2, bias_initializer=bias_init_conv_2 ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Lambda(lambda x: x/std_var ) )
model.add(Activation('relu'))
model.add(Dense(200, kernel_initializer=weight_init_fc_1, bias_initializer=bias_init_fc_1))
model.add(Activation('relu'))
model.add(Dense(100, kernel_initializer=weight_init_fc_2, bias_initializer=bias_init_fc_2))
model.add(Activation('relu'))
model.add(Dense(num_classes,  kernel_initializer=weight_init_fc_3, bias_initializer=bias_init_fc_3))
model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.001),
              metrics=['accuracy'])

# scores=model.evaluate(x_train, y_train, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Add ops to save and restore all the variables.
import tensorflow as tf
saver = tf.train.Saver()
sess = keras.backend.get_session()
save_path = saver.save(sess, "FF_init_model/FF_init_model_v2.ckpt")
print('model saved!')


# batch_size = 200
# epochs=5
# training_history=model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_data=(x_test, y_test),
#           shuffle=True)

# Save model and weights
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)

# Score trained model.
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
