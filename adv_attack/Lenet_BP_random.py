import keras
from keras.datasets import mnist
import pickle
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras import optimizers
from keras.models import load_model
import matplotlib.pyplot as plt

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from keras import backend as K

K.tensorflow_backend._get_available_gpus()

# load parameters
fr = open('pca_params.pkl', 'rb')
pca_params = pickle.load(fr)
fr.close()

fr = open('llsr_weights.pkl', 'rb')
weights = pickle.load(fr)
fr.close()

fr = open('llsr_bias.pkl', 'rb')
biases = pickle.load(fr)
fr.close()

# read data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)
train_images = train_images / 255.
test_images = test_images / 255.
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)
print('Training image size:', train_images.shape)
print('Testing_image size:', test_images.shape)


def PlotHistory(train_value, test_value, value_is_loss_or_acc):
    f, ax = plt.subplots()
    ax.plot([None] + train_value, 'o-', linewidth=0.8, markersize=3)
    ax.plot([None] + test_value, 'x-', linewidth=0.8, markersize=3)
    # Plot legend and use the best location automatically: loc = 0.
    ax.legend(['Train ' + value_is_loss_or_acc, 'Validation ' + value_is_loss_or_acc], loc=0)
    ax.set_title('Training/Validation ' + value_is_loss_or_acc + ' per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(value_is_loss_or_acc)
    plt.show()
    plt.savefig('History.png')


def main():

    lenet=Sequential()
    lenet.add(Conv2D(6,kernel_size=5,strides=1,padding='same',kernel_initializer='random_normal',data_format='channels_last', input_shape=(28, 28, 1),activation = 'relu'))
    lenet.add(MaxPool2D(pool_size=2,strides=2))
    lenet.add(Conv2D(16,kernel_size=5,strides=1,padding='valid',kernel_initializer='random_normal',activation = 'relu'))
    lenet.add(MaxPool2D(pool_size=2,strides=2))
    lenet.add(Flatten())
    lenet.add(Dense(120,activation='relu',kernel_initializer='random_normal'))
    lenet.add(Dense(84,activation='relu',kernel_initializer='random_normal'))
    lenet.add(Dense(10,activation='softmax',kernel_initializer='random_normal'))
    lenet.summary()

    lenet.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    # score, acc = lenet.evaluate(train_images, train_labels, 10, 1)
    # print 'Train accuracy: ', acc

    score, acc = lenet.evaluate(test_images, test_labels, 10, 1)
    print 'Test accuracy: ', acc


    training_history=lenet.fit(train_images,train_labels,batch_size=128 ,epochs=30,
                               validation_data=[train_images,train_labels])
    #
    # score, acc = lenet.evaluate(train_images, train_labels, 10, 1)
    # print 'Train accuracy: ', acc
    #
    # score, acc = lenet.evaluate(test_images, test_labels, 10, 1)
    # print 'Test accuracy: ', acc
    #
    # PlotHistory(training_history.history['acc'], training_history.history['val_acc'], 'Accuracy')
    # pickle.dump(training_history.history, open('training_history_rn.pkl','wb'))
    # lenet.save('lenet_rn.h5')


if __name__ == '__main__':
    main()
