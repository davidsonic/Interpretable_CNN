'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from keras import backend as K
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

batch_size = 500
num_classes = 10
epochs = 50
data_augmentation = False
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'final_BP_random_70.h5'
resume_weights = os.path.join(save_dir, model_name)


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# model construction
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# model.add(Dropout(0.5))  # 1600->400
model.add(Dense(200))
model.add(Activation('relu'))
# model.add(Dropout(0.5))  # 510->120
model.add(Dense(100))
model.add(Activation('relu'))
# model.add(Dropout(0.5))  # 120->80
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
if os.path.isfile(resume_weights):
    print("Resumed model's weights from {}".format(resume_weights))
    # load weights
    model.load_weights(resume_weights)

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])



checkpoint = ModelCheckpoint(resume_weights,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_acc', value=0.613, verbose=0, test_images=None, test_labels=None):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        # self.test_images=test_images
        # self.test_labels=test_labels

    # def on_batch_end(self, batch, logs={}):
    #     current = logs.get(self.monitor)
    #     score,acc=self.model.evaluate(test_images, test_labels, 100, 1)
    #     if acc>self.value:
    #         if self.verbose > 0:
    #             print("Current val acc is: ", current)
    #         self.model.stop_training = True
    #         self.model.save('lenet_weight1.h5', overwrite=True)

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        # score, acc = self.model.evaluate(self.test_images, self.test_labels, 500, 1)
        if current > self.value:
            if self.verbose > 0:
                print("Current val acc is: ", current)
            self.model.stop_training = True
            self.model.save('lenet_weight2.h5', overwrite=True)


callbacks_list = [

    keras.callbacks.TensorBoard(
        log_dir='lenet_weight_log',
        histogram_freq=1,
    ),
    # EarlyStoppingByLossVal(
    #     monitor='val_acc',
    #     verbose=1,
    #     test_images=x_test,
    #     test_labels=y_test,
    # ),
    checkpoint,
]

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks_list)

# Save model and weights
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#
# feat_train={}
# output=model.layers[5]
# f=K.function([model.input],[output])
# feat_train['1600']=f([x_train])
#
# output=model.layers[9]
# f=K.function([model.input],[output])
# feat_train['200']=f([x_train])
#
# output=model.layers[11]
# f=K.function([model.input],[output])
# feat_train['100']=f([x_train])
#
# output=model.layers[13]
# f=K.function([model.input],[output])
# feat_train['10']=f([x_train])
#
# with open('cifar_train_feat.pkl','wb') as fw:
#     pickle.dump(feat_train, fw)
#     print('cifar_train_feat stored!')
