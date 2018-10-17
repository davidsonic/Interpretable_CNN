
# coding: utf-8

# In[ ]:


from cleverhans.utils_mnist import data_mnist
x_train, y_train, x_test, y_test = data_mnist(train_start=0,
                                                  train_end=60000,
                                                  test_start=0,
                                                  test_end=10000)

x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')


# In[ ]:


print(x_test.shape)


# In[ ]:


# test ff_v2 on ff_v1
import numpy as np
import matplotlib.pyplot as plt
import pickle

data_dir='mnist_ff_model_v2/'
# FGSM data
with open(data_dir+'mnist_FGSM_data.pkl','rb') as fr:
    fgsm_data=pickle.load(fr)
with open(data_dir+'mnist_BIM_data.pkl','rb') as fr:
    bim_data=pickle.load(fr)
with open(data_dir+'mnist_DeepFool_data.pkl','rb') as fr:
    DF_data=pickle.load(fr)    


# In[ ]:


print(fgsm_data.shape)


# In[9]:


model_dir='mnist_ff_model'
filename='FF_init_model_v2.ckpt'
import os

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

import keras
from lenet_model import mnist_model
from cleverhans.utils_tf import train, model_eval


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=(None, 32, 32,
                                          1))
y = tf.placeholder(tf.float32, shape=(None, 10))

model = mnist_model(img_rows=32, img_cols=32,
                      channels=1, nb_filters=64,
                      nb_classes=10)
preds=model(x)

sess=tf.Session(config=config)
keras.backend.set_session(sess)

saver=tf.train.Saver()
saver.restore(sess, os.path.join(model_dir, filename))
eval_params = {'batch_size': 32}
acc = model_eval(sess, x, y, preds, fgsm_data, y_test, args=eval_params)
print('acc:', acc)
acc = model_eval(sess, x, y, preds, bim_data, y_test, args=eval_params)
print('acc:', acc)
acc = model_eval(sess, x, y, preds, DF_data, y_test, args=eval_params)
print('acc:', acc)


# In[15]:


model_dir='mnist_BP_model'
filename='mnist.ckpt'
import os

import tensorflow as tf

import keras
from lenet_model import mnist_model
from cleverhans.utils_tf import train, model_eval


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=(None, 32, 32,
                                          1))
y = tf.placeholder(tf.float32, shape=(None, 10))

model = mnist_model(img_rows=32, img_cols=32,
                      channels=1, nb_filters=64,
                      nb_classes=10)
preds=model(x)


sess=tf.Session()
keras.backend.set_session(sess)

saver=tf.train.Saver()
saver.restore(sess, os.path.join(model_dir, filename))
eval_params = {'batch_size': 32}
acc = model_eval(sess, x, y, preds, fgsm_data, y_test, args=eval_params)
print('acc:', acc)


# In[17]:


acc = model_eval(sess, x, y, preds, DF_data, y_test, args=eval_params)
print('acc:', acc)


# In[18]:


acc = model_eval(sess, x, y, preds, bim_data, y_test, args=eval_params)
print('acc:', acc)


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pickle
data_dir='cifar_ff_model_v2/'
# FGSM data
with open(data_dir+'cifar_FGSM_data.pkl','rb') as fr:
    fgsm_data=pickle.load(fr)
with open(data_dir+'cifar_BIM_data.pkl','rb') as fr:
    bim_data=pickle.load(fr)
with open(data_dir+'cifar_DeepFool_data.pkl','rb') as fr:
    DF_data=pickle.load(fr) 


# In[2]:


print(fgsm_data.shape)


# In[3]:


from keras.datasets import cifar10
import keras
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(y_test.shape)
y_test = keras.utils.to_categorical(y_test, 10)


# In[5]:


# test FF_v1 on FF_v2
model_dir='cifar_ff_model/'
filename='FF_init_model.ckpt'
import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

import keras
from cifar_ff_model import cifar_ff_model
from cleverhans.utils_tf import train, model_eval


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=(None, 32, 32,
                                          3))
y = tf.placeholder(tf.float32, shape=(None, 10))

# test ff_model
model = cifar_ff_model()
preds=model(x)


sess=tf.Session(config=config)
keras.backend.set_session(sess)

saver=tf.train.Saver()
saver.restore(sess, os.path.join(model_dir, filename))
eval_params = {'batch_size': 32}
acc = model_eval(sess, x, y, preds, fgsm_data, y_test, args=eval_params)
print('acc:', acc)
acc = model_eval(sess, x, y, preds, bim_data, y_test, args=eval_params)
print('acc:', acc)
acc = model_eval(sess, x, y, preds, DF_data, y_test, args=eval_params)
print('acc:', acc)


# In[12]:


model_dir='cifar_BP_model'
filename='cifar.ckpt'
import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

import keras
from lenet_model import cifar_model
from cleverhans.utils_tf import train, model_eval


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=(None, 32, 32,
                                          3))
y = tf.placeholder(tf.float32, shape=(None, 10))

model = cifar_model(img_rows=32, img_cols=32,
                      channels=3, nb_filters=64,
                      nb_classes=10)
preds=model(x)


sess=tf.Session(config=config)
keras.backend.set_session(sess)

saver=tf.train.Saver()
saver.restore(sess, os.path.join(model_dir, filename))
eval_params = {'batch_size': 32}
acc = model_eval(sess, x, y, preds, fgsm_data, y_test, args=eval_params)
print('acc:', acc)
acc = model_eval(sess, x, y, preds, DF_data, y_test, args=eval_params)
print('acc:', acc)
acc = model_eval(sess, x, y, preds, bim_data, y_test, args=eval_params)
print('acc:', acc)

