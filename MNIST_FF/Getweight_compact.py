# running order: GetKernel.py -> pca_params for each layer
#  GetFeature.py -> data_samples * kernels = Features
# Getweight.py ->
import data
import pickle
import numpy as np
import sklearn
import cv2
import keras
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras import backend as K

K.tensorflow_backend._get_available_gpus()
import random
random.seed(9001)

def main():
    # load data
    fr = open('pca_params_compact.pkl', 'rb')
    pca_params = pickle.load(fr)
    fr.close()

    # read data
    train_images, train_labels, test_images, test_labels, class_list = data.import_data("0-9")
    print('Training image size:', train_images.shape)
    print('Testing_image size:', test_images.shape)

    # load feature
    fr = open('feat_compact.pkl', 'rb')
    feat = pickle.load(fr)
    fr.close()
    feature = feat['feature']
    print 'feature type: ', feature.dtype
    print("S4 shape:", feature.shape)
    print('--------Finish Feature Extraction subnet--------')

    feature=feature.reshape(60000, 16, 5, 5)
    feature=np.moveaxis(feature, 1, 3)
    feature=feature.reshape(-1, 5*5*16)

    # feature normalization
    # std_var = (np.std(feature, axis=0)).reshape(1, -1)
    # feature = feature / std_var

    num_clusters = [120, 84, 10]
    use_classes = 10
    weights = {}
    bias = {}
    for k in range(len(num_clusters)):
        if k != len(num_clusters) - 1:
            # Kmeans
            kmeans = KMeans(n_clusters=num_clusters[k]).fit(feature)
            pred_labels = kmeans.labels_
            num_clas = np.zeros((num_clusters[k], use_classes), dtype=np.float32)
            for i in range(num_clusters[k]):
                for t in range(use_classes):
                    for j in range(feature.shape[0]):
                        if pred_labels[j] == i and train_labels[j] == t:
                            num_clas[i, t] += 1
            acc_train = np.sum(np.amax(num_clas, axis=1)) / feature.shape[0]
            print(k, ' layer Kmean (just ref) training acc is {}'.format(acc_train))

            # Compute centroids
            clus_labels = np.argmax(num_clas, axis=1)
            centroid = np.zeros((num_clusters[k], feature.shape[1]), dtype=np.float32)
            for i in range(num_clusters[k]):
                t = 0
                for j in range(feature.shape[0]):
                    if pred_labels[j] == i and clus_labels[i] == train_labels[j]:
                        if t == 0:
                            feature_test = feature[j].reshape(1, -1)
                        else:
                            feature_test = np.concatenate((feature_test, feature[j].reshape(1, -1)), axis=0)
                        t += 1
                centroid[i] = np.mean(feature_test, axis=0, keepdims=True)

            # Compute one hot vector
            t = 0
            labels = np.zeros((feature.shape[0], num_clusters[k]), dtype=np.float32)
            for i in range(feature.shape[0]):
                if clus_labels[pred_labels[i]] == train_labels[i]:
                    labels[i, pred_labels[i]] = 1
                else:
                    distance_assigned = euclidean_distances(feature[i].reshape(1, -1),
                                                            centroid[pred_labels[i]].reshape(1, -1))
                    cluster_special = [j for j in range(num_clusters[k]) if clus_labels[j] == train_labels[i]]
                    distance = np.zeros(len(cluster_special))
                    for j in range(len(cluster_special)):
                        distance[j] = euclidean_distances(feature[i].reshape(1, -1),
                                                          centroid[cluster_special[j]].reshape(1, -1))
                    labels[i, cluster_special[np.argmin(distance)]] = 1

            # least square regression
            A = np.ones((feature.shape[0], 1), dtype=np.float32)
            feature = np.concatenate((A, feature), axis=1)
            weight = np.matmul(LA.pinv(feature), labels)
            feature = np.matmul(feature, weight)
            print 'weight {}  dtype: {} '.format(i, weight.dtype)
            print 'weights save....'
            weights['%d LLSR weight' % k] = weight[1:weight.shape[0]]
            print 'weights saved!'
            bias['%d LLSR bias' % k] = weight[0].reshape(1, -1)
            print(k, ' layer LSR weight shape:', weight.shape)
            print(k, ' layer LSR output shape:', feature.shape)

            pred_labels = np.argmax(feature, axis=1)
            num_clas = np.zeros((num_clusters[k], use_classes), dtype=np.float32)
            for i in range(num_clusters[k]):
                for t in range(use_classes):
                    for j in range(feature.shape[0]):
                        if pred_labels[j] == i and train_labels[j] == t:
                            num_clas[i, t] += 1
            acc_train = np.sum(np.amax(num_clas, axis=1)) / feature.shape[0]
            print(k, ' layer LSR training acc is {}'.format(acc_train))

            # Relu
            for i in range(feature.shape[0]):
                for j in range(feature.shape[1]):
                    if feature[i, j] < 0:
                        feature[i, j] = 0

                        # # Double relu
                        # for i in range(feature.shape[0]):
                        # 	for j in range(feature.shape[1]):
                        # 		if feature[i,j]<0:
                        # 			feature[i,j]=0
                        # 		elif feature[i,j]>1:
                        # 			feature[i,j]=1
        else:
            # least square regression
            labels = keras.utils.to_categorical(train_labels, 10)
            A = np.ones((feature.shape[0], 1), dtype=np.float32)
            feature = np.concatenate((A, feature), axis=1)
            weight = np.matmul(LA.pinv(feature), labels).astype(np.float32)
            print 'weight {}  dtype: {} '.format(i, weight.dtype)
            feature = np.matmul(feature, weight)
            weights['%d LLSR weight' % k] = weight[1:weight.shape[0]]
            bias['%d LLSR bias' % k] = weight[0].reshape(1, -1)
            print(k, ' layer LSR weight shape:', weight.shape)
            print(k, ' layer LSR output shape:', feature.shape)

            pred_labels = np.argmax(feature, axis=1)
            acc_train = sklearn.metrics.accuracy_score(train_labels, pred_labels)
            print('training acc is {}'.format(acc_train))
    # save data
    fw = open('llsr_weights_compact_v2.pkl', 'wb')
    pickle.dump(weights, fw, protocol=2)
    fw.close()
    fw = open('llsr_bias_compact_v2.pkl', 'wb')
    pickle.dump(bias, fw, protocol=2)
    fw.close()


if __name__ == '__main__':
    main()

