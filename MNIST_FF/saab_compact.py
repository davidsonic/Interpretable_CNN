import numpy as np
from skimage.util.shape import view_as_windows

from sklearn.decomposition import PCA
from numpy import linalg as LA
from skimage.measure import block_reduce

import matplotlib.pyplot as plt
from itertools import product


def parse_list_string(list_string):
    """Convert the class string to list."""
    elem_groups = list_string.split(",")
    results = []
    for group in elem_groups:
        term = group.split("-")
        if len(term) == 1:
            results.append(int(term[0]))
        else:
            start = int(term[0])
            end = int(term[1])
            results += range(start, end + 1)
    return results


# convert responses to patches representation
def window_process(samples, kernel_size, stride):
    '''
    Create patches
    :param samples: [num_samples, feature_height, feature_width, feature_channel]
    :param kernel_size: int i.e. patch size
    :param stride: int
    :return patches: flattened, [num_samples, output_h, output_w, feature_channel*kernel_size^2]

    '''
    n, h, w, c = samples.shape
    output_h = (h - kernel_size) // stride + 1
    output_w = (w - kernel_size) // stride + 1
    patches = view_as_windows(samples, (1, kernel_size, kernel_size, c), step=(1, stride, stride, c))
    patches = patches.reshape(n, output_h, output_w, c * kernel_size * kernel_size)
    return patches


def window_process2(samples, kernel_size, stride):
    n, c, h, w= samples.shape
    output_h = (h - kernel_size) // stride + 1
    output_w = (w - kernel_size) // stride + 1
    # samples2=np.moveaxis(samples, 3, 1)
    patches=view_as_windows(samples, (1, c, kernel_size, kernel_size), step=(1, c, stride, stride))
    patches=patches.reshape(n,output_h, output_w, c*kernel_size*kernel_size)
    return patches


def window_process3(samples, kernel_size, stride):
    n, c, h, w=samples.shape
    print samples.shape
    out_h=(h-kernel_size)//stride+1
    out_w=(w-kernel_size)//stride+1
    idx1 = range(0, h-kernel_size+1, 1)
    print idx1
    idx2 = [i + kernel_size  for i in idx1]
    print idx2
    data_lattice = np.array([samples[:, :, i:j, k:l] for ((i, j), (k, l)) in product(zip(idx1, idx2), zip(idx1, idx2))])
    print data_lattice.shape
    data_lattice=np.moveaxis(data_lattice, 1,0)
    patches=data_lattice.reshape(n,out_h, out_w, c*kernel_size*kernel_size )
    return patches

def remove_mean(features, axis):
    '''
    Remove the dataset mean.
    :param features [num_samples,...]
    :param axis the axis to compute mean

    '''
    feature_mean = np.mean(features, axis=axis, keepdims=True)
    feature_remove_mean = features - feature_mean
    return feature_remove_mean, feature_mean


def select_balanced_subset(images, labels, use_num_images, use_classes):
    '''
    select equal number of images from each classes
    '''
    # Shuffle
    num_total = images.shape[0]
    shuffle_idx = np.random.permutation(num_total)
    images = images[shuffle_idx]
    labels = labels[shuffle_idx]

    num_class = len(use_classes)
    num_per_class = int(use_num_images / num_class)
    selected_images = np.zeros((use_num_images, images.shape[1], images.shape[2], images.shape[3]))
    selected_labels = np.zeros(use_num_images)
    for i in range(num_class):
        images_in_class = images[labels == i]
        selected_images[i * num_per_class:(i + 1) * num_per_class] = images_in_class[:num_per_class]
        selected_labels[i * num_per_class:(i + 1) * num_per_class] = np.ones((num_per_class)) * i

    # Shuffle again
    shuffle_idx = np.random.permutation(num_per_class * num_class)
    selected_images = selected_images[shuffle_idx]
    selected_labels = selected_labels[shuffle_idx]
    # For test
    # print(selected_images.shape)
    # print(selected_labels[0:10])
    # plt.figure()
    # for i in range (10):
    # 	img=selected_images[i,:,:,0]
    # 	plt.imshow(img)
    # 	plt.show()
    return selected_images, selected_labels


def find_kernels_pca(samples, num_kernels, energy_percent):
    '''
    Do the PCA based on the provided samples.
    If num_kernels is not set, will use energy_percent.
    If neither is set, will preserve all kernels.

    :param samples: [num_samples, feature_dimension]
    :param num_kernels: num kernels to be preserved
    :param energy_percent: the percent of energy to be preserved
    :return: kernels, sample_mean
    '''
    if num_kernels:
        num_components = num_kernels
        pca = PCA(n_components=num_components, svd_solver='full')
    else:
        pca = PCA(n_components=samples.shape[1], svd_solver='full')

    pca.fit(samples)

    # Compute the number of kernels corresponding to preserved energy
    if energy_percent:
        energy = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.sum(energy < energy_percent) + 1

    kernels = pca.components_[:num_components, :]
    mean = pca.mean_

    print("Num of kernels: %d" % num_components)
    print("Energy percent: %f" % np.cumsum(pca.explained_variance_ratio_)[num_components - 1])
    return kernels, mean


def multi_Saab_transform(images, labels, kernel_sizes, num_kernels, energy_percent, use_num_images, use_classes):
    '''
    Do the PCA "training".
    :param images: [num_images, height, width, channel]
    :param labels: [num_images]
    :param kernel_sizes: list, kernel size for each stage,
           the length defines how many stages conducted
    :param num_kernels: list the number of kernels for each stage,
           the length should be equal to kernel_sizes.
    :param energy_percent: the energy percent to be kept in all PCA stages.
           if num_kernels is set, energy_percent will be ignored.
    :param use_num_images: use a subset of train images
    :param use_classes: the classes of train images
    return: pca_params: PCA kernels and mean
    '''
    num_total_images = images.shape[0]
    if use_num_images < num_total_images and use_num_images > 0:
        sample_images, selected_labels = select_balanced_subset(images, labels, use_num_images, use_classes)
    else:
        sample_images = images
    # sample_images=images
    num_samples = sample_images.shape[0]
    num_layers = len(kernel_sizes)
    pca_params = {}
    pca_params['num_layers'] = num_layers
    pca_params['kernel_size'] = kernel_sizes

    for i in range(num_layers):
        print('--------stage %d --------' % i)
        # Create patches
        # sample_patches=window_process(sample_images,kernel_sizes[i],kernel_sizes[i]) # nonoverlapping
        # sample_patches = window_process(sample_images, kernel_sizes[i], 1)  # overlapping
        sample_patches = window_process2(sample_images, kernel_sizes[i], 1)  # overlapping
        h = sample_patches.shape[1]
        w = sample_patches.shape[2]
        # Flatten
        sample_patches = sample_patches.reshape([-1, sample_patches.shape[-1]])

        # Remove feature mean (Set E(X)=0 for each dimension)
        sample_patches_centered, feature_expectation = remove_mean(sample_patches, axis=0)
        # print 'sample_patches_centered.shape: {}'.format(sample_patches_centered.shape)

        # Remove patch mean
        training_data, dc = remove_mean(sample_patches_centered, axis=1)
        print 'training_data.shape: {}'.format(training_data.shape)

        # Compute PCA kernel

        if not num_kernels is None:
            num_kernel = num_kernels[i]
        kernels, mean = find_kernels_pca(training_data, num_kernel, energy_percent)

        # Add DC kernel
        # num_channels = sample_patches.shape[-1]
        num_channels = sample_patches.shape[1]
        dc_kernel = 1 / np.sqrt(num_channels) * np.ones((1, num_channels))
        kernels = np.concatenate((dc_kernel, kernels), axis=0)

        if i == 0:
            # Transform to get data for the next stage
            # print 'sample_patches_centered: {}'.format(sample_patches_centered)
            transformed = np.matmul(sample_patches_centered, np.transpose(kernels))
        else:
            # Compute bias term
            bias = LA.norm(sample_patches, axis=1)
            bias = np.max(bias)
            pca_params['Layer_%d/bias' % i] = bias
            # Add bias
            sample_patches_centered_w_bias = sample_patches_centered + 1 / np.sqrt(num_channels) * bias
            # Transform to get data for the next stage
            transformed = np.matmul(sample_patches_centered_w_bias, np.transpose(kernels))
            # Remove bias
            e = np.zeros((1, kernels.shape[0]))
            e[0, 0] = 1
            transformed -= bias * e

        # Reshape: place back as a 4-D feature map
        sample_images = transformed.reshape(num_samples, h, w, -1)
        sample_images=np.moveaxis(sample_images, 3,1)
        # sample_images = transformed.reshape(num_samples, -1, h, w)

        # Maxpooling
        # sample_images = block_reduce(sample_images, (1, 2, 2, 1), np.max)
        sample_images = block_reduce(sample_images, (1, 1, 2, 2), np.max)

        print('Sample patches shape after flatten:', sample_patches.shape)
        print('Kernel shape:', kernels.shape)
        print('Transformed shape:', transformed.shape)
        print('Sample images shape:', sample_images.shape)

        pca_params['Layer_%d/feature_expectation' % i] = feature_expectation
        pca_params['Layer_%d/kernel' % i] = kernels
        pca_params['Layer_%d/pca_mean' % i] = mean

    return pca_params


# Initialize
def initialize(sample_images, pca_params):
    num_layers = pca_params['num_layers']
    kernel_sizes = pca_params['kernel_size']

    for i in range(num_layers):
        print('--------stage %d --------' % i)
        # Extract parameters
        feature_expectation = pca_params['Layer_%d/feature_expectation' % i].astype(np.float32)
        kernels = pca_params['Layer_%d/kernel' % i].astype(np.float32)
        # mean = pca_params['Layer_%d/pca_mean' % i].astype(np.float32)
        print 'sample images shape: {}'.format(sample_images.shape)
        # Create patches
        # print 'sample_images {}: {}', sample_images.shape, sample_images[-2:]
        if i==0:
            sample_patches = window_process2(sample_images, kernel_sizes[i], 1)  # overlapping
            print 'sample_patches1: {}'.format(sample_patches.shape)
        elif i==1:
            sample_patches=window_process3(sample_images, kernel_sizes[i], 1)
            print 'sample_patches1: {}'.format(sample_patches.shape)

        h = sample_patches.shape[1]
        w = sample_patches.shape[2]
        # Flatten
        # (n*output_h*output_w, c*kernel_size*kernel_size)
        sample_patches = sample_patches.reshape([-1, sample_patches.shape[-1]])
        print 'sample_patches2: {}'.format(sample_patches.shape)



        # Remove feature mean (Set E(X)=0 for each dimension)
        # feature after removing mean, mean
        # sample_patches_centered, feature_expectation = remove_mean(sample_patches, axis=0)
        sample_patches_centered = sample_patches - feature_expectation
        # sample_patches_centered = sample_patches

         # sample_patches_centered=sample_patches-feature_expectation

        # Remove patch mean
        # training_data, dc = remove_mean(sample_patches_centered, axis=1)

        # num_channels = sample_patches.shape[-1]
        if i == 0:
            # Transform to get data for the next stage
            # print 'train data:{}'.format(sample_patches_centered[:5])
            # print 'sample shape: {}'.format(sample_patches_centered.shape)
            # print 'data shape: {}'.format(sample_patches_centered.shape)
            # print 'kernels: {}, shape:{}'.format(kernels, kernels.shape)
            # print 'kernel: {} /{}'.format(kernels, kernels.dtype)
            transformed = np.matmul(sample_patches_centered, np.transpose(kernels))
            # temp=transformed.reshape(10000, 28, 28, 6)
            # temp=np.moveaxis(temp, 3, 1)
            # print 'layer conv1 output: {}, shape: {}'.format( temp[:1].shape, temp[:1])
        # elif i==1:
        #     transformed= np.matmul(sample_patches_centered, np.transpose(kernels))
        #     temp=transformed.reshape(10000, 10, 10, 16)
        #     temp=np.moveaxis(temp, 3,1)
        #     print 'conv2  {},output: {}'.format(temp[:1].shape, temp[:1])
        else:
            # weight=(kernels)
            # print 'conv2 temp_bias: ', np.matmul(-1*feature_expectation, weight)
            # print 'conv2 kernel: {}, shape: {}'.format(kernels, kernels.shape)
            # print 'sample_patches_centered: ',sample_patches_centered.shape
            bias = pca_params['Layer_%d/bias' % i].astype(np.float32)
            # Add bias
            # print 'num_channels: {}'.format(num_channels)
            sample_patches_centered_w_bias = sample_patches_centered + 1 / np.sqrt(150) * bias
            # bias1_tmp=np.zeros(150)+1/np.sqrt(num_channels)* bias
            # print 'conv2 bias1: ', np.matmul(bias1_tmp, weight)
            # Transform to get data for the next stage
            transformed = np.matmul(sample_patches_centered_w_bias, np.transpose(kernels))
            # Remove bias
            e = np.zeros((1, kernels.shape[0]),dtype=np.float32)
            e[0, 0] = 1
            transformed -= bias * e
            # print 'conv2 bias2: ', -bias*e
            # print 'bias: {}'.format(bias)
            # temp=transformed.reshape(10000, 10, 10, 16)
            # temp=np.moveaxis(temp, 3, 1)
            # print 'conv2_2 shape {}, output: {}'.format(temp[:1].shape, temp[:1])

        # Reshape: place back as a 4-D feature map
        num_samples = sample_images.shape[0]
        sample_images = transformed.reshape(num_samples, h, w, -1)
        sample_images=np.moveaxis(sample_images, 3, 1)
        # sample_images = transformed.reshape(num_samples, -1, h, w)
        # print 'sample_images1.shape: {}'.format(sample_images.shape)

        # Maxpooling
        # sample_images = block_reduce(sample_images, (1, 2, 2, 1), np.max)
        sample_images = block_reduce(sample_images, (1, 1, 2, 2), np.max)
        # print 'Maxpool2D {}/ shape:{} / value: {} '.format(i, sample_images[:1].shape, sample_images[:1])

        print('Sample patches shape after flatten:', sample_patches.shape)
        print('Kernel shape:', kernels.shape)
        print('Transformed shape:', transformed.shape)
        print('Sample images shape:', sample_images.shape)
    return sample_images
