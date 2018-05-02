"""
Image manipulation functions for CIFAR10

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

# from __future__ import print_function
import numpy as np
# from matplotlib import pyplot as plt
# from neural_network import *
# from keras.layers import Activation
##from vis import utils
##from vis.utils import apply_modifications
##import vis
# from keras.models import Model

import sys

sys.path.append('../../')
from utils import *


# manipulate row*col pixels (:, :) at each time
def influential_pixel_manipulation(image, neural_network, max_index, pixel_num, ii, jj, kk, l, layer_functions):
    # print('\nGetting influential pixel manipulations: ')

    new_pixel_list = np.linspace(0, 1, pixel_num)
    # print('Pixel manipulation: ', new_pixel_list)

    image_batch = np.kron(np.ones((pixel_num, 1, 1, 1)), image)

    manipulated_images = []
    (row, col, chl) = image.shape
    for i in range(0, row):
        for j in range(0, col):
            changed_image_batch = image_batch.copy()
            for p in range(0, pixel_num):
                changed_image_batch[p, i, j, :] = new_pixel_list[p]
            manipulated_images.append(changed_image_batch)  # each loop append [pixel_num, row, col]
    manipulated_images = np.asarray(manipulated_images)  # [row*col, pixel_num, row, col]
    # print('Manipulated images: ', manipulated_images.shape)

    manipulated_images = manipulated_images.reshape(row * col * pixel_num, row, col, chl)
    print('Reshape dimensions to put into neural network: ', manipulated_images.shape)

    activations = eval_batch(layer_functions, manipulated_images)
    features_list = activations[l]

    feature_change = features_list[:, ii, jj, kk].reshape(-1, pixel_num).transpose()
    # print (feature_change)

    min_indices = np.argmax(feature_change, axis=0)
    min_values = np.amax(feature_change, axis=0)
    min_idx_values = min_indices.astype('float32') / (pixel_num - 1)

    [x, y] = np.meshgrid(np.arange(row), np.arange(col))
    x = x.flatten('F')  # to flatten in column-major order
    y = y.flatten('F')  # to flatten in column-major order

    target_feature_list = np.hstack((np.split(x, len(x)),
                                     np.split(y, len(y)),
                                     np.split(min_values, len(min_values)),
                                     np.split(min_idx_values, len(min_idx_values))))

    sorted_feature_map = target_feature_list[(target_feature_list[:, 2]).argsort()]
    sorted_feature_map = np.flip(sorted_feature_map, 0)

    return sorted_feature_map


def accumulated_pixel_manipulation(image, neural_network, sorted_feature_map, mani_range, label, ii, jj, kk, l,
                                   layer_functions):
    # print('\nLooking for adversary images...')
    # print('Manipulation range: ', mani_range)

    manipulated_images = []
    mani_image = image.copy()
    (row, col, chl) = image.shape
    for i in range(0, mani_range):
        # change row and col from 'float' to 'int'
        pixel_row = sorted_feature_map[i, 0].astype('int')
        pixel_col = sorted_feature_map[i, 1].astype('int')
        pixel_value = sorted_feature_map[i, 3]
        mani_image[pixel_row][pixel_col] = pixel_value
        # need to be very careful about image.copy()
        manipulated_images.append(mani_image.copy())

    manipulated_images = np.asarray(manipulated_images)
    # layer_output = get_current_layer_output([manipulated_images.reshape(len(manipulated_images), row, col, 1)])[0]
    # manipulated_labels=layer_output[:,ii, jj, kk]
    activations = eval_batch(layer_functions, manipulated_images.reshape(len(manipulated_images), row, col, chl))
    manipulated_labels = activations[l][:, ii, jj, kk]
    ##print('Manipulated labels: ', manipulated_labels.shape, '\n', manipulated_labels)

    adversary_images = manipulated_images[manipulated_labels > 0, :, :]
    adversary_labels = manipulated_labels[manipulated_labels > 0]
    # print('Adversary images: ', adversary_images.shape)
    ##print('Adversary labels: ', adversary_labels.shape, '\n', adversary_labels)

    if adversary_labels.any():
        success_flag = 1
        #       idx_first = (manipulated_labels != label).nonzero()[0][0]
        idx_first = np.amin((manipulated_labels != label).nonzero(), axis=1)
        # print('First adversary image found after', idx_first+1, 'pixel manipulations.')
    else:
        success_flag = 0
        idx_first = np.nan
        # print('Adversary image not found.')

    return adversary_images, adversary_labels, idx_first, success_flag


def refine_adversary_image(image, neural_network, adv_image_first, sorted_features, idx_first, label, ii, jj, kk, l,
                           layer_functions):
    (row, col, chl) = image.shape
    refined_adversary = adv_image_first.copy()
    # print('Evaluating individual pixels: \nNo. ', end='')
    total_idx = 0
    idx_range = np.arange(idx_first)
    go_deeper = True
    while go_deeper:
        length = len(idx_range)
        for i in range(0, idx_first):
            pixel_row = sorted_features[i, 0].astype('int')
            pixel_col = sorted_features[i, 1].astype('int')
            refined_adversary[pixel_row, pixel_col] = image[pixel_row, pixel_col]
            # refined_label = neural_network.predict_classes(refined_adversary.reshape(1, row, col, 1), verbose=0)
            # refined_label = get_current_layer_output([refined_adversary.reshape(1, row, col, 1)])[0][0][ii][jj][kk]
            ###
            activations = eval_batch(layer_functions, refined_adversary.reshape(1, row, col, chl))
            refined_label = activations[l][0][ii][jj][kk]
            ###
            if refined_label < 0:  # == label:
                refined_adversary[pixel_row, pixel_col] = sorted_features[i, 3]
            else:
                total_idx = total_idx + 1
                idx_range = idx_range[~(idx_range == i)]
        if len(idx_range) == length:
            go_deeper = False

    ##refined_label = neural_network.predict_classes(refined_adversary.reshape(1, row, col, 1), verbose=0)
    # refined_label = get_current_layer_output([refined_adversary.reshape(1, row, col, 1)])[0][0][ii][jj][kk]
    activations = eval_batch(layer_functions, refined_adversary.reshape(1, row, col, chl))
    refined_label = activations[l][0][ii][jj][kk]

    if (refined_adversary == adv_image_first).all():
        success_flag = 0
    else:
        success_flag = 1

    return refined_adversary, refined_label, success_flag


# manipulate row*col pixels (:, :) at each time
def influential_pixel_manipulation_dense(image, neural_network, max_index, pixel_num, hh, l, layer_functions):
    # print('\nGetting influential pixel manipulations: ')

    new_pixel_list = np.linspace(0, 1, pixel_num)
    # print('Pixel manipulation: ', new_pixel_list)

    image_batch = np.kron(np.ones((pixel_num, 1, 1, 1)), image)

    manipulated_images = []
    (row, col, chl) = image.shape
    for i in range(0, row):
        for j in range(0, col):
            # need to be very careful about image.copy()
            changed_image_batch = image_batch.copy()
            for p in range(0, pixel_num):
                changed_image_batch[p, i, j, :] = new_pixel_list[p]
            # changed_image_batch[:, i, j] = np.array(new_pixel_list)
            #           responseAllMat = np.concatenate((responseAllMat,changed_image_3d), axis=0)
            manipulated_images.append(changed_image_batch)  # each loop append [pixel_num, row, col]
    manipulated_images = np.asarray(manipulated_images)  # [row*col, pixel_num, row, col]
    # print('Manipulated images: ', manipulated_images.shape)

    manipulated_images = manipulated_images.reshape(row * col * pixel_num, row, col, chl)
    # print('Reshape dimensions to put into neural network: ', manipulated_images.shape)

    ##print(neural_network.layers[l].output_shape)

    # get_current_layer_output = K.function([neural_network.layers[0].get_input_at(1)], [neural_network.layers[l].get_output_at(1)])
    # features_list = get_current_layer_output([manipulated_images])[0]
    activations = eval_batch(layer_functions, manipulated_images)
    features_list = activations[l]

    feature_change = features_list[:, hh].reshape(-1, pixel_num).transpose()

    min_indices = np.argmax(feature_change, axis=0)
    min_values = np.amax(feature_change, axis=0)
    min_idx_values = min_indices.astype('float32') / (pixel_num - 1)

    [x, y] = np.meshgrid(np.arange(row), np.arange(col))
    x = x.flatten('F')  # to flatten in column-major order
    y = y.flatten('F')  # to flatten in column-major order

    target_feature_list = np.hstack((np.split(x, len(x)),
                                     np.split(y, len(y)),
                                     np.split(min_values, len(min_values)),
                                     np.split(min_idx_values, len(min_idx_values))))

    sorted_feature_map = target_feature_list[(target_feature_list[:, 2]).argsort()]
    sorted_feature_map = np.flip(sorted_feature_map, 0)

    # print('Sorted feature map: ', sorted_feature_map.shape)
    # print(sorted_feature_map)

    return sorted_feature_map


def accumulated_pixel_manipulation_dense(image, neural_network, sorted_feature_map, mani_range, label, hh, l,
                                         layer_functions):
    # print('\nLooking for adversary images...')
    # print('Manipulation range: ', mani_range)

    manipulated_images = []
    mani_image = image.copy()
    (row, col, chl) = image.shape
    for i in range(0, mani_range):
        # change row and col from 'float' to 'int'
        pixel_row = sorted_feature_map[i, 0].astype('int')
        pixel_col = sorted_feature_map[i, 1].astype('int')
        pixel_value = sorted_feature_map[i, 3]
        mani_image[pixel_row][pixel_col] = pixel_value
        # need to be very careful about image.copy()
        manipulated_images.append(mani_image.copy())

    manipulated_images = np.asarray(manipulated_images)
    # get_current_layer_output = K.function([neural_network.layers[0].input], [neural_network.layers[l].output])
    # get_current_layer_output = K.function([neural_network.layers[0].get_input_at(1)], [neural_network.layers[l].get_output_at(1)])
    # layer_output = get_current_layer_output([manipulated_images.reshape(len(manipulated_images), row, col, 1)])[0]
    # manipulated_labels=layer_output[:, hh]
    activations = eval_batch(layer_functions, manipulated_images.reshape(len(manipulated_images), row, col, chl))
    manipulated_labels = activations[l][:, hh]

    adversary_images = manipulated_images[manipulated_labels > 0, :, :]
    adversary_labels = manipulated_labels[manipulated_labels > 0]

    if adversary_labels.any():
        success_flag = 1
        idx_first = np.amin((manipulated_labels != label).nonzero(), axis=1)
        # print('First adversary image found after', idx_first+1, 'pixel manipulations.')
    else:
        success_flag = 0
        idx_first = np.nan
        # print('Adversary image not found.')

    return adversary_images, adversary_labels, idx_first, success_flag


def refine_adversary_image_dense(image, neural_network, adv_image_first, sorted_features, idx_first, label, hh, l,
                                 layer_functions):
    # print('\nRefining found adversary image...')

    # get_current_layer_output = K.function([neural_network.layers[0].get_input_at(1)], [neural_network.layers[l].get_output_at(1)])

    (row, col, chl) = image.shape
    refined_adversary = adv_image_first.copy()
    # print('Evaluating individual pixels: \nNo. ', end='')
    total_idx = 0
    idx_range = np.arange(idx_first)
    go_deeper = True
    while go_deeper:
        length = len(idx_range)
        for i in range(0, idx_first):
            pixel_row = sorted_features[i, 0].astype('int')
            pixel_col = sorted_features[i, 1].astype('int')
            refined_adversary[pixel_row, pixel_col] = image[pixel_row, pixel_col]
            # refined_label = get_current_layer_output([refined_adversary.reshape(1, row, col, 1)])[0][0][hh]
            activations = eval_batch(layer_functions, refined_adversary.reshape(1, row, col, chl))
            refined_label = activations[l][0][hh]
            if refined_label < 0:  # == label:
                refined_adversary[pixel_row, pixel_col] = sorted_features[i, 3]
            else:
                total_idx = total_idx + 1
                idx_range = idx_range[~(idx_range == i)]
                # print(i+1, end=' ')
        if len(idx_range) == length:
            go_deeper = False

    # refined_label = neural_network.predict_classes(refined_adversary.reshape(1, row, col, 1), verbose=0)
    # refined_label = get_current_layer_output([refined_adversary.reshape(1, row, col, 1)])[0][0][hh]
    activations = eval_batch(layer_functions, refined_adversary.reshape(1, row, col, chl))
    refined_label = activations[l][0][hh]

    if (refined_adversary == adv_image_first).all():
        success_flag = 0
    else:
        success_flag = 1

    return refined_adversary, refined_label, success_flag
