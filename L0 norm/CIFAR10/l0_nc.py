"""
Neuron Cover of CIFAR10 based on L0 Norm

Author: Youcheng Sun
Email: youcheng.sun@cs.ox.ac.uk
"""

# from __future__ import print_function
# from matplotlib import pyplot as plt
from numpy import linalg as LA
import time
from pixel_nc import *
# import os
import sys
import numpy as np

sys.path.append('../../')
from utils import *


# from keras.models import Model
# from keras import backend as K
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# import keras
# from keras.models import model_from_json
# from keras.datasets import mnist
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def negate(model, activations, nc_layer, nc_index, image, layer_functions):
    idx_min = 0
    idx_max = 10

    pixel_num = 2
    mani_range = 100
    adv = 0

    (row, col, chl) = image.shape

    l = nc_layer.layer_index

    if is_conv_layer(model.layers[l]):

        pos = np.unravel_index(nc_index, activations[l].shape)
        ii = pos[1]
        jj = pos[2]
        kk = pos[3]
        label = None  ## the label does not matter here

        tic = time.time()
        sorted_features = influential_pixel_manipulation(image, model, label, pixel_num, ii, jj, kk, l, layer_functions)
        (adv_images, adv_labels, idx_first, success_flag) = accumulated_pixel_manipulation(image, model,
                                                                                           sorted_features, mani_range,
                                                                                           label, ii, jj, kk, l,
                                                                                           layer_functions)
        elapsed = time.time() - tic
        print('\nElapsed time: ', elapsed)

        result = []
        if success_flag == 1:
            adv_image_first = adv_images[0]
            adv_label_first = adv_labels[0]
            image_diff = np.abs(adv_image_first - image)
            # note pixels have been transformed from [0,255] to [0,1]
            L0_distance = (image_diff * 255 > 1).sum()
            L1_distance = image_diff.sum()
            L2_distance = LA.norm(image_diff)

            (refined_adversary, refined_label, success_flag) = refine_adversary_image(image, model, adv_image_first,
                                                                                      sorted_features, idx_first, label,
                                                                                      ii, jj, kk, l, layer_functions)

            image_diff = np.abs(refined_adversary - image)
            L0_distance = (image_diff * 255 > 1).sum()
            L1_distance = image_diff.sum()
            L2_distance = LA.norm(image_diff)

            return True, L0_distance, refined_adversary
        else:
            return False, -1, -1
    else:
        pos = np.unravel_index(nc_index, activations[l].shape)
        hh = pos[1]
        label = None

        tic = time.time()
        sorted_features = influential_pixel_manipulation_dense(image, model, label, pixel_num, hh, l, layer_functions)
        (adv_images, adv_labels, idx_first, success_flag) = accumulated_pixel_manipulation_dense(image, model,
                                                                                                 sorted_features,
                                                                                                 mani_range, label, hh,
                                                                                                 l, layer_functions)
        elapsed = time.time() - tic
        print('\nElapsed time: ', elapsed)

        if success_flag == 1:
            adv_image_first = adv_images[0]
            adv_label_first = adv_labels[0]
            image_diff = np.abs(adv_image_first - image)
            L0_distance = (image_diff * 255 > 1).sum()
            L1_distance = image_diff.sum()
            L2_distance = LA.norm(image_diff)

            (refined_adversary, refined_label, success_flag) = refine_adversary_image_dense(image, model,
                                                                                            adv_image_first,
                                                                                            sorted_features, idx_first,
                                                                                            label, hh, l,
                                                                                            layer_functions)

            image_diff = np.abs(refined_adversary - image)
            L0_distance = (image_diff * 255 > 1).sum()
            L1_distance = image_diff.sum()
            L2_distance = LA.norm(image_diff)

            return True, L0_distance, refined_adversary
        else:
            return False, -1, -1
