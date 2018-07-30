import argparse
import sys
import os
from datetime import datetime

import keras
from keras.models import *
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.layers import *
from keras import *
from utils import *
from nc_setup import *
from ssc import *

def run_ssc(test_object, outs):
  print ('To run ssc\n')
  
  f_results, layer_functions, cover_layers, activations, test_cases, adversarials=nc_setup(test_object, outs)

  while True:
    dec_layer_index, dec_pos=get_ssc_next(cover_layers)

    print ('== to cover (dec_layer_index, dec_pos)', dec_layer_index, dec_pos)

    ###
    cond_layer=cover_layers[dec_layer_index-1]
    dec_layer=cover_layers[dec_layer_index]
    cond_cover=np.ones(cond_layer.ssc_map.shape, dtype=bool)
    ###

    for cond_pos in range(0, cond_cover.size):
      feasible, d, new_image, old_image=ssc_search(test_object, layer_functions, cond_layer, cond_pos, dec_layer, dec_pos)

      if feasible:
        test_cases.append((new_image, old_image))
      else:
        print ("not feasible")

