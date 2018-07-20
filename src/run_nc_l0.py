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
from nc_l0 import *

def run_nc_l0(test_object, outs):
  print ('\n == is coming soon... ==\n')
  nc_results, layer_functions, cover_layers, activations, test_cases, adversarials=nc_setup(test_object, outs)

  while True:
    index_nc_layer, nc_pos, nc_value=get_nc_next(cover_layers)
    nc_layer=cover_layers[index_nc_layer]

    shape=np.array(nc_layer.activations).shape
    pos=np.unravel_index(nc_pos, shape)
    im=test_cases[pos[0]]
    act_inst=eval(layer_functions, im)
    s=pos[0]*int(shape[1]*shape[2])
    if nc_layer.is_conv:
      s*=int(shape[3])*int(shape[4])

    feasible, d, new_im = l0_negate(test_object.dnn, act_inst, [im], nc_layer, nc_pos-s)

    cover_layers[index_nc_layer].disable_by_pos(pos)
    if feasible:
      test_cases.append(new_im)
      update_nc_map_via_inst(cover_layers, eval(layer_functions, new_im))
      y1 =(np.argmax(test_object.dnn.predict(np.array([im])))) 
      y2= (np.argmax(test_object.dnn.predict(np.array([im]))))
      if y1 != y2: adversarials.append([im, new_im])
      old_acts=eval(layer_functions, im)
      new_acts=eval(layer_functions, new_im)
    covered, not_covered=nc_report(cover_layers)
    f = open(nc_results, "a")
    f.write('NC-cover: {0} #test cases: {1} #adversarial examples: {2}\n'.format(1.0 * covered / (covered + not_covered), len(test_cases), len(adversarials)))
    f.close()



