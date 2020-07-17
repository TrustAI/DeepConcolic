assert False

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
from nc_lp import *
from lp_encoding import *
from nc_setup import nc_setup

def run_nc_linf(test_object, outs):

  nc_results, cover_layers, activations, test_cases = nc_setup(test_object, outs)
  adversarials, d_advs = [], []

  base_constraints = create_base_constraints (test_object.dnn)

  while True:
    nc_layer, nc_pos, nc_value = test_object.get_nc_next(cover_layers)
    pos, nc_location = nc_layer.locate (nc_pos)
    
    #print (nc_layer.layer_index, nc_pos, nc_value/nc_layer.pfactor)
    # print (np.array(nc_layer.activations).shape)
    # shape=np.array(nc_layer.activations).shape
    # pos=np.unravel_index(nc_pos, shape)

    im = test_cases[pos[0]]
    y_im = np.argmax(test_object.dnn.predict(np.array([im])))

    # s=pos[0]*int(shape[1]*shape[2])
    # if nc_layer.is_conv:
    #   s*=int(shape[3])*int(shape[4])
    print ('\n::', nc_pos, pos, nc_location)
    print (nc_layer.layer, nc_layer.layer_index)
    print ('the max v', nc_value)

    feasible, d, new_im = negate (test_object.dnn, test_object.eval (im), [im], nc_layer, nc_location,
                                  constraints_for_cover_layer (base_constraints, nc_layer))

    nc_layer.disable_by_pos(pos)

    if feasible:
      print ('\nis feasible!!!\n')
      test_cases.append(new_im)
      test_object.eval_and_update (cover_layers, new_im)
      y = np.argmax(test_object.dnn.predict(np.array([new_im])))
      if y != y_im:
        adversarials.append([im, new_im])
        inp_ub=test_object.inp_ub
        save_adversarial_examples([new_im/(inp_ub*1.0), '{0}-adv-{1}'.format(len(adversarials), y)], [im/(inp_ub*1.0), '{0}-original-{1}'.format(len(adversarials), y_im)], None, nc_results.split('/')[0]) 
        d_advs.append(np.amax(np.absolute(im-new_im)))
        if len(d_advs)%100==0:
          print_adversarial_distribution(d_advs, nc_results.replace('.txt', '')+'-adversarial-distribution.txt')
      #old_acts=eval(layer_functions, im)
      #new_acts=eval(layer_functions, new_im)
      #if nc_layer.is_conv:
      #  print ('\n should be < 0', old_acts[nc_layer.layer_index][pos[1]][pos[2]][pos[3]][pos[4]], '\n')
      #  print ('\n should be > 0', new_acts[nc_layer.layer_index][pos[1]][pos[2]][pos[3]][pos[4]], '\n')
    else:
      print ('\nis NOT feasible!!!\n')
      
    covered, not_covered = test_object.nc_report (cover_layers)
    append_in_file (nc_results,
                    'NC-cover: {0} #test cases: {1} #adversarial examples: {2}\n'
                    .format(1.0 * covered / (covered + not_covered),
                            len(test_cases), len(adversarials)))

