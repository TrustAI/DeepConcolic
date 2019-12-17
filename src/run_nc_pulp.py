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
from nc_pulp import *
#from lp_encoding import *
from pulp_encoding import *
from nc_setup import nc_setup

assert False

def run_nc_linf(test_object, outs):

  nc_results, cover_layers, _, test_cases = nc_setup(test_object, outs)
  adversarials, d_advs = [], []

  base_constraints, var_names = create_base_prob (test_object.dnn)

  while True:
    nc_layer, nc_pos, nc_value = test_object.get_nc_next(cover_layers)
    pos, nc_location = nc_layer.locate (nc_pos)

    im = test_cases[pos[0]]
    y_im = np.argmax(test_object.dnn.predict(np.array([im])))

    print ('\n::', pos, nc_layer.layer, nc_layer.layer_index, 'the max v::', nc_value/nc_layer.pfactor)

    feasible, d, new_im = negate (test_object.dnn, test_object.eval (im), [im], nc_layer, nc_location,
                                  constraints_for_cover_layer (base_constraints, nc_layer),
                                  var_names)

    nc_layer.disable_by_pos(pos)

    if feasible:
      if linf_filtered(test_object.raw_data.data, new_im): 
        print ('does not linf post filter')
        continue
      print ('\nis feasible!!!\n')
      test_cases.append(new_im)
      test_object.eval_and_update (cover_layers, new_im)
      y = np.argmax(test_object.dnn.predict(np.array([new_im])))
      d_adv = np.amax(np.absolute(im-new_im))
      print (d, d_adv)
      if y != y_im:
        adversarials.append([im, new_im])
        inp_ub=test_object.inp_ub
        save_adversarial_examples([new_im/(inp_ub*1.0), '{0}-adv-{1}'.format(len(adversarials), y)], [im/(inp_ub*1.0), '{0}-original-{1}'.format(len(adversarials), y_im)], None, nc_results.split('/')[0]) 
        d_advs.append(d_adv)
        if len(d_advs)%100==0:
          print_adversarial_distribution(d_advs, nc_results.replace('.txt', '')+'-adversarial-distribution.txt')
    else:
      print ('\nis NOT feasible!!!\n')

    covered, not_covered = test_object.nc_report (cover_layers)
    nc_percentage = 1.0 * covered / (covered + not_covered)
    append_in_file (nc_results,
                    'NC-cover: {0} #test cases: {1} #adversarial examples: {2}\n'
                    .format(nc_percentage, len(test_cases), len(adversarials)))

    if nc_percentage>0.9999: break
    #break

