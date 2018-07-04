import argparse
import sys
from datetime import datetime

import keras
from keras.models import *
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras import *


from utils import *

class raw_datat:
  def __init__(self, data, labels):
    self.data=data
    self.labels=labels

class test_objectt:
  def __init__(self, dnn, raw_data, criterion, norm):
    self.dnn=dnn
    self.raw_data=raw_data
    ## test config
    self.norm=norm
    self.criterion=criterion
    self.channels_last=True

def deepconcolic(test_object):
  print('\n== Start DeepConcolic testing ==\n')

  layer_functions=get_layer_functions(test_object.dnn)
  print('\n== Got layer functions: {0} ==\n'.format(len(layer_functions)))
  cover_layers=get_cover_layers(test_object.dnn)
  print('\n== Got cover layers: {0} ==\n'.format(len(cover_layers)))

  #for c_layer in cover_layers:
  #  isp=c_layer.layer.input.shape
  #  osp=c_layer.layer.output.shape
  #  if c_layer.is_conv:
  #    ## output 
  #    for o_i in range(0, osp[3]): # by default, we assume channel last
  #      for o_j in range(0, osp[1]):
  #        for o_k in range(0, osp[2]):
  #          ## input 
  #          for i_i in range(0, isp[3]): # by default, we assume channel last
  #            for i_j in range(0, isp[1]):
  #              for i_k in range(0, isp[2]):
  #                sscover_lp(c_layer.layer_index, [o_j, o_k, o_i], [i_j, i_k, i_i], test_object, layer_functions)
  #                #print("SSCover lp: {0}-{1}-{2}".format(c_layer.layer_index, [o_j, o_k, o_i], [i_j, i_k, i_i]))


def cover(test_object):
  ## we start from SSC
  if (criterion=='SSC'):
    sscover(test_object)
  else:
    print('More to be added...')
    return

def main():
  ## for testing purpose we fix the aicover configuration
  ##
  dnn=load_model("../saved_models/cifar10_complicated.h5")
  #dnn=VGG16()
  ##
  criterion='SSC'
  img_rows, img_cols = 32, 32
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
  x_test = x_test.astype('float32')
  x_test /= 255
  raw_data=raw_datat(x_test, y_test)

  deepconcolic(test_objectt(dnn, raw_data, criterion, "linf"))

if __name__=="__main__":
  main()
