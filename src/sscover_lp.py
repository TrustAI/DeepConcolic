
import cplex
import sys
import numpy as np

from utils import *

## To build the base LP constraints that are shared
## by different activation patterns
def base_lp(test_object):
  var_names_vect=[]
  objective=[]
  lower_bounds=[]
  upper_bounds=[]
  var_names=[]

  for l in range(0, len(test_object.dnn.layers)):
    layer=test_object.dnn.layers[l]
    if is_conv_layer(layer) or is_maxpooling_layer(layer):
      if l==0:
        isp=layer.input.shape
        var_names.append(np.empty((1, isp[1], isp[2], isp[3]), dtype="S40"))


def sscover_lp(layer_index, o_pos, i_pos, test_object, layer_functions):

  indices=np.random.permutation(len(test_object.raw_data))

  for index in indices:
    x=test_object.raw_data[index]
    activations=eval(layer_functions, x)
