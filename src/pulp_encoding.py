
from pulp import *

import sys
import numpy as np

from utils import *

import copy

epsilon=0.001
UPPER_BOUND=100000000
LOWER_BOUND=-100000000

def create_base_prob(dnn):

  base_prob = LpProblem("base_prob", LpMinimize)
  var_names=[]
  var_names_vect=[]
  the_index=0

  for l in range(0, len(dnn.layers)):
    if l==len(dnn.layers)-1: continue

    layer=dnn.layers[l]

    if is_input_layer(layer):
      osp=layer.input.shape ## the output at this layer (e.g.) 28x28x1, 32x32x3

      if len(osp)<=2:
        print ('== well, we do not think the input layer shall be non-convolutional... == \n')
        sys.exit(0)

      var_names.append(np.empty((1, osp[1], osp[2], osp[3]), dtype="S40"))
      for I in range(0, 1):
        for J in range(0, osp[1]):
          for K in range(0, osp[2]):
            for L in range(0, osp[3]):
              var_name='x_{0}_{1}_{2}_{3}_{4}'.format(the_index, I, J, K, L)
              var_names[the_index][I][J][K][L]=var_name
              x_var = LpVariable(var_name, lowBound=None, upBound=None)
              var_names_vect.append(x_var)
    elif is_conv_layer(layer):
      if l==0:
        isp=layer.input.shape
        var_names.append(np.empty((1, isp[1], isp[2], isp[3]), dtype="S40"))
        for I in range(0, 1):
          for J in range(0, isp[1]):
            for K in range(0, isp[2]):
              for L in range(0, isp[3]):
                var_name='x_{0}_{1}_{2}_{3}_{4}'.format(the_index, I, J, K, L)
                var_names[the_index][I][J][K][L]=var_name
                x_var = LpVariable(var_name, lowBound=None, upBound=None)
                var_names_vect.append(x_var)


      ### normal conv layer
      the_index+=1
      osp=layer.output.shape ## the output at this layer
      var_names.append(np.empty((1, osp[1], osp[2], osp[3]), dtype="S40"))
      for I in range(0, 1):
        for J in range(0, osp[1]):
          for K in range(0, osp[2]):
            for L in range(0, osp[3]):
              var_name='x_{0}_{1}_{2}_{3}_{4}'.format(the_index, I, J, K, L)
              var_names[the_index][I][J][K][L]=var_name
              x_var = LpVariable(var_name, lowBound=None, upBound=None)
              var_names_vect.append(x_var)

      ### conv layer + relu activation function
      if act_in_the_layer(layer)=='relu':
        ##
        the_index+=1
        osp=layer.output.shape ## the output at this layer
        var_names.append(np.empty((1, osp[1], osp[2], osp[3]), dtype="S40"))
        for I in range(0, 1):
          for J in range(0, osp[1]):
            for K in range(0, osp[2]):
              for L in range(0, osp[3]):
                var_name='x_{0}_{1}_{2}_{3}_{4}'.format(the_index, I, J, K, L)
                var_names[the_index][I][J][K][L]=var_name
                x_var = LpVariable(var_name, lowBound=None, upBound=None)
                var_names_vect.append(x_var)
    elif is_dense_layer(layer):
      if l==0: ## well, let us not allow to define a net starting with dense a layer...
        print ('well, let us not allow to define a net starting with dense a layer... ==\n')
        sys.exit(0)
      the_index+=1
      osp=layer.output.shape
      var_names.append(np.empty((1, osp[1]), dtype="S40"))
      for I in range(0, 1):
        for J in range(0, osp[1]):
          var_name='x_{0}_{1}_{2}'.format(the_index, I, J)
          var_names[the_index][I][J]=var_name
          x_var = LpVariable(var_name, lowBound=None, upBound=None)
          var_names_vect.append(x_var)
      if act_in_the_layer(layer)=='relu':
        ##
        the_index+=1
        osp=layer.output.shape
        var_names.append(np.empty((1, osp[1]), dtype="S40"))
        for I in range(0, 1):
          for J in range(0, osp[1]):
            var_name='x_{0}_{1}_{2}'.format(the_index, I, J)
            var_names[the_index][I][J]=var_name
            x_var = LpVariable(var_name, lowBound=None, upBound=None)
            var_names_vect.append(x_var)
    #####

  ## now, it comes the encoding of constraints
  weight_index=-1
  the_index=0
  tot_weights=dnn.get_weights()

  for l in range(0, len(dnn.layers)):
    ## we skip the last layer
    if l==len(dnn.layers)-1: continue

    layer=dnn.layers[l]
    if is_input_layer(layer):
      continue
    elif is_conv_layer(layer):
      if l==0:
        pass
      the_index+=1
      #print ('length of var_names: ', len(var_names))
      isp=var_names[the_index-1].shape
      osp=var_names[the_index].shape
      kernel_size=layer.kernel_size
      weight_index+=1
      weights=tot_weights[weight_index]
      weight_index+=1
      biases=tot_weights[weight_index]
      for I in range(0, osp[0]):
        for J in range(0, osp[1]):
          for K in range(0, osp[2]):
            for L in range(0, osp[3]):
              LpAffineExpression_list=[]
              out_neuron_var_name=var_names[the_index][I][J][K][L]
              #print ('out neuron var name: '+ out_neuron_var_name)
              LpAffineExpression_list.append((out_neuron_var_name, -1))
              for II in range(0, kernel_size[0]):
                for JJ in range(0, kernel_size[1]):
                  for KK in range(0, weights.shape[2]):
                    try:
                      in_neuron_var_name=var_names[the_index-1][0][J+II][K+JJ][KK]
                      LpAffineExpression_list.append((in_neuron_var_name, float(weights[II][JJ][KK][L])))
                    except: pass
              #LpAffineconstraints.append(constraint)
              c = LpAffineExpression(LpAffineExpression_list)
              constraint = LpConstraint(c, LpConstraintEQ, 'c_name_{0}'.format(out_neuron_var_name), -float(biases[L]))
              base_prob+=constraint





  
