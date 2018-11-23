
from pulp import *

import sys
import numpy as np

from utils import *

import copy

epsilon=0.0001
UPPER_BOUND=100000000
LOWER_BOUND=-100000000

## to create the base constraint for any AP
def create_base_prob(dnn):

  base_prob = LpProblem("base_prob", LpMinimize)
  var_names=[]
  var_names_vect=[]
  the_index=0
  
  for l in range(0, len(dnn.layers)):
    ## There is no intention to DIRECTLY change the classification label 
    if l==len(dnn.layers)-1: continue
  
    layer=dnn.layers[l]
  
    print ('== Create base variables: layer {0} == \n'.format(l), layer)
  
    if is_input_layer(layer):
      osp=layer.input.shape ## the output at this layer (e.g.) 28x28x1, 32x32x3
      if len(osp)<=2:
          print ('=== We assume the input layer be conv... === \n')
          sys.exit(0)
      gen_vars(the_index, osp, var_names, var_names_vect)

    elif is_conv_layer(layer):
      ### conv layer is the 1st layer
      if l==0:
        ## create variables for layer INPUT neurons
        isp = layer.input.shape
        gen_vars(the_index, isp, var_names, var_names_vect)
  
      ### create variables for layer OUTPUT neurons
      the_index+=1
      osp=layer.output.shape
      gen_vars(the_index, osp, var_names, var_names_vect)
  
      ### 'conv+relu'
      if act_in_the_layer(layer)=='relu':
        the_index+=1
        osp=layer.output.shape
        gen_vars(the_index, osp, var_names, var_names_vect)
  
    elif is_dense_layer(layer):
      if l==0:
        print ('=== We assume the input layer not be dense... === \n')
        sys.exit(0)
      the_index+=1
      osp=layer.output.shape
      gen_vars(the_index, osp, var_names, var_names_vect)
  
      ### 'dense+relu'
      if act_in_the_layer(layer)=='relu':
        the_index+=1
        osp=layer.output.shape
        gen_vars(the_index, osp, var_names, var_names_vect)
  
    elif is_activation_layer(layer):
      if get_activation(layer)!='relu':
        print ('=== We assume ReLU activation layer: layer {0}, {1} ==='.format(l, layer))
        continue 
      the_index+=1
      osp=layer.output.shape
      gen_vars(the_index, osp, var_names, var_names_vect)
  
    elif is_maxpooling_layer(layer):
        the_index+=1
        osp=layer.output.shape
        gen_vars(the_index, osp, var_names, var_names_vect)
  
    elif is_flatten_layer(layer):
        the_index+=1
        isp=layer.input.shape
        gen_vars_flattened(the_index, isp, var_names, var_names_vect)
  
    else:
        print ('Unknown layer: layer {)}, {1}'.format(l, layer))
        sys.exit(0)

  num_vars=0
  for x in var_names:
    num_vars+=x.size
  print ('LP variables have all been collected, #variables: {0}'.format(num_vars))
  
  
  ## now, it comes the encoding of constraints
  ## reset the_index
  weight_index=-1
  the_index=0
  tot_weights=dnn.get_weights()
  base_prob_dict=dict()

  for l in range(0, len(dnn.layers)):
    ## we skip the last layer
    if l==len(dnn.layers)-1: continue

    layer=dnn.layers[l]

    print ('== Create base constraint: layer {0} == \n'.format(l), layer)

    if is_input_layer(layer):
      ## nothing to constrain for InputLayer
      continue
    elif is_conv_layer(layer):
      if l==0:
        pass
      the_index+=1
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
              LpAffineExpression_list.append((out_neuron_var_name, -1))
              for II in range(0, kernel_size[0]):
                for JJ in range(0, kernel_size[1]):
                  for KK in range(0, weights.shape[2]):
                    try:
                      in_neuron_var_name=var_names[the_index-1][0][J+II][K+JJ][KK]
                      LpAffineExpression_list.append((in_neuron_var_name, float(weights[II][JJ][KK][L])))
                    except:
                      ## padding
                      pass
              #LpAffineconstraints.append(constraint)
              c = LpAffineExpression(LpAffineExpression_list)
              constraint = LpConstraint(c, LpConstraintEQ, 'c_name_{0}'.format(out_neuron_var_name), -float(biases[L]))
              base_prob+=constraint
      if act_in_the_layer(layer)=='relu':
          the_index+=1
          base_constraints_dict[l]=base_prob.copy()

    elif is_dense_layer(layer):
      the_index+=1
      isp=var_names[the_index-1].shape
      osp=var_names[the_index].shape
      weight_index+=1
      weights=tot_weights[weight_index]
      weight_index+=1
      biases=tot_weights[weight_index]
      for I in range(0, osp[0]):
        for J in range(0, osp[1]):
          LpAffineExpression_list=[]
          out_neuron_var_name=var_names[the_index][I][J]
          LpAffineExpression_list.append((out_neuron_var_name, -1))
          for II in range(0, isp[1]):
            in_neuron_var_name=var_names[the_index-1][0][II]
            LpAffineExpression_list.append((in_neuron_var_name, float(weights[II][J])))
          c = LpAffineExpression(LpAffineExpression_list)
          constraint = LpConstraint(c, LpConstraintEQ, 'c_name_{0}'.format(out_neuron_var_name), -float(biases[J]))
          base_prob+=constraint

      if act_in_the_layer(layer)=='relu':
        the_index+=1
        base_prob_dict[l]=base_prob.copy()

    elif is_flatten_layer(layer):
      the_index+=1
      isp=var_names[the_index-1].shape
      osp=var_names[the_index].shape

      tot=isp[1]*isp[2]*isp[3]
      for I in range(0, tot):
        d0=int(I)//(int(isp[2])*int(isp[3]))
        d1=(int(I)%(int(isp[2])*int(isp[3])))//int(isp[3])
        d2=int(I)-int(d0)*(int(isp[2])*int(isp[3]))-d1*int(isp[3])
        LpAffineExpression_list=[]
        out_neuron_var_name=var_names[the_index][0][I]
        LpAffineExpression_list.append((out_neuron_var_name, -1))
        in_neuron_var_name=var_names[the_index-1][0][int(d0)][int(d1)][int(d2)]
        LpAffineExpression_list.append((in_neuron_var_name, +1))
        c = LpAffineExpression(LpAffineExpression_list)
        constraint = LpConstraint(c, LpConstraintEQ, 'c_name_{0}'.format(out_neuron_var_name), 0)
        base_prob+=constraint
    elif is_activation_layer(layer):
      if get_activation(layer)!='relu': 
        print ('### We assume ReLU activation layer... ')
        continue
      the_index+=1
      base_prob_dict[l]=base_prob.copy()
      continue
    elif is_maxpooling_layer(layer):
      the_index+=1
      #isp=var_names[the_index-1].shape
      #osp=var_names[the_index].shape
      continue
    else:
      print ('Unknown layer', layer)
      sys.exit(0)

  return base_prob_dict, var_names

def build_conv_constraint(the_index, ll, I, J, K, L, act_inst, var_names, has_input_layer):
  l=ll
  #if not has_input_layer: l=ll-1
  osp=var_names[the_index].shape
  res=[]
  if act_inst[l][I][J][K][L]>0: ## we know what to do
    ## C1:
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index][I][J][K][L])
    #constraint[1].append(1)
    #constraint[0].append(var_names[the_index-1][I][J][K][L])
    #constraint[1].append(-1)
    #res.append([constraint, 0, 'E', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index][I][J][K][L], +1))
    LpAffineExpression_list.append((var_names[the_index-1][I][J][K][L], -1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)
    ## C2: >=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J][K][L])
    #constraint[1].append(1)
    #res.append([constraint, epsilon, 'G', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index-1][I][J][K][L], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintGE, '', epsilon)
    res.append(constraint)
  else:
    ## C1:
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index][I][J][K][L])
    #constraint[1].append(1)
    #res.append([constraint, 0, 'E', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index][I][J][K][L], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)
    ## C2: <=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J][K][L])
    #constraint[1].append(1)
    #res.append([constraint, -epsilon, 'L', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index-1][I][J][K][L], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintLE, '', -epsilon)
    res.append(constraint)
  return res

def build_dense_constraint(the_index, ll, I, J, act_inst, var_names, has_input_layer):
  osp=var_names[the_index].shape
  res=[]

  l=ll
  #if not has_input_layer: l=ll-1

  if act_inst[l][I][J]>0: ## do something
    ## C1:
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index][I][J])
    #constraint[1].append(1)
    #constraint[0].append(var_names[the_index-1][I][J])
    #constraint[1].append(-1)
    #res.append([constraint, 0, 'E', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index][I][J], +1))
    LpAffineExpression_list.append((var_names[the_index-1][I][J], -1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)
    ## C2: >=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J])
    #constraint[1].append(1)
    #res.append([constraint, epsilon, 'G', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index-1][I][J], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintGE, '', epsilon)
    res.append(constraint)
  else:
    ## C1:
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index][I][J])
    #constraint[1].append(1)
    #res.append([constraint, 0, 'E', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index][I][J], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)
    ## C2: <=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J])
    #constraint[1].append(1)
    #res.append([constraint, -epsilon, 'L', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index-1][I][J], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintLE, '', -epsilon)
    res.append(constraint)

  return res

def build_conv_constraint_neg(the_index, ll, I, J, K, L, act_inst, var_names, has_input_layer):
  if (act_inst[ll][I][J][K][L]>0):
    print ('activated neuron')
    sys.exit(0)
  l=ll
  #if not has_input_layer: l=ll-1
  #print ('**** to confirm act value: ', act_inst[l-1][I][J][K][L])
  #print (l, I, J, K, L)
  osp=var_names[the_index].shape
  res=[]
  if not(act_inst[l][I][J][K][L]>0): ## we know what to do
    ## C1:
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index][I][J][K][L])
    #constraint[1].append(1)
    #constraint[0].append(var_names[the_index-1][I][J][K][L])
    #constraint[1].append(-1)
    #res.append([constraint, 0, 'E', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index][I][J][K][L], +1))
    LpAffineExpression_list.append((var_names[the_index-1][I][J][K][L], -1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)

    ## C2: >=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J][K][L])
    #constraint[1].append(1)
    #res.append([constraint, epsilon, 'G', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index-1][I][J][K][L], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintGE, '', epsilon)
    res.append(constraint)
  else:
    ## C1:
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index][I][J][K][L])
    #constraint[1].append(1)
    #res.append([constraint, 0, 'E', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index][I][J][K][L], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)
    ## C2: <=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J][K][L])
    #constraint[1].append(1)
    #res.append([constraint, -epsilon, 'L', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index-1][I][J][K][L], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintLE, '', -epsilon)
    res.append(constraint)
  return res

def build_dense_constraint_neg(the_index, ll, I, J, act_inst, var_names, has_input_layer):
  osp=var_names[the_index].shape
  res=[]

  l=ll
  #if not has_input_layer: l=ll-1
  #print ('\n**** to confirm act value: ', act_inst[l-1][I][J])
  #print (the_index, ll, I, J)
  if (act_inst[ll][I][J]>0):
    print ('activated neuron')
    sys.exit(0)

  if not (act_inst[l][I][J]>0): ## do something
    ## C1:
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index][I][J])
    #constraint[1].append(1)
    #constraint[0].append(var_names[the_index-1][I][J])
    #constraint[1].append(-1)
    #res.append([constraint, 0, 'E', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index][I][J], +1))
    LpAffineExpression_list.append((var_names[the_index-1][I][J], -1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)
    ## C2: >=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J])
    #constraint[1].append(1)
    #res.append([constraint, epsilon, 'G', ''])
    LpAffineExpression_list=[]
    #LpAffineExpression_list.append((var_names[the_index-1][I][J], +1))
    LpAffineExpression_list.append((var_names[the_index][I][J], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintGE, '', epsilon)
    res.append(constraint)
  else:
    ## C1:
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index][I][J])
    #constraint[1].append(1)
    #res.append([constraint, 0, 'E', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index][I][J], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)
    ## C2: <=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J])
    #constraint[1].append(1)
    #res.append([constraint, -epsilon, 'L', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((var_names[the_index-1][I][J], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintLE, '', -epsilon)
    res.append(constraint)

  return res

def gen_vars(the_index, sp, var_names, var_names_vect):
  sp_len = len(sp)
  if sp_len==4: ## conv
    #var_names.append(np.empty((1, sp[1], sp[2], sp[3]), dtype="S40"))
    var_names.append(np.empty((1, sp[1], sp[2], sp[3]), dtype=LpVariable))
    for I in range(0, 1):
      for J in range(0, sp[1]):
        for K in range(0, sp[2]):
          for L in range(0, sp[3]):
            var_name='x_{0}_{1}_{2}_{3}_{4}'.format(the_index, I, J, K, L)
            #var_names[the_index][I][J][K][L]=var_name
            x_var = LpVariable(var_name, lowBound=LOWER_BOUND, upBound=UPPER_BOUND)
            var_names[the_index][I][J][K][L]=x_var
            var_names_vect.append(x_var)
  elif sp_len==2: ## not conv
    #var_names.append(np.empty((1, sp[1]), dtype="S40"))
    var_names.append(np.empty((1, sp[1]), dtype=LpVariable))
    for I in range(0, 1):
      for J in range(0, sp[1]):
        var_name='x_{0}_{1}_{2}'.format(the_index, I, J)
        #var_names[the_index][I][J]=var_name
        x_var = LpVariable(var_name, lowBound=LOWER_BOUND, upBound=UPPER_BOUND)
        var_names[the_index][I][J]=x_var
        var_names_vect.append(x_var)
  else:
    print ('## Unrecognised shape in gen_vars: {0}...'.format(sp))

def gen_vars_flattened(the_index, sp, var_names, var_names_vect):
  tot=int(sp[1]) * int(sp[2]) * int(sp[3])
  #var_names.append(np.empty((1, tot), dtype="S40"))
  var_names.append(np.empty((1, tot), dtype=LpVariable))
  for I in range(0, 1):
    for J in range(0, tot):
      var_name='x_{0}_{1}_{2}'.format(the_index, I, J)
      #var_names[the_index][I][J]=var_name
      x_var = LpVariable(var_name, lowBound=LOWER_BOUND, upBound=UPPER_BOUND)
      var_names[the_index][I][J]=x_var
      var_names_vect.append(x_var)
  
