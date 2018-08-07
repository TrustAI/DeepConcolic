try:
  import cplex
except:
  from solver import *

import sys
import numpy as np

from utils import *

import copy

epsilon=0.001 #1.0/(255)

class base_constraintst:
  def __init__(self, objective, lower_bounds, upper_bounds, var_names_vect, var_names, constraints, constraint_senses, rhs, constraint_names):
    print ('base_constraints', len(objective), len(lower_bounds), len(upper_bounds), len(var_names_vect))
    self.obj=copy.copy(objective)
    self.lb=copy.copy(lower_bounds)
    self.ub=copy.copy(upper_bounds)
    self.names=copy.copy(var_names_vect)
    self.var_names=copy.copy(var_names)
    self.lin_expr=copy.copy(constraints)
    self.senses=copy.copy(constraint_senses)
    self.rhs=copy.copy(rhs)
    self.constraint_names=copy.copy(constraint_names)

def create_base_constraints(dnn):

  ## to return
  base_constraints_dict=dict()

  var_names=[]
  var_names_vect=[]
  objective=[] ## do we need 'objective' as base?
  lower_bounds=[]
  upper_bounds=[]

  the_index=0

  for l in range(0, len(dnn.layers)):
    ## we skip the last layer
    if l==len(dnn.layers)-1: continue

    layer=dnn.layers[l]

    print ('== {0} == \n'.format(l), layer)

    if is_input_layer(layer):
      osp=layer.input.shape ## the output at this layer
      if len(osp)<=2: 
        print ('== well, we do not think the input layer shall be non-convolutional... == \n')
        sys.exit(0)
      var_names.append(np.empty((1, osp[1], osp[2], osp[3]), dtype="S40"))
      for I in range(0, 1):
        for J in range(0, osp[1]):
          for K in range(0, osp[2]):
            for L in range(0, osp[3]):
              var_name='x_{0}_{1}_{2}_{3}_{4}'.format(the_index, I, J, K, L)
              objective.append(0)
              lower_bounds.append(-cplex.infinity)
              upper_bounds.append(cplex.infinity)
              var_names[the_index][I][J][K][L]=var_name
              var_names_vect.append(var_name)
    elif is_conv_layer(layer):
      if l==0: ## to deal with the input variables
        isp=layer.input.shape
        var_names.append(np.empty((1, isp[1], isp[2], isp[3]), dtype="S40"))
        for I in range(0, 1):
          for J in range(0, isp[1]):
            for K in range(0, isp[2]):
              for L in range(0, isp[3]):
                var_name='x_{0}_{1}_{2}_{3}_{4}'.format(the_index, I, J, K, L)
                objective.append(0)
                lower_bounds.append(-cplex.infinity)
                upper_bounds.append(cplex.infinity)
                var_names[the_index][I][J][K][L]=var_name
                var_names_vect.append(var_name)

      ##
      the_index+=1
      osp=layer.output.shape ## the output at this layer
      var_names.append(np.empty((1, osp[1], osp[2], osp[3]), dtype="S40"))
      for I in range(0, 1):
        for J in range(0, osp[1]):
          for K in range(0, osp[2]):
            for L in range(0, osp[3]):
              var_name='x_{0}_{1}_{2}_{3}_{4}'.format(the_index, I, J, K, L)
              objective.append(0)
              lower_bounds.append(-cplex.infinity)
              upper_bounds.append(cplex.infinity)
              var_names[the_index][I][J][K][L]=var_name
              var_names_vect.append(var_name)
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
                objective.append(0)
                lower_bounds.append(-cplex.infinity)
                upper_bounds.append(cplex.infinity)
                var_names[the_index][I][J][K][L]=var_name
                var_names_vect.append(var_name)
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
          objective.append(0)
          lower_bounds.append(-cplex.infinity)
          upper_bounds.append(cplex.infinity)
          var_names[the_index][I][J]=var_name
          var_names_vect.append(var_name)
      if act_in_the_layer(layer)=='relu':
        ##
        the_index+=1
        osp=layer.output.shape
        var_names.append(np.empty((1, osp[1]), dtype="S40"))
        for I in range(0, 1):
          for J in range(0, osp[1]):
            var_name='x_{0}_{1}_{2}'.format(the_index, I, J)
            objective.append(0)
            lower_bounds.append(-cplex.infinity)
            upper_bounds.append(cplex.infinity)
            var_names[the_index][I][J]=var_name
            var_names_vect.append(var_name)
    elif is_activation_layer(layer):
      if str(layer.activation).find('relu')<0: continue ## well, we only consider ReLU activation layer
      the_index+=1
      osp=layer.output.shape
      if len(osp) > 2: ## multiple feature maps
        var_names.append(np.empty((1, osp[1], osp[2], osp[3]), dtype="S40"))
        for I in range(0, 1):
          for J in range(0, osp[1]):
            for K in range(0, osp[2]):
              for L in range(0, osp[3]):
                var_name='x_{0}_{1}_{2}_{3}_{4}'.format(the_index, I, J, K, L)
                objective.append(0)
                lower_bounds.append(-cplex.infinity)
                upper_bounds.append(cplex.infinity)
                var_names[the_index][I][J][K][L]=var_name
                var_names_vect.append(var_name)
      else:
        var_names.append(np.empty((1, osp[1]), dtype="S40"))
        for I in range(0, 1):
          for J in range(0, osp[1]):
            var_name='x_{0}_{1}_{2}'.format(the_index, I, J)
            objective.append(0)
            lower_bounds.append(-cplex.infinity)
            upper_bounds.append(cplex.infinity)
            var_names[the_index][I][J]=var_name
            var_names_vect.append(var_name)
    elif is_maxpooling_layer(layer):
      the_index+=1
      osp=layer.output.shape
      var_names.append(np.empty((1, osp[1], osp[2], osp[3]), dtype="S40"))
      for I in range(0, 1):
        for J in range(0, osp[1]):
          for K in range(0, osp[2]):
            for L in range(0, osp[3]):
              var_name='x_{0}_{1}_{2}_{3}_{4}'.format(the_index, I, J, K, L)
              objective.append(0)
              lower_bounds.append(-cplex.infinity)
              upper_bounds.append(cplex.infinity)
              var_names[the_index][I][J][K][L]=var_name
              var_names_vect.append(var_name)
    elif is_flatten_layer(layer):
      the_index+=1
      isp=layer.input.shape
      tot=int(isp[1]) * int(isp[2]) * int(isp[3])
      var_names.append(np.empty((1, tot), dtype="S40"))
      for I in range(0, 1):
        for J in range(0, tot):
          var_name='x_{0}_{1}_{2}'.format(the_index, I, J)
          objective.append(0)
          lower_bounds.append(-cplex.infinity)
          upper_bounds.append(cplex.infinity)
          var_names[the_index][I][J]=var_name
          var_names_vect.append(var_name)
    else:
      print ('Unknown layer', layer)
      sys.exit(0)

  constraints=[]
  rhs=[]
  constraint_senses=[]
  constraint_names=[]

  ## now, it comes the encoding of constraints
  weight_index=-1
  the_index=0
  tot_weights=dnn.get_weights()
  for l in range(0, len(dnn.layers)):
    ## we skip the last layer
    if l==len(dnn.layers)-1: continue

    layer=dnn.layers[l]

    print ('== {0} == \n'.format(l), layer)

    if is_input_layer(layer):
      continue
    elif is_conv_layer(layer):
      if l==0:
        #the_index+=1
        pass
      the_index+=1
      isp=var_names[the_index-1].shape
      osp=var_names[the_index].shape
      kernel_size=layer.kernel_size
      weight_index+=1
      weights=tot_weights[weight_index]
      weight_index+=1
      biases=tot_weights[weight_index]
      #print ('the_index', the_index)
      #print (osp)
      for I in range(0, osp[0]):
        for J in range(0, osp[1]):
          for K in range(0, osp[2]):
            for L in range(0, osp[3]):
              #print ('== ', l, ' == ', I, J, K, L)
              constraint=[[], []]
              constraint[0].append(var_names[the_index][I][J][K][L])
              constraint[1].append(-1)
              try:
                for II in range(0, kernel_size[0]):
                  for JJ in range(0, kernel_size[1]):
                    for KK in range(0, weights.shape[2]):
                      #print (weights.shape)
                      constraint[0].append(var_names[the_index-1][0][J+II][K+JJ][KK])
                      constraint[1].append(float(weights[II][JJ][KK][L]))
                constraints.append(constraint)
                rhs.append(-float(biases[L]))
                constraint_senses.append('E')
                constraint_names.append('')
              except: ## this is due to padding
                rhs.append(0)
                constraint_senses.append('E')
                constraint_names.append('')
      if act_in_the_layer(layer)=='relu':
        the_index+=1
        base_constraints_dict[l]=base_constraintst(objective, lower_bounds, upper_bounds, var_names_vect, var_names, constraints, constraint_senses, rhs, constraint_names)
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
          #print ('dense == ', l, ' == ', I, J)
          constraint=[[], []]
          constraint[0].append(var_names[the_index][I][J])
          constraint[1].append(-1)
          for II in range(0, isp[1]):
            #print (weights.shape)
            constraint[0].append(var_names[the_index-1][0][II])
            constraint[1].append(float(weights[II][J]))

          constraints.append(constraint)
          rhs.append(-float(biases[J])) 
          constraint_senses.append('E')
          constraint_names.append('')
      if act_in_the_layer(layer)=='relu':
        the_index+=1
        base_constraints_dict[l]=base_constraintst(objective, lower_bounds, upper_bounds, var_names_vect,var_names, constraints, constraint_senses, rhs, constraint_names)
    elif is_flatten_layer(layer):
      the_index+=1
      isp=var_names[the_index-1].shape
      osp=var_names[the_index].shape

      #print (isp, osp)

      tot=isp[1]*isp[2]*isp[3]
      for I in range(0, tot):
        #print ('flatten, == ', l, ' == ', I)
        d0=int(I)//(int(isp[2])*int(isp[3]))
        d1=(int(I)%(int(isp[2])*int(isp[3])))//int(isp[3])
        d2=int(I)-int(d0)*(int(isp[2])*int(isp[3]))-d1*int(isp[3])
        constraint=[[], []]
        constraint[0].append(var_names[the_index][0][I])
        constraint[1].append(-1)
        constraint[0].append(var_names[the_index-1][0][int(d0)][int(d1)][int(d2)])
        constraint[1].append(+1)

        constraints.append(constraint)
        constraint_senses.append('E')
        rhs.append(0)
        constraint_names.append('')
    elif is_activation_layer(layer):
      if str(layer.activation).find('relu')<0: continue ## well, we only consider ReLU activation layer
      the_index+=1
      print ('add one relu layer')
      base_constraints_dict[l]=base_constraintst(objective, lower_bounds, upper_bounds, var_names_vect, var_names, constraints, constraint_senses, rhs, constraint_names)
      continue
    elif is_maxpooling_layer(layer):
      the_index+=1
      isp=var_names[the_index-1].shape
      osp=var_names[the_index].shape
      continue
    else:
      print ('Unknown layer', layer)
      sys.exit(0)

  #return base_constraintst(objective, lower_bounds, upper_bounds, var_names_vect, constraints, constraint_senses, rhs, constraint_names)
  return base_constraints_dict


def build_conv_constraint(the_index, ll, I, J, K, L, act_inst, var_names, has_input_layer):
  #print (' == build conv constraints == ', the_index, l)
  #print (var_names[the_index].shape, var_names[the_index-1].shape)
  #print (var_names[0].shape)
  #print (var_names[1].shape)
  #print (var_names[2].shape)
  l=ll
  #if not has_input_layer: l=ll-1
  osp=var_names[the_index].shape
  res=[]
  if act_inst[l][I][J][K][L]>0: ## we know what to do
    ## C1:
    constraint = [[], []]
    constraint[0].append(var_names[the_index][I][J][K][L])
    constraint[1].append(1)
    constraint[0].append(var_names[the_index-1][I][J][K][L])
    constraint[1].append(-1)
    res.append([constraint, 0, 'E', ''])
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('E')
    #constraint_names.append('')
    ## C2: >=0
    constraint = [[], []]
    constraint[0].append(var_names[the_index-1][I][J][K][L])
    constraint[1].append(1)
    res.append([constraint, epsilon, 'G', ''])
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('G')
    #constraint_names.append('')
  else:
    ## C1:
    constraint = [[], []]
    constraint[0].append(var_names[the_index][I][J][K][L])
    constraint[1].append(1)
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('E')
    #constraint_names.append('')
    res.append([constraint, 0, 'E', ''])
    ## C2: <=0
    constraint = [[], []]
    constraint[0].append(var_names[the_index-1][I][J][K][L])
    constraint[1].append(1)
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('L')
    #constraint_names.append('')
    res.append([constraint, -epsilon, 'L', ''])
  return res

def build_dense_constraint(the_index, ll, I, J, act_inst, var_names, has_input_layer):
  osp=var_names[the_index].shape
  res=[]

  l=ll
  if not has_input_layer: l=ll-1

  if act_inst[l][I][J]>0: ## do something
    ## C1:
    constraint = [[], []]
    constraint[0].append(var_names[the_index][I][J])
    constraint[1].append(1)
    constraint[0].append(var_names[the_index-1][I][J])
    constraint[1].append(-1)
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('E')
    #constraint_names.append('')
    res.append([constraint, 0, 'E', ''])
    ## C2: >=0
    constraint = [[], []]
    constraint[0].append(var_names[the_index-1][I][J])
    constraint[1].append(1)
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('G')
    #constraint_names.append('')
    res.append([constraint, epsilon, 'G', ''])
  else:
    ## C1:
    constraint = [[], []]
    constraint[0].append(var_names[the_index][I][J])
    constraint[1].append(1)
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('E')
    #constraint_names.append('')
    res.append([constraint, 0, 'E', ''])
    ## C2: <=0
    constraint = [[], []]
    constraint[0].append(var_names[the_index-1][I][J])
    constraint[1].append(1)
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('L')
    #constraint_names.append('')
    res.append([constraint, -epsilon, 'L', ''])

  return res

def build_conv_constraint_neg(the_index, ll, I, J, K, L, act_inst, var_names, has_input_layer):
  if (act_inst[ll][I][J][K][L]>0):
    print ('activated neuron')
    sys.exit(0)
  #print (' == build conv constraints == ', the_index, l)
  #print (var_names[the_index].shape, var_names[the_index-1].shape)
  #print (var_names[0].shape)
  #print (var_names[1].shape)
  #print (var_names[2].shape)
  l=ll
  osp=var_names[the_index].shape
  res=[]
  if not(act_inst[l][I][J][K][L]>0): ## we know what to do
    ## C1:
    constraint = [[], []]
    constraint[0].append(var_names[the_index][I][J][K][L])
    constraint[1].append(1)
    constraint[0].append(var_names[the_index-1][I][J][K][L])
    constraint[1].append(-1)
    res.append([constraint, 0, 'E', ''])
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('E')
    #constraint_names.append('')
    ## C2: >=0
    constraint = [[], []]
    constraint[0].append(var_names[the_index-1][I][J][K][L])
    constraint[1].append(1)
    res.append([constraint, epsilon, 'G', ''])
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('G')
    #constraint_names.append('')
  else:
    ## C1:
    constraint = [[], []]
    constraint[0].append(var_names[the_index][I][J][K][L])
    constraint[1].append(1)
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('E')
    #constraint_names.append('')
    res.append([constraint, 0, 'E', ''])
    ## C2: <=0
    constraint = [[], []]
    constraint[0].append(var_names[the_index-1][I][J][K][L])
    constraint[1].append(1)
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('L')
    #constraint_names.append('')
    res.append([constraint, -epsilon, 'L', ''])
  return res

def build_dense_constraint_neg(the_index, ll, I, J, act_inst, var_names, has_input_layer):
  osp=var_names[the_index].shape
  res=[]

  l=ll
  if not has_input_layer: l=ll-1

  if not (act_inst[l][I][J]>0): ## do something
    ## C1:
    constraint = [[], []]
    constraint[0].append(var_names[the_index][I][J])
    constraint[1].append(1)
    constraint[0].append(var_names[the_index-1][I][J])
    constraint[1].append(-1)
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('E')
    #constraint_names.append('')
    res.append([constraint, 0, 'E', ''])
    ## C2: >=0
    constraint = [[], []]
    constraint[0].append(var_names[the_index-1][I][J])
    constraint[1].append(1)
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('G')
    #constraint_names.append('')
    res.append([constraint, epsilon, 'G', ''])
  else:
    ## C1:
    constraint = [[], []]
    constraint[0].append(var_names[the_index][I][J])
    constraint[1].append(1)
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('E')
    #constraint_names.append('')
    res.append([constraint, 0, 'E', ''])
    ## C2: <=0
    constraint = [[], []]
    constraint[0].append(var_names[the_index-1][I][J])
    constraint[1].append(1)
    #constraints.append(constraint)
    #rhs.append(0)
    #constraint_senses.append('L')
    #constraint_names.append('')
    res.append([constraint, -epsilon, 'L', ''])

  return res

