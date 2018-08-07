try:
  import cplex
except:
  from solver import *

import sys
import numpy as np

from utils import *
from lp_encoding import *
import copy

epsilon=1.0/(255)

def negate(dnn, act_inst, test, nc_layer, nc_pos, base_constraints):
  var_names=copy.copy(base_constraints.var_names)
  var_names_vect=copy.copy(base_constraints.names)
  objective=copy.copy(base_constraints.obj)
  lower_bounds=copy.copy(base_constraints.lb)
  upper_bounds=copy.copy(base_constraints.ub)

  constraints=copy.copy(base_constraints.lin_expr)
  rhs=copy.copy(base_constraints.rhs)
  constraint_senses=copy.copy(base_constraints.senses)
  constraint_names=copy.copy(base_constraints.constraint_names)


  # distance variable 
  var_names_vect.append('d')
  objective.append(1) ## 1 shall be minimum
  lower_bounds.append(1.0/255)
  upper_bounds.append(1.0)

  ## so, we go directly to convolutional input layer
  for I in range(0, var_names[0].shape[0]):
    for J in range(0, var_names[0].shape[1]):
      for K in range(0, var_names[0].shape[2]):
        for L in range(0, var_names[0].shape[3]):
          vn=var_names[0][I][J][K][L]
          v=test[I][J][K][L]
          # x <= x0 + d
          constraints.append([['d', vn], [-1, 1]])
          rhs.append(float(v))
          constraint_senses.append("L")
          constraint_names.append("x_0_{0}_{1}_{2}_{3}<=x0+d".format(I, J, K, L))
          # x >= x0 - d
          constraints.append([['d', vn], [1, 1]])
          rhs.append(float(v))
          constraint_senses.append("G")
          constraint_names.append("x_0_{0}_{1}_{2}_{3}>=x0-d".format(I, J, K, L))
          # x<=1
          constraints.append([[vn], [1]])
          rhs.append(1.0)
          constraint_senses.append("L")
          constraint_names.append("x<=1")
          ## x>=0
          constraints.append([[vn], [1]])
          rhs.append(0.0)
          constraint_senses.append("G")
          constraint_names.append("x_0_{0}_{1}_{2}_{3}>=0".format(I, J, K, L))

  ## to collect activation constraints
  has_input_layer=is_input_layer(dnn.layers[0])
  #stopped=False
  #try:
  the_index=0
  stopped=False
  for l in range(0, len(dnn.layers)):

    if stopped: break

    if l==len(dnn.layers)-1: continue
    print (' == collecting act constraints == ', l)

    layer=dnn.layers[l]

    to_stop=False
    npos=[]
    if (l==nc_layer.layer_index and act_in_the_layer(layer)=='relu'):
      to_stop=True
    elif (l==nc_layer.layer_index+1 and is_activation_layer(layer)):
      to_stop=True

    if is_input_layer(layer):
      continue
    elif is_conv_layer(layer):
      if l==0: pass
      the_index+=1
      if not (act_in_the_layer(layer)=='relu'): continue
      the_index+=1 ## here is the activation thing


      osp=var_names[the_index].shape

      if to_stop: npos.append(np.unravel_index(nc_pos, osp))
      
      for I in range(0, osp[0]):
        if stopped: break
        for J in range(0, osp[1]):
          if stopped: break
          for K in range(0, osp[2]):
            if stopped: break
            for L in range(0, osp[3]):
              if stopped: break
              if to_stop and I==npos[0][0] and J==npos[0][1] and K==npos[0][2]  and L==npos[0][3]:
                res=build_conv_constraint_neg(the_index, l, I, J, K, L, act_inst, var_names, has_input_layer)
                stopped=True
                print ('stopped:', l, I, J, K, L)
              elif not to_stop:
                res=build_conv_constraint(the_index, l, I, J, K, L, act_inst, var_names, has_input_layer)
              else: continue
              for iindex in range(0, len(res)):
                constraints.append(res[iindex][0])
                rhs.append(res[iindex][1])
                constraint_senses.append(res[iindex][2])
                constraint_names.append(res[iindex][3])

    elif is_dense_layer(layer):
      the_index+=1
      if not (act_in_the_layer(layer)=='relu'): continue
      the_index+=1
      osp=var_names[the_index].shape
      if to_stop: npos.append(np.unravel_index(nc_pos, osp))
      for I in range(0, osp[0]):
        if stopped: break
        for J in range(0, osp[1]):
          if stopped: break
          if to_stop and I==npos[0][0] and J==npos[0][1]:
            stopped=True
            print ('stopped:', l, I, J)
            res=build_dense_constraint_neg(the_index, l, I, J, act_inst, var_names, has_input_layer)
          elif not to_stop:
            res=build_dense_constraint(the_index, l, I, J, act_inst, var_names, has_input_layer)
          else: continue
          for iindex in range(0, len(res)):
            constraints.append(res[iindex][0])
            rhs.append(res[iindex][1])
            constraint_senses.append(res[iindex][2])
            constraint_names.append(res[iindex][3])

    elif is_activation_layer(layer):
      the_index+=1
      osp=var_names[the_index].shape
      if to_stop: npos.append(np.unravel_index(nc_pos, osp))
      if len(osp) > 2: 
        for I in range(0, osp[0]):
          if stopped: break
          for J in range(0, osp[1]):
            if stopped: break
            for K in range(0, osp[2]):
              if stopped: break
              for L in range(0, osp[3]):
                if stopped: break
                if to_stop and I==npos[0][0] and J==npos[0][1] and K==npos[0][2]  and L==npos[0][3]:
                  res=build_conv_constraint_neg(the_index, l, I, J, K, L, act_inst, var_names, has_input_layer)
                  stopped=True
                  print ('stopped:', l, I, J, K, L)
                elif not to_stop:
                  res=build_conv_constraint(the_index, l, I, J, K, L, act_inst, var_names, has_input_layer)
                else: continue
                for iindex in range(0, len(res)):
                  constraints.append(res[iindex][0])
                  rhs.append(res[iindex][1])
                  constraint_senses.append(res[iindex][2])
                  constraint_names.append(res[iindex][3])
        
      else:
        for I in range(0, osp[0]):
          if stopped: break
          for J in range(0, osp[1]):
            if stopped: break
            if to_stop and I==npos[0][0] and J==npos[0][1]:
              res=build_dense_constraint_neg(the_index, l, I, J, act_inst, var_names, has_input_layer)
              stopped=True
              print ('stopped:', l, I, J)
            elif not to_stop:
              res=build_dense_constraint(the_index, l, I, J, act_inst, var_names, has_input_layer)
            else: continue
            for iindex in range(0, len(res)):
              constraints.append(res[iindex][0])
              rhs.append(res[iindex][1])
              constraint_senses.append(res[iindex][2])
              constraint_names.append(res[iindex][3])
 
    elif is_maxpooling_layer(layer):
      the_index+=1
      pool_size = layer.pool_size
      max_found = False
      osp=var_names[the_index].shape
      for I in range(0, osp[1]):
        for J in range(0, osp[2]):
          for K in range(0, osp[3]):
            for II in range(I * pool_size[0], (I + 1) * pool_size[0]):
              for JJ in range(J * pool_size[1], (J + 1) * pool_size[1]):
                constraint = [[], []]
                constraint[0].append(var_names[the_index][0][I][J][K])
                constraint[1].append(1)
                constraint[0].append(var_names[the_index-1][0][II][JJ][K])
                constraint[1].append(-1)
                constraints.append(constraint)
                rhs.append(0)
                constraint_senses.append('G')
                constraint_names.append('')
                ll=l
                #if not has_input_layer: ll=l-1 
                if ((not max_found) and act_inst[ll][0][I][J][K] == act_inst[ll - 1][0][II][JJ][K]):
                  max_found = True
                  constraint = [[], []]
                  constraint[0].append(var_names[the_index][0][I][J][K])
                  constraint[1].append(1)
                  constraint[0].append(var_names[the_index-1][0][II][JJ][K])
                  constraint[1].append(-1)
                  constraints.append(constraint)
                  rhs.append(0)
                  constraint_senses.append('E')
                  constraint_names.append('')


    elif is_flatten_layer(layer):
      the_index+=1
    else:
      print ('Unknown layer', layer)
      sys.exit(0)
  
  try:
      # print 'the valuie of d is '
      print ('cplex now...')
      problem = cplex.Cplex()
      #print (len(objective), len(lower_bounds), len(upper_bounds), len(var_names_vect))
      problem.variables.add(obj=objective,
                            lb=lower_bounds,
                            ub=upper_bounds,
                            names=var_names_vect)
      problem.linear_constraints.add(lin_expr=constraints,
                                     senses=constraint_senses,
                                     rhs=rhs,
                                     names=constraint_names)
      print ('-- to solve----')
      timeLimit = 60 * 5
      problem.parameters.timelimit.set(60 * 5)
      problem.solve()
      ##
      d = problem.solution.get_values("d")
      print ('d:', d)
      if d == 0:
          return False, -1, None
      #print '***the target var: ', problem.solution.get_values(target_var)
      # print 'distance d is: {0}'.format(d)
      new_x = np.zeros((var_names[0].shape[1], var_names[0].shape[2], var_names[0].shape[3]))
      for I in range(0, var_names[0].shape[1]):
          for J in range(0, var_names[0].shape[2]):
              for K in range(0, var_names[0].shape[3]):
                  # print I, J, K
                  v = problem.solution.get_values(var_names[0][0][I][J][K])
                  # print v
                  if v < 0 or v > 1:
                      return False, -1, None
                  new_x[I][J][K] = v
      return True, d, new_x
  except:
      print ('there is one except')
      return False, -1, None

  return None, None, None
