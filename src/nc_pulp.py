import pulp
try:
  import cplex
  cplex_flag=True
except:
  cplex_flag=False
from pulp import *
import sys
import numpy as np
from utils import *
from pulp_encoding import *
import copy

epsilon=1.0/(255)

def negate(dnn, act_inst, test, nc_layer, nc_pos, base_prob_, var_names_, LB=0., UB=1.):
  to_stop_index=-1
  ##
  base_prob=base_prob_.copy()
  var_names=copy.copy(var_names_)

  # distance variable 
  d_var=LpVariable('d', lowBound=np.random.uniform(0./255, 1./255), upBound=UB)
  #d_var=LpVariable('d', lowBound=1./255, upBound=UB)
  base_prob+=d_var

  ## so, we go directly to convolutional input layer
  for I in range(0, var_names[0].shape[0]):
    for J in range(0, var_names[0].shape[1]):
      for K in range(0, var_names[0].shape[2]):
        for L in range(0, var_names[0].shape[3]):
          vn=var_names[0][I][J][K][L]
          v=test[I][J][K][L]
          # x <= x0 + d
          LpAffineExpression_list=[]
          LpAffineExpression_list.append((d_var, -1))
          LpAffineExpression_list.append((vn, +1))
          c = LpAffineExpression(LpAffineExpression_list)
          constraint = LpConstraint(c, LpConstraintLE, 'x_0_{0}_{1}_{2}_{3}<=x0+d'.format(I, J, K, L), float(v))
          base_prob+=constraint

          # x >= x0 - d
          LpAffineExpression_list.append((d_var, +1))
          LpAffineExpression_list.append((vn, +1))
          c = LpAffineExpression(LpAffineExpression_list)
          constraint = LpConstraint(c, LpConstraintGE, 'x_0_{0}_{1}_{2}_{3}>=x0+d'.format(I, J, K, L), float(v))
          base_prob+=constraint

          # x<=1
          LpAffineExpression_list=[]
          LpAffineExpression_list.append((vn, +1))
          c = LpAffineExpression(LpAffineExpression_list)
          constraint = LpConstraint(c, LpConstraintLE, 'x_0_{0}_{1}_{2}_{3}<=ub'.format(I, J, K, L), float(UB))
          base_prob+=constraint
          ## x>=0
          LpAffineExpression_list=[]
          LpAffineExpression_list.append((vn, +1))
          c = LpAffineExpression(LpAffineExpression_list)
          constraint = LpConstraint(c, LpConstraintGE, 'x_0_{0}_{1}_{2}_{3}>=lb'.format(I, J, K, L), float(LB))
          base_prob+=constraint

  ## to collect activation constraints
  has_input_layer=is_input_layer(dnn.layers[0])
  the_index=0
  stopped=False
  for l in range(0, len(dnn.layers)):

    if stopped: break

    if l==len(dnn.layers)-1: continue

    layer=dnn.layers[l]

    print (' == collecting act constraints: layer {0}, {1} =='.format(l, layer))

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
                to_stop_index=the_index
                #print ('to_stop_index', to_stop_index)
                #print ('stopped:', l, I, J, K, L)
              elif not to_stop:
                res=build_conv_constraint(the_index, l, I, J, K, L, act_inst, var_names, has_input_layer)
              else: continue
              for iindex in range(0, len(res)):
                base_prob+=res[iindex]

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
            to_stop_index=the_index
            #print ('\nstopped:', l, I, J)
            res=build_dense_constraint_neg(the_index, l, I, J, act_inst, var_names, has_input_layer)
          elif not to_stop:
            res=build_dense_constraint(the_index, l, I, J, act_inst, var_names, has_input_layer)
          else: continue
          for iindex in range(0, len(res)):
            base_prob+=res[iindex]

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
                  to_stop_index=the_index
                  #print ('stopped:', l, I, J, K, L)
                elif not to_stop:
                  res=build_conv_constraint(the_index, l, I, J, K, L, act_inst, var_names, has_input_layer)
                else: continue
                for iindex in range(0, len(res)):
                  base_prob+=res[iindex]
        
      else:
        for I in range(0, osp[0]):
          if stopped: break
          for J in range(0, osp[1]):
            if stopped: break
            if to_stop and I==npos[0][0] and J==npos[0][1]:
              res=build_dense_constraint_neg(the_index, l, I, J, act_inst, var_names, has_input_layer)
              stopped=True
              to_stop_index=the_index
              #print ('stopped:', l, I, J)
            elif not to_stop:
              res=build_dense_constraint(the_index, l, I, J, act_inst, var_names, has_input_layer)
            else: continue
            for iindex in range(0, len(res)):
              base_prob+=res[iindex]
 
    elif is_maxpooling_layer(layer):
      the_index+=1
      pool_size = layer.pool_size
      osp=var_names[the_index].shape
      for I in range(0, osp[1]):
        for J in range(0, osp[2]):
          for K in range(0, osp[3]):
            max_found = False
            for II in range(I * pool_size[0], (I + 1) * pool_size[0]):
              for JJ in range(J * pool_size[1], (J + 1) * pool_size[1]):
                #constraint = [[], []]
                #constraint[0].append(var_names[the_index][0][I][J][K])
                #constraint[1].append(1)
                #constraint[0].append(var_names[the_index-1][0][II][JJ][K])
                #constraint[1].append(-1)
                #constraints.append(constraint)
                #rhs.append(0)
                #constraint_senses.append('G')
                #constraint_names.append('')
                LpAffineExpression_list=[]
                LpAffineExpression_list.append((var_names[the_index][0][I][J][K], +1))
                LpAffineExpression_list.append((var_names[the_index-1][0][II][JJ][K], -1))
                c = LpAffineExpression(LpAffineExpression_list)
                constraint = LpConstraint(c, LpConstraintGE, '', 0.)
                base_prob+=constraint
                ll=l
                #if not has_input_layer: ll=l-1 
                if ((not max_found) and act_inst[ll][0][I][J][K] == act_inst[ll - 1][0][II][JJ][K]):
                  max_found = True
                  #constraint = [[], []]
                  #constraint[0].append(var_names[the_index][0][I][J][K])
                  #constraint[1].append(1)
                  #constraint[0].append(var_names[the_index-1][0][II][JJ][K])
                  #constraint[1].append(-1)
                  #constraints.append(constraint)
                  #rhs.append(0)
                  #constraint_senses.append('E')
                  #constraint_names.append('')
                  LpAffineExpression_list=[]
                  LpAffineExpression_list.append((var_names[the_index][0][I][J][K], +1))
                  LpAffineExpression_list.append((var_names[the_index-1][0][II][JJ][K], -1))
                  c = LpAffineExpression(LpAffineExpression_list)
                  constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
                  base_prob+=constraint
            if not max_found:
              print ('not max fuond')
              sys.exit(0)

    elif is_flatten_layer(layer):
      the_index+=1
    else:
      print ('Unknown layer', layer)
      sys.exit(0)
  
  print ('### to solve...')
  lp_status_b=True
  if cplex_flag:
    print ('### Using CPLEX backend')
    base_prob.solve(CPLEX(timeLimit=5*60))
    #base_prob.solve()
  else:
    print ('### Using default CBC backend')
    print ('### WARNING: CBC does not support time limit for the LP solving procedure')
    base_prob.solve()
  print ('### solved!')

  lp_status=LpStatus[base_prob.status]
  if lp_status!='Optimal': lp_status_b=False
  #print ('Status:', lp_status, lp_status_b)

  if not lp_status_b:
    return False, -1, None

  d_v=pulp.value(d_var)
  print ('min distance:', d_v)
  #if len(npos[0])>3:
  #  print ('@@@', to_stop_index, npos[0][0], npos[0][1], npos[0][2], npos[0][3])
  #  print (var_names[to_stop_index].shape)
  #  print (pulp.value(var_names[to_stop_index][npos[0][0]][npos[0][1]][npos[0][2]][npos[0][3]]))
  #else:
  #  print ('@@@', to_stop_index, npos[0][0], npos[0][1])
  #  print (var_names[to_stop_index].shape)
  #  print (pulp.value(var_names[to_stop_index][npos[0][0]][npos[0][1]]))
  #  print (pulp.value(var_names[to_stop_index-1][npos[0][0]][npos[0][1]]))
  new_x = np.zeros((var_names[0].shape[1], var_names[0].shape[2], var_names[0].shape[3]))
  for I in range(0, var_names[0].shape[1]):
    for J in range(0, var_names[0].shape[2]):
      for K in range(0, var_names[0].shape[3]):
        v = pulp.value(var_names[0][0][I][J][K])
        ##if v-LB<-epsilon or v-UB>epsilon:
        #if v<LB or v>UB:
        #    print ('\n\n**** WARNING *** THIS IS THE pixel value', v)
        #    #sys.exit(0)
        #    return False, -1, None
        ##if v<LB: v=LB
        ##if v>UB: v=UB
        new_x[I][J][K] = v

  #save_an_image(new_x,'new_x','./')
  return lp_status_b, d_v, new_x
