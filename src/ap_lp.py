try:
  import cplex
except:
  from solver import *
import sys
import numpy as np

from utils import *

epsilon=1.0/(255)

##  The AP method encodes a DNN instance into LP constraints
##  im:         the reference input
##  target_o:   the target output
##  d_vect:     the distance in each dimension
##  Max:        the optimisation goal (max or min)

def AP(model, activations, im, target_o, d_vect=None, Max=True):

  target_var=''
  var_names_vect=[]
  objective=[]
  lower_bounds=[]
  upper_bounds=[]
  var_names=[]

  print 'To encode the DNN instance...'

  for l in range(0, len(model.layers)):

    if l==len(model.layers)-1: continue ## skip the softmax

    layer=model.layers[l]

    if is_conv_layer(layer) or is_maxpooling_layer(layer):
      if l==0:
        isp=layer.input.shape
        var_names.append(np.empty((1, isp[1], isp[2], isp[3]), dtype="S40"))
        for I in range(0, 1):
          for J in range(0, isp[1]):
            for K in range(0, isp[2]):
              for L in range(0, isp[3]):
                var_name='x_{0}_{1}_{2}_{3}_{4}'.format(l, I, J, K, L)
                objective.append(0)
                lower_bounds.append(-cplex.infinity)
                upper_bounds.append(cplex.infinity)
                var_names[l][I][J][K][L]=var_name
                var_names_vect.append(var_name)
      isp=layer.output.shape
      var_names.append(np.empty((1, isp[1], isp[2], isp[3]), dtype="S40"))
      for I in range(0, 1):
        for J in range(0, isp[1]):
          for K in range(0, isp[2]):
            for L in range(0, isp[3]):
              var_name='x_{0}_{1}_{2}_{3}_{4}'.format(l+1, I, J, K, L)
              objective.append(0)
              lower_bounds.append(-cplex.infinity)
              upper_bounds.append(cplex.infinity)
              var_names[l+1][I][J][K][L]=var_name
              var_names_vect.append(var_name)
    elif is_dense_layer(layer):
      isp=layer.output.shape
      var_names.append(np.empty((1, isp[1]), dtype="S40"))
      for I in range(0, 1):
        for J in range(0, isp[1]):
          var_name='x_{0}_{1}_{2}'.format(l+1, I, J)
          if l==len(model.layers)-2 and J==target_o: ## to locate the target output

            target_var=var_name

            if Max:
              objective.append(-1)
            else:
              objective.append(+1)
            lower_bounds.append(-100000)
            upper_bounds.append(+100000)

          else:
            objective.append(0)
            lower_bounds.append(-cplex.infinity)
            upper_bounds.append(cplex.infinity)
          var_names[l+1][I][J]=var_name
          var_names_vect.append(var_name)
    elif is_activation_layer(layer):
      isp=layer.output.shape
      if len(isp)>2:  ##  multiple feature maps
        var_names.append(np.empty((1, isp[1], isp[2], isp[3]), dtype="S40"))
        for I in range(0, 1):
          for J in range(0, isp[1]):
            for K in range(0, isp[2]):
              for L in range(0, isp[3]):
                var_name='x_{0}_{1}_{2}_{3}_{4}'.format(l+1, I, J, K, L)
                objective.append(0)
                lower_bounds.append(0)
                upper_bounds.append(cplex.infinity)
                var_names[l+1][I][J][K][L]=var_name
                var_names_vect.append(var_name)
      else:  ##  fully connected
        var_names.append(np.empty((1, isp[1]), dtype="S40"))
        for I in range(0, 1):
          for J in range(0, isp[1]):
            var_name='x_{0}_{1}_{2}'.format(l+1, I, J)

            #if l==len(model.layers)-2 and J==target_o: ## to locate the target output

            #  target_var=var_name

            #  if Max:
            #    objective.append(+1)
            #  else:
            #    objective.append(-1)

            #else:
            objective.append(0)

            lower_bounds.append(-cplex.infinity)
            upper_bounds.append(cplex.infinity)
            var_names[l+1][I][J]=var_name
            var_names_vect.append(var_name)
    elif is_flatten_layer(layer):
      isp=model.layers[l].input.shape
      tot=isp[1]*isp[2]*isp[3]
      var_names.append(np.empty((1, tot), dtype="S40"))
      for I in range(0, 1):
        for J in range(0, tot):
          var_name='x_{0}_{1}_{2}'.format(l+1, I, J)
          objective.append(0)
          lower_bounds.append(0)
          upper_bounds.append(cplex.infinity)
          var_names[l+1][I][J]=var_name
          var_names_vect.append(var_name)
    else:
      print 'Un-expected layer!!!', layer
      sys.exit(0)

  constraints=[]
  rhs=[]
  constraint_senses=[]
  constraint_names=[]

  for I in range(0, var_names[0].shape[0]):
    for J in range(0, var_names[0].shape[1]):
      for K in range(0, var_names[0].shape[2]):
        for L in range(0, var_names[0].shape[3]):
          vn=var_names[0][I][J][K][L]
          v=im[J][K][L]
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

          if not (d_vect is None):
            # x <= v+d_vect[J][K][L]
            constraints.append([[vn], [1]])
            rhs.append(v+d_vect[J][K][L])
            constraint_senses.append("L")
            constraint_names.append("")
            # x >= v-d_vect[J][K][L]
            constraints.append([[vn], [1]])
            rhs.append(v-d_vect[J][K][L])
            constraint_senses.append("G")
            constraint_names.append("")

  iw=0
  for l in range(0, len(model.layers)-1):
    layer=model.layers[l]
    weights=None
    biases=None
    if is_conv_layer(layer) or is_dense_layer(layer):
      weights=model.get_weights()[iw]
      biases=model.get_weights()[iw+1]
      iw+=2

    isp=var_names[l].shape
    osp=var_names[l+1].shape
    if is_conv_layer(layer):

      print '## the convolutional layer {0}'.format(l)

      kernel_size=layer.kernel_size
      for I in range(0, osp[0]):
        for J in range(0, osp[1]):
          for K in range(0, osp[2]):
            for L in range(0, osp[3]):
              constraint=[[], []]
              constraint[0].append(var_names[l+1][I][J][K][L])
              constraint[1].append(-1)
              for II in range(0, kernel_size[0]):
                for JJ in range(0, kernel_size[1]):
                  for KK in range(0, weights.shape[2]):
                    constraint[0].append(var_names[l][0][J+II][K+JJ][KK])
                    constraint[1].append(float(weights[II][JJ][KK][L]))

              constraints.append(constraint)
              rhs.append(-float(biases[L]))
              constraint_senses.append('E')
              constraint_names.append('eq: x_{0}_{1}_{2}_{3}_{4}'.format(l+1, I, J, K, L))

    elif is_dense_layer(layer):

      print '## the dense layer {0}'.format(l)

      for I in range(0, osp[0]):
        for J in range(0, osp[1]):
          constraint=[[], []]
          constraint[0].append(var_names[l+1][I][J])
          constraint[1].append(-1)
          for II in range(0, isp[1]):
            constraint[0].append(var_names[l][0][II])
            constraint[1].append(float(weights[II][J]))

          constraints.append(constraint)
          rhs.append(-float(biases[J])) 
          constraint_senses.append('E')
          constraint_names.append('eq: x_{0}_{1}_{2}'.format(l+1, I, J))

    elif is_flatten_layer(layer):

      print '## the flatten layer {0}'.format(l)

      tot=isp[1]*isp[2]*isp[3]
      for I in range(0, tot):
        d0=I/(isp[2]*isp[3])
        d1=(I%(isp[2]*isp[3]))/isp[3]
        d2=I-d0*(isp[2]*isp[3])-d1*isp[3]
        constraint=[[], []]
        constraint[0].append(var_names[l+1][0][I])
        constraint[1].append(-1)
        constraint[0].append(var_names[l][0][d0][d1][d2])
        constraint[1].append(+1)

        constraints.append(constraint)
        constraint_senses.append('E')
        rhs.append(0)
        constraint_names.append('eq: x_{0}_{1}_{2}'.format(l+1, 0, I))

    elif is_maxpooling_layer(layer):

      print '## the maxpooling layer {0}'.format(l)

      pool_size=layer.pool_size
      for I in range(0, osp[1]):
        for J in range(0, osp[2]):
          for K in range(0, osp[3]):
            max_found=False
            for II in range(I*pool_size[0], (I+1)*pool_size[0]):
              for JJ in range(J*pool_size[1], (J+1)*pool_size[1]):
                constraint=[[], []]
                constraint[0].append(var_names[l+1][0][I][J][K])
                constraint[1].append(1)
                constraint[0].append(var_names[l][0][II][JJ][K])
                constraint[1].append(-1)
                constraints.append(constraint)
                rhs.append(0)
                constraint_senses.append('G')
                constraint_names.append('maxpooling:  x_{0}_{1}_{2}_{3}_{4}'.format(l+1, 0, I, J, K))
                if ((not max_found) and activations[l][0][I][J][K]==activations[l-1][0][II][JJ][K]):
                  max_found=True
                  constraint=[[], []]
                  constraint[0].append(var_names[l+1][0][I][J][K])
                  constraint[1].append(1)
                  constraint[0].append(var_names[l][0][II][JJ][K])
                  constraint[1].append(-1)
                  constraints.append(constraint)
                  rhs.append(0)
                  constraint_senses.append('E')
                  constraint_names.append('maxpooling eq:  x_{0}_{1}_{2}_{3}_{4}'.format(l+1, 0, I, J, K))
              #if max_found is False:
              #    print "maxpooling fails..."
              #    sys.exit(0)
    elif is_activation_layer(layer):
      ## for simplicity, we assume that activations are ReLU 

      print '## the ReLU activation layer {0}'.format(l)

      if len(osp)>2:
        for I in range(0, osp[1]):
          for J in range(0, osp[2]):
            for K in range(0, osp[3]):
              constraint=[[], []]
              constraint[0].append(var_names[l+1][0][I][J][K])
              constraint[1].append(1)
              if activations[l][0][I][J][K]==0:
                constraints.append(constraint)
                rhs.append(0)
                constraint_senses.append('E')
                constraint_names.append('relu not activated:  x_{0}_{1}_{2}_{3}_{4}'.format(l+1, 0, I, J, K))
              else:
                constraint[0].append(var_names[l][0][I][J][K])
                constraint[1].append(-1)
                constraints.append(constraint)
                rhs.append(0)
                constraint_senses.append('E')
                constraint_names.append('relu activated:  x_{0}_{1}_{2}_{3}_{4}'.format(l+1, 0, I, J, K))
      else:
        for I in range(0, osp[1]):
              constraint=[[], []]
              constraint[0].append(var_names[l+1][0][I])
              constraint[1].append(1)
              if activations[l][0][I]==0:
                constraints.append(constraint)
                rhs.append(0)
                constraint_senses.append('E')
                constraint_names.append('relu not activated:  x_{0}_{1}_{2}'.format(l+1, 0, I))
              else:
                constraint[0].append(var_names[l][0][I])
                constraint[1].append(-1)
                constraints.append(constraint)
                rhs.append(0)
                constraint_senses.append('E')
                constraint_names.append('relu activated:  x_{0}_{1}_{2}'.format(l+1, 0, I))
    else:
      print 'Unexpected layer', model.layers[l]
      sys.exit(0)

  try:
    print 'The LP encoding phase is done!'

    print 'To solve the LP constraints'

    problem=cplex.Cplex()
    problem.variables.add(obj = objective,
                          lb = lower_bounds,
                          ub = upper_bounds,
                          names = var_names_vect)
    problem.linear_constraints.add(lin_expr=constraints,
                                   senses = constraint_senses,
                                   rhs = rhs,
                                   names = constraint_names)

    ### 5 minutes threshold
    timeLimit = 60*5
    problem.parameters.timelimit.set(60*5)
    problem.solve()
    ###

    print 'Solved!!!'


    print '***the target var is {0}\n'.format(target_var)
    res=problem.solution.get_values(target_var)
    print '***the target var: ', res

    return res

  except:
    print 'There is one Exception'
    return None
