"""
Boundary Cover of CIFAR10 based on L-infinity Norm

Author: Youcheng Sun
Email: youcheng.sun@cs.ox.ac.uk
"""

import cplex
import sys
import numpy as np

from utils import *

epsilon = 1.0 / (2 * 255)


def boundary_cover(model, activations, nc_layer, nc_index, im, bc_udi):
    var_names_vect = ['d']
    objective = [1]
    lower_bounds = [1.0 / 255]
    upper_bounds = [0.3]
    target_v = ''

    var_names = []
    for l in range(0, len(model.layers)):
        layer = model.layers[l]
        # print layer.name
        # print layer.input.shape

        if is_conv_layer(layer) or is_maxpooling_layer(layer):
            if l == 0:
                isp = layer.input.shape
                var_names.append(np.empty((1, isp[1], isp[2], isp[3]), dtype="S40"))
                for I in range(0, 1):
                    for J in range(0, isp[1]):
                        for K in range(0, isp[2]):
                            for L in range(0, isp[3]):
                                var_name = 'x_{0}_{1}_{2}_{3}_{4}'.format(l, I, J, K, L)
                                objective.append(0)
                                lower_bounds.append(-cplex.infinity)
                                upper_bounds.append(cplex.infinity)
                                var_names[l][I][J][K][L] = var_name
                                var_names_vect.append(var_name)
            isp = layer.output.shape
            var_names.append(np.empty((1, isp[1], isp[2], isp[3]), dtype="S40"))
            for I in range(0, 1):
                for J in range(0, isp[1]):
                    for K in range(0, isp[2]):
                        for L in range(0, isp[3]):
                            var_name = 'x_{0}_{1}_{2}_{3}_{4}'.format(l + 1, I, J, K, L)
                            objective.append(0)
                            lower_bounds.append(-cplex.infinity)
                            upper_bounds.append(cplex.infinity)
                            var_names[l + 1][I][J][K][L] = var_name
                            var_names_vect.append(var_name)
        elif is_dense_layer(layer):
            isp = layer.output.shape
            var_names.append(np.empty((1, isp[1]), dtype="S40"))
            for I in range(0, 1):
                for J in range(0, isp[1]):
                    var_name = 'x_{0}_{1}_{2}'.format(l + 1, I, J)
                    objective.append(0)
                    lower_bounds.append(-cplex.infinity)
                    upper_bounds.append(cplex.infinity)
                    var_names[l + 1][I][J] = var_name
                    var_names_vect.append(var_name)
        elif is_activation_layer(layer):
            if l == len(model.layers) - 1: continue  ## skip the softmax
            isp = layer.output.shape
            if len(isp) > 2:
                var_names.append(np.empty((1, isp[1], isp[2], isp[3]), dtype="S40"))
                for I in range(0, 1):
                    for J in range(0, isp[1]):
                        for K in range(0, isp[2]):
                            for L in range(0, isp[3]):
                                var_name = 'x_{0}_{1}_{2}_{3}_{4}'.format(l + 1, I, J, K, L)
                                objective.append(0)
                                lower_bounds.append(0)
                                upper_bounds.append(cplex.infinity)
                                var_names[l + 1][I][J][K][L] = var_name
                                var_names_vect.append(var_name)
            else:
                var_names.append(np.empty((1, isp[1]), dtype="S40"))
                for I in range(0, 1):
                    for J in range(0, isp[1]):
                        var_name = 'x_{0}_{1}_{2}'.format(l + 1, I, J)
                        objective.append(0)
                        lower_bounds.append(0)
                        upper_bounds.append(cplex.infinity)
                        var_names[l + 1][I][J] = var_name
                        var_names_vect.append(var_name)
        elif is_flatten_layer(layer):
            isp = model.layers[l].input.shape
            # print 'input shape: {0}'.format(isp)
            tot = isp[1] * isp[2] * isp[3]
            var_names.append(np.empty((1, tot), dtype="S40"))
            for I in range(0, 1):
                for J in range(0, tot):
                    var_name = 'x_{0}_{1}_{2}'.format(l + 1, I, J)
                    objective.append(0)
                    lower_bounds.append(0)
                    upper_bounds.append(cplex.infinity)
                    var_names[l + 1][I][J] = var_name
                    var_names_vect.append(var_name)
        else:
            print 'Un-expected layer!!!'
            print layer
            sys.exit(0)

    constraints = []
    rhs = []
    constraint_senses = []
    constraint_names = []

    for I in range(0, var_names[0].shape[0]):
        for J in range(0, var_names[0].shape[1]):
            for K in range(0, var_names[0].shape[2]):
                for L in range(0, var_names[0].shape[3]):
                    vn = var_names[0][I][J][K][L]
                    v = im[J][K][L]
                    # print '{0}<={1}+d'.format(vn, v)
                    # x<=x0+d
                    constraints.append([['d', vn], [-1, 1]])
                    # if v<0: print v
                    rhs.append(float(v))
                    constraint_senses.append("L")
                    constraint_names.append("x_0_{0}_{1}_{2}_{3}<=x0+d".format(I, J, K, L))
                    # x>=x0-d
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

    target_index = nc_layer.layer_index
    target_sp = np.unravel_index(nc_index, var_names[target_index + 1].shape)

    iw = 0
    for l in range(0, len(model.layers) - 1):
        if l > target_index: break
        # print 'layer {0}: '.format(l)
        layer = model.layers[l]
        weights = None
        biases = None
        if is_conv_layer(layer) or is_dense_layer(layer):
            weights = model.get_weights()[iw]
            biases = model.get_weights()[iw + 1]
            # print '=============================================='
            # print 'input shape: {0}'.format(layer.input.shape)
            # print 'output shape: {0}'.format(layer.output.shape)
            # print 'weights shape: {0}'.format(weights.shape)
            # print 'biases shape: {0}'.format(biases.shape)
            # print '=============================================='
            iw += 2

        isp = var_names[l].shape
        osp = var_names[l + 1].shape
        if is_conv_layer(layer):
            kernel_size = layer.kernel_size
            for I in range(0, osp[0]):
                for J in range(0, osp[1]):
                    for K in range(0, osp[2]):
                        for L in range(0, osp[3]):
                            if l == target_index and not (
                                    I == target_sp[0] and J == target_sp[1] and K == target_sp[2] and L == target_sp[
                                3]):
                                continue
                            else:
                                constraint = [[], []]
                                constraint[0].append(var_names[l + 1][I][J][K][L])
                                constraint[1].append(-1)
                                for II in range(0, kernel_size[0]):
                                    for JJ in range(0, kernel_size[1]):
                                        for KK in range(0, weights.shape[2]):
                                            constraint[0].append(var_names[l][0][J + II][K + JJ][KK])
                                            constraint[1].append(float(weights[II][JJ][KK][L]))

                                constraints.append(constraint)
                                rhs.append(-float(biases[L]))
                                constraint_senses.append('E')
                                constraint_names.append('eq: x_{0}_{1}_{2}_{3}_{4}'.format(l + 1, I, J, K, L))
                                if l == target_index and (
                                        I == target_sp[0] and J == target_sp[1] and K == target_sp[2] and L ==
                                        target_sp[3]):
                                    target_var = (var_names[l + 1][I][J][K][L])
                                    constraint = [[], []]
                                    constraint[0].append(var_names[l + 1][I][J][K][L])
                                    constraint[1].append(1)
                                    constraints.append(constraint)
                                    if bc_udi:
                                        constraint_senses.append('G')
                                        rhs.append(nc_layer.hk[I][J][K][L])
                                    else:
                                        print "#### ", activations[l][I][J][K][L], nc_layer.lk[I][J][K][L]
                                        constraint_senses.append('L')
                                        rhs.append(nc_layer.lk[I][J][K][L])
                                    # if activations[l][I][J][K][L]>0:
                                    #  print activations[l][I][J][K][L]
                                    #  constraint_senses.append('G')
                                    #  rhs.append(nc_layer.hk[I][J][K][L])
                                    #  #constraint_senses.append('L')
                                    #  #print 'to negate activated neuron...'
                                    #  #print 'negate: x_{0}_{1}_{2}_{3}_{4}'.format(l+1, I, J, K, L)
                                    #  #print activations[l][I][J][K][L]
                                    #  #print activations[l].shape
                                    #  if not udi:
                                    #    print 'udi must be specified'
                                    #    sys.exit(0)
                                    # else:
                                    #  constraint_senses.append('L')
                                    #  rhs.append(nc_layer.lk[I][J][K][L])
                                    #  if udi:
                                    #    print 'not udi must be specified'
                                    #    sys.exit(0)
                                    #  #constraint_senses.append('G')
                                    #  #print 'negate: x_{0}_{1}_{2}_{3}_{4}'.format(l+1, I, J, K, L)
                                    constraint_names.append('negate: x_{0}_{1}_{2}_{3}_{4}'.format(l + 1, I, J, K, L))


        elif is_dense_layer(layer):
            for I in range(0, osp[0]):
                for J in range(0, osp[1]):
                    if l == target_index and not (I == target_sp[0] and J == target_sp[1]):
                        continue
                    else:
                        constraint = [[], []]
                        constraint[0].append(var_names[l + 1][I][J])
                        constraint[1].append(-1)
                        for II in range(0, isp[1]):
                            constraint[0].append(var_names[l][0][II])
                            constraint[1].append(float(weights[II][J]))

                        constraints.append(constraint)
                        rhs.append(-float(biases[J]))
                        constraint_senses.append('E')
                        constraint_names.append('eq: x_{0}_{1}_{2}'.format(l + 1, I, J))
                        if l == target_index and (I == target_sp[0] and J == target_sp[1]):
                            target_var = (var_names[l + 1][I][J])
                            constraint = [[], []]
                            constraint[0].append(var_names[l + 1][I][J])
                            constraint[1].append(1)
                            constraints.append(constraint)
                            rhs.append(epsilon)
                            if activations[l][I][J] > 0:
                                constraint_senses.append('L')
                            else:
                                constraint_senses.append('G')
                            constraint_names.append('negate: x_{0}_{1}_{2}'.format(l + 1, I, J))
        elif is_flatten_layer(layer):
            tot = isp[1] * isp[2] * isp[3]
            for I in range(0, tot):
                d0 = I / (isp[2] * isp[3])
                d1 = (I % (isp[2] * isp[3])) / isp[3]
                d2 = I - d0 * (isp[2] * isp[3]) - d1 * isp[3]
                constraint = [[], []]
                constraint[0].append(var_names[l + 1][0][I])
                constraint[1].append(-1)
                constraint[0].append(var_names[l][0][d0][d1][d2])
                constraint[1].append(+1)

                constraints.append(constraint)
                constraint_senses.append('E')
                rhs.append(0)
                constraint_names.append('eq: x_{0}_{1}_{2}'.format(l + 1, 0, I))
        elif is_maxpooling_layer(layer):
            pool_size = layer.pool_size
            max_found = False
            for I in range(0, osp[1]):
                for J in range(0, osp[2]):
                    for K in range(0, osp[3]):
                        for II in range(I * pool_size[0], (I + 1) * pool_size[0]):
                            for JJ in range(J * pool_size[1], (J + 1) * pool_size[1]):
                                constraint = [[], []]
                                constraint[0].append(var_names[l + 1][0][I][J][K])
                                constraint[1].append(1)
                                constraint[0].append(var_names[l][0][II][JJ][K])
                                constraint[1].append(-1)
                                constraints.append(constraint)
                                rhs.append(0)
                                constraint_senses.append('G')
                                constraint_names.append('maxpooling:  x_{0}_{1}_{2}_{3}_{4}'.format(l + 1, 0, I, J, K))
                                if ((not max_found) and activations[l][0][I][J][K] == activations[l - 1][0][II][JJ][K]):
                                    max_found = True
                                    constraint = [[], []]
                                    constraint[0].append(var_names[l + 1][0][I][J][K])
                                    constraint[1].append(1)
                                    constraint[0].append(var_names[l][0][II][JJ][K])
                                    constraint[1].append(-1)
                                    constraints.append(constraint)
                                    rhs.append(0)
                                    constraint_senses.append('E')
                                    constraint_names.append(
                                        'maxpooling eq:  x_{0}_{1}_{2}_{3}_{4}'.format(l + 1, 0, I, J, K))
        elif is_activation_layer(layer):
            ## for simplicity, we assume that activations are ReLU
            if len(osp) > 2:
                for I in range(0, osp[1]):
                    for J in range(0, osp[2]):
                        for K in range(0, osp[3]):
                            constraint = [[], []]
                            constraint[0].append(var_names[l + 1][0][I][J][K])
                            constraint[1].append(1)
                            if activations[l][0][I][J][K] == 0:
                                constraints.append(constraint)
                                rhs.append(0)
                                constraint_senses.append('E')
                                constraint_names.append(
                                    'relu not activated:  x_{0}_{1}_{2}_{3}_{4}'.format(l + 1, 0, I, J, K))
                            else:
                                constraint[0].append(var_names[l][0][I][J][K])
                                constraint[1].append(-1)
                                constraints.append(constraint)
                                rhs.append(0)
                                constraint_senses.append('E')
                                constraint_names.append(
                                    'relu activated:  x_{0}_{1}_{2}_{3}_{4}'.format(l + 1, 0, I, J, K))
            else:
                for I in range(0, osp[1]):
                    constraint = [[], []]
                    constraint[0].append(var_names[l + 1][0][I])
                    constraint[1].append(1)
                    if activations[l][0][I] == 0:
                        constraints.append(constraint)
                        rhs.append(0)
                        constraint_senses.append('E')
                        constraint_names.append('relu not activated:  x_{0}_{1}_{2}'.format(l + 1, 0, I))
                    else:
                        constraint[0].append(var_names[l][0][I])
                        constraint[1].append(-1)
                        constraints.append(constraint)
                        rhs.append(0)
                        constraint_senses.append('E')
                        constraint_names.append('relu activated:  x_{0}_{1}_{2}'.format(l + 1, 0, I))

    try:
        # print 'the valuie of d is '
        problem = cplex.Cplex()
        problem.variables.add(obj=objective,
                              lb=lower_bounds,
                              ub=upper_bounds,
                              names=var_names_vect)
        problem.linear_constraints.add(lin_expr=constraints,
                                       senses=constraint_senses,
                                       rhs=rhs,
                                       names=constraint_names)
        # print '--before solve----'
        problem.solve()
        ##
        d = problem.solution.get_values("d")
        if d == 0:
            return False, -1, None
        print '***the target var: ', problem.solution.get_values(target_var)
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
        print 'there is one except'
        return False, -1, None
