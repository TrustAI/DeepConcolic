from abc import abstractmethod
from typing import *
import sys
import numpy as np
from pulp import *
from utils import *

epsilon=0.0001
UPPER_BOUND=100000000
LOWER_BOUND=-100000000


# ---


class PulpLayerOutput:

  def __init__(self, **kwds):
    super().__init__(**kwds)


  @abstractmethod
  def pulp_out_exprs(self):
    raise NotImplementedError


# ---


class BasicPulpInputLayerEncoder (PulpLayerOutput):

  def __init__(self, var_names):
    self.var_names = var_names

  def pulp_in_vars(self):
    return self.var_names[0]

  def pulp_out_exprs(self):
    return self.var_names


# ---


class PulpLayerEncoder:

  def __init__(self, **kwds):
    super().__init__(**kwds)


  @abstractmethod
  def pulp_gen_vars(self, idx: int, var_names: dict) -> int:
    raise NotImplementedError


  @abstractmethod
  def pulp_gen_base_constraints(self, base_prob: pulp.LpProblem, base_prob_dict: dict,
                                prev: PulpLayerOutput) -> None:
    raise NotImplementedError


  @abstractmethod
  def pulp_replicate_activations(self, base_prob, ap_x, prev: PulpLayerOutput,
                                 exclude = (lambda _: False)) -> None:
    raise NotImplementedError


# ---


class PulpStrictLayerEncoder (PulpLayerEncoder, PulpLayerOutput):

  def __init__(self, l, layer, **kwds):
    super().__init__(**kwds)
    self.layer_index = l
    self.layer = layer


  def pulp_gen_vars(self, idx, var_names):
    layer = self.layer
    tp1 ('Creating base variables for layer {} ({})'.format(layer.name, self.layer_index))

    # ## Create variables for INPUT neurons
    # if self.layer_index == 0:
    #   # if is_input_layer(layer) or is_conv_layer(layer):
    #   idx = gen_vars(idx, layer.input.shape, var_names)
    #   # else:
    #   #   sys.exit('We assume the first layer to be input or conv...')

    ## create variables for layer OUTPUT neurons
    if is_input_layer(layer):
      # already done: just check the input shape?
      if len(layer.input.shape) <= 2:
        sys.exit('We assume the input layer to be conv...')

    elif is_conv_layer(layer):
      idx = gen_vars(idx, layer.output.shape, var_names)
      if activation_is_relu (layer):    # 'conv+relu'
        idx = gen_vars(idx, layer.output.shape, var_names)

    elif is_dense_layer(layer):
      idx = gen_vars(idx, layer.output.shape, var_names)
      if activation_is_relu (layer):    # 'dense+relu'
        idx = gen_vars(idx, layer.output.shape, var_names)

    elif is_activation_layer(layer):
      if not activation_is_relu (layer):
        p1 ('Assuming {} is ReLU ({})'.format(layer.name, self.layer_index))
      idx = gen_vars(idx, layer.output.shape, var_names)

    elif is_maxpooling_layer(layer):
      idx = gen_vars(idx, layer.output.shape, var_names)

    elif is_flatten_layer(layer):
      # NB: why not use output shape?
      idx = gen_vars(idx, layer.input.shape, var_names, flatten = True)
  
    else:
      sys.exit ('Unknown layer: layer {0}, {1}'.format(self.layer_index, layer.name))

    self.output_var_names = var_names[-1]
    return idx


  def pulp_out_exprs(self):
    return self.output_var_names


  def pulp_gen_base_constraints(self, base_prob, base_prob_dict, prev):
    layer = self.layer
    tp1 ('Creating base constraints for layer {} ({})'
         .format(layer.name, self.layer_index))

    assert isinstance (prev, PulpLayerOutput)
    in_exprs = prev.pulp_out_exprs ()
    out_vars = self.output_var_names
    isp = in_exprs.shape
    osp = out_vars.shape

    if is_input_layer(layer):
      ## nothing to constrain for InputLayer
      pass

    elif is_conv_layer(layer):
      weights = layer.get_weights ()[0]
      biases = layer.get_weights ()[1]
      for nidx in np.ndindex (osp):
        out_var = self.output_var_names[nidx]
        affine_expr = [(out_var, -1)]
        for kidx in np.ndindex(layer.kernel_size):
          for KK in range(0, weights.shape[-1]):
            try:
              in_expr = in_exprs[0][nidx[1]+kidx[0]][nidx[2]+kidx[1]][KK]
              affine_expr.append((in_expr, float(weights[kidx][KK][nidx[-1]])))
            except:
              ## padding
              pass
        base_prob += LpConstraint(LpAffineExpression(affine_expr),
                                  LpConstraintEQ,
                                  'c_name_{0}'.format(out_var),
                                  -float(biases[nidx[-1]]))
 
      if activation_is_relu (layer):
          base_prob_dict[self.layer_index] = base_prob.copy()

    elif is_dense_layer(layer):
      weights = layer.get_weights ()[0]
      biases = layer.get_weights ()[1]
      for nidx in np.ndindex(osp):
        out_var = self.output_var_names[nidx]
        affine_expr = [(out_var, -1)]
        for II in range(0, isp[-1]):
          affine_expr.append((in_exprs[0][II], float(weights[II][nidx[-1]])))
        base_prob += LpConstraint(LpAffineExpression(affine_expr),
                                  LpConstraintEQ,
                                  'c_name_{0}'.format(out_var),
                                  -float(biases[nidx[-1]]))

      if activation_is_relu (layer):
        base_prob_dict[self.layer_index] = base_prob.copy()

    elif is_flatten_layer(layer):
      tot = isp[1]*isp[2]*isp[3]
      for I in range(0, tot):
        d0 = int(I)//(int(isp[2])*int(isp[3]))
        d1 = (int(I)%(int(isp[2])*int(isp[3])))//int(isp[3])
        d2 = int(I)-int(d0)*(int(isp[2])*int(isp[3]))-d1*int(isp[3])
        out_var = self.output_var_names[0][I]
        affine_expr = [(out_var, -1),
                       (in_exprs[0][int(d0)][int(d1)][int(d2)], +1)]
        base_prob += LpConstraint(LpAffineExpression(affine_expr),
                                  LpConstraintEQ,
                                  'c_name_{0}'.format(out_var),
                                  0)

    elif is_maxpooling_layer(layer):
      pass

    elif is_activation_layer (layer):   # Assuming ReLU activation
      base_prob_dict[self.layer_index] = base_prob.copy()

    else:
      sys.exit ('Unknown layer: layer {0}, {1}'.format(self.layer_index, layer.name))


  def pulp_replicate_activations(self, base_prob, ap_x, prev: PulpLayerOutput,
                                 exclude = (lambda _: False)) -> None:
    layer = self.layer
    if (is_conv_layer (layer) and not activation_is_relu (layer) or
        is_dense_layer (layer) and not activation_is_relu (layer) or
        is_input_layer (layer) or
        is_flatten_layer (layer)):
      return                            # skip

    out_vars = self.output_var_names
    in_exprs = prev.pulp_out_exprs ()
    osp = out_vars.shape

    if (is_conv_layer (layer) or
        is_activation_layer (layer) and len(osp) > 2):
      for oidx in np.ndindex (osp):
        if exclude (oidx): continue
        for r in build_conv_constraint(out_vars, in_exprs, oidx, ap_x[self.layer_index]):
          base_prob += r

    elif (is_dense_layer (layer) or
          is_activation_layer (layer) and len(osp) <= 2):
      for oidx in np.ndindex (osp):
        if exclude (oidx): continue
        for r in build_dense_constraint(out_vars, in_exprs, oidx, ap_x[self.layer_index]):
          base_prob += r

    elif is_maxpooling_layer (layer):
      # XXX: ignoreing oidx here...
      pool_size = layer.pool_size
      for oidx in np.ndindex (osp):
        for II in range(oidx[0] * pool_size[0], (oidx[0] + 1) * pool_size[0]):
          for JJ in range(oidx[1] * pool_size[1], (oidx[1] + 1) * pool_size[1]):
            c = LpAffineExpression([(out_vars[0][oidx[0]][oidx[1]][oidx[2]], +1),
                                    (in_exprs[0][II][JJ][oidx[2]], -1)])
            base_prob += LpConstraint(c, LpConstraintGE, '', 0.)

    else:
      sys.exit ('Unknown layer: layer {0}, {1}'.format(self.layer_index, layer.name))


  def pulp_negate_activation(self, base_prob, ap_x, oidx, prev: PulpLayerOutput) -> None:
    layer = self.layer
    assert not is_input_layer (layer)
    assert not is_flatten_layer (layer)
    if (is_conv_layer (layer) and not activation_is_relu (layer) or
        is_dense_layer (layer) and not activation_is_relu (layer)):
      return                            # skip

    in_exprs = prev.pulp_out_exprs ()
    out_vars = self.output_var_names
    osp = out_vars.shape

    if (is_conv_layer (layer) or
        is_activation_layer (layer) and len(osp) > 2):
      for r in build_conv_constraint_neg(out_vars, in_exprs, oidx, ap_x[self.layer_index]):
        base_prob += r

    elif (is_dense_layer (layer) or
          is_activation_layer (layer) and len(osp) <= 2):
      for r in build_dense_constraint_neg(out_vars, in_exprs, oidx, ap_x[self.layer_index]):
        base_prob += r

    elif is_maxpooling_layer (layer):
      # XXX: Ignoring oidx and constrain activation of max.
      pool_size = layer.pool_size
      max_found = False
      for oidx in np.ndindex (osp):
        for II in range(oidx[0] * pool_size[0], (oidx[0] + 1) * pool_size[0]):
          for JJ in range(oidx[1] * pool_size[1], (oidx[1] + 1) * pool_size[1]):
            if not max_found and (ap_x[self.layer_index][0][oidx[0]][oidx[1]][oidx[2]] ==
                                  ap_x[self.layer_index - 1][0][II][JJ][oidx[2]]):
              max_found = True
              c = LpAffineExpression([(out_vars[0][oidx[0]][oidx[1]][oidx[2]], +1),
                                      (in_exprs[0][II][JJ][oidx[2]], -1)])
              base_prob += LpConstraint(c, LpConstraintEQ, '', 0.)

    else:
      sys.exit ('Unknown layery: layer {0}, {1}'.format(self.layer_index, layer.name))


# ---


def setup_layer_encoders (dnn, first_layer = 0, upto = None):
  upto = -1 if upto == None else max(-1, upto + 1)
  lc, var_names = [], []
  ## Create variables for INPUT neurons
  idx = gen_vars(0, dnn.layers[0].input.shape, var_names)
  for l, layer in enumerate(dnn.layers[first_layer:upto]):
    lcl = PulpStrictLayerEncoder (l, layer)
    idx = lcl.pulp_gen_vars (idx, var_names)
    lc.append (lcl)
  return lc, BasicPulpInputLayerEncoder (var_names[0]), var_names


# ---


def create_base_problem (layer_encoders, input_layer_encoder):
  base_prob = LpProblem("base_prob", LpMinimize)
  base_prob_dict = dict()
  # prev = BasicPulpInputLayerEncoder(var_names[0])
  prev = input_layer_encoder
  for l in layer_encoders:
    l.pulp_gen_base_constraints (base_prob, base_prob_dict, prev)
    prev = l
  return base_prob_dict


# ---


# ## to create the base constraint for any AP
# def create_base_prob(dnn, first_layer = 0, upto = None):
#   upto = -1 if upto == None else max(-1, upto + 1)

#   base_prob = LpProblem("base_prob", LpMinimize)
#   var_names=[]
#   idx = 0

#   # Enumerate all but the last layer, as there is no intention to
#   # DIRECTLY change the classification label.
#   for l, layer in enumerate(dnn.layers[first_layer:upto]):
#     tp1 ('Creating base variables for layer {} ({})'
#          .format(layer.name, l))

#     ## Create variables for INPUT neurons
#     if l == first_layer:
#       # if is_input_layer(layer) or is_conv_layer(layer):
#       idx = gen_vars(idx, layer.input.shape, var_names)
#       # else:
#       #   sys.exit('We assume the first layer to be input or conv...')
  
#     ## create variables for layer OUTPUT neurons
#     if is_input_layer(layer):
#       # already done: just check the input shape?
#       if len(layer.input.shape) <= 2:
#         sys.exit('We assume the input layer to be conv...')

#     elif is_conv_layer(layer):
#       idx = gen_vars(idx, layer.output.shape, var_names)
#       if activation_is_relu (layer):    # 'conv+relu'
#         idx = gen_vars(idx, layer.output.shape, var_names)

#     elif is_dense_layer(layer):
#       idx = gen_vars(idx, layer.output.shape, var_names)
#       if activation_is_relu (layer):    # 'dense+relu'
#         idx = gen_vars(idx, layer.output.shape, var_names)

#     elif is_activation_layer(layer):
#       if not activation_is_relu (layer):
#         p1 ('Assuming {} is ReLU ({})'.format(layer.name, l))
#       idx = gen_vars(idx, layer.output.shape, var_names)
  
#     elif is_maxpooling_layer(layer):
#       idx = gen_vars(idx, layer.output.shape, var_names)

#     elif is_flatten_layer(layer):
#       # NB: why not use output shape?
#       idx = gen_vars(idx, layer.input.shape, var_names, flatten = True)
  
#     else:
#       sys.exit('Unknown layer: layer {0}, {1}'.format(l, layer.name))
  
#   tp1 ('{} LP variables have been collected.'
#       .format(sum(x.size for x in var_names)))

#   ## now, it comes the encoding of constraints
#   ## reset idx
#   the_index = 0
#   base_prob_dict=dict()

#   # Enumerate all but the last layer:
#   for l, layer in enumerate(dnn.layers[first_layer:upto]):
#     tp1 ('Creating base constraints for layer {} ({})'
#          .format(layer.name, l))
#     # print ('prev_var_names.size =', var_names[the_index].size if the_index > 0 else 0)
#     # print ('output_var_names.size =', var_names[the_index+1].size)

#     if is_input_layer(layer):
#       ## nothing to constrain for InputLayer
#       continue
#     elif is_conv_layer(layer):
#       the_index+=1
#       osp=var_names[the_index].shape
#       kernel_size=layer.kernel_size
#       weights = layer.get_weights ()[0]
#       biases = layer.get_weights ()[1]
#       for I in range(0, osp[0]):
#         for J in range(0, osp[1]):
#           for K in range(0, osp[2]):
#             for L in range(0, osp[3]):
#               LpAffineExpression_list=[]
#               out_neuron_var_name=var_names[the_index][I][J][K][L]
#               if I == 0 and J == 0 and K == 0 and L == 0:
#                 print (out_neuron_var_name, the_index-1)
#               LpAffineExpression_list.append((out_neuron_var_name, -1))
#               for II in range(0, kernel_size[0]):
#                 for JJ in range(0, kernel_size[1]):
#                   for KK in range(0, weights.shape[2]):
#                     try:
#                       in_neuron_var_name=var_names[the_index-1][0][J+II][K+JJ][KK]
#                       LpAffineExpression_list.append((in_neuron_var_name, float(weights[II][JJ][KK][L])))
#                     except:
#                       ## padding
#                       pass
#               #LpAffineconstraints.append(constraint)
#               c = LpAffineExpression(LpAffineExpression_list)
#               constraint = LpConstraint(c, LpConstraintEQ, 'c_name_{0}'.format(out_neuron_var_name), -float(biases[L]))
#               base_prob+=constraint

#       if activation_is_relu (layer):
#           the_index+=1
#           base_prob_dict[l]=base_prob.copy()

#     elif is_dense_layer(layer):
#       the_index+=1
#       isp=var_names[the_index-1].shape
#       osp=var_names[the_index].shape
#       weights = layer.get_weights ()[0]
#       biases = layer.get_weights ()[1]
#       for I in range(0, osp[0]):
#         for J in range(0, osp[1]):
#           LpAffineExpression_list=[]
#           out_neuron_var_name=var_names[the_index][I][J]
#           LpAffineExpression_list.append((out_neuron_var_name, -1))
#           for II in range(0, isp[1]):
#             in_neuron_var_name=var_names[the_index-1][0][II]
#             LpAffineExpression_list.append((in_neuron_var_name, float(weights[II][J])))
#           c = LpAffineExpression(LpAffineExpression_list)
#           constraint = LpConstraint(c, LpConstraintEQ, 'c_name_{0}'.format(out_neuron_var_name), -float(biases[J]))
#           base_prob+=constraint

#       if activation_is_relu (layer):
#         the_index+=1
#         base_prob_dict[l]=base_prob.copy()

#     elif is_flatten_layer(layer):
#       the_index+=1
#       isp=var_names[the_index-1].shape
#       osp=var_names[the_index].shape

#       tot=isp[1]*isp[2]*isp[3]
#       for I in range(0, tot):
#         d0=int(I)//(int(isp[2])*int(isp[3]))
#         d1=(int(I)%(int(isp[2])*int(isp[3])))//int(isp[3])
#         d2=int(I)-int(d0)*(int(isp[2])*int(isp[3]))-d1*int(isp[3])
#         LpAffineExpression_list=[]
#         out_neuron_var_name=var_names[the_index][0][I]
#         LpAffineExpression_list.append((out_neuron_var_name, -1))
#         in_neuron_var_name=var_names[the_index-1][0][int(d0)][int(d1)][int(d2)]
#         LpAffineExpression_list.append((in_neuron_var_name, +1))
#         c = LpAffineExpression(LpAffineExpression_list)
#         constraint = LpConstraint(c, LpConstraintEQ, 'c_name_{0}'.format(out_neuron_var_name), 0)
#         base_prob+=constraint
#     elif is_activation_layer (layer):
#       # if not activation_is_relu (layer):
#       #   # print ('### We assume ReLU activation layer... ')
#       #   continue
#       the_index+=1
#       base_prob_dict[l]=base_prob.copy()
#       continue
#     elif is_maxpooling_layer(layer):
#       the_index+=1
#       #isp=var_names[the_index-1].shape
#       #osp=var_names[the_index].shape
#       continue
#     else:
#       print ('Unknown layer', layer)
#       sys.exit(0)

#   return base_prob_dict, var_names


# ---


def gen_vars(layer_index, sp, var_names, flatten = False):
  shape = (1,) + ((np.prod (sp[1:]),) if flatten else tuple(sp[1:]))
  var_names.append (np.empty (shape, dtype = LpVariable))
  for idx in np.ndindex (*shape):
    var = LpVariable('_'.join(str(i) for i in ("x", layer_index) + idx),
                     lowBound = LOWER_BOUND, upBound = UPPER_BOUND)
    var_names[layer_index][idx] = var
  return layer_index + 1


# ---


# def constraints_for_cover_layer(constraints, clayer):
#   return constraints[clayer.layer_index + (0 if activation_is_relu (clayer.layer) else 1)]


# ---


def build_conv_constraint(out_vars, in_exprs, idx, ap_x):
  (I, J, K, L) = idx
  #if not has_input_layer: l=ll-1
  osp = out_vars.shape
  res = []
  if ap_x[idx]>0: ## we know what to do
    ## C1:
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index][I][J][K][L])
    #constraint[1].append(1)
    #constraint[0].append(var_names[the_index-1][I][J][K][L])
    #constraint[1].append(-1)
    #res.append([constraint, 0, 'E', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((out_vars[idx], +1))
    LpAffineExpression_list.append((in_exprs[idx], -1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)
    ## C2: >=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J][K][L])
    #constraint[1].append(1)
    #res.append([constraint, epsilon, 'G', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((in_exprs[idx], +1))
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
    LpAffineExpression_list.append((out_vars[idx], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)
    ## C2: <=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J][K][L])
    #constraint[1].append(1)
    #res.append([constraint, -epsilon, 'L', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((in_exprs[idx], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintLE, '', -epsilon)
    res.append(constraint)
  return res


def build_dense_constraint(out_vars, in_exprs, idx, ap_x):
  (I, J) = idx
  osp = out_vars.shape
  res = []

  #if not has_input_layer: l=ll-1

  if ap_x[idx]>0: ## do something
    ## C1:
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index][I][J])
    #constraint[1].append(1)
    #constraint[0].append(var_names[the_index-1][I][J])
    #constraint[1].append(-1)
    #res.append([constraint, 0, 'E', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((out_vars[idx], +1))
    LpAffineExpression_list.append((in_exprs[idx], -1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)
    ## C2: >=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J])
    #constraint[1].append(1)
    #res.append([constraint, epsilon, 'G', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((in_exprs[idx], +1))
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
    LpAffineExpression_list.append((out_vars[idx], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)
    ## C2: <=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J])
    #constraint[1].append(1)
    #res.append([constraint, -epsilon, 'L', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((in_exprs[idx], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintLE, '', -epsilon)
    res.append(constraint)

  return res


def build_conv_constraint_neg(out_vars, in_exprs, idx, ap_x):
  (I, J, K, L) = idx
  osp = out_vars.shape
  res = []
  if (ap_x[idx]>0):
    sys.exit('Activated neuron!')
  #if not has_input_layer: l=ll-1
  #print ('**** to confirm act value: ', ap_x[l-1][I][J][K][L])
  #print (l, I, J, K, L)
  if not(ap_x[idx]>0): ## we know what to do
    ## C1:
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index][I][J][K][L])
    #constraint[1].append(1)
    #constraint[0].append(var_names[the_index-1][I][J][K][L])
    #constraint[1].append(-1)
    #res.append([constraint, 0, 'E', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((out_vars[idx], +1))
    LpAffineExpression_list.append((in_exprs[idx], -1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)
    ## C2: >=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J][K][L])
    #constraint[1].append(1)
    #res.append([constraint, epsilon, 'G', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((in_exprs[idx], +1))
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
    LpAffineExpression_list.append((out_vars[idx], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)
    ## C2: <=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J][K][L])
    #constraint[1].append(1)
    #res.append([constraint, -epsilon, 'L', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((in_exprs[idx], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintLE, '', -epsilon)
    res.append(constraint)
  return res


def build_dense_constraint_neg(out_vars, in_exprs, idx, ap_x):
  (I, J) = idx
  osp = out_vars.shape
  res = []

  #if not has_input_layer: l=ll-1
  #print ('\n**** to confirm act value: ', ap_x[l-1][I][J])
  #print (the_index, ll, I, J)
  if (ap_x[idx]>0):
    sys.exit('Activated neuron!')

  if not (ap_x[idx]>0): ## do something
    ## C1:
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index][I][J])
    #constraint[1].append(1)
    #constraint[0].append(var_names[the_index-1][I][J])
    #constraint[1].append(-1)
    #res.append([constraint, 0, 'E', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((out_vars[idx], +1))
    LpAffineExpression_list.append((in_exprs[idx], -1))
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
    LpAffineExpression_list.append((out_vars[idx], +1))
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
    LpAffineExpression_list.append((out_vars[idx], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintEQ, '', 0.)
    res.append(constraint)
    ## C2: <=0
    #constraint = [[], []]
    #constraint[0].append(var_names[the_index-1][I][J])
    #constraint[1].append(1)
    #res.append([constraint, -epsilon, 'L', ''])
    LpAffineExpression_list=[]
    LpAffineExpression_list.append((in_exprs[idx], +1))
    c = LpAffineExpression(LpAffineExpression_list)
    constraint = LpConstraint(c, LpConstraintLE, '', -epsilon)
    res.append(constraint)

  return res
  
