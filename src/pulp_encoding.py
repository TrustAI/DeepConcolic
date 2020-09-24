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
  """
  Abstract representation to obtain symbolic expressions that encode
  layer outputs.
  """

  @abstractmethod
  def pulp_out_exprs(self):
    raise NotImplementedError


# ---


class BasicPulpInputLayerEncoder (PulpLayerOutput):
  """
  Input layer encoder for pulp.
  """

  def __init__(self, var_names):
    self.var_names = var_names

  def pulp_in_vars(self):
    return self.var_names[0]

  def pulp_out_exprs(self):
    return self.var_names


# ---


class PulpLayerEncoder:
  """
  Generic layer encoder for pulp.
  """

  @abstractmethod
  def pulp_gen_vars(self, idx: int, var_names: dict) -> int:
    raise NotImplementedError


  @abstractmethod
  def pulp_gen_base_constraints(self, base_prob: LpProblem, base_prob_dict: dict,
                                prev: PulpLayerOutput) -> None:
    raise NotImplementedError


  @abstractmethod
  def pulp_replicate_activations(self, ap_x, prev: PulpLayerOutput,
                                 exclude = (lambda _: False)) -> Sequence[LpConstraint]:
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

    ## create variables for layer OUTPUT neurons
    if is_input_layer(layer):
      # already done: just check the input shape?
      if len(layer.input.shape) <= 2:
        sys.exit('We assume the input layer to be conv...')

    elif is_conv_layer(layer):
      idx = gen_vars(idx, layer.output.shape, var_names)
      self.u_var_names = var_names[-1]
      if activation_is_relu (layer):    # 'conv+relu'
        idx = gen_vars(idx, layer.output.shape, var_names)

    elif is_dense_layer(layer):
      idx = gen_vars(idx, layer.output.shape, var_names)
      self.u_var_names = var_names[-1]
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
      idx = gen_vars(idx, layer.output.shape, var_names)
  
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
      u_vars = self.u_var_names
      weights = layer.get_weights ()[0]
      biases = layer.get_weights ()[1]
      for nidx in np.ndindex (u_vars.shape):
        u_var = u_vars[nidx]
        affine_expr = [(u_var, -1)]
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
                                  'c_name_{0}'.format(u_var),
                                  -float(biases[nidx[-1]]))

      if activation_is_relu (layer):
        base_prob_dict[self.layer_index] = base_prob.copy()

    elif is_dense_layer(layer):
      u_vars = self.u_var_names
      weights = layer.get_weights ()[0]
      biases = layer.get_weights ()[1]
      for nidx in np.ndindex(u_vars.shape):
        u_var = u_vars[nidx]
        affine_expr = [(u_var, -1)]
        for II in range(0, isp[-1]):
          affine_expr.append((in_exprs[0][II], float(weights[II][nidx[-1]])))
        base_prob += LpConstraint(LpAffineExpression(affine_expr),
                                  LpConstraintEQ,
                                  'c_name_{0}'.format(u_var),
                                  -float(biases[nidx[-1]]))

      if activation_is_relu (layer):
        base_prob_dict[self.layer_index] = base_prob.copy()

    elif is_flatten_layer(layer):
      for idx in range (np.prod (isp[1:])):
        out_var = self.output_var_names[0][idx]
        iix = in_exprs[0][np.unravel_index (idx, in_exprs[0].shape)]
        base_prob += LpConstraint (LpAffineExpression ([(out_var, -1), (iix, +1)]),
                                   LpConstraintEQ, 'c_name_{0}'.format(out_var), 0.)

    elif is_maxpooling_layer(layer):
      pass

    elif is_activation_layer(layer):    # Assuming ReLU activation
      base_prob_dict[self.layer_index] = base_prob.copy()

    else:
      sys.exit ('Unknown layer: layer {0}, {1}'.format(self.layer_index, layer.name))


  def pulp_replicate_activations(self, ap_x, prev: PulpLayerOutput,
                                 exclude = (lambda _: False)) -> Sequence[LpConstraint]:
    layer = self.layer
    if is_input_layer (layer) or is_flatten_layer (layer):
      return []
    # assert not is_input_layer (layer)
    # assert not is_flatten_layer (layer)

    u_exprs = prev.pulp_out_exprs ()
    v_vars = self.output_var_names
    v_sp = v_vars.shape
    cstrs = []

    if (is_conv_layer (layer) or is_dense_layer (layer) or is_activation_layer (layer)):
      constrain_output = not (is_conv_layer (layer) and not activation_is_relu (layer) or
                              is_dense_layer (layer) and not activation_is_relu (layer))
      u_exprs = u_exprs if is_activation_layer (layer) else self.u_var_names
      v_vars = v_vars if constrain_output else None
      v_idx = self.layer_index - 1 if is_activation_layer (layer) else self.layer_index
      for oidx in np.ndindex (v_sp):
        if exclude (oidx): continue
        cstrs.extend (same_act (self.layer_index, v_vars, u_exprs, oidx, ap_x[v_idx]))

    elif is_maxpooling_layer (layer):
      # XXX: ignoreing oidx here...
      pool_size = layer.pool_size
      for oidx in np.ndindex (v_vars.shape):
        for II in range(oidx[0] * pool_size[0], (oidx[0] + 1) * pool_size[0]):
          for JJ in range(oidx[1] * pool_size[1], (oidx[1] + 1) * pool_size[1]):
            c = LpAffineExpression([(v_vars[0][oidx[0]][oidx[1]][oidx[2]], +1),
                                    (u_exprs[0][II][JJ][oidx[2]], -1)])
            cname = '_'.join(str(i) for i in ("mpcr__", self.layer_index, ) + oidx + (II, JJ,))
            cstrs.append(LpConstraint(c, LpConstraintGE, cname, 0.))

    else:
      sys.exit ('Unknown layer: layer {0}, {1}'.format(self.layer_index, layer.name))

    return cstrs


  def pulp_negate_activation(self, ap_x, oidx,
                             prev: PulpLayerOutput) -> Sequence[LpConstraint]:
    layer = self.layer
    assert not is_input_layer (layer)
    assert not is_flatten_layer (layer)

    u_exprs = prev.pulp_out_exprs ()
    v_vars = self.output_var_names
    cstrs = []

    if (is_conv_layer (layer) or is_dense_layer (layer) or is_activation_layer (layer)):
      constrain_output = not (is_conv_layer (layer) and not activation_is_relu (layer) or
                              is_dense_layer (layer) and not activation_is_relu (layer))
      u_exprs = u_exprs if is_activation_layer (layer) else self.u_var_names
      v_vars = v_vars if constrain_output else None
      v_idx = self.layer_index - 1 if is_activation_layer (layer) else self.layer_index
      cstrs.extend (neg_act (self.layer_index, v_vars, u_exprs, oidx, ap_x[v_idx]))

    elif is_maxpooling_layer (layer):
      # XXX: Ignoring oidx and constrain activation of max.
      pool_size = layer.pool_size
      max_found = False
      for oidx in np.ndindex (v_vars.shape):
        for II in range(oidx[0] * pool_size[0], (oidx[0] + 1) * pool_size[0]):
          for JJ in range(oidx[1] * pool_size[1], (oidx[1] + 1) * pool_size[1]):
            if not max_found and (ap_x[self.layer_index][0][oidx[0]][oidx[1]][oidx[2]] ==
                                  ap_x[self.layer_index - 1][0][II][JJ][oidx[2]]):
              max_found = True
              cname = '_'.join(str(i) for i in ("mpcn__", self.layer_index, ) + oidx + (II, JJ,))
              c = LpAffineExpression([(v_vars[0][oidx[0]][oidx[1]][oidx[2]], +1),
                                      (u_exprs[0][II][JJ][oidx[2]], -1)])
              cstrs.append(LpConstraint(c, LpConstraintEQ, cname, 0.))

    else:
      sys.exit ('Unknown layery: layer {0}, {1}'.format(self.layer_index, layer.name))

    return cstrs

# ---


strict_encoder = PulpStrictLayerEncoder


def setup_layer_encoders (dnn, build_encoder, first_layer = 0, upto = None):
  upto = -1 if upto == None else max(-1, upto + 1)
  lc, var_names = [], []
  ## Create variables for INPUT neurons
  isp = dnn.layers[0].input.shape
  idx = gen_vars(0, isp, var_names)
  for l, layer in enumerate(dnn.layers[first_layer:upto]):
    lcl = build_encoder (l, layer)
    assert isinstance (lcl, PulpLayerEncoder)
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

def same_act (base_name, v_vars, u_exprs, pos, ap_x):
  """
  Returns a set of constraints that reproduces the activation pattern
  `ap_x` at position `pos` ((0,) + neuron), based on:

  - pulp variables `v_vars` that encode neuron outputs; and

  - pulp expressions `u_exprs` that encode the pre-activation value of
    each neuron.
  """

  cname = '_'.join(str(i) for i in ("sa__", base_name, ) + pos)

  if ap_x [pos] >= 0:
    x  = [ LpConstraint (LpAffineExpression ([(v_vars[pos], +1), (u_exprs[pos], -1)]),
                        LpConstraintEQ, cname + '_eq', 0.) ] if v_vars is not None else []
    x += [ LpConstraint (LpAffineExpression ([(u_exprs[pos], +1)]),
                         LpConstraintGE, cname + '_ge', epsilon)]
    return x
  else:
    x  = [ LpConstraint (LpAffineExpression ([(v_vars[pos], +1)]),
                         LpConstraintEQ, cname + '_eq', 0.) ] if v_vars is not None else []
    x += [ LpConstraint (LpAffineExpression ([(u_exprs[pos], +1)]),
                         LpConstraintLE, cname + '_le', -epsilon) ]
    return x

def neg_act (base_name, v_vars, u_exprs, pos, ap_x):
  """
  Returns a set of constraints that negates the activation pattern
  `ap_x` at position `pos` ((0,) + neuron), based on:

  - pulp variables `v_vars` that encode neuron outputs; and

  - pulp expressions `u_exprs` that encode the pre-activation value of
    each neuron.
  """

  cname = '_'.join(str(i) for i in ("na__", base_name, ) + pos)

  # print (pos, u_exprs[pos], v_vars[pos] if v_vars is not None else None, ap_x [pos])
  if ap_x [pos] < 0:
    x = [ LpConstraint (LpAffineExpression ([(v_vars[pos], +1), (u_exprs[pos], -1)]),
                        LpConstraintEQ, cname + '_eq', 0.) ] if v_vars is not None else []
    x += [ LpConstraint (LpAffineExpression ([(u_exprs[pos], +1)]),
                         LpConstraintGE, cname + '_ge', epsilon) ]
    return x
  else:
    x = [ LpConstraint (LpAffineExpression ([(v_vars[pos], +1)]),
                        LpConstraintEQ, cname + '_eq', 0.) ] if v_vars is not None else []
    x += [ LpConstraint (LpAffineExpression ([(u_exprs[pos], +1)]),
                         LpConstraintLE, cname + '_le', -epsilon) ]
    return x

# Those are now just aliases:

def build_conv_constraint(base_name, v_vars, u_exprs, pos, ap_x):
  """Alias for :func:`same_act`"""
  return same_act (base_name, v_vars, u_exprs, pos, ap_x)

def build_dense_constraint(base_name, v_vars, u_exprs, pos, ap_x):
  """Alias for :func:`same_act`"""
  return same_act (base_name, v_vars, u_exprs, pos, ap_x)

def build_conv_constraint_neg(base_name, v_vars, u_exprs, pos, ap_x):
  """Alias for :func:`neg_act`"""
  return neg_act (base_name, v_vars, u_exprs, pos, ap_x)

def build_dense_constraint_neg(base_name, v_vars, u_exprs, pos, ap_x):
  """Alias for :func:`neg_act`"""
  return neg_act (base_name, v_vars, u_exprs, pos, ap_x)
  
