from typing import *
from pulp import *
from utils import *
from pulp_encoding import *
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA

# ---

from dbnc import BFcLayer
from lp import PulpLinearMetric, PulpSolver4DNN


class PulpBFcAbstrLayerEncoder (PulpStrictLayerEncoder):

  def __init__(self, fl: BFcLayer, *args, **kwds):
    super().__init__(*args, **kwds)
    self.flayer = fl


  def pulp_constrain_outputs_in_feature_part (self, feature: int, feature_part: int):

    assert isinstance (self.flayer.transform[-1], (PCA, FastICA))

    o = self.pulp_out_exprs ()
    m = self.flayer.transform[-1].mean_
    c = self.flayer.transform[-1].components_

    if isinstance (self.flayer.transform[0], StandardScaler):
      assert len (self.flayer.transform) == 2
      u = self.flayer.transform[0].mean_
      s = self.flayer.transform[0].scale_
      lin_expr = lpSum ([LpAffineExpression ([(o, c / s)], - (u / s) * c - m * c)
                         for (o, u, s, m, c) in
                         zip(o.flatten (), u, s, m, c[feature].T)])
    else:
      assert len (self.flayer.transform) == 1

      lin_expr = lpSum ([LpAffineExpression ([(o, c)], - m * c)
                         for (o, m, c) in
                         zip(o.flatten (), m, c[feature].T)])

    # if True:                 # Filter-out terms where coeff < epsilon:
    #   lin_expr = LpAffineExpression ([ (x, a) for (x, a) in lin_expr.items ()
    #                                    if abs (a) >= epsilon * 10. ],
    #                                  lin_expr.constant)

    low, up = self.flayer.discr.part_edges (feature, feature_part)
    assert low == -np.inf or up == np.inf or low <= up - epsilon

    # mean = self.flayer.discr.part_mean (feature, feature_part)

    cstrs = []
    if low != -np.inf:
      constr = LpConstraint (lin_expr, LpConstraintGE,
                             '{}_low_{}'.format(self.flayer, feature), low)
      cstrs.append(constr)
    if up != np.inf:
      constr = LpConstraint (lin_expr, LpConstraintLE, 
                             '{}_up_{}'.format(self.flayer, feature), up - epsilon/100.)
      cstrs.append(constr)

    return cstrs


  def pulp_replicate_activations_outside_of_feature (self, feature: int, ap_x,
                                                     prev: PulpLayerOutput):
    o = self.pulp_out_exprs ()
    c = self.flayer.transform[-1].components_[feature].reshape(o.shape)
    m = self.flayer.transform[-1].mean_.reshape(o.shape)
    return self.pulp_replicate_activations (
      ap_x, prev, exclude = lambda i: abs(c[i] # * m[i]
                                          ) > epsilon)


# ---


def abstracted_layer_encoder (flayers):
  flayers = { fl.layer_index: fl for fl in flayers if isinstance (fl, BFcLayer) }
  return (lambda i, l: (PulpBFcAbstrLayerEncoder (flayers[i], i, l) if i in flayers else
                        PulpStrictLayerEncoder (i, l)))


class _BFcPulpAnalyzer (PulpSolver4DNN):
  
  def __init__(self, input_metric: PulpLinearMetric = None, **kwds):
    assert isinstance (input_metric, PulpLinearMetric)
    super().__init__(**kwds)
    self.metric = input_metric


  def finalize_setup(self, clayers):
    super().setup (self.dnn, self.metric,
                   build_encoder = abstracted_layer_encoder (clayers),
                   upto = deepest_tested_layer (self.dnn, clayers))


  def input_metric(self):
    return self.metric

  
  def actual_search(self, problem, x, extra_constrs, target):

    res = self.find_constrained_input (problem, self.metric, x, extra_constrs)

    if not res:
      return None
    else:
      if not (target.check (res[1])):
        cp1 ('| Missed target {}'.format (target))
        # return None
      dist = self.metric.distance (x, res[1])
      return dist, res[1]
    


# ---

from dbnc import BFcTarget, BFcAnalyzer


class BFcPulpAnalyzer (_BFcPulpAnalyzer, BFcAnalyzer):

  def search_input_close_to(self, x, target: BFcTarget):
    lc = self.layer_encoders[target.fnode.flayer.layer_index]
    problem = self.for_layer (target.fnode.flayer)
    cstrs = []

    cstrs.extend(lc.pulp_constrain_outputs_in_feature_part (
      target.fnode.feature, target.feature_part))

    # cstrs.extend(lc.pulp_replicate_activations_outside_of_feature (
    #   target.fnode.feature, self.eval (x),
    #   self.layer_encoders[target.fnode.flayer.layer_index - 1]))

    return self.actual_search (problem, x, cstrs, target)

# ---

from dbnc import BFDcTarget, BFDcAnalyzer


class BFDcPulpAnalyzer (_BFcPulpAnalyzer, BFDcAnalyzer):

  def search_input_close_to(self, x, target: BFDcTarget):
    lc0 = self.layer_encoders[target.flayer0.layer_index]
    lc1 = self.layer_encoders[target.fnode1.flayer.layer_index]
    problem = self.for_layer (target.fnode1.flayer)
    cstrs = []

    cstrs.extend(lc1.pulp_constrain_outputs_in_feature_part (
      target.fnode1.feature, target.feature_part1))

    for f0, f0p in enumerate (target.feature_parts0):
      cstrs.extend(lc0.pulp_constrain_outputs_in_feature_part (
        f0, f0p))

    # cstrs.extend(lc.pulp_replicate_activations_outside_of_feature (
    #   target.fnode.feature, self.eval (x),
    #   self.layer_encoders[target.fnode.flayer.layer_index - 1]))

    return self.actual_search (problem, x, cstrs, target)


# ---
