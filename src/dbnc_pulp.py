from typing import *
from utils import *
from pulp import *
from pulp_encoding import *

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA

# ---

from engine import Analyzer
from dbnc import BFcLayer
from lp import PulpLinearMetric, PulpSolver4DNN


class PulpBFcAbstrLayerEncoder (PulpStrictLayerEncoder):

  def __init__(self, fl: BFcLayer, *args, **kwds):
    super().__init__(*args, **kwds)
    self.flayer = fl


  def pulp_output_feature_linear_expression (self, feature: int):

    transform = self.flayer.transform

    if len (transform) not in (1, 2) or \
       len (transform) == 2 and not isinstance (transform[0], StandardScaler):
      raise ValueError ('Unsupported feature extraction pipeline: {}.\n'
                        'Pipeline may only comprize a scaler (optional), '
                        'followed by either a PCA or ICA transform.'
                        .format (transform))

    if not isinstance (transform[-1], (PCA, FastICA)):
      raise ValueError ('Unsupported feature extraction transform: {}.\n'
                        'Transform may only be either PCA or ICA.'
                        .format (transform[-1]))

    o = self.pulp_out_exprs ()
    m = transform[-1].mean_
    c = transform[-1].components_

    if isinstance (transform[0], StandardScaler):
      u = transform[0].mean_
      s = transform[0].scale_
      lin_expr = lpSum ([LpAffineExpression ([(o, c / s)], - (u / s) * c - m * c)
                         for (o, u, s, m, c) in
                         zip (o.flatten (), u, s, m, c[feature].T)])
    else:
      lin_expr = lpSum ([LpAffineExpression ([(o, c)], - m * c)
                         for (o, m, c) in
                         zip (o.flatten (), m, c[feature].T)])

    # if True:                 # Filter-out terms where coeff < epsilon:
    #   lin_expr = LpAffineExpression ([ (x, a) for (x, a) in lin_expr.items ()
    #                                    if abs (a) >= epsilon * 10. ],
    #                                  lin_expr.constant)

    return lin_expr



  def pulp_constrain_outputs_in_feature_part (self, feature: int, feature_part: int):

    lin_expr = self.pulp_output_feature_linear_expression (feature)
    low, up = self.flayer.discr.part_edges (feature, feature_part)
    assert low == -np.inf or up == np.inf or low <= up - epsilon

    # mean = self.flayer.discr.part_mean (feature, feature_part)

    cstrs = []
    if low != -np.inf:
      cstrs.append (LpConstraint (lin_expr, LpConstraintGE,
                                  '{}_low_{}'.format(self.flayer, feature),
                                  low + epsilon))
    if up != np.inf:
      cstrs.append (LpConstraint (lin_expr, LpConstraintLE,
                                  '{}_up_{}'.format(self.flayer, feature),
                                  up - 2 * epsilon))

    return cstrs


  def pulp_replicate_feature_value (self, feature: int, value: float, approx = False):
    lin_expr = self.pulp_output_feature_linear_expression (feature)
    if approx:
      return [ LpConstraint (lin_expr, LpConstraintGE,
                             '{}_low_{}'.format(self.flayer, feature),
                             value - epsilon),
               LpConstraint (lin_expr, LpConstraintLE,
                             '{}_up_{}'.format(self.flayer, feature),
                             value + epsilon)]
    else:
      return [ LpConstraint (lin_expr, LpConstraintEQ,
                             '{}_eq_{}'.format(self.flayer, feature),
                             value) ]


  # def pulp_replicate_activations_outside_of_feature (self, feature: int, ap_x,
  #                                                    prev: PulpLayerOutput):
  #   o = self.pulp_out_exprs ()
  #   c = self.flayer.transform[-1].components_[feature].reshape(o.shape)
  #   m = self.flayer.transform[-1].mean_.reshape(o.shape)
  #   return self.pulp_replicate_activations (
  #     ap_x, prev, exclude = lambda i: abs(c[i] # * m[i]
  #                                         ) > epsilon)


# ---


def abstracted_layer_encoder (flayers):
  flayers = { fl.layer_index: fl for fl in flayers if isinstance (fl, BFcLayer) }
  return (lambda i, l: (PulpBFcAbstrLayerEncoder (flayers[i], i, l) if i in flayers else
                        PulpStrictLayerEncoder (i, l)))


class _BFcPulpAnalyzer (Analyzer, PulpSolver4DNN):

  def __init__(self, input_metric: PulpLinearMetric = None, **kwds):
    assert isinstance (input_metric, PulpLinearMetric)
    super().__init__(**kwds)
    self.metric = input_metric


  def finalize_setup(self, clayers):
    super().setup (self.dnn, self.metric, self._input_bounds,
                   build_encoder = abstracted_layer_encoder (clayers),
                   upto = deepest_tested_layer (self.dnn, clayers))


  def input_metric(self) -> PulpLinearMetric:
    return self.metric


  def actual_search(self, problem, x, extra_constrs, target):

    res = self.find_constrained_input (problem, self.metric, x, extra_constrs)

    if not res:
      return None
    else:
      if target.measure_progress (res[1]) <= 0.0:
        return None
      return self.metric.distance (x, res[1]), res[1]


# ---

from dbnc import BFcTarget, BFcAnalyzer


class BFcPulpAnalyzer (_BFcPulpAnalyzer, BFcAnalyzer):

  def __init__(self, fix_other_features = False, **kwds):
    super ().__init__(**kwds)
    self.fix_other_features = fix_other_features


  def search_input_close_to(self, x, target: BFcTarget):
    lc = self.layer_encoders[target.fnode.flayer.layer_index]
    problem = self.for_layer (target.fnode.flayer)
    dimred_activations = lc.flayer.dimred_activations (self.eval (x))[0]
    cstrs = []

    for feature in lc.flayer.range_features ():
      if feature == target.fnode.feature:
        cstrs.extend (lc.pulp_constrain_outputs_in_feature_part \
                      (target.fnode.feature, target.feature_part))
      elif self.fix_other_features:
        cstrs.extend (lc.pulp_replicate_feature_value \
                      (feature, dimred_activations[feature],
                       approx = False))

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

    cstrs.extend (lc1.pulp_constrain_outputs_in_feature_part \
                  (target.fnode1.feature, target.feature_part1))

    for f0, f0p in enumerate (target.feature_parts0):
      cstrs.extend (lc0.pulp_constrain_outputs_in_feature_part \
                    (f0, f0p))

    # cstrs.extend(lc.pulp_replicate_activations_outside_of_feature (
    #   target.fnode.feature, self.eval (x),
    #   self.layer_encoders[target.fnode.flayer.layer_index - 1]))

    return self.actual_search (problem, x, cstrs, target)


# ---
