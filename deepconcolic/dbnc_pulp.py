from typing import *
from utils import *
from pulp import *
from pulp_encoding import *
from amplif import AnalyzerWithLinearExtrapolation

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA

# ---

from engine import LayerLocalAnalyzer, CoverableLayer
from dbnc import BFcLayer
from dbnc import BFcTarget, BFcAnalyzer
from dbnc import BFDcTarget, BFDcAnalyzer
from lp import PulpLinearMetric, PulpSolver4DNN


class PulpBFcAbstrLayerEncoder (PulpStrictLayerEncoder):

  def __init__(self, fl: BFcLayer, *args, **kwds):
    super().__init__(*args, **kwds, nonact_layers = True)
    self.flayer = fl


  def pulp_output_component_linear_expression (self, component: int):

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
      lin_expr = lpSum ([LpAffineExpression \
                         ([(o, float (c / s))], float (- (u / s) * c - m * c))
                         for (o, u, s, m, c) in
                         zip (o.flatten (), u, s, m, c[component].T)])
    else:
      lin_expr = lpSum ([LpAffineExpression \
                         ([(o, float (c))], float (- m * c))
                         for (o, m, c) in
                         zip (o.flatten (), m, c[component].T)])

    # if True:                 # Filter-out terms where coeff < epsilon:
    #   lin_expr = LpAffineExpression ([ (x, a) for (x, a) in lin_expr.items ()
    #                                    if abs (a) >= epsilon * 10. ],
    #                                  lin_expr.constant)

    return lin_expr


  def pulp_output_feature_linear_expression (self, feature: int):
    return self.pulp_output_component_linear_expression (self.flayer.first + feature)



  def pulp_constrain_outputs_in_feature_part (self, feature: int, feature_part: int):

    lin_expr = self.pulp_output_feature_linear_expression (feature)
    low, up = self.flayer.discr.part_edges (feature, feature_part)
    assert low == -np.inf or up == np.inf or low <= up - act_epsilon

    # mean = self.flayer.discr.part_mean (feature, feature_part)

    cstrs = []
    if low != -np.inf:
      cstrs.append (LpConstraint (lin_expr, LpConstraintGE,
                                  '{}_low_{}'.format(self.flayer, feature),
                                  float (low + act_epsilon)))
    if up != np.inf:
      cstrs.append (LpConstraint (lin_expr, LpConstraintLE,
                                  '{}_up_{}'.format(self.flayer, feature),
                                  float (up - act_epsilon - lt_epsilon)))

    return cstrs



  def pulp_replicate_component_value (self, component: int, value: float, approx = False):
    lin_expr = self.pulp_output_component_linear_expression (component)
    if approx:
      return [ LpConstraint (lin_expr, LpConstraintGE,
                             '{}_low_{}'.format(self.flayer, component),
                             float (value - act_epsilon)),
               LpConstraint (lin_expr, LpConstraintLE,
                             '{}_up_{}'.format(self.flayer, component),
                             float (value + act_epsilon))]
    else:
      return [ LpConstraint (lin_expr, LpConstraintEQ,
                             '{}_eq_{}'.format(self.flayer, component),
                             value) ]


  def pulp_replicate_feature_value (self, feature: int, value: float, **kwds):
    return pulp_replicate_component_value (self.flayer.first + feature, value, **kwds)


# ---


def abstracted_layer_encoder (flayers):
  flayers = { fl.layer_index: fl for fl in flayers if isinstance (fl, BFcLayer) }
  return (lambda i, l: (PulpBFcAbstrLayerEncoder (flayers[i], i, l) if i in flayers else
                        PulpStrictLayerEncoder (i, l)))


class _BasePulpAnalyzer (LayerLocalAnalyzer, PulpSolver4DNN):

  def __init__(self,
               input_metric: PulpLinearMetric = None,
               fix_untargetted_components = False,
               **kwds):
    assert isinstance (input_metric, PulpLinearMetric)
    super().__init__(**kwds)
    self.metric = input_metric
    self.fix_untargetted_components = fix_untargetted_components


  def finalize_setup(self, clayers):
    super().setup (self.dnn, self.metric,
                   self._input_bounds, self._postproc_inputs,
                   build_encoder = abstracted_layer_encoder (clayers),
                   upto = deepest_tested_layer (self.dnn, clayers))


  def for_layer(self, cl: CoverableLayer) -> pulp.LpProblem:
    return self.base_constraints[cl.layer_index]


  def input_metric(self) -> PulpLinearMetric:
    return self.metric


  def actual_search(self, problem, x, extra_constrs, target):

    res = self.find_constrained_input (problem, self.metric, x, extra_constrs)

    if not res:
      return None
    else:
      if target.measure_progress (res[1]) < 0.0:
        return None
      return self.metric.distance (x, res[1]), res[1]


  def constrain_target_interval(self, lc, feature, feature_part, activations):
    cstrs = lc.pulp_constrain_outputs_in_feature_part (feature, feature_part)

    if self.fix_untargetted_components:
      dimred = lc.flayer.dimred_activations (activations,
                                             feature_space = False)[0]
      for component in lc.flayer.range_components ():
        cfeat = lc.flayer.feature_of_component (component)
        if cfeat is not None and cfeat == feature:
          continue
        cstrs.extend (lc.pulp_replicate_component_value \
                         (component, dimred[component], approx = False))

    return cstrs


# ---

class BFcPulpAnalyzer (_BasePulpAnalyzer, BFcAnalyzer):

  def search_input_close_to(self, x, target: BFcTarget):
    lc = self.layer_encoders[target.fnode.flayer.layer_index]
    problem = self.for_layer (target.fnode.flayer)
    activations = self.eval (x)
    cstrs = []

    prev = self.input_layer_encoder
    for le in self.layer_encoders[:target.fnode.flayer.layer_index]:
      cstrs.extend (le.pulp_replicate_behavior (activations, prev))
      prev = le

    cstrs.extend (self.constrain_target_interval ( \
      lc, target.fnode.feature, target.feature_part, activations))

    return self.actual_search (problem, x, cstrs, target)



class BFcPulpAnalyzerWithLinearExtrapolation \
          (AnalyzerWithLinearExtrapolation, BFcPulpAnalyzer):
  pass


# ---


class BFDcPulpAnalyzer (BFcPulpAnalyzer, BFDcAnalyzer):

  def search_input_close_to(self, x, target: Union[BFcTarget,BFDcTarget]):
    if isinstance (target, BFcTarget):
      return super ().search_input_close_to (x, target)

    lc0 = self.layer_encoders[target.flayer0.layer_index]
    lc1 = self.layer_encoders[target.fnode1.flayer.layer_index]
    problem = self.for_layer (target.fnode1.flayer)
    activations = self.eval (x)
    cstrs = []

    prev = self.input_layer_encoder
    for le in self.layer_encoders[:target.fnode1.flayer.layer_index]:
      cstrs.extend (le.pulp_replicate_behavior (activations, prev))
      prev = le

    for f0, f0p in enumerate (target.feature_parts0):
      cstrs.extend (lc0.pulp_constrain_outputs_in_feature_part (f0, f0p))

    cstrs.extend (self.constrain_target_interval ( \
      lc1, target.fnode1.feature, target.feature_part1, activations))

    return self.actual_search (problem, x, cstrs, target)


class BFDcPulpAnalyzerWithLinearExtrapolation \
          (AnalyzerWithLinearExtrapolation, BFDcPulpAnalyzer):
  pass


# ---
