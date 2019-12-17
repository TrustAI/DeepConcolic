from typing import *
from pulp import *
from utils import *
from pulp_encoding import *
import numpy as np

# ---

from ssc import SScAnalyzer4RootedSearch, SScTarget
from lp import PulpLinearMetric, PulpSolver4DNN


class SScPulpAnalyzer (SScAnalyzer4RootedSearch, PulpSolver4DNN):

  def __init__(self, clayers,
               input_metric: PulpLinearMetric = None,
               analyzed_dnn = None,
               **kwds):
    assert isinstance (input_metric, PulpLinearMetric)
    self.metric = input_metric
    super().__init__(analyzed_dnn = analyzed_dnn, lp_dnn = analyzed_dnn,
                     upto = deepest_tested_layer (analyzed_dnn, clayers),
                     **kwds)


  def input_metric(self):
    return self.metric


  def search_input_close_to(self, x, target: SScTarget) -> Optional[Tuple[float, Any, Any]]:
    activations = eval_batch (self.dnn, np.array([x]))

    dec_layer, dec_pos, dec_neuron, cond_layer, cond_pos, cond_neuron = (
      target.decision_layer, target.decision_position, target.decision_neuron,
      target.condition_layer, target.condition_position, target.condition_neuron)
    assert cond_layer is not None
    
    problem = self.for_layer (dec_layer)

    # Augment problem with activation constraints up to condition
    # layer:
    prev = self.input_layer_encoder
    for lc in self.layer_encoders:
      if lc.layer_index < cond_layer.layer_index: # < k
        lc.pulp_replicate_activations (problem, activations, prev)
      elif lc.layer_index == cond_layer.layer_index: # == k
        lc.pulp_replicate_activations (problem, activations, prev,
                                       exclude = (lambda nidx: nidx == cond_neuron))
        lc.pulp_negate_activation (problem, activations, cond_neuron, prev)
      elif lc.layer_index == dec_layer.layer_index:  # == k + 1
        lc.pulp_negate_activation (problem, activations, dec_neuron, prev)
        break
      prev = lc

    res = self.find_constrained_input (problem, self.metric, x)
    if not res:
      return None
    else:
      dist = self.metric.distance (x, res[1])
      return dist, res[1]


# ---
