from typing import *
from pulp import *
from utils import *
from pulp_encoding import *
import numpy as np

# ---

from nc import NcAnalyzer, NcTarget
from lp import PulpLinearMetric, PulpSolver4DNN


class NcPulpAnalyzer (NcAnalyzer, PulpSolver4DNN):

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


  def search_input_close_to(self, x, target: NcTarget):
    problem = self.for_layer (target.layer)
    activations = eval_batch (self.dnn, np.array([x]))

    # Augment problem with activation constraints up to layer of
    # target:
    target_neuron = target.position[0]
    prev = self.input_layer_encoder
    for lc in self.layer_encoders:
      if lc.layer_index < target.layer.layer_index:
        lc.pulp_replicate_activations (problem, activations, prev)
        prev = lc
      else:
        lc.pulp_replicate_activations (problem, activations, prev,
                                       exclude = (lambda nidx: nidx == target_neuron))
        lc.pulp_negate_activation (problem, activations, target_neuron, prev)
        break

    res = self.find_constrained_input (problem, self.metric, x)
    if not res:
      return None
    else:
      dist = self.metric.distance (x, res[1])
      return dist, res[1]


# ---
