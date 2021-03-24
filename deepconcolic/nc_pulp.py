from utils import *
from pulp import *
from pulp_encoding import *

# ---

from engine import LayerLocalAnalyzer, CoverableLayer, Input
from nc import NcAnalyzer, NcTarget
from lp import PulpLinearMetric, PulpSolver4DNN


class NcPulpAnalyzer (NcAnalyzer, LayerLocalAnalyzer, PulpSolver4DNN):
  """
  Pulp-based analyzer for neuron coverage.
  """

  def __init__(self, input_metric: PulpLinearMetric = None, **kwds):
    assert isinstance (input_metric, PulpLinearMetric)
    super().__init__(**kwds)
    self.metric = input_metric


  def finalize_setup(self, clayers: Sequence[CoverableLayer]):
    super().setup (self.dnn, self.metric,
                   self._input_bounds, self._postproc_inputs,
                   upto = deepest_tested_layer (self.dnn, clayers))


  def input_metric(self) -> PulpLinearMetric:
    return self.metric


  def search_input_close_to(self, x: Input, target: NcTarget) -> Optional[Tuple[float, Any]]:
    problem = self.for_layer (target.layer)
    activations = self.eval (x)
    cstrs = []

    # Augment problem with activation constraints up to layer of
    # target:
    target_neuron = target.position
    prev = self.input_layer_encoder
    for lc in self.layer_encoders:
      if lc.layer_index < target.layer.layer_index:
        cstrs.extend(lc.pulp_replicate_activations (activations, prev))
        prev = lc
      else:
        cstrs.extend(lc.pulp_replicate_activations (activations, prev,
                                                    exclude = (lambda nidx: nidx == target_neuron)))
        cstrs.extend(lc.pulp_negate_activation (activations, target_neuron, prev))
        break

    res = self.find_constrained_input (problem, self.metric, x,
                                       extra_constrs = cstrs)

    if not res:
      return None
    else:
      dist = self.metric.distance (x, res[1])
      activations2 = self.eval (res[1])
      i = target.layer.layer_index
      if (np.sign (activations2[i][target_neuron]) ==
          np.sign (activations[i][target_neuron])):
        p1 ('| Missed activation target '
            f'(original = {float (activations[i][target_neuron]):.8}, '
            f'new = {float (activations2[i][target_neuron]):.8})')
        # Keep as the new input may trigger unseen activations anyways
      return dist, res[1]


# ---
