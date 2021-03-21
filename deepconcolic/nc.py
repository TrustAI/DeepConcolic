from utils_io import *
from utils import *
from engine import (Input, TestTarget,
                    BoolMappedCoverableLayer, LayerLocalCriterion,
                    Criterion4RootedSearch,
                    Analyzer4RootedSearch)
from l0_encoding import L0EnabledTarget


# ---


class NcLayer (BoolMappedCoverableLayer):
  '''
  Covered layer that tracks per-neuron activation.
  '''
  pass


# ---


class NcTarget (NamedTuple, L0EnabledTarget, TestTarget):
  """Inherits :class:`engine.TestTarget` as well."""
  layer: NcLayer
  position: Tuple[int, ...]


  def cover(self, acts) -> None:
    self.layer.cover_neuron (self.position[1:])


  def __repr__(self) -> str:
    return 'activation of {} in {}'.format(xtuple (self.position[1:]),
                                           self.layer)


  def log_repr(self) -> str:
    return '#layer: {} #pos: {}'.format(self.layer.layer_index,
                                        xtuple (self.position[1:]))


  def eval_inputs (self, inputs: Sequence[Input], eval_batch = None) \
      -> Sequence[float]:
    """
    Measures how a new input `t` improves towards fulfilling the
    target.  A negative returned value indicates that no progress is
    being achieved by the given input.
    """
    acts = eval_batch (inputs, layer_indexes = (self.layer.layer_index,))
    acts = acts[self.layer.layer_index][(Ellipsis,) + self.position[1:]]
    return acts


  def valid_inputs (self, evals: Sequence[float]) -> Sequence[bool]:
    return evals > 0


# ---


class NcAnalyzer (Analyzer4RootedSearch):
  '''
  Analyzer that finds inputs by focusing on a target within a
  designated layer.
  '''

  @abstractmethod
  def search_input_close_to(self, x: Input, target: NcTarget) -> Optional[Tuple[float, Input]]:
    raise NotImplementedError


# ---


class NcCriterion (LayerLocalCriterion, Criterion4RootedSearch):
  """
  Neuron coverage criterion
  """

  def __init__(self, clayers: Sequence[NcLayer], analyzer: NcAnalyzer, **kwds):
    assert isinstance (analyzer, NcAnalyzer)
    super().__init__(clayers = clayers, analyzer = analyzer, **kwds)


  def __repr__(self):
    return "NC"


  def find_next_rooted_test_target(self) -> Tuple[Input, NcTarget]:
    cl, nc_pos, nc_value, test_case = self.get_max ()
    cl.inhibit_activation (nc_pos)
    return test_case, NcTarget(cl, nc_pos[1:])


# ---


from engine import setup as engine_setup, Engine

def setup (test_object = None,
           setup_analyzer: Callable[[dict], NcAnalyzer] = None,
           criterion_args: dict = {},
           **kwds) -> Engine:
  """
  Helper to build an engine for neuron-coverage (using
  :class:`NcCriterion` and an analyzer constructed using
  `setup_analyzer`).

  Extra arguments are passed to `setup_analyzer`.
  """

  setup_layer = (
    lambda l, i, **kwds: NcLayer (layer = l, layer_index = i,
                                  feature_indices = test_object.feature_indices,
                                  **kwds))
  cover_layers = get_cover_layers (test_object.dnn, setup_layer,
                                   layer_indices = test_object.layer_indices,
                                   exclude_direct_input_succ = False)
  return engine_setup (test_object = test_object,
                       cover_layers = cover_layers,
                       setup_analyzer = setup_analyzer,
                       setup_criterion = NcCriterion,
                       criterion_args = criterion_args,
                       **kwds)


# ---
