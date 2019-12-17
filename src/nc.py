from typing import *
from utils import *
from engine import (Input, TestTarget,
                    BoolMappedCoverableLayer, LayerLocalCriterion,
                    Criterion4RootedSearch,
                    Analyzer4RootedSearch)
import numpy as np


# ---


class NcLayer (BoolMappedCoverableLayer):
  '''
  Covered layer that tracks per-neuron activation.
  '''

  def update_with_new_activations(self, act) -> None:
    if activation_is_relu (self.layer): # todo (???)
      sys.exit ('Unsupported NC-update for activation layer (bug/todo?)')
    super().update_with_new_activations (act)


# ---


class NcTarget (NamedTuple, TestTarget):
  layer: NcLayer
  position: Tuple[int, ...]


  def cover(self) -> None:
    self.layer.cover (self.position[1:])


  def __repr__(self) -> str:
    return 'activation of {} in {}'.format(xtuple (self.position[1:]),
                                           self.layer)


  def log_repr(self) -> str:
    return '#layer: {} #pos: {}'.format(self.layer.layer_index,
                                        xtuple (self.position[1:]))


# ---


class NcAnalyzer (Analyzer4RootedSearch):
  '''
  Analyzer that finds inputs by focusing on a target within a
  designated layer.
  '''

  def __init__(self, **kwds):
    super().__init__(**kwds)


  @abstractmethod
  def search_input_close_to(self, x, target: NcTarget) -> Optional[Tuple[float, Input]]:
    pass


# ---


class NcCriterion (LayerLocalCriterion, Criterion4RootedSearch):

  def __init__(self, clayers: Sequence[NcLayer], analyzer: NcAnalyzer, **kwds):
    assert isinstance (analyzer, NcAnalyzer)
    super().__init__(clayers = clayers, analyzer = analyzer, **kwds)


  def __repr__(self):
    return "NC"


  def find_next_rooted_test_target(self) -> Tuple[Input, NcTarget]:
    cl, nc_pos, nc_value, test_case = self.get_max ()
    # ppos = (lambda p: p if len(p) > 1 else p[0])(nc_pos[2:])
    # p1 ('Targeting activation of {} in {} (value = {})'.format(ppos, cl, nc_value))
    cl.inhibit_activation (nc_pos)
    return test_case, NcTarget(cl, nc_pos[1:])


# ---


from engine import setup as engine_setup

def setup (test_object = None,
           **kwds):

  setup_layer = (
    lambda l, i, **kwds: NcLayer (layer = l, layer_index = i,
                                  feature_indices = test_object.feature_indices,
                                  **kwds))
  cover_layers = get_cover_layers (test_object.dnn, setup_layer,
                                   layer_indices = test_object.layer_indices,
                                   exclude_direct_input_succ = False)
  return engine_setup (test_object = test_object,
                       cover_layers = cover_layers,
                       setup_criterion = NcCriterion,
                       **kwds)


# ---
