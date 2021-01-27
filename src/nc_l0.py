from typing import *
from time import time
from norms import L0
from engine import Input
from nc import NcAnalyzer, NcTarget
from l0_encoding import L0Analyzer
import numpy as np


class NcL0Analyzer (NcAnalyzer, L0Analyzer):
  """
  Neuron-cover analyzer that is dedicated to find close inputs w.r.t
  L0 norm.
  """

  def __init__(self, l0_args = {}, **kwds):
    super().__init__(**kwds)
    self.norm = L0 (**l0_args)


  def input_metric(self) -> L0:
    return self.norm


  def search_input_close_to(self, x: Input, target: NcTarget) -> Optional[Tuple[float, Any]]:
    mani_range = 100

    tic = time()
    sorted_features = self.sort_features (x, target)
    res = self.accumulate (x, target, sorted_features, mani_range)
    elapsed = time() - tic
    #print ('\n == Elapsed time: ', elapsed)

    if res:
      act_inputs, idx_first = res
      new_input = self.refine (x, target, sorted_features, act_inputs[0], idx_first)
      new_input = self._postproc_inputs (new_input.reshape (x.shape))
      return self.norm.distance (x, new_input), new_input

    return None

