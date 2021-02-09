# Basic amplification of input features

from typing import *
from utils_io import p1, tp1
from utils import np
from engine import Input, InputsDict, Bounds, TestTarget, Analyzer

# ---

class LinearExtrapolationAmplifier:
  '''
  Basic utility that allows one to alleviate the cost of symbolic
  analysis by attempting pixel-wise linear extrapolations when
  generated inputs are re-used as candidates of the test target they
  originate from.
  '''
  def __init__(self, *args, extrapolation_factor = 1., **kwds):
    super().__init__(*args, **kwds)
    self.extrapolation_factor = extrapolation_factor
    self.memory = dict ()

  def register_derived_input (self, target: TestTarget, orig: Input, new: Input):
    if target not in self.memory:
      self.memory[target] = InputsDict ()
    # TODO: lru-style cleanup to avoid bloating
    self.memory[target][new] = orig

  def try_extrapolate (self, target: TestTarget, x: Input,
                       input_bounds: Optional[Bounds]) -> Optional[Input]:
    if target not in self.memory:
      return None
    
    origins = self.memory[target]
    if x not in origins:
      return None

    y = x - origins[x]                  # just the raw diff
    np.multiply (self.extrapolation_factor, y, out = y)
    np.add (x, y, out = y)
    if input_bounds is not None:
      np.clip (y, a_min = input_bounds.low, a_max = input_bounds.up, out = y)
    if np.allclose (x, y):
      del y
      return None
    return y

# ---

class AnalyzerWithLinearExtrapolation (Analyzer, LinearExtrapolationAmplifier):
  '''
  Analyzers that inherit this class first attempt to extrapolate based
  on previously generated inputs, and resort symbolic analysis if this
  fails to generate any input.

  This is to be combined with an actual symbolic analyzer, by
  inserting it BEFORE any actual symbolic analyzer in the class
  inheritance list.
  '''
  def __init__(self, *args, enable_linear_extrapolation = True, **kwds):
    super ().__init__(*args, **kwds)
    self.enable_linear_extrapolation = enable_linear_extrapolation


  def search_input_close_to (self, x: Input, t: TestTarget):
    if self.enable_linear_extrapolation:
      extrapolation_attempt = self.try_extrapolate (t, x, self._input_bounds)
      if extrapolation_attempt is not None:
        d = self.metric.distance (x, extrapolation_attempt)
        p1 (f'| New input obtained via linear extrapolation')
        self.register_derived_input (t, x, extrapolation_attempt)
        return d, extrapolation_attempt

    res = super ().search_input_close_to (x, t)
    if res is not None and self.enable_linear_extrapolation:
      self.register_derived_input (t, x, res[1])
    return res

# ---
