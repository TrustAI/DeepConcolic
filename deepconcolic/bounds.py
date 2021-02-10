from utils_io import *
from utils import *
from engine import _InputsStatBasedInitializable, Input, Bounds


# ---


class UniformBounds (Bounds):
  """
  Basic class to represent any uniform bounds on inputs.
  """

  def __init__ (self, LB = 0.0, UB = 1.0, **kwds):
    super().__init__(**kwds)
    self.LB = LB
    self.UB = UB

  @property
  def low (self):
    return np.array([self.LB])

  @property
  def up (self):
    return np.array([self.UB])

  def __getitem__ (self, _idx: Tuple[int, ...]) -> Tuple[float, float]:
    return self.LB, self.UB


# ---


class StatBasedInputBounds (Bounds, _InputsStatBasedInitializable):
  """
  Stat-based bounds for generating inputs.

  Analyzes given training samples to compute per-component bounds for
  inputs.

  - `looseness` is a factor that widens the range by some amount (0.1%
    by default).

  - `hard_bounds` is an optional object of type :class:`Bounds`, that
    is used to restrict the bounds after they have been widenned as
    above.
  """

  def __init__(self, looseness: float = .001, hard_bounds: Bounds = None, **kwds):
    assert hard_bounds is None or isinstance (hard_bounds, Bounds)
    self.looseness = looseness
    self.hard_bounds = hard_bounds
    super ().__init__(**kwds)


  def inputs_stat_initialize (self,
                              train_data: raw_datat = None,
                              test_data: raw_datat = None):

    if isinstance (self.hard_bounds, _InputsStatBasedInitializable):
      # Forward call to to hard_bounds, in case.
      self.hard_bounds.inputs_stat_initialize (train_data, test_data)

    np1 ('Initializing stat-based input bounds with {} training samples... '
         .format (len (train_data.data)))
    ptp = np.ptp (train_data.data, axis = 0)
    self._up = np.amax (train_data.data, axis = 0) + self.looseness * ptp
    self._low = np.amin (train_data.data, axis = 0) - self.looseness * ptp
    if self.hard_bounds is not None:
      np.minimum (self._up, self.hard_bounds.up, out = self._up)
      np.maximum (self._low, self.hard_bounds.low, out = self._low)
    c1 ('done')


  @property
  def low (self) -> np.array(float):
    return self._low


  @property
  def up (self) -> np.array(float):
    return self._up


  def __getitem__ (self, idx: Tuple[int, ...]) -> Tuple[float, float]:
    return self._low[idx], self._up[idx]


# ---
