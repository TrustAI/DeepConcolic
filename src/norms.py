from abc import abstractmethod
from engine import Metric
import numpy as np


# ---


class Norm (Metric):
  '''
  Just an alias for norms.
  '''


# ---


class L0 (Norm):
  '''
  L0 norm.
  '''

  def __init__(self, scale = 255, **kwds):
    super().__init__(scale = scale, **kwds)
    

  def __repr__(self):
    return 'L0'


  @property
  def is_int(self):
    return True


  def distance(self, x, y):
    return (np.abs (x - y) * self.scale > 1).sum()


  def close_to (self, refs, x):
    size = refs.size * self.factor
    for ref in refs:
      if np.count_nonzero (ref - x) < size:
        return True
    return False


# ---


class LInf (Norm):
  '''
  L-inf norm.
  '''

  def __repr__(self):
    return 'Linf'


  def distance(self, x, y):
    return np.amax (np.absolute (x - y) * self.scale)


  def close_to(self, refs, x):
    size = refs.size * self.factor
    for index, ref in np.ndenumerate (refs):
      diff_abs = np.array(np.abs(ref - x))
      diff_abs[diff_abs <= self.factor] = 0.0
      if np.count_nonzero(diff_abs) < size:
        return True
    return False


# ---
