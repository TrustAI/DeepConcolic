# Input filters

from utils_io import *
from utils import *
from engine import _InputsStatBasedInitializable, StaticFilter, Input
from sklearn.neighbors import LocalOutlierFactor

choices = []

# ---

choices += ['LOF']
class LOFNoveltyFilter (StaticFilter, _InputsStatBasedInitializable):

  def __init__(self, name = 'LOF-based novelty', sample_size = 3000, metric = 'cosine', lof_kwds = {}, **kwds):
    assert (isinstance (sample_size, int) and 1 <= sample_size)
    self.name = name
    self.sample_size = sample_size
    self.lof_threshold = 0.0
    self.lof = LocalOutlierFactor (**lof_kwds, metric = metric, novelty = True)
    super().__init__(**kwds)

  def inputs_stat_initialize (self,
                              train_data: raw_datat = None,
                              test_data: raw_datat = None):
    sample_size = min (self.sample_size, train_data.data.shape[0])
    np1 ('Initializing LOF-based novelty estimator with {} training samples... '
         .format (sample_size))
    # TODO: random sampling (& shuffle)?.
    self.lof.fit (train_data.data[:sample_size])
    c1 ('done')
    p1 ('{} offset is {}'.format (self.name, self.lof.offset_))

  def close_enough (self, i: Input):
    lof = self.lof.decision_function (i.reshape (1, -1))
    # p1 ('{}: {}'.format (self.name, lof))
    return lof > self.lof_threshold


# ---


def by_name (name, **kwds):
  if name is None or name in ('none', 'None'):
    return None
  elif name in ('LOF', 'lof'):
    return LOFNoveltyFilter (**kwds)
  else:
    raise ValueError ("Unknown input filter name `{}'".format (name))
