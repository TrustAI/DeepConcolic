from typing import *
from collections import UserDict
import random
import numpy as np

# ---

def xtuple(t):
  return t if len(t) > 1 else t[0]

def xlist(t):
  return [t] if t is not None else []

def seqx(t):
  return [] if t is None else t if isinstance (t, (list, tuple)) else [t]

def id(x):
  return x

def some(a, d):
  return a if a is not None else d

def appopt(f, a):
  return f(a) if a is not None else a

# ---

def validate_strarg (valid, spec):
  def aux (v, s):
    if s is not None and s not in valid:
      raise ValueError ('Unknown {} `{}\' for argument `{}\': expected one of '
                        '{}'.format (spec, s, v, valid))
  return aux

def validate_inttuplearg (v, s):
  if isinstance (s, tuple) and all (isinstance (se, int) for se in s):
    return
  raise ValueError ('Invalid value for argument `{}\': expected tuple of ints'
                    .format (v))


# ---


try:
  import pandas as pd
except:
  pd = None
  pass


def as_numpy (d):
  if pd is not None:
    return d.to_numpy () if isinstance (d, pd.core.frame.DataFrame) else d
  else:
    return d


# ---


def rng_seed (seed: Optional[int]):
  if seed is None:
    seed = int (np.random.uniform (2**32-1))
  print ('RNG seed:', seed) # Log seed to help some level of reproducibility
  np.random.seed (seed)
  # In case one also uses pythons' stdlib ?
  random.seed (seed)

def randint ():
  return int (np.random.uniform (2**32-1))


# ---


try:
  # Use xxhash if available as it's probably more efficient
  import xxhash
  __h = xxhash.xxh64 ()
  def np_hash (x):
    __h.reset ()
    __h.update (x)
    return __h.digest ()
except:
  def np_hash (x):
    return hash (x.tobytes ())
  # NB: In case we experience too many collisions:
  # import hashlib
  # def np_hash (x):
  #   return hashlib.md5 (x).digest ()

class NPArrayDict (UserDict):
  '''
  Custom dictionary that accepts numpy arrays as keys.
  '''

  def __getitem__(self, x: np.ndarray):
    return self.data[np_hash (x)]

  def __delitem__(self, x: np.ndarray):
    del self.data[np_hash (x)]

  def __setitem__(self, x: np.ndarray, val):
    x.flags.writeable = False
    self.data[np_hash (x)] = val

  def __contains__(self, x: np.ndarray):
    return np_hash (x) in self.data


# ---


D, C = TypeVar ('D'), TypeVar ('C')


class LazyLambda:
  '''
  Lazy eval on an unknown domain.
  '''

  def __init__(self, f: Callable[[D], C], **kwds):
    super ().__init__(**kwds)
    self.f = f

  def __getitem__(self, x: D) -> C:
    return self.f (x)

  def __len__(self) -> int:
    return self.f (None)


class LazyLambdaDict (Dict[D, C]):
  '''
  Lazy function eval on a fixed domain.
  '''

  def __init__(self, f: Callable[[D], C], domain: Set[D], **kwds) -> Dict[D, C]:
    super ().__init__(**kwds)
    self.domain = domain
    self.f = f

  def __getitem__(self, x: D) -> D:
    if x not in self.domain:
      return KeyError
    return self.f (x)

  def __contains__(self, x: D) -> bool:
    return x in self.domain

  def __iter__(self) -> Iterator[D]:
    return self.domain.__iter__ ()

  def __setitem__(self,_):
    raise RuntimeError ('Invalid item assignment on `LazyLambdaDict` object')

  def __delitem__(self,_):
    raise RuntimeError ('Invalid item deletion on `LazyLambdaDict` object')
