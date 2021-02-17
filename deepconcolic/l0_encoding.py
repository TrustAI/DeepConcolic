from utils import *
from itertools import product
from engine import Input, Analyzer4RootedSearch
from norms import L0
import numpy as np

# ---

MAX_DIM = 50

def _unsqueeze_shape (shape, single = False):
  unsqueeze = len (shape) < (2 if single else 3)
  return (shape + (1,)) if unsqueeze else shape

def _unsqueeze (data, **kwds):
  return data.reshape (_unsqueeze_shape (data.shape, **kwds))


# ---


class L0EnabledTarget:

  @abstractmethod
  def eval_inputs (self, inputs: Sequence[Input], eval_batch = None) \
      -> Sequence[float]:
    raise NotImplementedError

  @abstractmethod
  def valid_inputs (self, evals: Sequence[float]) -> Sequence[bool]:
    raise NotImplementedError


# ---


class L0Analyzer:
  """
  Custom analyzer for generating new inputs, based on the L0 norm.
  """

  def __init__(self, input_shape, eval_batch, gran = 2):
    self.input_shape = input_shape
    self.shape = _unsqueeze_shape (input_shape)
    self.dims = tuple (min (i, MAX_DIM) for i in self.shape[:-1])
    grid = np.meshgrid (*(np.arange (d) for d in self.dims))
    self.flat = tuple (np.split (fc, len (fc))
                       for fc in (c.flatten ('F') for c in grid))
    self.gran = gran
    self.eval_batch = eval_batch
    super().__init__()


  def input_metric(self) -> L0:
    return self.norm


  def eval_inputs (self, inputs, n, target: L0EnabledTarget):
    return target.eval_inputs (inputs.reshape((n,) + self.input_shape), self.eval_batch)


  def sort_input_features (self, input, target: L0EnabledTarget):
    sort_list = np.linspace (0, 1, self.gran)
    input_batch = np.kron (np.ones((self.gran,) + (1,) * len (self.shape)),
                           _unsqueeze (input, single = True))

    selected = tuple (np.random.choice (d, dl)
                      for d, dl in zip (self.shape[:-1], self.dims))
    inputs = []
    for idx in product (*selected):
      new_input_batch = input_batch.copy()
      for g in range(0, self.gran):
        new_input_batch[g][idx] = sort_list[g]
      inputs.append(new_input_batch)

    inputs = np.asarray (inputs)
    target_change = \
      self.eval_inputs (inputs, np.prod (self.dims) * self.gran, target) \
          .reshape(-1, self.gran).T

    min_indices = np.argmax(target_change, axis=0)
    min_evals = np.amax(target_change, axis=0)
    min_idx_evals = min_indices.astype('float32') / (self.gran - 1)
    target_list = np.hstack((*self.flat,
                             np.split(min_evals, len(min_evals)),
                             np.split(min_idx_evals, len(min_idx_evals))))

    sorted_map = target_list[(target_list[:, -2]).argsort()]
    sorted_map = np.flipud (sorted_map)
    for i in range (len (sorted_map)):
      for d in range (len (self.dims)):
        sorted_map[i][d] = selected[d][int(sorted_map[i][d])]

    return sorted_map


  def accumulate (self, input, target: L0EnabledTarget, sorted_features, mani_range):
    inputs = []
    mani_input = _unsqueeze (input.copy(), single = True)
    for i in range(0, min (mani_range, len (sorted_features))):
      idx = tuple (sorted_features[i, :len(self.dims)].astype (int))
      mani_input[idx] = sorted_features[i, -1]
      assert mani_input[idx] == sorted_features[i, -1]
      inputs.append (mani_input.copy ())

    inputs = np.asarray(inputs)
    evals = self.eval_inputs (inputs, len (inputs), target)
    valid_evals = target.valid_inputs (evals)

    new_inputs = inputs[valid_evals]
    if new_inputs.any ():
      return new_inputs, np.amin (valid_evals.nonzero (), axis = 1)
    else:
      return None


  def refine (self, input, target: L0EnabledTarget, sorted_features, act_first, idx_first):
    input = _unsqueeze (input, single = True)
    refined = act_first.copy ()
    total_idx = 0
    idx_range = np.arange (idx_first)
    while True:
      length = len (idx_range)
      for i in range(0, idx_first[0]):
        idx = tuple (sorted_features[i, :len(self.dims)].astype (int))
        refined[idx] = input[idx]
        refined_evals = self.eval_inputs (refined, 1, target)
        valid = target.valid_inputs (refined_evals)

        if not valid.any ():    # == label:
          refined[idx] = sorted_features[i, -1]
        else:
          total_idx = total_idx + 1
          idx_range = idx_range[~(idx_range == i)]

      if len(idx_range) == length:
        break

    return refined


# ---


class GenericL0Analyzer (Analyzer4RootedSearch, L0Analyzer):
  """Generic analyzer that is dedicated to find close inputs w.r.t L0 norm.
  """

  def __init__(self, l0_args = {}, **kwds):
    super().__init__(**kwds)
    if 'LB_hard' in l0_args:
      l0_args = dict (**l0_args, scale = 1 / l0_args['LB_hard'])
      del l0_args['LB_hard']
    self.norm = L0 (**l0_args)


  def input_metric(self) -> L0:
    return self.norm


  def search_input_close_to(self, x: Input, target: L0EnabledTarget) -> Optional[Tuple[float, Any]]:
    mani_range = 100

    sorted_features = self.sort_input_features (x, target)
    res = self.accumulate (x, target, sorted_features, mani_range)

    if res:
      act_inputs, idx_first = res
      new_input = self.refine (x, target, sorted_features, act_inputs[0], idx_first)
      new_input = self._postproc_inputs (new_input.reshape (x.shape))
      return self.norm.distance (x, new_input), new_input

    return None


# ---
