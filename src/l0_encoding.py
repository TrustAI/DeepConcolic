import numpy as np
from utils import *
from itertools import product

DIM=50

def _unsqueeze_shape (shape, single = False):
  unsqueeze = len (shape) < (2 if single else 3)
  return (shape + (1,)) if unsqueeze else shape

def _unsqueeze (data, **kwds):
  return data.reshape (_unsqueeze_shape (data.shape, **kwds))

class L0Analyzer:
  """
  Custom analyzer for generating new inputs, based on the L0 norm.
  """

  def __init__(self, input_shape, eval_batch, gran = 2):
    self.input_shape = input_shape
    self.shape = _unsqueeze_shape (input_shape)
    self.dims = tuple (min (i, DIM) for i in self.shape[:-1])
    grid = np.meshgrid (*(np.arange (d) for d in self.dims))
    self.flat = tuple (np.split (fc, len (fc))
                       for fc in (c.flatten ('F') for c in grid))
    self.gran = gran
    self.sort_size = np.prod (self.dims) * self.gran
    self.eval_batch = eval_batch
    super().__init__()


  def eval_change(self, inputs, n, target):
    nc_layer, pos = target.layer, target.position
    inputs = inputs.reshape((n,) + self.input_shape)
    activations = self.eval_batch (inputs, layer_indexes = (nc_layer.layer_index,))
    return activations[nc_layer.layer_index][(Ellipsis,) + pos[1:]]


  def sort_features(self, input, nc_target):
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
      self.eval_change (inputs, self.sort_size, nc_target) \
          .reshape(-1, self.gran).T

    min_indices = np.argmax(target_change, axis=0)
    min_values = np.amax(target_change, axis=0)
    min_idx_values = min_indices.astype('float32') / (self.gran - 1)
    target_list = np.hstack((*self.flat,
                             np.split(min_values, len(min_values)),
                             np.split(min_idx_values, len(min_idx_values))))

    sorted_map = target_list[(target_list[:, len (self.dims)]).argsort()]
    sorted_map = np.flipud (sorted_map)
    for i in range (len (sorted_map)):
      for d in range (len (self.dims)):
        sorted_map[i][d] = selected[d][int(sorted_map[i][d])]

    return sorted_map


  def accumulate(self, input, nc_target, sorted_features, mani_range):
    inputs = []
    mani_input = _unsqueeze (input.copy(), single = True)
    for i in range(0, min (mani_range, len (sorted_features))):
      idx = tuple (sorted_features[i, np.arange(len(self.dims))].astype (int))
      mani_input[idx] = sorted_features[i, -1]
      assert mani_input[idx] == sorted_features[i, -1]
      inputs.append (mani_input.copy ())

    inputs = np.asarray(inputs)
    nc_acts = self.eval_change (inputs, len (inputs), nc_target)
    pos_acts = (nc_acts > 0)

    new_inputs = inputs[pos_acts, :, :]
    if new_inputs.any ():
      return new_inputs, np.amin (pos_acts.nonzero (), axis = 1)
    else:
      return None


  def refine(self, input, nc_target, sorted_features, act_first, idx_first):
    input = _unsqueeze (input, single = True)
    refined = act_first.copy ()
    total_idx = 0
    idx_range = np.arange (idx_first)
    while True:
      length = len (idx_range)
      for i in range(0, idx_first[0]):
        idx = tuple (sorted_features[i, np.arange(len(self.dims))].astype (int))
        refined[idx] = input[idx]
        refined_activation = self.eval_change (refined, 1, nc_target)

        if refined_activation < 0:  # == label:
          refined[idx] = sorted_features[i, -1]
        else:
          total_idx = total_idx + 1
          idx_range = idx_range[~(idx_range == i)]
  
      if len(idx_range) == length:
        break
  
    return refined



# ---
