from abc import abstractmethod
from utils_io import *
from utils_funcs import *
import sys, copy

# NB: importing cv2 and sklearn before tensorflow seems to solve an
# issue with static TLS I've been having on an "oldish" version of
# Linux (cf
# https://github.com/scikit-learn/scikit-learn/issues/14485#issuecomment-633452991):
import sklearn
import tensorflow as tf
from tensorflow import keras

print ("Using TensorFlow version:", tf.__version__, file = sys.stderr)

# ---

#the_dec_pos=0
MIN=-100000
BUFFER_SIZE=20
#ssc_ratio=0.005 #0.1 #0.05 #0.01

# Some type for any DNN layer
Layer = keras.layers.Layer

## some DNN model has an explicit input layer
def is_input_layer(layer):
  return isinstance (layer, keras.layers.InputLayer)

def is_reshape_layer(layer):
  return isinstance (layer, keras.layers.Reshape)

def is_conv_layer(layer):
  return isinstance (layer, (keras.layers.Conv1D,
                             keras.layers.Conv2D))

def is_dense_layer(layer):
  return isinstance (layer, keras.layers.Dense)

def is_activation_layer(layer):
  return isinstance (layer, keras.layers.Activation)

def is_relu_layer(layer):
  return isinstance (layer, keras.layers.ReLU)

# def act_in_the_layer(layer):
#   try:
#     act = str(layer.activation)
#     if act.find('relu')>=0: return 'relu'
#     elif act.find('softmax')>=0: return 'softmax'
#     else: return ''
#   except:
#     return ''

# def activation_is_relu(layer):
#   return act_in_the_layer(layer)=='relu'
#   # try:
#   #   print (layer.activation)
#   #   return isinstance (layer.activation, layers.ReLU)
#   # except:
#   #   return False

def is_maxpooling_layer(layer):
  return isinstance (layer, (keras.layers.MaxPooling1D,
                             keras.layers.MaxPooling2D,
                             keras.layers.MaxPooling3D))

def is_flatten_layer(layer):
  return isinstance (layer, keras.layers.Flatten)

def is_dropout_layer(layer):
  return isinstance (layer, keras.layers.Dropout)

# def act_in_the_layer(layer):
#   try:
#     act=str(layer.activation)
#     if act.find('relu')>=0: return 'relu'
#     elif act.find('softmax')>=0: return 'softmax'
#     else: return ''
#   except:
#     return ''

def activation_is_relu(layer):
  try: return (layer.activation == keras.activations.relu)
  except: return False

# def is_relu_layer (layer):
#   return activation_is_relu(layer)

# def get_activation(layer):
#   if str(layer.activation).find('relu')>=0: return 'relu'
#   elif  str(layer.activation).find('linear')>=0: return 'linear'
#   elif  str(layer.activation).find('softmax')>=0: return 'softmax'
#   else: return ''

# ---

class cover_layert:
  pass

# Basic helper to build more polymorphic functions
def actual_layer(l):
  return l.layer if isinstance (l, cover_layert) else l

# ---

def post_activation_layer (dnn, idx):
  return min((i for i, layer in enumerate(dnn.layers)
              if (i >= idx and (is_activation_layer (layer) or
                                activation_is_relu (layer)))))


def deepest_tested_layer (dnn, clayers):
  return post_activation_layer (dnn, max((l.layer_index for l in clayers)))


def post_conv_or_dense (dnn, idx):
  prev = dnn.layers[idx - 1] if idx > 0 else None
  return prev is not None and (is_conv_layer (prev) or is_dense_layer (prev))


def activation_of_conv_or_dense (dnn, idx):
  layer = dnn.layers[idx]
  return \
    (is_activation_layer (layer) and post_conv_or_dense (dnn, idx)) or \
    ((is_conv_layer (layer) or is_dense_layer (layer)) and activation_is_relu (layer))


def testable_layer_function (dnn, idx,
                             exclude_output_layer = True,
                             exclude_direct_input_succ = False):
  layer = dnn.layers[idx]
  input_succ = idx == 0 or idx == 1 and is_input_layer (dnn.layers[0])
  non_output = idx != len (dnn.layers) - 1
  return \
    (not input_succ if exclude_direct_input_succ else True) and \
    (non_output if exclude_output_layer else True)#  and \


def get_cover_layers (dnn, constr, layer_indices = None,
                      activation_of_conv_or_dense_only = True,
                      **kwds):
  def a_(l):
    in_layer_act = \
      (is_conv_layer (dnn.layers[l]) or is_dense_layer (dnn.layers[l])) and \
      activation_is_relu (dnn.layers[l])
    return l if in_layer_act else l - 1

  def flt (l):
    return(activation_of_conv_or_dense (dnn, l) and
           testable_layer_function (dnn, a_(l), **kwds)) if activation_of_conv_or_dense_only \
      else testable_layer_function (dnn, l, **kwds)

  def fun (l):
    return (a_(l), dnn.layers[a_(l)]) if activation_of_conv_or_dense_only \
      else (l, dnn.layers[l])

  cls = [ fun (l) for l, layer in enumerate (dnn.layers) if
          (layer_indices is None or l in layer_indices) and flt (l) ]

  return [ constr (layer[1], layer[0],
                   prev = (cls[l-1][0] if l > 0 else layer[0]-1 if layer[0] > 0 else None),
                   succ = (cls[l+1][1] if l < len(cls) - 1 else None))
           for l, layer in enumerate (cls) ]

# ---

# Do we really manipulate many DNNs at once?
from functools import lru_cache
@lru_cache(4)
def get_layer_functions(dnn):
  return ([ keras.backend.function([layer.input], [layer.output])
            for layer in dnn.layers ],
          is_input_layer (dnn.layers[0]))

# ---

_default_batch_size = 256
def batched_eval (f, X, axis = 0, batch_size = _default_batch_size):
  batch_size = batch_size or _default_batch_size
  X, Y = np.asarray (X), []
  for b in np.array_split (X, X.shape[axis] // batch_size + 1, axis = axis):
    Y += f (b)
  return np.concatenate (Y, axis = axis)

# ---

### given input images, evaluate activations
def eval_batch(o, ims, allow_input_layer = False, layer_indexes = None,
               batch_size = None):
  layer_functions, has_input_layer = (
    get_layer_functions (o) if isinstance (o, (keras.Sequential, keras.Model))
    # TODO: Check it's sequential? --------------------------------------^
    else o)
  activations = []
  deepest_layer_index = max (layer_indexes) if layer_indexes is not None else None
  prev, prevv = None, None
  for l, func in enumerate (layer_functions):
    prev = ([] if has_input_layer and l == 0 else \
            batched_eval (func,
                          ims if l == (1 if has_input_layer else 0) else prev,
                          batch_size = batch_size))
    if prevv is not None and activations[-1] is not prevv:
      del prevv
    activations.append (prev if layer_indexes is None or l in layer_indexes else [])
    if deepest_layer_index is not None and l == deepest_layer_index:
      break
    prevv = prev
  return activations

def eval(o, im, **kwds):
  return eval_batch (o, np.array([im]), **kwds)

def eval_batch_func (dnn):
  return lambda imgs, **kwds: eval_batch (dnn, imgs, **kwds)

def _prediction (f, x, top_classes = None):
  return \
    np.argmax (f (np.array ([x]))) if top_classes is None else \
    np.flip (np.argsort (dnn.predict (np.array ([x])))[0])[:top_classes]

def _predictions (f, xl, top_classes = None):
  return \
    np.argmax (f (np.array (xl)), axis = 1) if top_classes is None else \
    np.fliplr (np.argsort (f (np.array (xl))))[:top_classes]

def prediction (dnn, x, **_):
  return _prediction (dnn.predict, x, **_)

def predictions (dnn, x, **_):
  return _predictions (dnn.predict, x, **_)

# ---

class raw_datat:
  def __init__(self, data, labels, name = 'unknown'):
    self.data = as_numpy (data)
    self.labels = appopt (np.squeeze, as_numpy (labels))
    self.name = name


class fix_image_channels_:
  def __init__(self, up = 255., bounds = (0.0, 255.0), ctype = 'uint8', down = 255.):
    assert bounds is not None
    assert ctype is not None
    self.up, self.down = up, down
    self.bounds = bounds
    self.ctype = ctype

  def __call__ (self, x):
    with np.errstate (over = 'ignore', under = 'ignore'):
      if self.up is not None:
        np.multiply (x, self.up, out = x)
      x = np.clip (x, *self.bounds, out = x).astype (self.ctype).astype (float)
      if self.down is not None:
        np.divide (x, self.down, out = x)
      return x


def dataset_dict (name, save_input_args = ('new_inputs',)):
  import datasets
  np1 (f'Loading {name} dataset... ')
  (x_train, y_train), (x_test, y_test), dims, kind, labels = datasets.load_by_name (name)
  test_data = raw_datat (x_test, y_test, name)
  train_data = raw_datat (x_train, y_train, name)
  save_input = (save_an_image if kind in datasets.image_kinds else \
                save_in_csv (*save_input_args) if len (dims) == 1 else None)
  input_bounds = ((0., 1.) if kind in datasets.image_kinds else \
                  'normalized' if kind in datasets.normalized_kinds else None)
  postproc_inputs = fix_image_channels_ () if kind in datasets.image_kinds else id
  c1 ('done.')
  return dict (test_data = test_data, train_data = train_data,
               kind = kind, dims = dims, labels = labels,
               input_bounds = input_bounds,
               postproc_inputs = postproc_inputs,
               save_input = save_input)


# ---


def load_model (model_spec):
  # NB: Eager execution needs to be disabled before any model loading.
  tf.compat.v1.disable_eager_execution ()
  if model_spec == 'vgg16':
    return tf.keras.applications.VGG16 ()
  elif os.path.exists (model_spec):
    return tf.keras.models.load_model (model_spec)
  else:
    raise ValueError (f'Invalid specification for neural network model: `{model_spec}')


# ---


class test_objectt:
  def __init__(self, dnn, train_data, test_data):
    self.dnn = dnn
    self.train_data = train_data
    self.raw_data = test_data
    self.postproc_inputs = id
    # Most of what's below should not be needed anymore: one should
    # avoid populating that object with criteria/analyzer-specific
    # parameters.
    ## test config
    self.cond_ratio=None
    self.top_classes=None
    self.labels=None                    # only used in run_scc.run_svc
    self.trace_flag=None
    self.layer_indices=None
    self.feature_indices=None


  def layer_index (self, l):
    layer = self.dnn.get_layer (name = l) if isinstance (l, str) else \
            self.dnn.get_layer (index = int (l))
    return self.dnn.layers.index (layer)


  def set_layer_indices (self, ll):
    self.layer_indices = [ self.layer_index (l) for l in ll ]


  def tests_layer (self, cl):
    return self.layer_indices == None or cl.layer_index in self.layer_indices


  def check_layer_indices (self, criterion):
    mcdc = criterion in ('ssc', 'ssclp')
    dbnc = criterion in ('bfc', 'bfdc')
    testable_layers = get_cover_layers (self.dnn, lambda x, y, **_: (x, y),
                                        activation_of_conv_or_dense_only = not dbnc,
                                        exclude_direct_input_succ = mcdc,
                                        exclude_output_layer = not dbnc)
    print ('Testable function layers: {}'
           .format (', '.join (l.name for l, _ in testable_layers)))

    if self.layer_indices == None: return

    testable_idxs = tuple (l[1] for l in testable_layers)
    testable_idxs = tuple (i + 1 if (not dbnc and \
                                     not is_activation_layer (self.dnn.get_layer(index=i)) and\
                                     not activation_is_relu (self.dnn.get_layer(index=i))) \
                           else i
                           for i in testable_idxs)
    wrong_layer_indices = tuple (i for i in self.layer_indices if i not in testable_idxs)
    if wrong_layer_indices != ():
      sys.exit ('Untestable function {}layers: {}{}'
                .format('or non-activation ' if not dbnc else '',
                        ', '.join (self.dnn.layers[l].name for l in wrong_layer_indices),
                        '\nOnly activation layers may be specified for '
                        f'criterion {criterion}' if not dbnc else ''))

    tested_layers = get_cover_layers (self.dnn, lambda x, y, **_: (x, y),
                                      layer_indices = self.layer_indices,
                                      activation_of_conv_or_dense_only = not dbnc,
                                      exclude_direct_input_succ = mcdc,
                                      exclude_output_layer = not dbnc)

    if tested_layers == []:
      sys.exit ('No layer function is to be tested: aborting.')
    else:
      print ('Function layers to be tested: {}'
             .format (', '.join (l.name for l, _ in tested_layers)))

    if mcdc:
      self.find_mcdc_injecting_layer ([i for _, i in tested_layers],
                                      criterion in ('ssclp',))


  def find_mcdc_injecting_layer (self, tested_layer_indexes, concolic):

    injecting_layer_index = tested_layer_indexes[0] - 1
    if concolic:
      while injecting_layer_index >= 0 and \
            (activation_is_relu (self.dnn.layers[injecting_layer_index]) or \
             is_activation_layer (self.dnn.layers[injecting_layer_index]) or \
             is_maxpooling_layer (self.dnn.layers[injecting_layer_index])):
        injecting_layer_index -= 1

    if injecting_layer_index < 0:
      sys.exit ('DNN architecture not supported by concolic MC/DC-style '
                'citerion: no suitable activation-less condition layer found')

    return injecting_layer_index


# ---

# TODO: generalize to n-dimensional convolutional layers:
# Good starting point: `from tensorflow.python.keras.utils import conv_utils'
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/utils/conv_utils.py
def is_padding(dec_pos, dec_layer, cond_layer, post = True, unravel_pos = True):
  ## to check if dec_pos is a padding
  dec_layer = actual_layer (dec_layer)
  if is_conv_layer (dec_layer):
    cond_layer = actual_layer (cond_layer)
    kernel_size = dec_layer.kernel_size
    weights = dec_layer.get_weights()[0]
    (I, J, K) = (np.unravel_index(dec_pos, dec_layer.output.shape[1:])
                 if unravel_pos else dec_pos)
    # z = (zip ((I, J) pos_idx[:-1], cond_layer.output.shape[1:-1]) if post else
    #      zip ((J, K) pos_idx[1: ], cond_layer.output.shape[2:  ]))
    return ((I - kernel_size[0] < 0 or
             I + kernel_size[0] > cond_layer.output.shape[1] or
             J - kernel_size[1] < 0 or
             J + kernel_size[1] > cond_layer.output.shape[2] or
             weights.shape[1]   > cond_layer.output.shape[3]) if post else
            (J - kernel_size[0] < 0 or
             J + kernel_size[0] > cond_layer.output.shape[2] or
             K - kernel_size[1] < 0 or
             K + kernel_size[1] > cond_layer.output.shape[3] or
             weights.shape[0]   > cond_layer.output.shape[1]))
  return False

# TODO: stride & padding
def maxpool_idxs (oidx, pool_size) -> range:
  for pool_idx in np.ndindex (pool_size):
    yield (tuple (oidx[i] * pool_size[i] + pool_idx[i]
                  for i in range (len (pool_size))))

def get_ssc_next(clayers, layer_indices=None, feature_indices=None):
  #global the_dec_pos
  # clayers2=[]
  # if layer_indices==None:
  clayers2=clayers
  # else:
  #   for i in range(1, len(clayers)):
  #     if clayers[i].layer_index in layer_indices:
  #       clayers2.append(clayers[i])
  # if clayers2==[]:
  #   sys.exit('incorrect layer index specified (the layer tested shall be either conv or dense layer) {}'
  #            .format(layer_indices))
  #print (clayers2[0].layer_index)
  dec_layer_index_ret=None
  dec_pos_ret=None

  while True:
    dec_layer_index=np.random.randint(0, len(clayers2))
    ## todo: this is a shortcut
    #print ('#######',len(clayers2), dec_layer_index, clayers[1].layer)
    if not np.any(clayers2[dec_layer_index].ssc_map):
      print ('all decision features at layer {0} have been covered'.format(dec_layer_index))
      continue
      #sys.exit(0)

    tot_s = np.prod (clayers2[dec_layer_index].ssc_map.shape)

    the_dec_pos = np.random.randint(0, tot_s)
    if not feature_indices==None:
      the_dec_pos=np.argmax(clayers2[dec_layer_index].ssc_map.shape)
    # print (the_dec_pos, tot_s, np.count_nonzero (clayers2[dec_layer_index].ssc_map))
    found=False
    while the_dec_pos < tot_s:
      if not clayers2[dec_layer_index].ssc_map.item(the_dec_pos):
        the_dec_pos+=1
        continue
      else:
        found=True
        break
    #if the_dec_pos>=tot_s:
    #  print ('all decision features at layer {0} have been covered'.format(dec_layer_index))
    #  sys.exit(0)
    if found:
      dec_pos_ret=the_dec_pos
      for i in range(0, len(clayers)):
        if clayers[i].layer_index==clayers2[dec_layer_index].layer_index:
          dec_layer_index_ret=i
          break
      break
  if dec_layer_index_ret==None:
    print ('End of the testing')
    sys.exit(0)
  return dec_layer_index_ret, dec_pos_ret

def print_adversarial_distribution(advs, fname, int_flag=False):
  advs = np.sort(advs)
  ## average and std
  ave = np.mean(advs)
  std = np.std(advs)
  d_max = advs[len(advs)-1]
  xs = np.arange(1, d_max+1, 1) if int_flag else np.arange(0.001, d_max+0.001, 0.001)
  ys = np.zeros(len(xs))
  for i in range(0, len(xs)):
    for d in advs:
      if d <= xs[i]: ys[i] += 1
    ys[i] = ys[i] * 1.0 / len(advs)

  write_in_file (fname,
                 'adversarial examples:  (average distance, {0}), (standard variance, {1})\n'
                 .format(ave, std),
                 '#distance #accumulated adversarial examples fall into this distance\n',
                 *['{0} {1}\n'.format(xs[i], ys[i]) for i in range(0, len(xs))])

# ---

def lazy_activations_on_indexed_data (fnc, dnn, data: raw_datat,
                                      indexes, layer_indexes,
                                      pass_kwds = True):
  input_data = data.data[indexes]
  f = lambda j: LazyLambda \
    ( lambda i: (eval_batch (dnn, input_data[i], allow_input_layer = True,
                             layer_indexes = (j,))[j] if i is not None
                 else len (input_data)))
  if pass_kwds:
    return fnc (LazyLambdaDict (f, layer_indexes),
                input_data = input_data,
                true_labels = data.labels[indexes],
                pred_labels = predictions (dnn, input_data))
  else:
    return fnc (LazyLambdaDict (f, layer_indexes))


# TODO: customize default batch_size?
def lazy_activations_transform (acts, transform, batch_size = 100):
  yacc = None
  for i in range (0, len (acts), batch_size):
    imax = min (i + batch_size, len (acts))
    facts = acts[i:imax].copy ()
    x = facts.reshape (len (facts), -1)
    y = transform (x)
    yacc = np.vstack ((yacc, y)) if yacc is not None else y
    del facts, x
    if y is not yacc: del y
  return yacc

# ---
