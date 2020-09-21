#import matplotlib.pyplot as plt
from abc import abstractmethod
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from tensorflow import keras
# NB: Eager execution needs to be disabled before any model loading.
tf.compat.v1.disable_eager_execution ()

import numpy as np
import copy
import sys
import os
import cv2

COLUMNS = os.getenv ('COLUMNS', default = '80')
P1F = '{:<' + COLUMNS + '}'
N1F = '\n{:<' + COLUMNS + '}'

def tp1(x):
  print (P1F.format(x), end = '\r', flush = True)

def ctp1(x):
  print (N1F.format(x), end = '\r', flush = True)

def np1(x):
  print (x, end = '', flush = True)

def cnp1(x):
  print ('\n', x, sep = '', end = '', flush = True)

def p1(x, **k):
  print (P1F.format(x), **k)

def cp1(x, **k):
  print (N1F.format(x), **k)


def xtuple(t):
  return t if len(t) > 1 else t[0]


#the_dec_pos=0
MIN=-100000
DIM=50
BUFFER_SIZE=20
#ssc_ratio=0.005 #0.1 #0.05 #0.01

# Some type for any DNN layer
Layer = keras.layers.Layer

## some DNN model has an explicit input layer
def is_input_layer(layer):
  return isinstance (layer, keras.layers.InputLayer)

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
  return False ## we do not allow dropout

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

def setup_output_dir (outs, log = True):
  if not os.path.exists (outs):
    os.makedirs (outs)
  if not outs.endswith ('/'):
    outs += '/'
  if log: print ('Setting up output directory: {0}'.format (outs))
  return outs

# def setup_report_files (outs, ident, suff0 = '', suff = '.txt', log = True):
#   if not os.path.exists(outs):
#     sys.exit ('Output directory {0} was not initialized (internal bug)!'
#               .format (outs))
#   f = outs+ident+suff
#   if log: print ('Reporting into: {0}'.format (f))
#   return f, ident

class OutputDir:
  '''
  Class to help ensure output directory is created before starting any
  lengthy computations.
  '''
  def __init__(self, outs = '/tmp', log = None):
    self.dirpath = setup_output_dir (outs, log = log)

  @property
  def path(self) -> str:
    return self.dirpath

  def filepath(self, base) -> str:
    return self.dirpath + base

# ---

def _write_in_file (f, mode, *fmts):
  f = open (f, mode)
  for fmt in fmts: f.write (fmt)
  f.close ()

def write_in_file (f, *fmts):
  _write_in_file (f, "w", *fmts)

def append_in_file (f, *fmts):
  _write_in_file (f, "a", *fmts)

def save_an_image(im, name, directory = './', log = True):
  if not directory.endswith('/'): directory += '/'
  f = directory + name + '.png'
  if log: print ('Outputing image into {0}'.format (f))
  cv2.imwrite (f, im * 255)

def save_adversarial_examples(adv, origin, diff, di):
  save_an_image(adv[0], adv[1], di)
  save_an_image(origin[0], origin[1], di)
  if diff is not None:
    save_an_image(diff[0], diff[1], di)


# ---


class cover_layert:
  def __init__(self, layer, layer_index, prev: int = None, succ: int = None):
    self.layer = layer
    self.layer_index = layer_index
    self.is_conv = is_conv_layer(layer)
    self.prev_layer_index = prev
    self.succ_layer_index = succ
    self.activations = []  ## so, we need to store neuron activations?
    self.ssc_map = None ## 
    self.ubs = None ## 

  def __repr__(self):
    return self.layer.name

  # ssc/svc:

  def initialize_ssc_map(self, layer_feature = None):
    sp = self.layer.output.shape
    if self.is_conv:
      self.ssc_map = np.ones((1, sp[1], sp[2], sp[3]), dtype=bool)
      if layer_feature==None: return
      if layer_feature[0]==None: return
      if layer_feature[1]==None: return
      if self.layer_index in layer_feature[0]:
        sp=self.ssc_map.shape
        for i in range(0, sp[3]):
          if not i in layer_feature[1]:
            self.ssc_map[:,:,:,i]=False
    else:
      self.ssc_map = np.ones((1, sp[1]), dtype=bool)

  def initialize_ubs(self):
    self.ubs = np.zeros((1,) + tuple(self.layer.output.shape[1:]), dtype = float)

  # ---

  @abstractmethod
  def coverage(self):
    pass

# ---

# Basic helper to build more polymorphic functions
def actual_layer(l):
  return l.layer if isinstance (l, cover_layert) else l

# ---

def post_activation_layer (dnn, idx):
  return min((i for i, layer in enumerate(dnn.layers)
              if (i >= idx and (is_activation_layer (layer) or
                                activation_is_relu(layer)))))


def deepest_tested_layer (dnn, clayers):
  return post_activation_layer (dnn, max((l.layer_index for l in clayers)))


def testable_layer (dnn, idx,
                    exclude_direct_input_succ = False):
  layer = dnn.layers[idx]
  return ((is_conv_layer(layer) or is_dense_layer(layer)) and
          (idx != len(dnn.layers)-1 or activation_is_relu (layer)) and
          not (exclude_direct_input_succ and
               (idx == 0 or idx == 1 and is_input_layer (dnn.layers[0]))))

def get_cover_layers (dnn, constr, layer_indices = None,
                      exclude_direct_input_succ = False,
                      exclude_output_layer = True):
  # All coverable layers:
  layers = dnn.layers[:-1] if exclude_output_layer else dnn.layers
  cls = [ (l, layer) for l, layer in enumerate (layers) if testable_layer (dnn, l) ]
  return [ constr (layer[1], layer[0],
                   prev = (cls[l-1][0] if l > 0 else None),
                   succ = (cls[l+1][1] if l < len(cls) - 1 else None))
           for l, layer in enumerate(cls)
           if not (exclude_direct_input_succ and
                   (layer[0] == 0 or layer[0] == 1 and is_input_layer (dnn.layers[0])))
           and (layer_indices == None or layer[0] in layer_indices) ]

# ---

# Do we really manipulate many DNNs at once?
from functools import lru_cache
@lru_cache(4)
def get_layer_functions(dnn):
  return ([ keras.backend.function([layer.input], [layer.output])
            for layer in dnn.layers ],
          is_input_layer (dnn.layers[0]))

# ---

### given input images, evaluate activations
def eval_batch(o, ims, allow_input_layer = False):
  layer_functions, has_input_layer = (
    get_layer_functions (o) if isinstance (o, (keras.Sequential, keras.Model))
    # TODO: Check it's sequential? --------------------------------------^
    else o)
  having_input_layer = allow_input_layer and has_input_layer
  activations = []
  for l, func in enumerate(layer_functions):
    if not having_input_layer:
      if l==0:
        activations.append(func([ims])[0])
      else:
        activations.append(func([activations[l-1]])[0])
    else:
      if l==0:
        activations.append([]) #activations.append(func([ims])[0])
      elif l==1:
        activations.append(func([ims])[0])
      else:
        activations.append(func([activations[l-1]])[0])
  return activations

def eval(o, im, having_input_layer = False):
  return eval_batch (o, np.array([im]), having_input_layer)

def eval_batch_func (dnn):
  return lambda imgs, **kwds: eval_batch (dnn, imgs, **kwds)

# ---

class raw_datat:
  def __init__(self, data, labels, name = 'unknown'):
    self.data=data
    self.labels=labels
    self.name = name
    


class test_objectt:
  def __init__(self, dnn, test_data, train_data):
    self.dnn=dnn
    self.raw_data=test_data
    self.train_data = train_data
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
  

  def tests_layer(self, cl):
    return self.layer_indices == None or cl.layer_index in self.layer_indices


  def check_layer_indices (self, criterion):
    if self.layer_indices == None: return
    mcdc = criterion in ('ssc', 'ssclp')
    testable = lambda l: testable_layer (self.dnn, l, exclude_direct_input_succ = mcdc)
    testable_layers_indices = [ l for l in range(0, len(self.dnn.layers)) if testable (l) ]
    wrong_layer_indices = [ i for i in self.layer_indices if i not in testable_layers_indices ]
    if wrong_layer_indices != []:
      sys.exit ('Untestable layers: {}'
                .format([self.dnn.layers[l].name for l in wrong_layer_indices]))

# ---

# TODO: generalize to n-dimensional convolutional layers:
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


class Coverage:
  """Basic helper class to manipulate and type-annotate coverage measures."""

  def __init__(self, covered = None, total = None, non_covered = None):
    if total != None:
      self.total = total
    elif covered != None and non_covered != None:
      self.total = covered + non_covered
    elif covered != None:
      self.total = covered
    elif non_covered != None:
      self.total = non_covered
    else:
      self.total = 0

    if covered != None:
      self.c = covered
    elif non_covered != None and self.total > 0:
      self.c = self.total - non_covered
    else:
      self.c = 0


  def __add__(self, x):
    return Coverage (covered = self.c + x.c,
                     total = self.total + x.total)


  @property
  def done(self) -> bool:
    return self.total == self.c


  @property
  def as_prop(self) -> float:
    return (((1.0 * self.c) / (1.0 * self.total))
            if self.total != 0 else 0.0)


  def __repr__(self):
    return str(self.as_prop)


# ---


from collections import UserDict

class NPArrayDict (UserDict):
  '''
  Custom dictionary that accepts numpy arrays as keys.
  '''

  def __getitem__(self, x: np.ndarray):
    return self.data[hash (x.tobytes ())]

  def __delitem__(self, x: np.ndarray):
    del self.data[hash (x.tobytes ())]

  def __setitem__(self, x: np.ndarray, val):
    x.flags.writeable = False
    self.data[hash (x.tobytes ())] = val


# ---
