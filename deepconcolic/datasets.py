# NB: do NOT import utils as this disables eager execution that seems
# to be required for proper operations of `tf.summary`.
import os
import numpy as np
from tempfile import gettempdir
from sklearn.model_selection import train_test_split

# ---

default_datadir = os.getenv ('DC_DATADIR') or \
                  os.path.join (gettempdir (), 'sklearn_data')

image_kinds = set (('image', 'greyscale_image',))
normalized_kind = 'normalized'
unknown_kind = 'unknown'
normalized_kinds = set ((normalized_kind,))
kinds = image_kinds | normalized_kinds | set ((unknown_kind,))

choices = []
funcs = {}

def register_dataset (name, f):
  if name in funcs:
    print (f'Warning: a dataset named {name} already exists: replacing.')
  if not callable (f):
    raise ValueError (f'Second argument to `register_dataset\' must be a function')
  choices.append (name)
  choices.sort ()
  funcs[name] = f

# MNIST

def load_mnist_data (**_):
  import tensorflow as tf
  img_shape = 28, 28, 1
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data ()
  x_train = x_train.reshape (x_train.shape[0], *img_shape).astype ('float32') / 255
  x_test = x_test.reshape (x_test.shape[0], *img_shape).astype ('float32') / 255
  return (x_train, y_train), (x_test, y_test), img_shape, 'image', \
         [ str (i) for i in range (0, 10) ]
register_dataset ('mnist', load_mnist_data)

# Fashion-MNIST

def load_fashion_mnist_data (**_):
  import tensorflow as tf
  img_shape = 28, 28, 1
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data ()
  x_train = x_train.reshape (x_train.shape[0], *img_shape).astype ('float32') / 255
  x_test = x_test.reshape (x_test.shape[0], *img_shape).astype ('float32') / 255
  return (x_train, y_train), (x_test, y_test), img_shape, 'image', \
         [ 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot' ]
register_dataset ('fashion_mnist', load_fashion_mnist_data)

# CIFAR10

def load_cifar10_data (**_):
  import tensorflow as tf
  img_shape = 32, 32, 3
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data ()
  x_train = x_train.reshape (x_train.shape[0], *img_shape).astype ('float32') / 255
  x_test = x_test.reshape (x_test.shape[0], *img_shape).astype ('float32') / 255
  return (x_train, y_train), (x_test, y_test), img_shape, 'image', \
         [ 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
register_dataset ('cifar10', load_cifar10_data)

# ---

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

openml_choices = {}
openml_choices['har'] = {
  'shuffle_last': True,
  # , 'test_size': 0.3,
  'input_kind': normalized_kind,
}

def load_openml_data_generic (name, datadir = default_datadir,
                              input_kind = 'unknown',
                              shuffle_last = False,
                              test_size = None,
                              **_):
  # print ('Retrieving OpenML dataset:', name, end = '\r', flush = True)
  ds = fetch_openml (data_home = datadir, name = name)
  # print ('Setting up', len (ds.data), 'data samples', end = '\r', flush = True)
  x_train, x_test, y_train, y_test = train_test_split (ds.data, ds.target,
                                                       test_size = test_size,
                                                       shuffle = not shuffle_last)
  if shuffle_last:
    x_train, y_train = shuffle (x_train, y_train)
    x_test, y_test = shuffle (x_test, y_test)
  labels = np.unique (ds.target)
  labl2y_dict = { y : i for i, y in enumerate (labels) }
  labl2y = np.vectorize (lambda y: labl2y_dict[y])
  y_train, y_test = labl2y (y_train), labl2y (y_test)
  # print ('Loaded', len (y_train), 'training samples, '
  #        'and', len (y_test), 'test samples')
  return (x_train, y_train.astype (int)), (x_test, y_test.astype (int)), \
         (x_train.shape[1:]), input_kind, \
         [ str (c) for c in labels ]

def load_openml_data_lambda (name):
  return lambda **kwds: load_openml_data_generic (\
      name = name, **dict (**openml_choices[name], **kwds))

for c in openml_choices:
  register_dataset ('OpenML:' + str(c), load_openml_data_lambda (c))

# ---

def load_by_name (name, **kwds):
  if name in funcs:
    return funcs[name] (**kwds)
  else:
    raise ValueError (f'Unknown dataset name `{name}\'')

# ---

try:
  from utils_io import warnings, cv2, parse
  from utils_funcs import validate_strarg, validate_inttuplearg
  def images_from_dir (d,
                       raw = False,
                       raw_shape = None,
                       filename_pattern = '{id}-{kind}-{label:d}.{ext}',
                       resolution = None,
                       channels = 'grayscale',
                       normalize = True,
                       channel_bits = 8,
                       ):

    grey_channels = ('grayscale', 'greyscale',)
    rgb_channels = ('rgb', 'RGB',)
    other_channels = ('argb', 'ARGB', 'original')
    validate_strarg (grey_channels + rgb_channels + other_channels,
                     'image channels') ('channels', channels)
    if resolution is not None:
      validate_inttuplearg ('resolution', resolution)
    cv2_flag = cv2.IMREAD_GRAYSCALE if channels in grey_channels else \
               cv2.IMREAD_COLOR if channels in rgb_channels else \
               cv2.IMREAD_UNCHANGED
    def read_image (f):
      image = cv2.imread (f, cv2_flag).astype ('float')
      image = cv2.resize (image, resolution) if resolution is not None else image
      image = image[..., np.newaxis] if channels in grey_channels else image
      if normalize:
        np.divide (image, (2 ** channel_bits - 1), out = image)
      return image

    def add_rawdir (d):
      images = []
      def add_file (f):
        images.append (read_image (f))
      for dir, dirs, files in os.walk (d):
        for f in files:
          if not (f.endswith ('.png') or
                  f.endswith ('.jpg') or
                  f.endswith ('.jpeg')):
            continue
          images.append (read_image (os.path.join (dir, f)))
      if images == []: return None
      images = np.asarray (images)
      shape = raw_shape or images[0].shape
      return images.reshape (len (images), *shape)

    def add_outdir (d):
      filename_parser = parse.compile (filename_pattern)
      images, labels, adversarials = [], [], []
      def add_file (f, label):
        images.append (read_image (f))
        labels.append (label)
      for dir, dirs, files in os.walk (d):
        ff = {}
        for f in files:
          if not (f.endswith ('.png') or
                  f.endswith ('.jpg') or
                  f.endswith ('.jpeg')):
            continue
          info = filename_parser.parse (f)
          if info is None: continue
          info = info.named
          if 'id' not in info or 'kind' not in info: continue
          fid, kind = info['id'], info['kind']
          info['filename'] = os.path.join (dir, f)
          if fid not in ff:
            ff[fid] = { kind: info }
          else:
            ff[fid][kind] = info
        for fid in ff:
          for fid2 in ff[fid]:
            info = ff[fid][fid2]
            if info['kind'] == 'ok':
              add_file (info['filename'], info['label'])
            elif info['kind'] == 'adv' and 'original' in ff[fid]:
              add_file (info['filename'], ff[fid]['original']['label'])
              adversarials.append ((images[-1],
                                    read_image (ff[fid]['original']['filename'])))
      if images == []: return None
      images, labels = np.asarray (images), np.asarray (labels).astype (int)
      images = images.reshape (images.shape[0], *images[0].shape)
      return images, labels.astype (int), images.shape[1:], \
             [ str (i) for i in np.unique (labels) ], \
             adversarials

    if raw:
      return add_rawdir (d)
    else:
      return add_outdir (d)
except:
  # parse not available
  pass
