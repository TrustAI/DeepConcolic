# NB: do NOT import utils as this disables eager execution that seems
# to be required for proper operations of `tf.summary`.
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# ---

default_datadir = os.getenv ('DC_DATADIR') or \
                  os.getenv ('TMPDIR', default = '/tmp') + '/sklearn_data'

image_kinds = set (('image', 'greyscale_image',))
normalized_kind = 'normalized'
unknown_kind = 'unknown'
normalized_kinds = set ((normalized_kind,))
kinds = image_kinds | normalized_kinds | set ((unknown_kind,))

choices = []

# MNIST

choices += ['mnist']
def load_mnist_data ():
    img_rows, img_cols, img_channels = 28, 28, 1
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data ()
    x_train = x_train.reshape (x_train.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    x_test = x_test.reshape (x_test.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    return (x_train, y_train), (x_test, y_test), \
           (img_rows, img_cols, img_channels), 'image', \
           [ str (i) for i in range (0, 10) ]

# Fashion-MNIST

choices += ['fashion_mnist']
def load_fashion_mnist_data ():
    img_rows, img_cols, img_channels = 28, 28, 1
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data ()
    x_train = x_train.reshape (x_train.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    x_test = x_test.reshape (x_test.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    return (x_train, y_train), (x_test, y_test), \
           (img_rows, img_cols, img_channels), 'image', \
           [ 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot' ]

# CIFAR10

choices += ['cifar10']
def load_cifar10_data ():
    img_rows, img_cols, img_channels = 32, 32, 3
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data ()
    x_train = x_train.reshape (x_train.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    x_test = x_test.reshape (x_test.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    return (x_train, y_train), (x_test, y_test), \
           (img_rows, img_cols, img_channels), 'image', \
           [ 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ---

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

openml_choices = {}
openml_choices['har'] = {
    'shuffle_last': True,
    # , 'test_size': 0.3,
    'input_kind': normalized_kind,
}

choices += ['OpenML:' + str(c) for c in openml_choices]

def load_openml_data_generic (name, datadir = default_datadir,
                              input_kind = 'unknown',
                              shuffle_last = False,
                              test_size = None):
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

# ---

def load_by_name (name, datadir = None):
    if name == 'mnist':
        return load_mnist_data ()
    elif name == 'fashion_mnist':
        return load_fashion_mnist_data ()
    elif name == 'cifar10':
        return load_cifar10_data ()
    elif name.startswith (('OpenML:', 'openml:')):
        name = name[len ('OpenML:'):]
        return load_openml_data_generic (name,
                                         **openml_choices[name],
                                         datadir = datadir)
    else:
        raise ValueError ("Unknown dataset name `{}'".format (name))
