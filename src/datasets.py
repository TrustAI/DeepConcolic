from utils import *

def load_mnist_data ():
    img_rows, img_cols, img_channels = 28, 28, 1
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data ()
    x_train = x_train.reshape (x_train.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    x_test = x_test.reshape (x_test.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    return (x_train, y_train), (x_test, y_test), \
           (img_rows, img_cols, img_channels), \
           [ str (i) for i in range (0, 10) ]

def load_cifar10_data ():
    img_rows, img_cols, img_channels = 32, 32, 3
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data ()
    x_train = x_train.reshape (x_train.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    x_test = x_test.reshape (x_test.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    return (x_train, y_train), (x_test, y_test), \
           (img_rows, img_cols, img_channels), \
           [ 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_fashion_mnist_data ():
    img_rows, img_cols, img_channels = 28, 28, 1
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data ()
    x_train = x_train.reshape (x_train.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    x_test = x_test.reshape (x_test.shape[0], img_rows, img_cols, img_channels).astype ('float32') / 255
    return (x_train, y_train), (x_test, y_test), \
           (img_rows, img_cols, img_channels), \
           [ 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot' ]

# ---

choices = ['mnist', 'fashion_mnist', 'cifar10']

def load_by_name (name):
    if name == 'mnist':
        return load_mnist_data ()
    elif name == 'fashion_mnist':
        return load_fashion_mnist_data ()
    elif name == 'cifar10':
        return load_cifar10_data ()
    else:
        raise ValueError ("Unknown dataset name `{}'".format (name))
