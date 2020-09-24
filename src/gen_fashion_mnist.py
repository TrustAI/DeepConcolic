from training import *
from datasets import load_fashion_mnist_data as load_data

def make_small_model (input_shape, **kwds):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, (5, 5), # padding = 'same',
                               input_shape = input_shape),
        tf.keras.layers.Activation('relu'),
        # tf.keras.layers.Conv2D(32, (5, 5)),
        # tf.keras.layers.Activation('relu'),
        # tf.keras.layers.Conv2D(16, (5, 5)),
        # tf.keras.layers.Activation('relu'),
        # tf.keras.layers.Conv2D(5, (7, 7)),
        # tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax'),
    ], **kwds)

classifier (load_data, make_small_model,
            model_name = 'small_fashion_mnist',
            outdir = '/tmp',
            cm_plot_args = dict (hrotate = True))
