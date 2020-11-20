from training import *
from datasets import load_fashion_mnist_data as load_data

def make_small_model (input_shape, **kwds):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, (5, 5), input_shape = input_shape),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax'),
    ], **kwds)

def make_small_maxp_model (input_shape, **kwds):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, (5, 5), input_shape = input_shape),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax'),
    ], **kwds)

def make_medium_maxp_model (input_shape, **kwds):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape = input_shape),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (5, 5)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax'),
    ], **kwds)

# classifier (load_data, make_small_model,
#             model_name = 'fashion_mnist_small',
#             outdir = '/tmp',
#             cm_plot_args = dict (hrotate = True))

# classifier (load_data, make_small_maxp_model,
#             model_name = 'fashion_mnist_small_maxp',
#             outdir = '/tmp',
#             cm_plot_args = dict (hrotate = True))

classifier (load_data, make_medium_maxp_model,
            model_name = 'fashion_mnist_medium_maxp',
            outdir = '/tmp',
            epochs = 50,
            cm_plot_args = dict (hrotate = True))
