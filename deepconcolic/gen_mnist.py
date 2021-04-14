#!/usr/bin/env python3
from training import *
from datasets import load_mnist_data as load_data

def make_tiny_model (input_shape, **kwds):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, (3, 3), input_shape = input_shape),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax'),
    ], **kwds)

def make_small_model (input_shape, **kwds):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), input_shape = input_shape),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(42),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax'),
    ], **kwds)

def make_small_maxp_model (input_shape, **kwds):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), input_shape = input_shape),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(42),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax'),
    ], **kwds)

def make_medium_model (input_shape, **kwds):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(5, (3, 3), input_shape = input_shape),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(5, (5, 5)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(3, (7, 7)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax'),
    ], **kwds)

def make_large_model (input_shape, **kwds):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape = input_shape),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(16, (5, 5)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(8, (7, 7)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax'),
    ], **kwds)

# classifier (load_data, make_tiny_model,
#             model_name = 'mnist_tiny',
#             epochs = 20)

# classifier (load_data, make_small_model,
#             model_name = 'mnist_small',
#             epochs = 20)

# classifier (load_data, make_small_model,
#             model_name = 'mnist_small_overfitting',
#             early_stopping = False,
#             epochs = 50)

classifier (load_data, make_small_maxp_model,
            model_name = 'mnist_small_maxp',
            epochs = 20)

# classifier (load_data, make_medium_model,
#             model_name = 'mnist_medium',
#             epochs = 20)

# classifier (load_data, make_large_model,
#             model_name = 'mnist_overfitting',
#             early_stopping = False,
#             epochs = 20)
