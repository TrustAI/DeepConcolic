#!/usr/bin/env python3
#
# Human Activity Recognition
# https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
#
from training import *
from datasets import load_openml_data_lambda

common_args = dict (load_data_args = dict (datadir = '/tmp/sklearn_data'),
                    epochs = 40,
                    validate_on_test_data = True,
                    shuffle = False,
                    outdir = '/tmp')

def make_model (input_shape, **kwds):
    return tf.keras.models.Sequential([
        tf.keras.layers.Reshape ((187, 3), input_shape = input_shape),
        tf.keras.layers.Conv1D(filters = 24, kernel_size = 3, activation = 'relu'),
        tf.keras.layers.Conv1D(filters = 24, kernel_size = 3, activation = 'relu'),
        tf.keras.layers.MaxPooling1D((3,)),
        tf.keras.layers.Conv1D(filters = 24, kernel_size = 3, activation = 'relu'),
        tf.keras.layers.Conv1D(filters = 24, kernel_size = 3, activation = 'relu'),
        tf.keras.layers.MaxPooling1D((3,)),
        tf.keras.layers.Conv1D(filters = 16, kernel_size = 3, activation = 'relu'),
        tf.keras.layers.Conv1D(filters = 16, kernel_size = 3, activation = 'relu'),
        tf.keras.layers.MaxPooling1D((3,)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation = 'relu'),
        tf.keras.layers.Dense(92, activation = 'relu'),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(6),
        tf.keras.layers.Activation('softmax'),
    ], **kwds)

# classifier (load_openml_data_lambda ('har'),
#             make_model,
#             model_name = 'har_conv1d',
#             **common_args)

# def make_model (input_shape, **kwds):
#     return tf.keras.models.Sequential([
#         tf.keras.layers.Reshape ((187, 3), input_shape = input_shape),
#         tf.keras.layers.Conv1D(filters = 64, kernel_size = 3),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.Conv1D(filters = 32, kernel_size = 3),
#         tf.keras.layers.Activation('relu'),
#         # tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(200),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.Dense(100),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.Dense(6),
#         tf.keras.layers.Activation('softmax'),
#     ], **kwds)

# classifier (load_openml_data_lambda ('har'),
#             make_model,
#             model_name = 'har_conv1d',
#             **common_args)

# ---

# # Very bad small model, for testing purposes only
# def make_small_dense_model (input_shape, **kwds):
#     return tf.keras.models.Sequential([
#         tf.keras.layers.Dense(187, input_shape = input_shape),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.Dense(92),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.Dense(6),
#         tf.keras.layers.Activation('softmax'),
#     ], **kwds)

# Very bad small model, for testing purposes only
def make_dense_model (input_shape, **kwds):
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(192, activation = 'relu', input_shape = input_shape),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(92, activation = 'relu'),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(6),
        tf.keras.layers.Activation('softmax'),
    ], **kwds)

classifier (load_openml_data_lambda ('har'),
            make_dense_model,
            model_name = 'har_dense',
            **common_args)
