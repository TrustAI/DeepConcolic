from __future__ import absolute_import, division, print_function, unicode_literals
# NB: see head of `datasets.py'
from training_utils import *
from datasets import image_kinds

print ("Using TensorFlow version:", tf.__version__)

def train_n_save_classifier (model, class_names, input_kind,
                             train_data, test_data = None,
                             optimizer = 'adam',
                             kind = 'sparse_categorical',
                             outdir = '/tmp',
                             early_stopping = True,
                             validate_on_test_data = False,
                             cm_plot_args = {},
                             **kwds):
    x_train, y_train = train_data
    path = outdir +'/'+ model.name
    log_dir = path + '_logs'
    fw_train = tf.summary.create_file_writer (log_dir + '/train')
    fw_confision_matrix = tf.summary.create_file_writer (log_dir + '/confusion_matrix')

    # Very basic & dumb test for detecting images...
    if input_kind in image_kinds:
        log_25_img_dataset_grid (fw_train, class_names, 'Training data (some)', train_data)

    model.summary ()
    loss, metric = (tf.losses.SparseCategoricalCrossentropy (from_logits=True),
                    tf.metrics.SparseCategoricalAccuracy ()) # if kind = 'sparse_categorical' else ?
    model.compile (optimizer = optimizer,
                   loss = loss,
                   metrics = [metric])
    callbacks = [
      tf.keras.callbacks.ModelCheckpoint (
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath = path + "_{epoch}",
        save_best_only = True, # Only save a model if `val_loss` has improved.
        monitor = "val_loss",
        verbose = 1,
      ),
      tf.keras.callbacks.TensorBoard (
        log_dir = log_dir,
        histogram_freq = 1, # How often to log histogram visualizations
        embeddings_freq = 1, # How often to log embedding visualizations
        update_freq = "epoch", # How often to write logs (default: once per epoch)
      ),
    ] + ([
      # https://www.tensorflow.org/guide/keras/train_and_evaluate#checkpointing_models
      tf.keras.callbacks.EarlyStopping (
        # Stop training when `val_loss` is no longer improving
        monitor = "val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta = 1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience = 3,
        verbose = 1,
      ),
    ] if early_stopping else []) + ([
      log_confusion_matrix_callback (
        fw_confision_matrix,
        model, class_names, test_data,
        **cm_plot_args),
    ] if test_data is not None else [])
    valargs = dict (validation_data = test_data) \
              if validate_on_test_data and test_data is not None \
              else {}
    model.fit (x_train, y_train,
               callbacks = callbacks,
               **{'epochs': 20,         # some defaults:
                  'shuffle': True,
                  'batch_size': 64,
                  'validation_split': 0.2,
                  **valargs,
                  **kwds})
    if test_data is not None:
        x_test, y_test = test_data
        print ('Performing final validation on given test data:')
        # Just check and show accuracy on "official" test data:
        _, test_accuracy = model.evaluate (x_test, y_test, verbose = 1)
        print ('Validation accuracy on given test data:', test_accuracy)
    print ('Saving model in', path + '.h5')
    model.save (path + '.h5')

# ---

def classifier (load_data, make_model, model_name = None, load_data_args = {}, **kwds):
    train_data, test_data, input_shape, input_kind, class_names = load_data (**load_data_args)
    train_n_save_classifier (make_model (input_shape, name = model_name),
                             class_names, input_kind, train_data, test_data, **kwds)

# ---
