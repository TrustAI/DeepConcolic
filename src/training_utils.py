import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import io
import itertools

# https://www.tensorflow.org/tensorboard/image_summaries

def plot_to_image(figure):
  """
  Converts the matplotlib plot specified by 'figure' to a PNG image
  and returns it. The supplied figure is closed and inaccessible after
  this call.
  """
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def image_grid(dataset, class_names, cmap = plt.cm.binary):
  """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
  # Create a figure to contain the plot.
  fig = plt.figure (figsize = (10,10))
  for i in range(25):
    # Start next subplot.
    plt.subplot(5, 5, i + 1, title=class_names[dataset[1][i]])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(dataset[0][i], cmap = cmap)
  return fig

def plot_confusion_matrix(cm, class_names, hrotate = False):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45 if hrotate else None)
  plt.yticks(tick_marks, class_names)

  # Normalize the confusion matrix.
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

# ---

def log_confusion_matrix(fw, model, class_names, test_data, epoch, logs, **kwds):
  # Use the model to predict the values from the validation dataset.
  test_pred = np.argmax(model.predict (test_data[0]), axis=1)

  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix (test_data[1], test_pred)
  # Log the confusion matrix as an image summary.
  fig = plot_confusion_matrix (cm, class_names = class_names, **kwds)

  # Log the confusion matrix as an image summary.
  with fw.as_default():
    tf.summary.image ('Confusion matrix at epoch {}'.format (epoch),
                      plot_to_image (fig), step = epoch)

def log_confusion_matrix_callback (fw, model, class_names, test_data, **kwds):
  return tf.keras.callbacks.LambdaCallback (
    on_epoch_end = lambda epoch, logs: \
    log_confusion_matrix (fw, model, class_names, test_data, epoch, logs, **kwds))

# Image-specific

def log_img_dataset (fw, dataset_name, dataset):
  with fw.as_default ():
    images = np.reshape (dataset, (-1,) + dataset[0].shape)
    tf.summary.image (dataset_name, images, step = 0)

def log_25_img_dataset_grid (fw, class_names, dataset_name, dataset, **kwds):
  fig = image_grid (dataset, class_names, **kwds)
  with fw.as_default ():
    tf.summary.image (dataset_name, plot_to_image (fig), step = 0)

