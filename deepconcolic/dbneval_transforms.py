#!/usr/bin/env python3
from dbnabstr import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---

def imgen (test_object, test_size):
  datagen = ImageDataGenerator (
    # rotation_range = 90, # randomly rotate images in the range (degrees, 0 to 180)
    # zoom_range = 0.3,   # Randomly zoom image 
    # shear_range = 0.3, # shear angle in counter-clockwise direction in degrees  
    # width_shift_range=0.08, # randomly shift images horizontally (fraction of total width)
    # height_shift_range=0.08, # randomly shift images vertically (fraction of total height)
    # vertical_flip=True,      # randomly flip images
  )
  np1 ('| Fitting image data generator... ')
  datagen.fit (test_object.train_data.data)
  c1 ('done')
  X, Y = [], []
  test_size = min (test_size, len (test_object.raw_data.data))
  np1 (f'| Generating {test_size} new images... ')
  for x, y in datagen.flow (test_object.raw_data.data[:test_size],
                            test_object.raw_data.labels[:test_size]):
    X.extend (x)
    Y.extend (y)
    if len (X) >= test_size:
      break
  c1 ('done')
  return raw_datat (np.array (X), np.array (Y), 'transformed')

def check_transformed (test_object, *args, test_size = 100, **kwds):
  check (test_object, *args, test_size = test_size, **kwds, transformed_data = {
    'transformed': imgen (test_object, test_size)
  })

parser_check.set_defaults (func = check_transformed)

# ---

if __name__=="__main__":
  main ()

# ---
