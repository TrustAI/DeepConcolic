import argparse
import sys
import os
import numpy as np

from tensorflow import keras
from fuzz_variables import *

def flag_diff(origins, mutants):
    ## (TBD) if origins and mutants are different enough
    return False

def flag_coverage(origins, mutants):
    ## (TBD) if coverage is updated
    return True

def main():
  parser=argparse.ArgumentParser(description='This is a template to fuzz the DNN model' )
  parser.add_argument(
    '--model', dest='model', default='-1', help='the input neural network model (.h5)')
  parser.add_argument("--input-rows", dest="img_rows", default="28",
                    help="input rows", metavar="INT")
  parser.add_argument("--input-cols", dest="img_cols", default="28",
                    help="input cols", metavar="INT")
  parser.add_argument("--input-channels", dest="img_channels", default="1",
                    help="input channels", metavar="INT")
  parser.add_argument('--origins', action='store', nargs='+', help='the original inputs')
  parser.add_argument('--mutants', action='store', nargs='+', help='the mutant inputs')

  args = parser.parse_args()
  #
  img_rows, img_cols, img_channels = int(args.img_rows), int(args.img_cols), int(args.img_channels)
  #
  xs = []
  mutants = []
  try :
    ## to read mutants
    for i in range(0, len(args.mutants)):
        print (args.mutants[i])
        if img_channels == 1:
            x=keras.preprocessing.image.load_img(args.mutants[i], target_size=(img_rows, img_cols), color_mode = "grayscale")
            x=np.expand_dims(x,axis=2)
        else:
          x=keras.preprocessing.image.load_img(args.mutants[i], target_size=(img_rows, img_cols))
        x=np.expand_dims(x,axis=0)
        mutants.append(x)
    mutants = np.vstack(mutants)
    mutants = mutants.reshape(mutants.shape[0], img_rows, img_cols, img_channels)
    ## to read origins
    for i in range(0, len(args.origins)):
        print (args.origins[i])
        if img_channels == 1:
            x=keras.preprocessing.image.load_img(args.origins[i], target_size=(img_rows, img_cols), color_mode = "grayscale")
            x=np.expand_dims(x,axis=2)
        else:
            x=keras.preprocessing.image.load_img(args.origins[i], target_size=(img_rows, img_cols))
        x=np.expand_dims(x,axis=0)
        xs.append(x)
    xs = np.vstack(xs)
    xs = xs.reshape(xs.shape[0], img_rows, img_cols, img_channels)
    origins = xs
  except:
      print ('corrupt input...to exit...')
      #os.system('rm {0}'.format(args.mutants[0])) ## remove corrupted mutants
      sys.exit(SIG_NORMAL)

  if flag_diff(origins, mutants) or not flag_coverage(origins, mutants):
      #os.system('rm {0}'.format(args.mutants[0])) ## remove corrupted mutants
      sys.exit(SIG_NORMAL)

  dnn = keras.models.load_model(args.model)
  #dnn.summary()
  
  ys1 = dnn.predict_classes(origins)
  ys2 = dnn.predict_classes(mutants)

  print (ys1, ys2)

  if not ys1==ys2: sys.exit(SIG_ADV)
  else:
      #os.system('rm {0}'.format(args.mutants[0])) ## remove corrupted mutants
      sys.exit(SIG_COV)

if __name__=="__main__":
    main()

