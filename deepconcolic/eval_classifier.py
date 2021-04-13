#!/usr/bin/env python3
import argparse
import datasets
import plugins
from pathlib import Path
from utils_io import *
from utils_funcs import *

def report (dnn, test_data):
  from utils import predictions
  from sklearn.metrics import classification_report

  X_test, Y_test = test_data.data, test_data.labels
  h1 ('Classificaion Report')
  tp1 (f'Evaluating DNN on {len (X_test)} test samples...')
  Y_dnn = predictions (dnn, X_test)
  print (classification_report (Y_test, Y_dnn))

def main():

  parser=argparse.ArgumentParser(description='Concolic testing for neural networks' )
  parser.add_argument('--model', dest='model', default='-1',
                      help='the input neural network model (.h5)')
  parser.add_argument("--vgg16-model", dest='vgg16',
                      help="use keras's default VGG16 model (ImageNet)",
                      action="store_true")
  # parser.add_argument("--inputs", dest="inputs", default="-1",
  #                     help="the input test data directory", metavar="DIR")
  # parser.add_argument("--rng-seed", dest="rng_seed", metavar="SEED", type=int,
  #                     help="Integer seed for initializing the internal random number "
  #                     "generator, and therefore get some(what) reproducible results")
  parser.add_argument("--dataset", dest='dataset',
                      help="selected dataset", choices=datasets.choices)
  parser.add_argument("--extra-tests", dest='extra_testset_dirs', metavar="DIR",
                      type=Path, nargs="+",
                      help="additonal directories of test images")

  args=parser.parse_args()

  # # Initialize with random seed first, if given:
  # try: rng_seed (args.rng_seed)
  # except ValueError as e:
  #   sys.exit ("Invalid argument given for `--rng-seed': {}".format (e))

  dnn = None
  X_test, Y_test, X_train, Y_train = [], [], [], []

  # fuzzing_params
  # if args.inputs!='-1':
  #   file_list = []
  #   xs=[]
  #   np1 (f'Loading input data from `{args.inputs}\'... ')
  #   for path, subdirs, files in os.walk(args.inputs):
  #     for name in files:
  #       fname=(os.path.join(path, name))
  #       file_list.append(fname) # fuzzing params
  #       if fname.endswith('.jpg') or fname.endswith('.png'):
  #         try:
  #           image = cv2.imread(fname)
  #           image = cv2.resize(image, (img_rows, img_cols))
  #           image = image.astype('float')
  #           xs.append((image))
  #         except: pass
  #   X_test = np.asarray(xs)
  #   X_test = X_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
  #   c1 (f'{len(xs)} loaded.')
  # el
  assert args.dataset in datasets.choices
  from utils import dataset_dict, load_model

  dd = dataset_dict (args.dataset)
  test_data = dd['test_data']
  del dd

  if args.extra_testset_dirs is not None:
    for d in args.extra_testset_dirs:
      np1 (f'Loading extra image testset from `{str(d)}\'... ')
      x, y, _, _, _ = datasets.images_from_dir (str (d))
      X_test = np.concatenate ((X_test, x))
      y_test = np.concatenate ((Y_test, y))
      print ('done.')

  dnn = load_model (args.model)
  dnn.summary ()

  report (dnn, test_data)

if __name__=="__main__":
  try:
    main ()
  except KeyboardInterrupt:
    sys.exit('Interrupted.')
