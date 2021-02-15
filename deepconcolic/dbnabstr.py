#!/usr/bin/env python3
from utils import *
from utils_funcs import rng_seed
from utils_args import *
from dbnc import BNAbstraction, layer_setup, interval_repr
from tabulate import tabulate
import datasets
import plugins
import scipy

# ---

def load_model (filename, print_summary = True):
  tf.compat.v1.disable_eager_execution ()
  dnn = keras.models.load_model (filename)
  if print_summary:
    dnn.summary ()
  return dnn

def load_dataset (name):
  train, test, _, _, _ = datasets.load_by_name (name)
  return raw_datat (*train, name), raw_datat (*test, name)

def fit_data (dnn, bn_abstr, data, indexes):
  indexes = np.arange (len (data.data)) if indexes is None else indexes
  np1 (f'| Fitting BN with {len (indexes)} samples... ')
  lazy_activations_on_indexed_data \
    (bn_abstr.fit_activations, dnn, data, indexes,
     layer_indexes = [ fl.layer_index for fl in bn_abstr.flayers ],
     pass_kwds = False)
  c1 ('done')

def fit_data_sample (dnn, bn_abstr, data, size, rng):
  bn_abstr.reset_bn ()
  idxs = np.arange (len (data.data))
  if size is not None:
    idxs = rng.choice (a = idxs, axis = 0, size = min (size, len (idxs)))
  fit_data (dnn, bn_abstr, data, idxs)

def eval_coverages (dnn, bn_abstr, data, size, rng):
  fit_data_sample (dnn, bn_abstr, data, size, rng)
  return dict (bfc = bn_abstr.bfc_coverage (),
               bfdc = bn_abstr.bfdc_coverage ())

def eval_probas (dnn, bn_abstr, data, size, rng, indexes = None):
  if indexes is None:
    indexes = np.arange (len (data.data))
    if size is not None:
      indexes = rng.choice (a = indexes, axis = 0, size = min (size, len (indexes)))
  probas = lazy_activations_on_indexed_data \
    (bn_abstr.activations_probas, dnn, data, indexes,
     layer_indexes = [ fl.layer_index for fl in bn_abstr.flayers ],
     pass_kwds = False)
  return dict (probas = probas,
               stats = scipy.stats.describe (probas))

# ---

parser = argparse.ArgumentParser (description = 'BN abstraction manager')
parser.add_argument ('--dataset', dest='dataset', required = True,
                     help = "selected dataset", choices = datasets.choices)
parser.add_argument ('--model', dest='model', required = True,
                     help = 'neural network model (.h5)')
parser.add_argument ('--rng-seed', dest="rng_seed", metavar="SEED", type=int,
                     help="Integer seed for initializing the internal random number "
                    "generator, and therefore get some(what) reproducible results")
subparsers = parser.add_subparsers (title = 'sub-commands', required = True,
                                    dest = 'cmd')

# ---

ap_create = subparsers.add_parser ('create')
add_abstraction_arg (ap_create)
ap_create.add_argument ("--layers", dest = "layers", nargs = "+", metavar = "LAYER",
                        help = 'considered layers (given by name or index)')
ap_create.add_argument ('--train-size', '-ts', type = int,
                        help = 'train dataset size (default is all)',
                        metavar = 'INT')
ap_create.add_argument ('--feature-extraction', '-fe',
                        choices = ('pca', 'ipca', 'ica',), default = 'pca',
                        help = 'feature extraction technique (default is pca)')
ap_create.add_argument ('--num-features', '-nf', type = int, default = 2,
                        help = 'number of extracted features for each layer '
                        '(default is 2)', metavar = 'INT')
ap_create.add_argument ('--num-intervals', '-ni', type = int, default = 2,
                        help = 'number of intervals for each extracted feature '
                        '(default is 2)', metavar = 'INT')
ap_create.add_argument ('--discr-strategy', '-ds',
                        choices = ('uniform', 'quantile',), default = 'uniform',
                        help = 'discretisation strategy (default is uniform)')
ap_create.add_argument ('--extended-discr', '-xd', action = 'store_true',
                        help = 'use extended partitions')

def create (test_object,
            layers = None,
            train_size = None,
            feature_extraction = None,
            num_features = None,
            num_intervals = None,
            discr_strategy = None,
            extended_discr = False,
            abstraction = None,
            **_):
  if layers is not None:
    test_object.set_layer_indices (int (l) if l.isdigit () else l for l in layers)
  n_bins = num_intervals - 2 if extended_discr else num_intervals
  if n_bins < 1:
    raise ValueError (f'The total number of intervals for each extracted feature '
                      f'must be strictly positive (got {n_bins} '
                      f'with{"" if extended_discr else "out"} extended discretization)')
  feats = dict (decomp = feature_extraction, n_components = num_features)
  discr = dict (strategy = discr_strategy, n_bins = n_bins, extended = extended_discr)
  setup_layer = lambda l, i, **kwds: \
    layer_setup (l, i, feats, discr, discr_n_jobs = 8)
  clayers = get_cover_layers \
    (test_object.dnn, setup_layer, layer_indices = test_object.layer_indices,
     activation_of_conv_or_dense_only = False,
     exclude_direct_input_succ = False,
     exclude_output_layer = False)
  bn_abstr = BNAbstraction (clayers, dump_abstraction = False)
  lazy_activations_on_indexed_data \
    (bn_abstr.initialize, test_object.dnn, test_object.train_data,
     np.arange (min (train_size or sys.maxsize, len (test_object.train_data.data))),
     [fl.layer_index for fl in clayers])
  bn_abstr.dump_abstraction (pathname = abstraction_path (abstraction))

ap_create.set_defaults (cmd = create)

# ---

ap_show = subparsers.add_parser ('show')
add_abstraction_arg (ap_show)

def show (test_object,
          abstraction = None,
          **_):
  rng = np.random.default_rng (randint ())
  bn_abstr = BNAbstraction.from_file (test_object.dnn, abstraction_path (abstraction),
                                      log = False)
  table = [
    [str (fl)] +
    [ '\n'.join (str (f) for f in range (fl.num_features)) ] +
    [ '\n'.join (', '.join (interval_repr (i) for i in fi_intervals)
                 for fi_intervals in fl.intervals) ]
    for fl in bn_abstr.flayers
  ]
  h1 ('Extracted Features and Associated Intervals')
  p1 (tabulate (table, headers = ('Layer', 'Feature', 'Intervals')))

ap_show.set_defaults (cmd = show)

# ---

ap_check = subparsers.add_parser ('check')
add_abstraction_arg (ap_check)
ap_check.add_argument ('--train-size', '-ts', type = int,
                       help = 'train dataset size (default is all)')
# ap_check.add_argument ('--trained-bn', metavar = 'YML',
#                        help = 'BN fit with training data (.yml)')
ap_check.add_argument ('--size', '-s', dest = 'test_size',
                       type = int, default = 100,
                       help = 'test dataset size (default is 100)')
ap_check.add_argument ('--summarize-probas', '-p', action = 'store_true',
                       help = 'fit the BN with all training data and then '
                       'assess the probability of the test dataset')

def check (test_object,
           abstraction = None,
           test_size = 100,
           train_size = None,
           summarize_probas = False,
           summarize_coverages = True,
           transformed_data = {},
           **_):
  rng = np.random.default_rng (randint ())
  tests = dict (raw = test_object.raw_data, **transformed_data)
  bn_abstr = BNAbstraction.from_file (test_object.dnn, abstraction_path (abstraction))

  if summarize_probas:
    fit_data_sample (test_object.dnn, bn_abstr, test_object.train_data, train_size, rng)
    probs = {
      t: eval_probas (test_object.dnn, bn_abstr, tests[t], test_size, rng)
      for t in tests
    }
    print (probs)

  if summarize_coverages:
    covs = {
      t: eval_coverages (test_object.dnn, bn_abstr, tests[t], test_size, rng)
      for t in tests
    }
    print (covs)

ap_check.set_defaults (cmd = check)

# ---

def get_args (args = None, parser = parser):
  args = parser.parse_args () if args is None else args
  # Initialize with random seed first, if given:
  try: rng_seed (args.rng_seed)
  except ValueError as e:
    sys.exit (f'Invalid argument given for \`--rng-seed\': {e}')
  return args


def main (args = None, parser = parser, pp_args = (pp_abstraction_arg (),)):
  try:
    args = get_args (args, parser = parser)
    # args = reduce (lambda args, pp: pp (args), pp_args, args)
    for pp in pp_args: pp (args)
    test_object = test_objectt (load_model (args.model),
                                *load_dataset (args.dataset))

    if 'cmd' in args:
      args.cmd (test_object, **vars (args))
    else:
      parser.print_help ()
      sys.exit (1)
  except ValueError as e:
    sys.exit (f'Error: {e}')
  except FileNotFoundError as e:
    sys.exit (f'Error: {e}')
  except KeyboardInterrupt:
    sys.exit ('Interrupted.')

# ---

if __name__=="__main__":
  main ()

# ---
