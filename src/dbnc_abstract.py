from utils import *
from dbnc import BFcLayer, BNAbstraction, abstract_layer_setup
import argparse
import datasets

# ---

def load_model (filename):
  tf.compat.v1.disable_eager_execution ()
  return keras.models.load_model (filename)

def load_dataset (name):
  train, test, _, _, _ = datasets.load_by_name (name)
  return raw_datat (*train, name), raw_datat (*test, name)

def load_bn_abstr (dnn, filename):
  return BNAbstraction.from_file (dnn, filename)

def eval_coverages (dnn, bn_abstr, inputs):
  acts = eval_batch (dnn, inputs, allow_input_layer = True,
                     layer_indexes = [ fl.layer_index for fl in bn_abstr.flayers ])
  bn_abstr.fit_activations (acts)
  print ('BFC:', bn_abstr.bfc_coverage ())
  print ('BFdC:', bn_abstr.bfdc_coverage ())

# ---

parser = argparse.ArgumentParser (description = 'BN abstraction manager')
parser.add_argument ("--dataset", dest='dataset', required = True,
                     help = "selected dataset", choices = datasets.choices)
parser.add_argument ('--model', dest='model', required = True,
                     help = 'neural network model (.h5)')
parser.add_argument ('--rng-seed', dest="rng_seed", metavar="SEED", type=int,
                     help="Integer seed for initializing the internal random number "
                    "generator, and therefore get some(what) reproducible results")
subparsers = parser.add_subparsers (title = 'sub-commands', required = True)

# ---

parser_create = subparsers.add_parser ('create')
parser_create.add_argument ("--layers", dest = "layers", nargs = "+", metavar = "LAYER",
                            help = 'considered layers (given by name or index)')
parser_create.add_argument ('output', metavar = 'PKL',
                            help = 'output BN abstraction (.pkl)')

def create (test_object, args):
  if args.layers is not None:
    try:
      test_object.set_layer_indices (int (l) if l.isdigit () else l
                                     for l in args.layers)
    except ValueError as e:
      sys.exit (e)
  setup_layer = lambda l, i, **kwds: \
    abstract_layer_setup (l, i, 2, 5, discr_n_jobs = 8)
  clayers = get_cover_layers (test_object.dnn, setup_layer,
                              layer_indices = test_object.layer_indices,
                              activation_of_conv_or_dense_only = False,
                              exclude_direct_input_succ = False,
                              exclude_output_layer = False)
  bn_abstr = BNAbstraction (clayers, dump_abstraction = False)
  indexes = np.arange (min (1000, len (test_object.train_data.data)))
  lazy_activations_on_indexed_data (bn_abstr.initialize,
                                    test_object.dnn,
                                    test_object.train_data,
                                    indexes,
                                    [fl.layer_index for fl in clayers])
  bn_abstr.dump_abstraction (pathname = args.output)

parser_create.set_defaults (func = create)

# ---

parser_check = subparsers.add_parser ('check')
parser_check.add_argument ('input', metavar = 'PKL',
                           help = 'input BN abstraction (.pkl)')
parser_check.add_argument ('--size', '-s', type = int, default = 100,
                           help = 'test dataset size (default is 100)')

def check (test_object, args):
  bn_abstr = load_bn_abstr (test_object.dnn, args.input)
  test_sample = np.random.default_rng().choice \
    (a = test_object.raw_data.data, axis = 0,
     size = min (len (test_object.raw_data.data), args.size))
  eval_coverages (test_object.dnn, bn_abstr, test_sample)

parser_check.set_defaults (func = check)

# ---

if __name__=="__main__":
  args = parser.parse_args ()
  
  # Initialize with random seed first, if given:
  try: rng_seed (args.rng_seed)
  except ValueError as e:
    sys.exit ("Invalid argument given for `--rng-seed': {}".format (e))

  test_object = test_objectt (load_model (args.model),
                              *load_dataset (args.dataset))
  
  if 'func' in args:
    args.func (test_object, args)
  else:
    parser.print_help ()
    sys.exit (1)

# ---
