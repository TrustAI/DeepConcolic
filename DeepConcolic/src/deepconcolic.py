import argparse
import sys
sys.path.append('DeepConcolic/src')
import yaml
from pathlib import Path
from utils import *
from bounds import UniformBounds, StatBasedInputBounds
from deepconcolic_fuzz import deepconcolic_fuzz
import datasets
import filters

def deepconcolic(criterion, norm, test_object, report_args,
                 engine_args = {},
                 norm_args = {},
                 dbnc_spec = {},
                 input_bounds = None,
                 run_engine = True,
                 **engine_run_args):
  test_object.check_layer_indices (criterion)
  engine = None
  if criterion=='nc':                   ## neuron cover
    from nc import setup as nc_setup
    if norm=='linf':
      from pulp_norms import LInfPulp
      from nc_pulp import NcPulpAnalyzer
      engine = nc_setup (test_object = test_object,
                         engine_args = engine_args,
                         setup_analyzer = NcPulpAnalyzer,
                         input_metric = LInfPulp (**norm_args),
                         input_bounds = input_bounds)
    elif norm=='l0':
      from nc_l0 import NcL0Analyzer
      l0_args = copy.copy (norm_args)
      del l0_args['LB_hard']
      del l0_args['LB_noise']
      engine = nc_setup (test_object = test_object,
                         engine_args = engine_args,
                         setup_analyzer = NcL0Analyzer,
                         input_shape = test_object.raw_data.data[0].shape,
                         eval_batch = eval_batch_func (test_object.dnn),
                         l0_args = l0_args)
    else:
      print('\n not supported norm... {0}\n'.format(norm))
      sys.exit(0)
  elif criterion=='bfc':                ## feature cover
    from dbnc import setup as dbnc_setup
    from dbnc import BFcCriterion
    print ("DBNC Spec:\n", yaml.dump (dbnc_spec), sep='')
    if norm == 'linf':
      from pulp_norms import LInfPulp
      from dbnc_pulp import BFcPulpAnalyzerWithLinearExtrapolation as Analyzer
      engine = dbnc_setup (**dbnc_spec,
                           test_object = test_object,
                           engine_args = engine_args,
                           setup_criterion = BFcCriterion,
                           setup_analyzer = Analyzer,
                           input_metric = LInfPulp (**norm_args),
                           input_bounds = input_bounds,
                           outdir = report_args['outdir'])
    else:
      sys.exit ('\n not supported norm... {0}\n'.format(norm))
  elif criterion=='bfdc':               ## feature-dependence cover
    from dbnc import setup as dbnc_setup
    from dbnc import BFDcCriterion
    print ("DBNC Spec:\n", yaml.dump (dbnc_spec), sep='')
    if norm == 'linf':
      from pulp_norms import LInfPulp
      from dbnc_pulp import BFDcPulpAnalyzerWithLinearExtrapolation as Analyzer
      engine = dbnc_setup (**dbnc_spec,
                           test_object = test_object,
                           engine_args = engine_args,
                           setup_criterion = BFDcCriterion,
                           setup_analyzer = Analyzer,
                           input_metric = LInfPulp (**norm_args),
                           input_bounds = input_bounds,
                           outdir = report_args['outdir'])
    else:
      sys.exit ('\n not supported norm... {0}\n'.format(norm))
  elif criterion=='dbnc_stats':
    import dbnc_stats
    dbnc_stats.run (test_object, report_args['outdir'],
                    input_bounds = input_bounds)
  elif criterion=='ssc':
    from ssc import SScGANBasedAnalyzer, setup as ssc_setup
    linf_args = copy.copy (norm_args)
    del linf_args['LB_hard']
    del linf_args['LB_noise']
    engine = ssc_setup (test_object = test_object,
                        engine_args = engine_args,
                        setup_analyzer = SScGANBasedAnalyzer,
                        ref_data = test_object.raw_data,
                        input_bounds = input_bounds,
                        linf_args = linf_args)
  elif criterion=='ssclp':
    from pulp_norms import LInfPulp
    from mcdc_pulp import SScPulpAnalyzer
    from ssc import setup as ssc_setup
    engine = ssc_setup (test_object = test_object,
                        engine_args = engine_args,
                        setup_analyzer = SScPulpAnalyzer,
                        input_metric = LInfPulp (**norm_args),
                        input_bounds = input_bounds)
  elif criterion=='svc':
    from run_ssc import run_svc
    print('\n== Starting DeepConcolic tests for {0} =='.format (test_object))
    run_svc(test_object, report_args['outdir'].path)
  else:
    print('\n not supported coverage criterion... {0}\n'.format(criterion))
    sys.exit(0)

  if engine != None and run_engine:
    return engine, engine.run (**report_args, **engine_run_args)
  return engine


def main():

  parser=argparse.ArgumentParser(description='Concolic testing for neural networks' )
  parser.add_argument('--model', dest='model', default='-1',
                      help='the input neural network model (.h5)')
  parser.add_argument("--inputs", dest="inputs", default="-1",
                      help="the input test data directory", metavar="DIR")
  parser.add_argument("--outputs", dest="outputs", required=True,
                      help="the outputput test data directory", metavar="DIR")
  # parser.add_argument("--training-data", dest="training_data", default="-1",
  #                     help="the extra training dataset", metavar="DIR")
  parser.add_argument("--criterion", dest="criterion", default="nc",
                      help="the test criterion", metavar="nc, ssc...")
  parser.add_argument("--setup-only", dest="setup_only", action='store_true',
                      help="only setup the coverage critierion and analyzer, "
                      "and terminate before engine initialization and startup")
  parser.add_argument("--init", dest="init_tests", metavar="INT",
                      help="number of test samples to initialize the engine")
  parser.add_argument("--max-iterations", dest="max_iterations", metavar="INT",
                      help="maximum number of engine iterations (use < 0 for unlimited)",
                      default='-1')
  parser.add_argument("--save-all-tests", dest="save_all_tests", action="store_true",
                      help="save all generated tests in output directory; "
                      "only adversarial examples are kept by default")
  parser.add_argument("--rng-seed", dest="rng_seed", metavar="SEED", type=int,
                      help="Integer seed for initializing the internal random number "
                      "generator, and therefore get some(what) reproducible results")
  parser.add_argument("--labels", dest="labels", default="-1",
                      help="the default labels", metavar="FILE")
  parser.add_argument("--dataset", dest='dataset',
                      help="selected dataset", choices=datasets.choices)
  parser.add_argument("--extra-tests", dest='extra_testset_dirs', metavar="DIR",
                      type=Path, nargs="+",
                      help="additonal directories of test images")
  parser.add_argument("--vgg16-model", dest='vgg16',
                      help="vgg16 model", action="store_true")
  parser.add_argument("--filters", dest='filters', # nargs='+'
                      nargs=1, default=[],
                      help='additional filters used to put aside generated '
                      'test inputs that are too far from training data (there '
                      'is only one filter to choose from for now; the plural '
                      'is used for future-proofing)', choices=filters.choices)
  parser.add_argument("--norm", dest="norm", default="l0",
                      help="the norm metric", metavar="linf, l0")
  parser.add_argument("--input-rows", dest="img_rows", default="224",
                      help="input rows", metavar="INT")
  parser.add_argument("--input-cols", dest="img_cols", default="224",
                      help="input cols", metavar="INT")
  parser.add_argument("--input-channels", dest="img_channels", default="3",
                      help="input channels", metavar="INT")
  parser.add_argument("--cond-ratio", dest="cond_ratio", default="0.01",
                      help="the condition feature size parameter (0, 1]", metavar="FLOAT")
  parser.add_argument("--top-classes", dest="top_classes", default="1",
                      help="check the top-xx classifications", metavar="INT")
  parser.add_argument("--layers", dest="layers", nargs="+", metavar="LAYER",
                      help="test layers given by name or index")
  parser.add_argument("--feature-index", dest="feature_index", default="-1",
                      help="to test a particular feature map", metavar="INT")

  # fuzzing params
  parser.add_argument("--fuzzing", dest='fuzzing', help="to start fuzzing", action="store_true")
  parser.add_argument("--num-tests", dest="num_tests", default="1000",
                    help="number of tests to generate", metavar="INT")
  parser.add_argument("--num-processes", dest="num_processes", default="1",
                    help="number of processes to use", metavar="INT")
  parser.add_argument("--sleep-time", dest="stime", default="4",
                    help="fuzzing sleep time", metavar="INT")

  # DBNC-specific params
  parser.add_argument("--dbnc-spec", dest="dbnc_spec", default="{}",
                      help="Feature extraction and discretisation specification",
                      metavar="SPEC")
  
  args=parser.parse_args()

  # Initialize with random seed first, if given:
  try: rng_seed (args.rng_seed)
  except ValueError as e:
    sys.exit ("Invalid argument given for `--rng-seed': {}".format (e))

  outs = args.outputs
  criterion=args.criterion
  cond_ratio=float(args.cond_ratio)
  top_classes=int(args.top_classes)

  test_data=None
  train_data = None
  img_rows, img_cols, img_channels = int(args.img_rows), int(args.img_cols), int(args.img_channels)

  dnn = None
  inp_ub = 1
  save_input = None
  amplify_diffs = True
  lower_bound_metric_hard = .01
  lower_bound_metric_noise = 0
  input_bounds = UniformBounds (0.0, 1.0)

  # fuzzing_params
  if args.inputs!='-1':
    file_list = []
    xs=[]
    np1 ('Loading input data from {}... '.format (args.inputs))
    for path, subdirs, files in os.walk(args.inputs):
      for name in files:
        fname=(os.path.join(path, name))
        file_list.append(fname) # fuzzing params
        if fname.endswith('.jpg') or fname.endswith('.png'):
          try:
            image = cv2.imread(fname)
            image = cv2.resize(image, (img_rows, img_cols))
            image = image.astype('float')
            xs.append((image))
          except: pass
    x_test = np.asarray(xs)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
    test_data = raw_datat(x_test, None)
    print (len(xs), 'loaded.')
  elif args.dataset in datasets.choices:
    print ('Loading {} dataset... '.format (args.dataset), end = '', flush = True)
    (x_train, y_train), (x_test, y_test), dims, kind, _ = datasets.load_by_name (args.dataset)
    test_data = raw_datat(x_test, y_test, args.dataset)
    train_data = raw_datat(x_train, y_train, args.dataset)
    save_input = save_an_image if kind in datasets.image_kinds else \
                 save_in_csv ('new_inputs') if len (dims) == 1 else \
                 None
    amplify_diffs = kind in datasets.image_kinds
    lower_bound_metric_hard = 1 / 255
    lower_bound_metric_noise = 0.1
    input_bounds = UniformBounds () if kind in datasets.image_kinds else \
                   StatBasedInputBounds (hard_bounds = UniformBounds (-1.0, 1.0)) \
                   if kind in datasets.normalized_kinds else StatBasedInputBounds ()
    print ('done.')
  else:
    sys.exit ('Missing input dataset')

  if args.extra_testset_dirs is not None:
    for d in args.extra_testset_dirs:
      np1 (f'Loading extra image testset from `{str(d)}\'... ')
      x, y, _, _, _ = datasets.images_from_dir (str (d))
      x_test = np.concatenate ((x_test, x))
      y_test = np.concatenate ((y_test, y))
      print ('done')

  input_filters = []
  for f in args.filters:
    input_filters += xlist (filters.by_name (f))

  if args.fuzzing:
    pass
  elif args.model!='-1':
    # NB: Eager execution needs to be disabled before any model loading.
    tf.compat.v1.disable_eager_execution ()
    dnn = keras.models.load_model (args.model)
    dnn.summary()
  elif args.vgg16:
    # NB: Eager execution needs to be disabled before any model loading.
    tf.compat.v1.disable_eager_execution ()
    dnn = keras.applications.VGG16 ()
    inp_ub = 255
    lower_bound_metric_hard = 1/255
    dnn.summary()
    save_input = save_an_image
  else:
    sys.exit ('Missing input neural network')


  test_object=test_objectt(dnn, test_data, train_data)
  test_object.cond_ratio = cond_ratio
  test_object.top_classes = top_classes
  test_object.inp_ub = inp_ub
  if args.layers is not None:
    try:
      test_object.set_layer_indices (int (l) if l.isdigit () else l
                                     for l in args.layers)
    except ValueError as e:
      sys.exit (e)
    if args.feature_index!='-1':
      test_object.feature_indices=[]
      test_object.feature_indices.append(int(args.feature_index))
      print ('feature index specified:', test_object.feature_indices)
  # if args.training_data!='-1':          # NB: never actually used
  #   tdata=[]
  #   print ('To load the extra training data...')
  #   for path, subdirs, files in os.walk(args.training_data):
  #     for name in files:
  #       fname=(os.path.join(path, name))
  #       if fname.endswith('.jpg') or fname.endswith('.png'):
  #         try:
  #           image = cv2.imread(fname)
  #           image = cv2.resize(image, (img_rows, img_cols))
  #           image=image.astype('float')
  #           tdata.append((image))
  #         except: pass
  #   print ('The extra training data loaded: ', len(tdata))
  #   # test_object.training_data=tdata

  if args.labels!='-1':             # NB: only used in run_scc.run_svc
    labels=[]
    lines = [line.rstrip('\n') for line in open(args.labels)]
    for line in lines:
      for l in line.split():
        labels.append(int(l))
    test_object.labels=labels

  init_tests = int (args.init_tests) if args.init_tests is not None \
               else None
  max_iterations = int (args.max_iterations)

  # fuzzing params
  if args.fuzzing:
    deepconcolic_fuzz(test_object, outs, args.model, int(args.stime), file_list,
                      num_tests = int(args.num_tests),
                      num_processes = int(args.num_processes))
    sys.exit(0)

  # DBNC-specific parameters:
  try:
    if args.dbnc_spec != "{}" and os.path.exists(args.dbnc_spec):
      with open(args.dbnc_spec, 'r') as f:
        dbnc_spec = yaml.safe_load (f)
    else:
      dbnc_spec = yaml.safe_load (args.dbnc_spec)
  except yaml.YAMLError as exc:
    sys.exit(exc)

  deepconcolic (args.criterion, args.norm, test_object,
                report_args = { 'outdir': OutputDir (outs, log = True),
                                'save_new_tests': args.save_all_tests,
                                'save_input_func': save_input,
                                'amplify_diffs': amplify_diffs },
                norm_args = { 'factor': .25,
                              'LB_hard': lower_bound_metric_hard,
                              'LB_noise': lower_bound_metric_noise },
                engine_args = { 'custom_filters': input_filters },
                dbnc_spec = dbnc_spec,
                input_bounds = input_bounds,
                run_engine = not args.setup_only,
                initial_test_cases = init_tests,
                max_iterations = max_iterations)

if __name__=="__main__":
  try:
    main ()
  except KeyboardInterrupt:
    sys.exit('Interrupted.')
