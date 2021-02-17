#!/usr/bin/env python3
import argparse
from utils_funcs import random, rng_seed
from utils_io import *
from utils import *
from plotting import plt
from bounds import UniformBounds
from main import deepconcolic
import scripting
import json
import datasets
import plotting

# ---

# Some defaults:

train_size = 20000
max_iterations = 100

init_tests_range = (10, 100, 1000)
n_components_range = tuple (range (1, 6))
n_bins_range = tuple (range (1, 6, 2))
all_feat_extr_techs = ('pca', 'ica')

input_bounds = UniformBounds (0.0, 1.0)
norm_args = dict (factor = .25,
                  LB_hard = 1 / 255,
                  LB_noise = 0.1)

base_report_args = dict (save_new_tests = True,
                         save_input_func = save_an_image,
                         amplify_diffs = True)

# ---

# Arg parsing and general dataset and model setup
parser = argparse.ArgumentParser ( \
  description = 'Concolic testing for neural networks (stats script)')
subparsers = parser.add_subparsers (title = 'sub-commands')

# Options (potentially) shared by every sub-command:
parser.add_argument ("--outputs", dest = "outputs", required = True,
                     help = "the output directory", metavar = "DIR")

# ---

def load_dataset (name):
  print ('Loading {} dataset... '.format (name), end = '', flush = True)
  (x_train, y_train), (x_test, y_test), dims, kind, _ = datasets.load_by_name (name)
  train_data = raw_datat (x_train, y_train, name)
  test_data = raw_datat (x_test, y_test, name)
  print ('done.')
  return train_data, test_data

def load_model (model):
  # NB: Eager execution needs to be disabled before any model loading.
  tf.compat.v1.disable_eager_execution ()
  dnn = keras.models.load_model (model)
  dnn.summary()
  return dnn

def add_common_run_args (parser):
  parser.add_argument ('--model', dest='model', required = True,
                       help='the input neural network model (.h5)')
  parser.add_argument ("--dataset", dest='dataset', required = True,
                       help="selected dataset", choices=datasets.choices)
  parser.add_argument ("--layers", dest="layers", nargs="+", required = True,
                       help="test layers given by name or index")
  parser.add_argument ("--max-iterations", dest = "max_iterations",
                       metavar = "INT", type = int, default = max_iterations,
                       help = "maximum number of engine iterations "
                       f'(default is {max_iterations})')

def setup_run_common (args):
  test_object = test_objectt (load_model (args.model),
                              *load_dataset (args.dataset))
  test_object.set_layer_indices (int (l) if l.isdigit () else l
                                 for l in args.layers)
  return (test_object, args.outputs)

# ---

# NC: Engine setup and helper functions

def setup_results_file (go):
  return scripting.setup_results_file \
         (go, 'crit', 'run',
          'init_tests', 'total_iterations',
          'setup_time', 'init_time', 'run_time',
          'init_coverage', 'final_coverage',
          'num_tests', 'num_adversarials')

def generic_setup (outdir, init_tests, crit, test_object, **dc_kwargs):
  report_args = dict (**base_report_args, outdir = outdir)
  return deepconcolic (crit, 'linf',
                       test_object, report_args,
                       **dc_kwargs,
                       initial_test_cases = init_tests,
                       max_iterations = 0)

def generic_run (test_object,
                 outdir,
                 append_results,
                 init_tests,
                 setup_args,
                 max_iterations = max_iterations,
                 **analyzer_args):

  tic, get_times = scripting.init_tics ()
  engine, report = generic_setup (outdir, init_tests, *setup_args,
                                  test_object,
                                  norm_args = norm_args,
                                  input_bounds = input_bounds,
                                  **analyzer_args)
  init_coverage = engine.criterion.coverage ().as_prop

  tic ()

  report = engine.run (report = report, max_iterations = max_iterations)
  final_coverage = engine.criterion.coverage ().as_prop

  tic ()

  append_results (str (c) for c in
                  (*setup_args,
                   init_tests, report.nsteps,
                   *get_times (),
                   init_coverage, final_coverage,
                   report.num_tests,
                   report.num_adversarials))

# ---

def add_generic_args (parser):
  add_common_run_args (parser)
  parser.add_argument ('-c', '--criterion', # required = True,
                       choices = ('nc',), default ='nc',
                       help = 'criterion to focus on')
  parser.add_argument ('-n', '--total-runs', dest = 'total_runs',
                       type = int, default = 1, metavar = 'INT',
                       help = 'total number of runs (default is 1)',)

def generic (args):
  test_object, outs = setup_run_common (args)
  max_iterations = args.max_iterations
  crit = args.criterion

  global_outdir = OutputDir (f'{outs}/{crit}/', log = False)
  append_results = setup_results_file (global_outdir)

  init_tests_range = (1000,)

  _run = 0
  while _run < args.total_runs:
    _run += 1

    # draw parameters (not much...)
    init_tests = random.choice (init_tests_range)

    # setup output directory for this run
    basename = f'{crit}-X{init_tests}'
    outdir = global_outdir.fresh_dir (basename, enable_stamp = False,
                                      log = True)

    generic_run (test_object, outdir, append_results,
                 init_tests, (crit,),
                 max_iterations = max_iterations)

parser_generic = subparsers.add_parser ('generic')
parser_generic.set_defaults (func = generic)
add_generic_args (parser_generic)

# ---

# DBNC: Engine setup and helper functions

def add_common_dbnc_run_args (parser):
  add_common_run_args (parser)
  parser.add_argument ("--train-size", dest = "train_size",
                       metavar = "INT", type = int, default = train_size,
                       help = 'size of training dataset (default is '
                       f'{train_size})')

# Setup outputs

def setup_dbnc_results_file (go, discr_fields = ()):
  return scripting.setup_results_file \
         (go, 'crit', 'tech', 'N', 'skip', 'focus',
          'discr', *discr_fields, 'run',
          'min_n_bins', 'mean_n_bins', 'max_n_bins',
          'init_tests', 'total_iterations',
          'init_time', 'run_time',
          'init_coverage', 'final_coverage',
          'num_tests', 'num_adversarials')

def setup_init_coverages_file (outdir, init_tests, discr_fields = [], discr_fields_fmt = ()):
  init_coverages_file = outdir.stamped_filepath (f'init_coverages-{init_tests}',
                                                 suff = '.csv')
  init_coverages_dtype = ([('tech', 'U4'), ('discr', 'U10')] + discr_fields +
                          [('N', 'i4'), ('crit', 'U6'), ('run', 'O')] +
                          [(f'f{i}', 'f8') for i in range (1, init_tests + 1)])
  init_coverages_header = '\t'.join (f[0] for f in init_coverages_dtype)
  init_coverages_fmt = ('%s', '%s', *discr_fields_fmt, '%d', '%s', '%d',) + ('%f',) * init_tests
  init_coverages = []
  def write ():
    ic = np.asarray (init_coverages, dtype = np.dtype (init_coverages_dtype,
                                                       (len (init_coverages),)))
    np.savetxt (init_coverages_file, ic, delimiter = '\t', encoding = 'utf8',
                header = init_coverages_header, fmt = init_coverages_fmt)
  return init_coverages.append, write

# DBNC specifications

def base_dbnc_spec (bn_abstr_train_size = train_size, **k):
  return dict (bn_abstr_train_size = bn_abstr_train_size,
               report_on_feature_extractions = False,
               bn_abstr_n_jobs = 8,
               discr_n_jobs = 8,
               **k)

def feats_pca_spec (n_components = 1, **k):
  return dict (decomp = 'pca',
               n_components = n_components,
               # svd_solver = 'full',
               # svd_solver = 'arpack',
               # svd_solver = 'randomized',
               **k)

def feats_ipca_spec (n_components = 1, **k):
  return dict (decomp = 'ipca',
               n_components = n_components,
               **k)

def feats_ica_spec (n_components = 1, **k):
  return dict (decomp = 'ica',
               n_components = n_components,
               max_iter = 10000,
               tol = max (0.01, 0.01 * (n_components - 1)),
               **k)

def feats_spec (tech, n_components = None, **k):
  return feats_pca_spec (n_components = n_components, **k) if tech == 'pca' else \
         feats_ipca_spec (n_components = n_components, **k) if tech == 'ipca' else \
         feats_ica_spec (n_components = n_components, **k)

def discr_kde (extended = True, **k):
  return dict (strategy = 'kde',
               kde_dip_space = 'dens',
               extended = extended,
               **k)

def discr_uniform (extended = True, n_bins = 1, **k):
  return dict (strategy = 'uniform',
               n_bins = n_bins,
               extended = extended,
               **k)

def dbnc_setup (outdir, init_tests, crit, tech, N, skip, focus, test_object,
                train_size, discr = discr_kde, **dc_kwargs):
  dbnc_spec = dict (**base_dbnc_spec (bn_abstr_train_size = train_size),
                    feats = feats_spec (tech, n_components = N,
                                        skip = skip, focus = focus),
                    discr = discr ())
  report_args = dict (**base_report_args, outdir = outdir)
  return deepconcolic (crit, 'linf',
                       test_object, report_args,
                       **dc_kwargs,
                       dbnc_spec = dbnc_spec,
                       initial_test_cases = init_tests,
                       max_iterations = 0)

def acc_init_coverages (engine, report, init_tests, allow_skip = True):
  coverages = [ engine.criterion.coverage ().as_prop ]
  n = 1
  for i in range (1, init_tests):
    if coverages[-1] < 1. or not allow_skip:
      # skip step if full coverage (assumes monotonicity)
      report = engine.run (report = report,
                           initial_test_cases = 1,
                           max_iterations = 0)
      n += 1
    coverages += [ engine.criterion.coverage ().as_prop ]
  return coverages, n

def acc_init_dbnc_coverages (engine, report, init_tests):
  bfc_coverages = [ engine.criterion.bfc_coverage ().as_prop ]
  bfdc_coverages = [ engine.criterion.bfdc_coverage ().as_prop ]
  n = 1
  for i in range (1, init_tests):
    # skip step if full coverage (assumes monotonicity)
    report = engine.run (report = report,
                         initial_test_cases = 1,
                         max_iterations = 0)
    n += 1
    bfc_coverages += [ engine.criterion.bfc_coverage ().as_prop ]
    bfdc_coverages += [ engine.criterion.bfdc_coverage ().as_prop ]
  return bfc_coverages, bfdc_coverages, n


def dbnc_run (test_object,
              outdir,
              append_results,
              init_tests,
              setup_args,
              extra_descr = (),
              discr = None,
              max_iterations = max_iterations,
              train_size = train_size,
              **analyzer_args):

  tic, get_times = scripting.init_tics ()
  engine, report = dbnc_setup (outdir, init_tests, *setup_args,
                               test_object, train_size,
                               norm_args = norm_args,
                               input_bounds = input_bounds,
                               discr = discr,
                               **analyzer_args)
  init_coverage = engine.criterion.coverage ().as_prop

  tic ()

  report = engine.run (report = report, max_iterations = max_iterations)
  final_coverage = engine.criterion.coverage ().as_prop

  tic ()

  feature_parts = engine.criterion.BN.num_feature_parts
  append_results (str (c) for c in
                  (*setup_args, *extra_descr,
                   np.amin (feature_parts),
                   np.mean (feature_parts),
                   np.amax (feature_parts),
                   init_tests, report.nsteps,
                   *get_times (),
                   init_coverage, final_coverage,
                   report.num_tests,
                   report.num_adversarials))

# ---

num_runs = 3

def add_run_args (parser):
  add_common_dbnc_run_args (parser)
  parser.add_argument ("--runs", type = int, default = num_runs,
                       help = 'number of runs for each parameter selection (default is '
                       f'{num_runs})')

def run (args):
  test_object, outs = setup_run_common (args)
  max_iterations = args.max_iterations
  num_runs = args.runs
  train_size = args.train_size

  for crit in ('bfc', 'bfdc',):

    global_outdir = OutputDir (f'{outs}/{crit}/', log = True)
    append_results = setup_dbnc_results_file (global_outdir,
                                              discr_fields = ('n_bins',))

    for init_tests in init_tests_range:
      for tech in all_feat_extr_techs:
        for N in n_components_range:
          skip, focus = 0, N

          for run in range (num_runs):
            rng_seed (42 + run)
            outdir = OutputDir (global_outdir.filepath \
                                (f'{crit}-{tech}-N{N}-{skip}-{focus}-X{init_tests}-KDE-R{run}'),
                                enable_stamp = False, log = True)
            dbnc_run (test_object, outdir, append_results,
                      init_tests, (crit, tech, N, skip, focus),
                      extra_descr = ('kde', 0, run),
                      discr = discr_kde,
                      max_iterations = max_iterations,
                      train_size = train_size)

          for n_bins in n_bins_range:
            for run in range (num_runs):
              rng_seed (42 + run)
              outdir = OutputDir (global_outdir.filepath \
                                  (f'{crit}-{tech}-N{N}-{skip}-{focus}-X{init_tests}-U{n_bins}-R{run}'),
                                  enable_stamp = False, log = True)
              dbnc_run (test_object, outdir, append_results,
                        init_tests, (crit, tech, N, skip, focus),
                        extra_descr = ('uniform', n_bins, run),
                        discr = lambda : discr_uniform (n_bins = n_bins),
                        max_iterations = max_iterations,
                        train_size = train_size)


  # ---

  # global_outdir = OutputDir (f'{outs}/', log = True)

  # for init_tests in (# 1,
  #   # 2, 3, 4, 8, 16,
  #   30,):

  #   append_init_coverages, write_init_coverages = \
  #     setup_init_coverages_file (global_outdir, init_tests)

  #   for tech in ('pca', 'ica',):
  #     for N in (1, 2, 3, 4,):
  #       for run in range (num_runs):
  #         rng_seed (42 + run)
  #         outdir = OutputDir (global_outdir.filepath \
  #                             (f'{tech}-N{N}-X{init_tests}-R{run}'),
  #                             enable_stamp = False, log = True)

  #         engine, report = dbnc_setup (outdir, 1, 'bfc', tech, N,
  #                                      test_object, norm_args = norm_args,
  #                                      input_bounds = input_bounds)

  #         bfc_coverages, bfdc_coverages, n_init = \
  #             acc_init_dbnc_coverages (engine, report, init_tests)
  #         append_init_coverages ((tech, 'kde', N, 'bfc', run,) + tuple (bfc_coverages))
  #         append_init_coverages ((tech, 'kde', N, 'bfdc', run,) + tuple (bfdc_coverages))

  #         write_init_coverages ()

  # ---

  # global_outdir = OutputDir (f'{outs}/', log = True)

  # for init_tests in (# 1,
  #   # 2, 3, 4, 8, 16,
  #   100,):

  #   append_init_coverages, write_init_coverages = \
  #     setup_init_coverages_file (global_outdir, init_tests,
  #                                discr_fields = [('n_bins', 'i4')],
  #                                discr_fields_fmt = ('%d',))

  #   for tech in ('pca', 'ica',):
  #     for N in (1, 2, 3, 4, 5,):
  #       for n_bins in range (1, 5):
  #         for run in range (num_runs):
  #           rng_seed (42 + run)
  #           outdir = OutputDir (global_outdir.filepath \
  #                               (f'{tech}-N{N}-X{init_tests}-U{n_bins}-R{run}'),
  #                               enable_stamp = False, log = True)

  #           discr = lambda : discr_uniform (n_bins = n_bins)
  #           engine, report = dbnc_setup (outdir, 1, 'bfc', tech, N,
  #                                        test_object, norm_args = norm_args,
  #                                        input_bounds = input_bounds,
  #                                        discr = discr)

  #           bfc_coverages, bfdc_coverages, n_init = \
  #               acc_init_dbnc_coverages (engine, report, init_tests)
  #           append_init_coverages ((tech, 'uniform', n_bins, N, 'bfc', run,) + tuple (bfc_coverages))
  #           append_init_coverages ((tech, 'uniform', n_bins, N, 'bfdc', run,) + tuple (bfdc_coverages))

  #         write_init_coverages ()

  # ---

  # global_outdir = OutputDir (f'{outs}/', log = True)

  # for init_tests in (# 1,
  #   # 2, 3, 4, 8, 16,
  #   30,):

  #   append_init_coverages, write_init_coverages = \
  #     setup_init_coverages_file (global_outdir, init_tests,
  #                                discr_fields = [('n_bins', 'i4')],
  #                                discr_fields_fmt = ('%d',))

  #   for tech in ('pca', 'ica',):
  #     for N in (1, 2, 3, 4,):
  #       for n_bins in range (1, 4):
  #         for run in range (num_runs):
  #           rng_seed (42 + run)
  #           outdir = OutputDir (global_outdir.filepath \
  #                               (f'{tech}-N{N}-X{init_tests}-U{n_bins}-R{run}'),
  #                               enable_stamp = False, log = True)

  #           discr = lambda : discr_uniform (n_bins = n_bins)
  #           engine, report = dbnc_setup (outdir, init_tests, 'bfc', tech, N,
  #                                        test_object, norm_args = norm_args,
  #                                        input_bounds = input_bounds,
  #                                        discr = discr)

  #           bfc_coverages, bfdc_coverages, n_init = \
  #               acc_init_dbnc_coverages (engine, report, init_tests)
  #           append_init_coverages ((tech, 'uniform', n_bins, N, 'bfc', run,) + tuple (bfc_coverages))
  #           append_init_coverages ((tech, 'uniform', n_bins, N, 'bfdc', run,) + tuple (bfdc_coverages))

  #         write_init_coverages ()

parser_run = subparsers.add_parser ('run')
parser_run.set_defaults (func = run)
add_run_args (parser_run)

# ---

def add_randrun_args (parser):
  add_common_dbnc_run_args (parser)
  parser.add_argument ('-c', '--criterion', required = True,
                       choices = ('bfc', 'bfdc'),
                       help = 'criterion to focus on')
  parser.add_argument ('-n', '--total-runs', dest = 'total_runs',
                       type = int, default = 1, metavar = 'INT',
                       help = 'total number of runs (default is 1)',)

def randrun (args):
  test_object, outs = setup_run_common (args)
  max_iterations = args.max_iterations
  train_size = args.train_size
  crit = args.criterion

  global_outdir = OutputDir (f'{outs}/{crit}/', log = False)
  append_results = setup_dbnc_results_file (global_outdir,
                                            discr_fields = ('n_bins',))

  discr_strats = ('KDE',) + tuple (range (1, 6))

  _run = 0
  while _run < args.total_runs:
    _run += 1

    # draw parameters
    tech = random.choice (all_feat_extr_techs)
    N = random.randint (1, 5)                     # extract up to 5 features
    skip = 0                                      # fixed for now
    focus = min (random.randint (1, N - skip), 5) # cap to 5 to avoid too large BNs
    discr_strat = random.choice (discr_strats)
    init_tests = random.choice (init_tests_range)

    n_bins, discr_strat = (0, discr_strat) if discr_strat == 'KDE' else \
                          (discr_strat, f'U{discr_strat}')

    # setup output directory for this run
    basename = f'{crit}-{tech}-N{N}-{skip}-{focus}-X{init_tests}-{discr_strat}'
    outdir = global_outdir.fresh_dir (basename, enable_stamp = False,
                                      log = True)

    extra_descr = ('kde', 0, _run) if discr_strat == 'KDE' else \
                  ('uniform', n_bins, _run)
    discr = discr_kde if discr_strat == 'KDE' else \
            lambda : discr_uniform (n_bins = n_bins)
    dbnc_run (test_object, outdir, append_results,
              init_tests, (crit, tech, N, skip, focus),
              extra_descr = extra_descr,
              discr = discr,
              max_iterations = max_iterations,
              train_size = train_size)

parser_randrun = subparsers.add_parser ('randrun')
parser_randrun.set_defaults (func = randrun)
add_randrun_args (parser_randrun)

# ---

# see https://matplotlib.org/api/pyplot_api.html
plotting.generic_setup (**{
  'ytick.labelsize': 'small',
  'ytick.major.size': 4,
  'ytick.major.width': .4,
  'ytick.major.pad': 4,
  'ytick.direction': 'in',
  'xtick.labelsize': 'small',
  'xtick.major.size': 4,
  'xtick.major.width': .4,
  'xtick.major.pad': 4,
  'axes.labelsize': 'medium',
  'axes.labelpad': 2.,
  'axes.linewidth': .5,
  # 'xaxis.labellocation': 'right',
  'lines.markersize': 1.5,
  'lines.linewidth': .8,
})
plotting.pgf_setup (**{
  'ytick.labelsize': 'xx-small',
  'ytick.major.size': 2,
  'ytick.major.width': .2,
  'ytick.major.pad': 2,
  'ytick.direction': 'in',
  'xtick.labelsize': 'xx-small',
  'xtick.major.size': 1,
  'xtick.major.width': .1,
  'xtick.major.pad': 1,
  'axes.labelsize': 'x-small',
  'axes.titlesize': 'small',
  'axes.formatter.limits': (-2, 2),
  'axes.formatter.useoffset': True,
  'axes.formatter.use_mathtext': True,
  'lines.markersize': .2,
  'lines.linewidth': .2,
})

def add_plots_args (parser):
  parser.add_argument ('--reports-dir', dest = 'dir', metavar = "DIR", required = True,
                       help = 'directory where all execution reports are to be found')
  parser.add_argument ('-p', '--outputs-prefix', dest = 'prefix', type = str,
                       help = 'prefix of output filenames (e.g. PNGs, PDFs, PGFs...)')
  parser.add_argument ('-c', '--criterion', required = True, choices = ('bfc', 'bfdc'),
                       help = 'criterion to focus on')
  parser.add_argument ('--no-pca-progress', dest='no_pca_progress', action = 'store_true',
                       help = 'disable plotting of PCA distances and progress')
  parser.add_argument ('--no-ica-progress', dest='no_ica_progress', action = 'store_true',
                       help = 'disable plotting of ICA distances and progress')
  parser.add_argument ('--no-summary', dest='no_summary', action = 'store_true',
                       help = 'disable plotting of overall summary')
  parser.add_argument ('--dnn-name', dest='dnn_name', default = r'\mathcal{N}',
                       help = 'name of the DNN (in LaTeX math)')
  parser.add_argument ('--hist-bins', dest='hist_bins', type = int, default = 200,
                       help = 'number of bins in each histogram')

def read_reports (dir):
  T = scripting.gather_all_reports \
      (dir,
       '{crit}-{tech}-N{N:g}-{skip:d}-{focus:d}-X{init_tests:d}-{discr}-{run}',
       [('crit', 'U4'),
        ('tech', 'U4'),
        ('N', 'f8'),
        ('skip', 'i4'),
        ('focus', 'i4'),
        ('init_tests', 'i4'),
        ('discr', 'U10'),
        ('run', 'O')],
       ignore_head = 1)                 # ignore first init entry
  print ('Found {} report{}'.format (*s_(len(T))))
  for k in ('crit', 'tech', 'N', 'skip', 'focus', 'discr', 'init_tests'):
    print (f'>> {k}:', *(np.unique (T[k])))
  return T

def plots (args):
  T = read_reports (args.dir)
  outdir = OutputDir (args.outputs, log = True)
  filename = lambda f: args.prefix + '-' + f if args.prefix is not None else f

  T = T[T['crit'] == args.criterion]
  T_init_tests = { n: T[T['init_tests'] == n] for n in np.unique(T['init_tests']) }

  for init_tests in T_init_tests:
    generated_tests = sum (run['report']['#tests'][-1] - init_tests
                           for run in T_init_tests[init_tests])
    n_runs = len (T_init_tests[init_tests])
    print (f'{generated_tests} tests generated for |X_0|={init_tests}'
           '(average = {} test{}/run).'
           .format (*s_(generated_tests * 1. / n_runs)))

  def tech_style (tech):
    return dict (color = 'blue' if tech == 'pca' else 'red')

  # Progress/ICA

  def plot_progress (tech, T):
    P_tech = T[T['tech'] == tech]['progress']
    P_tech = [ P for P in P_tech if len (P.shape) > 0 ]
    nDists = np.concatenate([P['new_dist'].astype (float) for P in P_tech ])
    oDists = np.concatenate([P['old_dist'].astype (float) for P in P_tech ])
    dDists = oDists - nDists

    fig, ax = plotting.subplots (1, 2,
                                 figsize_adjust = (1.0, 0.5),
                                 constrained_layout = True)
    ax[0].hist (nDists, bins = args.hist_bins, **tech_style (tech))
    ax[0].axvline (x = 0, lw = 1, color = 'black')
    ax[0].set_ylabel (r'\#steps where new distance is $d$')
    ax[0].set_xlabel (r'Distance ($d$ — '+tech+r')')
    ax[1].hist (dDists, bins = args.hist_bins, **tech_style (tech))
    ax[1].axvline (x = 0, lw = 1, color = 'black')
    ax[1].set_ylabel (r'\#steps where progress is $\delta$')
    ax[1].set_xlabel (r'Progress ($\delta$ — '+tech+r')')
    plotting.show (fig,
                   outdir = outdir,
                   basefilename = filename (tech + '-dist-n-progress'),
                   w_pad = 0.06)

  if not args.no_pca_progress:
    plot_progress ('pca', T)

  # Progress/ICA

  if not args.no_ica_progress:
    plot_progress ('ica', T)

  # Summary

  if not args.no_summary:
    def plot_style (report):
      return tech_style (report['tech'])

    def it_(ax):
      return ax if len (T_init_tests) > 1 else [ax]

    Nms = args.dnn_name# r'\mathcal{N}_{\mathsf{ms}}'
    cov_label_ = lambda d, n, x: r'\mathrm{'+ d +r'}(\mathcal{B}_{'+n+r', '+x+'})'
    cov_label = lambda n, x: \
                cov_label_ ('BFCov', n, x) if args.criterion == 'bfc' else \
                cov_label_ ('BFdCov', n, x)

    fig, ax = plotting.subplots (3, len (T_init_tests),
                                 sharex='col', sharey='row',
                                 constrained_layout = True)
    for axi in it_(ax[-1]):
      # unshare x axes for the bottom row:
      g = axi.get_shared_x_axes()
      g.remove (axi)
      for a in g.get_siblings(axi): g.remove (a)

    for init_tests, axi in zip (T_init_tests, it_(ax[0])):
      for run in T_init_tests[init_tests]:
        axi.plot (run['report']['#tests'] - init_tests,
                  **plot_style (run))

    from matplotlib.ticker import StrMethodFormatter
    for init_tests, axi in zip (T_init_tests, it_(ax[1])):
      for run in T_init_tests[init_tests]:
        if len (run['report']) == 0:
          continue
        axi.plot (run['report']['coverage'] - run['report']['coverage'][0],
                  **plot_style (run))
      axi.yaxis.set_major_formatter(StrMethodFormatter('{x:2.1f}'))
      axi.yaxis.set_ticks(np.arange (0, np.amax(axi.get_yticks()), step=0.1))

    for init_tests, axi in zip (T_init_tests, it_(ax[2])):
      init_covs  = [run['report']['coverage'][ 0]
                    for run in T_init_tests[init_tests]
                    if len (run['report']) > 0]
      final_covs = [run['report']['coverage'][-1]
                    for run in T_init_tests[init_tests]
                    if len (run['report']) > 0]
      bp = axi.boxplot ([init_covs, final_covs],
                        positions = [0, 20], widths = 6,
                        # labels = [r'initial ($i=0$)', 'final'],
                        flierprops = dict (marker='.', markersize = 1),
                        bootstrap = 1000,
                        manage_ticks = False)
      axi.yaxis.set_major_formatter(StrMethodFormatter('{x:2.1f}'))
      for box in bp['boxes']: box.set(linewidth=.5)
      for box in bp['caps']: box.set(linewidth=.5)
      plt.setp(axi.get_xticklabels(), visible=False)

    for init_tests, axi in zip (T_init_tests, it_(ax[1])):
      axi.xaxis.set_tick_params(which='both', labelbottom=True)

    # Set labels and column titles:
    for init_tests, axi in zip (T_init_tests, it_(ax[0])):
      axi.set_title (f'$|X_0| = {init_tests}$')
    for axi in it_(ax[-1]):
      axi.set_xlabel (r'iteration ($i$)')
    it_(ax[0])[0].set_ylabel (r'$|X_i| - |X_0|$')
    it_(ax[1])[0].set_ylabel (r'$' +
                              cov_label (Nms, r'X_i') +
                              '-' +
                              cov_label (Nms, r'X_0') +
                              '$')
    it_(ax[2])[0].set_ylabel (r'$' +
                              cov_label (Nms, r'X_i') +
                              '$')
    # it_(ax[-1])[(len (T_init_tests) - 1) // 2 + 1].set_xlabel (r'iteration ($i$)')
    plotting.show (fig,
                   basefilename = filename ('summary-per-X0'),
                   outdir = outdir,
                   rect = (.01, 0, 1, 1))

  # fig, ax = plotting.subplots (2, len (T_init_tests),
  #                              sharex='col', sharey='row')
  # for init_tests, axi in zip (T_init_tests, it_(ax[0])):
  #   for run in T_init_tests[init_tests]:
  #     axi.plot (run['report']['#tests'] - init_tests,
  #               **plot_style (run))
  # for init_tests, axi in zip (T_init_tests, it_(ax[1])):
  #   for run in T_init_tests[init_tests]:
  #     axi.plot (run['report']['coverage'] - run['report']['coverage'][0],
  #               **plot_style (run))
  #     # cov_diff = np.diff(run['report']['coverage'])
  #     # axi.plot (range (1, len(cov_diff) + 1), cov_diff,
  #     #           **plot_style (run))
  # # for init_tests, axi in zip (T_init_tests, it_(ax[2])):
  # #   init_covs  = [run['report']['coverage'][ 0] for run in T_init_tests[init_tests]]
  # #   final_covs = [run['report']['coverage'][-1] for run in T_init_tests[init_tests]]
  # #   axi.boxplot ([init_covs, final_covs],
  # #                labels = [r'initial ($i=0$)', 'final'], bootstrap = 1000)
  #   # unshare x axes:
  #   # g = axi.get_shared_x_axes()
  #   # g.remove (axi)
  #   # for a in g.get_siblings(axi): g.remove (a)
  # for init_tests, axi in zip (T_init_tests, it_(ax[0])):
  #   axi.set_title (f'$|X_0| = {init_tests}$')
  # # for init_tests, axi in zip (T_init_tests, it_(ax[-1])):
  # ax[-1][1].set_xlabel (r'iteration')
  # it_(ax[0])[0].set_ylabel (r'$\Delta(|X|)$')
  # it_(ax[1])[0].set_ylabel (r'$'+ bfc_label (mnist_small, r'X') + '-' +
  #                           bfc_label (mnist_small, r'X_0') + '$')
  # # it_(ax[2])[0].set_ylabel (r'$'+ bfc_label (mnist_small, r'X_i') + '$')
  # plotting.show (fig, basefilename = 'num_tests-per-X0')

  # ---

  # fig, ax = plotting.subplots (1, len (T_init_tests),
  #                              figsize_adjust = (1., .5),
  #                              sharex='col', sharey='row')
  # for init_tests, axi in zip (T_init_tests, it_(ax)):
  #   init_covs  = [run['report']['coverage'][ 0] for run in T_init_tests[init_tests]]
  #   final_covs = [run['report']['coverage'][-1] for run in T_init_tests[init_tests]]
  #   axi.boxplot ([init_covs, final_covs],
  #                labels = [r'initial ($i=0$)', 'final'],
  #                bootstrap = 1000)
  # it_(ax)[0].set_ylabel (r'$'+ bfc_label (mnist_small, r'X_i') + '$')
  # plotting.show (fig, basefilename = 'coverage-per-X0')


parser_plots = subparsers.add_parser ('plots')
parser_plots.set_defaults (func = plots)
add_plots_args (parser_plots)

# ---

if __name__=="__main__":
  args = parser.parse_args()
  if 'func' in args:
    args.func (args)
  else:
    parser.print_help ()
    sys.exit (1)

