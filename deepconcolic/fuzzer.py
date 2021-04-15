#!/usr/bin/env python3
from utils_io import *
from utils_args import *
from utils_funcs import rng_seed, random, randint
from utils_mp import init as mp_init, FFPool, forking, np_share
from pathlib import Path
import datasets
import plugins

# ## to be refined
# apps = ['./deepconcolic/run_template.py']

# def fuzz (test_object, outs, model_name, stime, file_list,
#           num_tests = 1000, num_processes = 1):
#   assert isinstance (outs, OutputDir)

#   mutant_path = outs.subdir ('mutants')
#   adv_path = outs.subdir ('advs')

#   data = test_object.raw_data.data
#   if data.shape[1] == 28: # todo: this is mnist hacking
#     # NB: check if that's really needed. data.shape[1:] may be enough?
#     img_rows, img_cols, img_channels = data.shape[1], data.shape[2], 1
#   else:
#     img_rows, img_cols, img_channels = data.shape[1], data.shape[2], data.shape[3]

#   num_crashes = 0
#   for i in range(num_tests):
#       processes = []
#       commandlines = []
#       fuzz_outputs = []
#       for j in range(0, num_processes):
#           file_choice = random.choice(file_list)
#           buf = bytearray(open(file_choice, 'rb').read())
#           numwrites = 1 # to keep a minimum change (hard coded for now)
#           for k in range(numwrites):
#               rbyte = random.randrange(256)
#               rn = random.randrange(len(buf))
#               buf[rn] = rbyte

#           fuzz_output = mutant_path + '/mutant-iter{0}-p{1}'.format(i, j)
#           fuzz_outputs.append(fuzz_output)
#           f = open(fuzz_output, 'wb')
#           f.write(buf)
#           f.close()

#           commandline = ['python3', apps[0], '--model', model_name, '--origins', file_choice, '--mutants', fuzz_output, '--input-rows', str(img_rows), '--input-cols', str(img_cols), '--input-channels', str(img_channels)]
#           commandlines.append(commandline)
#           process = subprocess.Popen(commandline)
#           processes.append(process)

#       time.sleep(stime) # (hard coded for now)
#       for j in range(0, num_processes):
#           process = processes[j]
#           fuzz_output = fuzz_outputs[j]
#           crashed = process.poll()
#           #print ('>>>>>', crashed)
#           if crashed == SIG_NORMAL:
#               process.terminate()
#           elif crashed == SIG_COV:
#               ## TODO coverage guided; add fuzz_output into the queue
#               #print (">>>> add fuzz_output into the queue")
#               process.terminate()
#           elif crashed == SIG_ADV:
#               num_crashes += 1
#               append_in_file (outs.filepath ("advs.list"),
#                               "Adv# {0}: command {1}\n".format(num_crashes, commandlines[j]))
#               adv_output = adv_path+'/' + fuzz_output.split('/')[-1]
#               f = open(adv_output, 'wb')
#               f.write(buf)
#               f.close()
#           else: pass

#   #print (report_args)


def gen_mutations (N, test_data, rng = None,
                   item_level_mutations = False,
                   byte_level_mutations = True):
  data = test_data.data
  rng = rng or np.random.default_rng ()

  if byte_level_mutations and not data.flags['C_CONTIGUOUS']:
    warnings.warn ('C-contiguous test data is required for byte-level mutations: '
                   'using item-level mutations instead.')
    byte_level_mutations = False
    item_level_mutations = True

  # just a lambda for test input selection:
  select_idx = lambda: rng.integers (len (data))

  def mut_set (idx):
    return tuple (rng.integers (data[idx].shape)), rng.random ()
  item_muts = (
    ('set', mut_set),
  )

  def mut_setbyte (idx):
    return tuple (rng.integers (data[idx].view ('B').shape)), rng.integers (256)
  byte_muts = (
    ('setbyte', mut_setbyte),
  )

  muts \
    = (item_muts if item_level_mutations else ()) \
    + (byte_muts if byte_level_mutations else ())

  assert muts != ()

  def gen_muts (idx):
    iid = rng.integers (len (muts))
    return ((muts[iid][0],) + muts[iid][1] (idx),)

  def gen_test ():
    idx = select_idx ()
    return idx, gen_muts (idx)

  return tuple (gen_test () for _ in range (N))


def mutate_test (x, muts):
  for mut in muts:
    op = mut[0]

    if op == 'set':
      idx, elt = mut[1:]
      x[idx] = elt

    elif op == 'setbyte':
      idx, elt = mut[1:]
      x.view ('B')[idx] = elt

    else:
      ValueError (f'Unknown mutation `{mut}\'')


def select_origins (test_data, mutation_descrs):
  return test_data.data[[idx for idx, _ in mutation_descrs]]


def select_labels (test_data, mutation_descrs):
  if test_data.labels is None:
    return [None] * len (mutation_descrs)
  return test_data.labels[[idx for idx, _ in mutation_descrs]]


def perform_mutations (origins, mutation_descrs):
  mutants = origins.copy ()
  for i, muts in enumerate (mut for _, mut in mutation_descrs):
    mutate_test (mutants[i], muts)
  return mutants


def np_neq (x1, x2):
  return (x1 != x2).any (axis = tuple (range (1 - len (x1.shape), 0)))


def np_true (x1, x2):
  return np.ones (len (x1), dtype = bool)


def test_mutations (model, test_data, postproc = id,
                    flag_diff = np_neq,
                    flag_cov = np_true):

  from utils import load_model, predictions

  dnn = load_model (model)

  def aux (mutation_descrs):
    with np.errstate (all = 'raise'): # Should we only expect numpy errors (?)
      origins = select_origins (test_data, mutation_descrs)
      mutants = perform_mutations (origins, mutation_descrs)
      Y_origins = predictions (dnn, origins)
      try:
        mutants = postproc (mutants)

        X_valid = flag_diff (origins, mutants) | ~flag_cov (origins, mutants)
        Y_mutants = predictions (dnn, mutants)

        return Y_origins, np.where (X_valid, Y_mutants, None)
      except:
        return Y_origins, [None] * len (Y_origins)
  return aux


def record_tests (id, test_data, mutation_descrs, test_results,
                  postproc = id, save_input = None, save_all_tests = False):

  origins = select_origins (test_data, mutation_descrs)
  labels = select_labels (test_data, mutation_descrs)
  with np.errstate (all = 'ignore'):
    mutants = perform_mutations (origins, mutation_descrs)
    mutants = postproc (mutants)

  new, adv = 0, 0

  for Y_origin, Y_mutant, origin, mutant, Y_official in \
      zip (*test_results, origins, mutants, labels):

    if Y_official is not None and Y_origin != Y_official:
      save_input (origin, f'{id}-official-{Y_official}')

    if Y_mutant is not None:                      # valid test
      new += 1

      if Y_mutant == Y_origin and save_all_tests: # ok
        save_input (origin, f'{id}-original-{Y_origin}')
        save_input (mutant, f'{id}-ok-{Y_mutant}')

      if Y_mutant != Y_origin:  # adversarial
        save_input (origin, f'{id}-original-{Y_origin}')
        save_input (mutant, f'{id}-adv-{Y_mutant}')
        adv += 1

    id += 1

  del origins, mutants
  return id, new, adv


def make_tester_ (*_):
  test_muts = test_mutations (*_)
  def aux (mutation_descrs):
    return mutation_descrs, test_muts (mutation_descrs)
  return aux


def run (dataset = None,
         testset_dir = None,
         extra_testset_dirs = None,
         sample = None,
         model = None,
         num_tests = 10,
         outdir = None,
         save_all_tests = True,
         processes = None,
         verbose = False):
  assert dataset in datasets.choices

  from utils import dataset_dict
  dd = dataset_dict (dataset)
  test_data, dims, save_input, postproc = \
    dd['test_data'], dd['dims'], dd['save_input'], dd['postproc_inputs']
  del dd

  if testset_dir is not None:
    np1 (f'Loading input data from `{testset_dir}\'... ')
    test_data.data = datasets.images_from_dir (testset_dir, raw = True)
    test_data.labels = None
    p1 ('done.')

  elif extra_testset_dirs is not None:
    for d in extra_testset_dirs:
      np1 (f'Loading extra image testset from `{str(d)}\'... ')
      x, y, _, _, _ = datasets.images_from_dir (str (d))
      test_data.data = np.concatenate ((test_data.data, x))
      test_data.labels = np.concatenate ((test_data.labels, y))
      p1 ('done.')

  # ---

  outdir = OutputDir (outdir, enable_stamp = False)
  save_test = lambda x, bn: save_input (x, bn, directory = outdir.path,
                                        log = verbose)
  max_pre_dispatch = 100
  rng = np.random.default_rng (randint ())

  # ---

  if sample is not None:
    if not (isinstance (sample, int) or \
            isinstance (sample, float) and sample > 0. and sample <= 1.):
      raise ValueError ('`sample\' must be an Integer or a value in (0,1] '
                        f'(got sample={sample})')
    idxs = rng.choice (a = np.arange (len (test_data.data)),
                       axis = 0, size = min (sample, len (test_data.data)))
    test_data.data = test_data.data[idxs]
    if test_data.labels is not None:
      test_data.labels = test_data.labels[idxs]

  # ---

  if forking ():
    def make_tester (model):
      test_muts = test_mutations (model, test_data, postproc)
      def aux (mutation_descrs):
        return mutation_descrs, test_muts (mutation_descrs)
      return aux
    ffpool_args = make_tester, model
  else:
    test_data.data = np_share (test_data.data)
    test_data.labels = np_share (test_data.labels)
    ffpool_args = make_tester_, model, test_data, postproc

  pool = FFPool (*ffpool_args,
                 processes = some (processes, 1),
                 verbose = verbose)
  pool.start ()

  # Feed some first work items to the pool:
  init_n = min (len (pool) + max_pre_dispatch, num_tests)
  for _ in range (init_n):
    pool.put (gen_mutations (1, test_data, rng))

  tid, tests, advs = (0,) * 3
  while tid < num_tests:
    mutation_descrs, test_results = pool.get ()

    # feed new work straight away
    if tid < num_tests - init_n:
      pool.put (gen_mutations (1, test_data, rng))

    # save mutants and/or adversarial examples
    tid, new_tests, new_advs = \
      record_tests (tid, test_data, mutation_descrs, test_results,
                    postproc = postproc,
                    save_input = save_test,
                    save_all_tests = save_all_tests)
    tests += new_tests
    advs += new_advs

    tp1 (f'{tid/num_tests:.2%}: |tests|={tests}, |adv|={advs}')

  p1 ('Terminating after {} iteration{}: '
      '{} test{} generated, {} of which {} adversarial.'
      .format (*s_(tid), *s_(tests), *is_are_(advs)))

  # terminate pool:
  pool.join ()
  if verbose:
    tp1 ('Exiting')

# ---

ap = argparse.ArgumentParser \
  (description = 'Fuzzer for Neural Networks',
   prog = 'python3 -m deepconcolic.fuzzer',
   prefix_chars = '-+',
   formatter_class = argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument ('--dataset', dest = 'dataset', required = True,
                 help = 'selected dataset', choices = datasets.choices)
ap.add_argument ('--model', required = True,
                 help = 'the input neural network model (.h5 file or "vgg16")')
ap.add_argument ("--outputs", '-o', '-d', dest = "outputs", required = True,
                 help = "the output test data directory", metavar = "DIR")
gp = ap.add_mutually_exclusive_group (required = False)
gp.add_argument ('--inputs', '-i', dest = 'testset_dir', metavar = 'DIR',
                 help = 'directory of test images')
gp.add_argument ('--extra-tests', '+i', dest = 'extra_testset_dirs', metavar = 'DIR',
                 type = Path, nargs = '+',
                 help = 'additonal directories of test images')
gp = ap.add_mutually_exclusive_group (required = False)
gp.add_argument ('--sample', type = int, metavar = 'N',
                 help = 'sample a subset of N inputs for testing')
gp.add_argument ('--sample-ratio', type = float, metavar = 'T',
                 help = 'sample a ratio T of inputs for testing')
ap.add_argument ('--num-tests', '-N', type = int, metavar = 'N', default = 100,
                 help = "number of tests to generate")
ap.add_argument ('--processes', '-P', '-J', type = int, default = 1, metavar = 'N',
                 help = 'use N parallel tester processes (default is 1---'
                 'use -1 to use all available CPUs)')
ap.add_argument ('--rng-seed', dest = 'rng_seed', metavar = 'SEED', type = int,
                 help = 'Integer seed for initializing the internal random number '
                 'generator')
add_verbose_flags (ap)

def get_args (args = None, parser = ap):
  args = parser.parse_args () if args is None else args
  # Initialize with random seed first, if given:
  try: rng_seed (args.rng_seed)
  except ValueError as e:
    sys.exit (f'Invalid argument given for \`--rng-seed\': {e}')
  return args

def main (args = None, parser = ap, pp_args = ()):
  try:
    args = get_args (args, parser = parser)
    for pp in pp_args: pp (args)
    run (dataset = args.dataset,
         testset_dir = args.testset_dir,
         extra_testset_dirs = args.extra_testset_dirs,
         sample = args.sample or args.sample_ratio,
         model = args.model,
         num_tests = args.num_tests,
         outdir = args.outputs,
         processes = args.processes,
         verbose = args.verbose)
  except ValueError as e:
    sys.exit (f'Error: {e}')
  except FileNotFoundError as e:
    sys.exit (f'Error: {e}')
  except KeyboardInterrupt:
    sys.exit ('Interrupted.')

# ---

if __name__=="__main__":
  mp_init ()
  main ()
