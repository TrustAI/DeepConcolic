from typing import *
from utils import *
from sklearn.model_selection import train_test_split


# ---


# Define an alias type for inputs
Input = NewType("Input", np.ndarray)


# ---


class Filter:
  '''
  Filters can be used to compare any concrete input against a
  reference set.
  '''

  @abstractmethod
  def close_to(self, refs: Sequence[Input], x: Input):
    raise NotImplementedError



# ---


class Metric (Filter):
  '''
  For now, we can assume that every metric can also be used as a
  filter to compare and assess concrete inputs.
  '''

  def __init__(self, factor = 0.25, scale = 1, **kwds):
    '''
    The `factor` argument determines closeness when the object is used
    as a filter; defaults to 1/4.  In turn, `scale` is applied on
    involved scalar values (e.g. pixels) when computing distances.
    '''
    self.factor = factor
    self.scale = scale
    super().__init__(**kwds)
    

  @abstractmethod
  def distance(self, x, y):
    '''
    Returns the distance between two concrete inputs `x` and `y`.
    '''
    raise NotImplementedError


  @property
  def is_int(self):
    '''
    Holds iff Integer metrics.
    '''
    return False


# ---


class TestTarget:
  '''
  Base record of test targets.
  '''

  @abstractmethod
  def cover(self) -> None:
    '''
    Record that the target has been covered.
    '''
    raise NotImplementedError

  
  def log_repr(self) -> str:
    '''
    Returns a single-line string representation of the target suitable
    for logging.
    '''
    raise NotImplementedError


# ---


class Analyzer:
  '''
  Base class for any kind of analyzer that is able to construct new
  concrete inputs.
  '''

  def __init__(self, analyzed_dnn = None, **kwds):
    assert analyzed_dnn is not None
    self._analyzed_dnn = analyzed_dnn
    super().__init__(**kwds)


  # ---

  # TODO: `dnn` and the two methods below (previously in test_objectt)
  # would deserve to be on their own as they are not strictly speaking
  # analyzer-dependent.  Yet they stay there for now as analyzers,
  # criteria, and engines rely on at least one of them.

  @property
  def dnn(self) -> keras.Model:
    '''
    The analyzed DNN.
    '''
    return self._analyzed_dnn


  def eval(self, i, allow_input_layer = False):
    '''
    Returns the activations associated to a given input.
    '''
    return eval (self.dnn, i, allow_input_layer)


  def eval_batch(self, i, allow_input_layer = False):
    '''
    Returns all activations associated to a given input batch.
    '''
    return eval_batch (self.dnn, i, allow_input_layer)


  @abstractmethod
  def input_metric(self) -> Metric:
    '''
    Returns the metric used to compare concrete inputs.
    '''
    raise NotImplementedError


# ---


class Analyzer4RootedSearch (Analyzer):
  '''
  Analyzers that are able to find new concrete inputs close to a given
  input should inherit this class.
  '''

  @abstractmethod
  def search_input_close_to(self, x, target: TestTarget) -> Optional[Tuple[float, Input]]:
    '''
    Generates a new concrete input close to `x`, that fulfills test
    target `target`.

    Returns a tuple `(d, y)`, that is a new concrete input `y` along
    with its distance `d` w.r.t the input metric, or `None` is
    unsuccessful.
    '''
    raise NotImplementedError


# ---


class Analyzer4FreeSearch (Analyzer):
  '''
  Analyzers that are able to find new concrete inputs close to any
  input from a give set of test cases.
  '''

  @abstractmethod
  def search_close_inputs(self, target: TestTarget) -> Optional[Tuple[float, Input, input]]:
    '''
    Generates a new concrete input that fulfills test target `target`.
    
    Returns a tuple `(d, base, new)` where `base` is a concrete
    element from a set given on initialization (typically for now, raw
    data from `test_object`) and `new` is a new concrete input at
    distance `d` from `base`, or `None` is unsuccessful.
    '''
    raise NotImplementedError


# ---


class Report:
  '''
  A simple class to take reporting stuff out from the engine.
  '''

  def __init__(self,
               base_name = '',
               outs = '/tmp',
               save_new_tests = False,
               adv_dist_period = 100,
               save_input_func = None,
               inp_up = 1,
               **kwds):

    self.adversarials = []
    self.base_name = base_name
    self.save_new_tests = save_new_tests
    self.adv_dist_period = adv_dist_period
    self.dir = setup_output_dir (outs)
    report_file, base = setup_output_files (self.dir, self.base_name,
                                            suff = '', log = False)
    self.report_file = report_file + '_report.txt'
    self.base = base
    self.save_input_func = save_input_func
    self.inp_ub = inp_up
    p1 ('Reporting into: {0}'.format (self.report_file))
    self.ntests = 0

    
  def _save_input(self, im, name, log = None, fit = False):
    if self.save_input_func != None:
      self.save_input_func (self.inp_ub * 0.5 + (im / self.inp_ub * 0.5) if fit
                            else (im / self.inp_ub * 1.0),
                            name, self.dir, log)


  def _save_derived_input(self, new, origin, diff = None, log = None):
    self._save_input (new[0], new[1], log)
    self._save_input (origin[0], origin[1], log)
    if diff is not None:
      self._save_input (diff[0], diff[1], log, fit = False)
                              

  def save_input(self, i, suff):
    self._save_input (i, self.base + '_' + suff)


  def new_test(self, new = (), orig = (), dist = None, is_int = None):
    if self.save_new_tests:
      diff = np.abs(new[0] - orig[0])
      diff *= 0.5 / np.max (diff)
      self._save_derived_input (
        (new[0], '{0.ntests}-ok-{1}'.format (self, new[1])),
        (orig[0], '{0.ntests}-original-{1}'.format (self, orig[1])),
        (diff, '{0.ntests}-diff-{1}'.format (self, orig[1])))
    self.ntests += 1


  @property
  def num_adversarials(self):
    return len(self.adversarials)


  def new_adversarial(self, new = (), orig = (), dist = None, is_int = None):
    self.adversarials.append ((orig, new, dist))
    self._save_derived_input (
      (new[0], '{0.ntests}-adv-{1}'.format (self, new[1])),
      (orig[0], '{0.ntests}-original-{1}'.format (self, orig[1])),
      (np.abs(new[0] - orig[0]), '{0.ntests}-diff-{1}'.format (self, orig[1])))
    if self.num_adversarials % self.adv_dist_period == 0:
      print_adversarial_distribution (
        [ d for o, n, d in self.adversarials ],
        self.dir + self.base + '_adversarial-distribution.txt',
        int_flag = is_int)
    self.ntests += 1


  def step(self, *args) -> None:
    '''
    Prints a single report line.
    '''
    append_in_file (self.report_file, *args)



# ---


class EarlyTermination (Exception):
  '''
  Exception raised by criteria when no new test target can be found.
  '''
  pass
  

# ---


class Criterion:
  '''
  Base class for test critieria.

  Note that a criterion MUST inherit either (or both)
  :class:`Criterion4FreeSearch` or :class:`Criterion4RootedSearch`.
  '''

  def __init__(self,
               analyzer: Analyzer = None,
               prefer_rooted_search = None,
               **kwds):
    '''
    A criterion operates based on an `analyzer` to find new concrete
    inputs.

    Flag `prefer_rooted_search` can be used in case both the criterion
    and the analyzer support the two kinds of search; the default
    behavior is to select rooted search.
    '''
    
    assert isinstance (analyzer, Analyzer)
    super().__init__(**kwds)
    self.analyzer = analyzer
    self.test_cases = []
    self.rooted_search = self._rooted_search (prefer_rooted_search)


  # True for rooted search, False for free search
  def _rooted_search(self, prefer_rooted_search = None):
    '''
    Holds if rooted-search mode is selected and the criterion and
    analyzer pair supports it.

    Parameters
    ----------
    prefer_rooted_search: bool, optional
     
    Returns
    -------
    whether rooted search mode is selected.
    
    '''
    rooted_ok = (isinstance (self.analyzer, Analyzer4RootedSearch) and
                 isinstance (self, Criterion4RootedSearch))
    free_ok = (isinstance (self.analyzer, Analyzer4FreeSearch) and
               isinstance (self, Criterion4FreeSearch))
    if not (free_ok or rooted_ok):
      sys.exit ('Incompatible pair criterion/analyzer')
    if free_ok and rooted_ok and prefer_rooted_search is None:
      p1 ('Arbitrarily selecting rooted search against free search.')
    return rooted_ok and (prefer_rooted_search is None or prefer_rooted_search)


  # ---


  @abstractmethod
  def __repr__(self):
    raise NotImplementedError


  @abstractmethod
  def finalize_setup(self):
    """
    Called once after any stat-based initialization (see, e.g.,
    :meth:`stat_based_basic_initializers`), and before any call to
    :meth:`add_new_test_cases`, :meth:`coverage`, and
    :meth:`search_next`.
    """
    raise NotImplementedError
    # self.analyzer.finalize_setup ()


  def reset(self):
    '''
    Empties the set of test cases
    '''
    self.test_cases = []


  @abstractmethod
  def coverage(self) -> Coverage:
    '''
    Returns a measure of the current coverage.
    '''
    raise NotImplementedError


  @property
  def metric(self) -> Metric:
    '''
    Returns the metric used by the analyzer to compare concrete
    inputs.
    '''
    return self.analyzer.input_metric ()


  @property
  def filter(self) -> Filter:
    '''
    Returns the filter used to compare concrete inputs.  By default
    this is in input metric used by the analyzer.
    '''
    return self.metric


  @property
  def num_test_cases(self) -> int:
    '''
    Returns the number of test cases.
    '''
    return len(self.test_cases)


  # final as well
  def add_new_test_cases(self, tl: Sequence[Input]) -> None:
    """
    As its name says, this method adds a given series of inputs into
    the set of test cases.  It then calls :meth:`update_coverage`.
    """
    tp1 ('Adding {} test case{}'.format (len (tl), 's' if len (tl) > 1 else ''))
    self.test_cases.extend (tl)
    self.update_coverage (tl)


  @abstractmethod
  def update_coverage(self, tl: Sequence[Input]) -> None:
    """
    Method called whenever new test cases are registered.  Overload
    this method to update coverage.
    """
    raise NotImplementedError


  def search_next(self) -> Tuple[Union[Tuple[Input, Input, float], None], TestTarget]:
    '''
    Selects a new test target based (see
    :class:`Criterion4RootedSearch` and
    :class:`Criterion4FreeSearch`), and then uses the analyzer to find
    a new concrete input.

    Returns a pair of:

    - either `None` in case of failure of the analyzer, or a triple
      `(x0, x1, d)`, `x1` being the new concrete input generated by
      the analyzer;

    - the test target considered.
    '''
    if self.rooted_search:
      x0, target = self.find_next_rooted_test_target ()
      tp1 ('| Targeting {}'.format(target))
      x1_attempt = self.analyzer.search_input_close_to (x0, target)
      if x1_attempt == None:
        return None, target
      else:
        d, x1 = x1_attempt
        return (x0, x1, d), target
    else:
      target = self.find_next_test_target ()
      tp1 ('| Targeting {}'.format(target))
      attempt = self.analyzer.search_close_inputs (target)
      if attempt == None:
        return None, target
      else:
        d, x0, x1 = attempt
        return (x0, x1, d), target


  # ---


  def stat_based_basic_initializers(self):
    """
    Stat-based initialization steps (non-batched).
    
    Returns a list of dictionaries (or `None`) with the following
    entries:

    - name: short description of what's computed;

    - layer_indexes: a list or set of indexes for layers whose
      activations values are needed;
    
    - once: a callable taking a mapping (as a dictionary) from each
      layer index given in `layer_indexes` to activation values for
      the corresponding layer; this is to be called only once during
      initialization of the analyzer;

    - print (optional): a function that prints a summary of results.
    """
    return None


  def stat_based_incremental_initializers(self):
    """
    Stat-based incremental initialization steps.

    Returns a list of dictionaries (or `None`) with the following
    entries:

    - name: short description of what's computed;
    
    - accum: a callable taking batched activation values for every
      layer and any accumulator that is (initially `None`), and
      returns a new or updated accumulator.  This is called at least
      once.

    - final: optional function that is called with the final
      accumulator once all batched activations have been passed to
      `accum`;

    - print (optional): a function that prints a summary of results.
    """
    return None


  def stat_based_train_cv_initializers(self):
    """
    Stat-based initialization steps with optional cross-validation
    performed on training data.

    Returns a list of dictionaries (or `None`) with the following
    entries:

    - name: short description of what's computed;

    - layer_indexes: a list or set of indexes for layers whose
      activations values are needed;

    - test_size & train_size: (as in
      `sklearn.model_selection.train_test_split`)

    - train: a callable taking some training data as mapping (as a
      dictionary) from each layer index given in `layer_indexes` to
      activation values for the corresponding layer, and returns some
      arbitrary object; this is to be called only once during
      initialization;

    - test: a callable taking some extra training data and associated
      labels as two separate mappings (as a dictionary) from each
      layer index given in `layer_indexes` to activation values for
      the corresponding layer.

    Any function given as entries above is always called before
    functions returned by `stat_based_test_cv_initializers`, but after
    those retured by `stat_based_basic_initializers` and
    `stat_based_incremental_initializers`.
    """
    return None


  def stat_based_test_cv_initializers(self):
    """
    Stat-based initialization steps with optional cross-validation
    performed on test data.

    Returns a list of dictionaries (or `None`) with the following
    entries:

    - name: short description of what's computed;

    - layer_indexes: a list or set of indexes for layers whose
      activations values are needed;

    - test_size & train_size: (as in
      `sklearn.model_selection.train_test_split`)

    - train: a callable taking some test data as mapping (as a
      dictionary) from each layer index given in `layer_indexes` to
      activation values for the corresponding layer, and returns some
      arbitrary object; this is to be called only once during
      initialization;

    - test: a callable taking some extra test data and associated
      labels as two separate mappings (as a dictionary) from each
      layer index given in `layer_indexes` to activation values for
      the corresponding layer.

    Any function given as entries above is always called last.
    """
    # - accum_test: a callable that is called with the object returned
    #   by `train`, along with batched activation values for every layer
    #   on the test data, and returns a new or updated accumulator.
    #   This is called at least once.
    #
    # - final_test: optional function that is called with the final test
    #   accumulator once all batched test activations have been passed
    #   to  `accum_test`.
    return None




# ---



class Criterion4RootedSearch (Criterion):
  '''
  Any criterion that can be used to find a pair of a base test case
  and a test target should inherit this class.
  '''

  @abstractmethod
  def find_next_rooted_test_target(self) -> Tuple[Input, TestTarget]:
    '''
    Seeks a new test target associated with an existing test input
    taken from the set of recorded test cases.

    Note this method MUST perform enough bookkeeping so that two
    successive calls that are not interleaved with any call to
    :meth:`Criterion.add_new_test_cases` return different results.
    This property is to enforce progress upon unsuccessful search of
    concrete inputs.
    '''
    raise NotImplementedError


# ---


class Criterion4FreeSearch (Criterion):
  '''
  Any criterion that can be used to select a test target without
  relying on activation data or previously inserted test cases should
  inherit this class.
  '''

  @abstractmethod
  def find_next_test_target(self) -> TestTarget:
    '''
    Seeks and returns a new test target.
    '''
    raise NotImplementedError


# ---


def setup_basic_report(criterion: Criterion, **kwds) -> Report:
  '''
  Returns a very basic report object that feeds files whose base names
  are constructed from the provided criterion.

  Extra keyword arguments are passed on to the constructor of
  :class:`Report`.
  '''
  return Report (base_name = '{0}_{0.metric}'.format (criterion),
                 **kwds)



class Engine:
  '''
  Core Deepconcolic engine.
  '''

  def __init__(self, ref_data, train_data,
               criterion: Criterion,
               custom_filter: Filter = None,
               **kwds):
    """
    Builds a test engine with the given DNN, reference data, and test
    criterion.  Uses the input metric provided by the
    criterion-specific analyzer as filter for assessing legitimacy of
    new test inputs, unless `custom_filter` is not `None`.
    """
    
    self.ref_data = ref_data
    self.train_data = train_data
    self.criterion = criterion
    self.filter = custom_filter or criterion.filter
    assert isinstance (self.filter, Filter)
    super().__init__(**kwds)
    self._stat_based_inits ()
    self.criterion.finalize_setup ()


  def __repr__(self):
    return 'criterion {0} with norm {0.metric}'.format (self.criterion)


  def _run_test(self, x):
    return np.argmax (self.criterion.analyzer.dnn.predict (np.array([x])))


  def _run_tests(self, xl):
    return np.argmax (self.criterion.analyzer.dnn.predict (np.array(xl)),
                      axis = 1)


  def _initialize_search (self, report: Report, initial_test_cases = None):
    '''
    Method called once at the beginning of search.
    '''
    xl = []
    if initial_test_cases is not None and initial_test_cases > 0:
      x = self.ref_data.data
      x = x[self._run_tests (x) == self.ref_data.labels]
      x = np.random.default_rng().choice (a = x, axis = 0,
                                          size = min (initial_test_cases, len (x)))
      p1 ('Initializing with {} randomly selected test case{} that {} correctly classified.'
          .format(len (x), 's' if len (x) > 1 else '', 'are' if len (x) > 1 else 'is'))
      self.criterion.add_new_test_cases (x)
    elif self.criterion.rooted_search:
      p1 ('Randomly selecting an input from test data.')
      x = np.random.default_rng().choice (a = self.ref_data.data, axis = 0)
      report.save_input (x, 'seed-input')
      self.criterion.add_new_test_cases ([x])


  def run(self,
          setup_report: Callable[[Criterion], Report] = setup_basic_report,
          initial_test_cases = None,
          max_iterations = None,
          **kwds):
    '''
    Uses `setup_report` to construct a helper for outputing logs and
    new test cases, and then starts the engine for either: up to
    `max_iterations` iterations (i.e. number of runs of the analyzer)
    if `max_iterations = None`, or until full coverage is reached, or
    the criterion is fulfilled (whichever happens first).
    '''

    criterion = self.criterion
    filter = self.filter

    p1 ('Starting tests for {}{}.'
        .format (self, '' if max_iterations == None else
                 ' ({} max iterations)'.format (max_iterations)))
    report = setup_report (criterion, **kwds)

    self._initialize_search (report, initial_test_cases)

    coverage = criterion.coverage ()
    p1 ('#0 {}: {.as_prop:10.8%}'.format(criterion, coverage))
    report.step ('{0}-cover: {1} #test cases: {0.num_test_cases} '
                 .format(criterion, coverage),
                 '#adversarial examples: 0\n')

    iteration = 1

    try:

      while ((max_iterations == None or iteration <= max_iterations) and
             not coverage.done):
  
        adversarial = False

        search_attempt, target = criterion.search_next ()
        if search_attempt != None:
          x0, x1, d = search_attempt
  
          # Test oracle for adversarial testing
          close_enough = filter.close_to (self.ref_data.data, x1)
          if close_enough:
            target.cover ()
            criterion.add_new_test_cases ([x1])
            coverage = criterion.coverage ()
            y0 = self._run_test (x0)
            y1 = self._run_test (x1)
  
            if y1 != y0:
              adversarial = True
              report.new_adversarial (new = (x1, y1), orig = (x0, y0), dist = d,
                                      is_int = criterion.metric.is_int)
            else:
              report.new_test (new = (x1, y1), orig = (x0, y0), dist = d,
                               is_int = criterion.metric.is_int)
  
        p1 ('#{} {}: {.as_prop:10.8%} {}'
            .format (iteration, criterion, coverage,
                     'with {} at {} distance {}: {}'
                     .format('new test case' if close_enough else 'failed attempt',
                             criterion.metric, d,
                             'too far from raw data' if not close_enough else
                             'adversarial' if adversarial else 'passed')
                     if search_attempt != None else 'after failed attempt'))
  
        report.step ('{0}-cover: {1} #test cases: {0.num_test_cases} '
                     .format(criterion, coverage),
                     '#adversarial examples: {0.num_adversarials} '
                     .format(report),
                     '#diff: {} {}\n'
                     .format(d if search_attempt != None else '_',
                             target.log_repr ()))
  
        iteration += 1

    except EarlyTermination as e:
      p1 ('{}'.format (e))


  def _stat_based_inits(self):
    '''
    Performs basic and incremental static initializations of the
    criterion (and its associated analyzer).
    '''
    
    ggi = self.criterion.stat_based_basic_initializers () or []
    gi = self.criterion.stat_based_incremental_initializers () or []
    trcv = self.criterion.stat_based_train_cv_initializers () or []
    tscv = self.criterion.stat_based_test_cv_initializers () or []

    # Run stats on batched activations, and/or accumulate for layers
    # that require full activations for their stats.

    if gi == [] and ggi == [] and trcv == [] and tscv == []:
      return

    if gi != [] or ggi != []:
      if gi != []:
        np1 ('Computing {}... '
             .format(' & '.join((map ((lambda gi: gi['name']), gi)))))
      else:
        np1 ('Aggregating activations required for {}... '
             .format(' & '.join((map ((lambda gg: gg['name']), ggi)))))
      acc = [ None for _ in gi ]
      gacc_indexes = set().union (*(gg['layer_indexes'] for gg in ggi))
      gacc = dict.fromkeys (gacc_indexes, np.array([]))
      for act in self._batched_activations_on_raw_data ():
        acc = [ g['accum'](act, acc) for g, acc in zip(gi, acc) ]
        if ggi != []:
          for j in gacc_indexes:
            gacc[j] = (np.concatenate ((gacc[j], act[j]), axis = 0)
                       if gacc[j].any () else np.copy (act[j]))
      for g, acc in zip(gi, acc):
        if 'final' in g: g['final'](acc)
      print ('done.')
      for g in gi:
        if 'print' in g: print (g['print']())
      print ('', end = '', flush = True)
  
      # Now we can pass the aggregated activations to basic stat
      # initializers.

      if ggi == []:
        return

      np1 ('Computing {}... '
           .format(' & '.join((map ((lambda gg: gg['name']), ggi)))))
      for gg in ggi:
        gg['once']({ j: gacc[j] for j in gg['layer_indexes']})
      print ('done.')
      for gg in ggi:
        if 'print' in gg: print (gg['print']())
      print (end = '', flush = True)

    if trcv != []:
      self._cv_init (trcv, self.train_data)

    if tscv != []:
      self._cv_init (tscv, self.test_data)


  def _cv_init (self, cv, data):
    idxs = np.arange (len (data.data))
    for x in cv:
      np1 ('Computing {}... ' .format(x['name']))
      train_idxs, test_idxs = train_test_split (
        idxs, test_size = x['test_size'], train_size = x['train_size'])
      acts = self._activations_on_indexed_data (data, train_idxs)
      acc = x['train']({ j: acts[j] for j in x['layer_indexes'] })

      if 'test' in x:
        acts = self._activations_on_indexed_data (data, test_idxs)
        lbls = data.labels[test_idxs]
        x['test']({ j: acts[j] for j in x['layer_indexes'] }, lbls)


  def _batched_activations_on_raw_data(self):
    batches = np.array_split (self.ref_data.data,
                              len (self.ref_data.data) // 1000 + 1)
    for batch in batches:
      yield (self.criterion.analyzer.eval_batch (batch, allow_input_layer = True))


  def _activations_on_indexed_data(self, data, indexes):
    batch = data.data[indexes]
    return self.criterion.analyzer.eval_batch (batch, allow_input_layer = True)


  # ---


# ---

CL = TypeVar('CL')                  # Type variable for covered layers

def setup (test_object: test_objectt = None,
           cover_layers: Sequence[CL] = None,
           setup_analyzer: Callable[[dict], Analyzer] = None,
           setup_criterion: Callable[[Sequence[CL], Analyzer, dict], Criterion] = None,
           criterion_args: dict = {},
           **kwds) -> Engine:
  """
  Helper to build engine instances.  Extra arguments are passed to the
  analyzer setup function (`setup_analyzer`).

  Note: only fields ``dnn``, ``raw_data``, and ``train_data`` are
  required from `test_object`.
  """

  print ('DNN under test has {0} layer functions, {1} of which {2} to be covered:'
         .format(len(get_layer_functions (test_object.dnn)[0]), len(cover_layers),
                 'is' if len(cover_layers) <= 1 else 'are'),
         [ cl for cl in cover_layers ],
         sep='\n', end = '\n\n')
  analyzer = setup_analyzer (analyzed_dnn = test_object.dnn, **kwds)
  criterion = setup_criterion (cover_layers, analyzer, **criterion_args)
  return Engine (test_object.raw_data, test_object.train_data, criterion)


# ------------------------------------------------------------------------------
# Provide slightly more specialized classes:


class CoverableLayer (cover_layert):
  '''
  Base class for any layer based on which coverability criteria are
  defined.

  Note: this reuses `cover_layert` to hold `layer` and `layer_index`,
  yet one should not rely on that as this is only temporary.
  '''

  def __init__(self, layer = None, layer_index = None, **kwds):
    super().__init__(layer, layer_index, **kwds)


  def __repr__(self):
    return self.layer.name


# ---


class BoolMappedCoverableLayer (CoverableLayer):
  '''
  Represents a layer where coverage is defined using a Boolean mapping
  from each neuron.
  '''

  def __init__(self,
               feature_indices = None,
               bottom_act_value = MIN,
               **kwds):
    super().__init__(**kwds)
    self._initialize_map (feature_indices)
    self.bottom_act_value = bottom_act_value
    self.filtered_out = 0


  def _initialize_map(self, feature_indices) -> None:
    shape = tuple(self.layer.output_shape)
    self.map = np.ones(shape[1:], dtype = bool)
    if self.is_conv and feature_indices != None:
      for i in range(0, self.map.shape[-1]):
        if not i in feature_indices:
          self.map[...,i] = False


  def filter_out_padding_against(self, prev_layer):
    if not self.is_conv: return
    tp1 ('Filtering out padding neurons from layer {}'.format(self))
    paddings = 0
    for n in np.ndindex (self.map.shape):
      if self.map[n]:
        if is_padding (n, self, prev_layer, post = True, unravel_pos = False):
          self.map[n] = False
          paddings += 1
    self.filtered_out += paddings


  def coverage(self, feature_indices) -> Coverage:
    if not self.is_conv or feature_indices == None:
      nc = np.count_nonzero (self.map)
      tot = np.prod (self.map.shape)
    else:
      nc, tot = 0, 0
      for i in range(0, self.map.shape[-1]):
        if not i in feature_indices: continue
        nc += np.count_nonzero (self.map[...,i])
        tot += self.map[...,i].size
    tot -= self.filtered_out
    return Coverage (covered = tot - nc, total = tot)


  ## to get the index of the next property to be satisfied
  # [ eq. (15,17,18)? ]
  def find(self, f):
    acts = np.array(self.activations)
    spos = f (acts)
    pos = np.unravel_index(spos, acts.shape)
    return pos, acts.item (spos)


  def cover(self, pos) -> None:
    self.map[pos] = False


  def inhibit_activation(self, pos) -> None:
    '''
    Inhibit any activation at the given position so that it is not
    returned by any direct subsequent call to `find` (i.e not
    precededed by a call to `update_with_new_activations`).
    '''
    act = self.activations
    while len(pos) != 1:
      act = act[pos[0]]
      pos = pos[1:]
    act[pos] = self.bottom_act_value


  def update_with_new_activations(self, act) -> None:
    act = copy.copy (act[self.layer_index])
    assert len (act) == 1
    # Keep only negative new activation values:
    # TODO: parameterize this (ditto bottom_act_value)
    act[act >= 0] = 0
    self.map = np.logical_and (self.map, act[0])
    # Append activations after map change
    self._append_activations (act)


  def _append_activations(self, act):
    '''
    Append given activations into the internal buffer.
    '''
    if len(self.activations) >= BUFFER_SIZE:
      self.activations.pop(-1)
    self.activations.insert(0, act)
    self._filter_out_covered_activations ()


  def _filter_out_covered_activations(self):
    for j in range(0, len(self.activations)):
      # Only keep values of non-covered activations
      self.activations[j] = np.multiply(self.activations[j], self.map)
      self.activations[j][self.activations[j] >= 0] = self.bottom_act_value


# ---


class LayerLocalAnalyzer (Analyzer):
  '''
  Analyzers that seek layer-local criteria must inherit this class to
  register the sequence of covered layers.
  '''

  @abstractmethod
  def finalize_setup(self, clayers: Sequence[CoverableLayer]):
    raise NotImplementedError


# ---


class LayerLocalCriterion (Criterion):
  '''
  Criteria whose definition involves layer-local coverage properties.
  '''

  def __init__(self,
               clayers: Sequence[BoolMappedCoverableLayer] = None,
               shallow_first = True,
               feature_indices = None,
               **kwds):
    self.shallow_first = shallow_first
    self.cover_layers = clayers
    self.feature_indices = feature_indices
    super().__init__(**kwds)
    for cl in self.cover_layers:
      assert isinstance (cl, BoolMappedCoverableLayer)


  def finalize_setup(self):
    if isinstance (self.analyzer, LayerLocalAnalyzer):
      self.analyzer.finalize_setup (self.cover_layers)


  @property
  def _updatable_layers(self):
    '''
    Gives the set of all internal objects that are updated upon
    insertion of a new test case.
    '''
    return self.cover_layers


  def stat_based_incremental_initializers(self):
    if len (self.cover_layers) <= 1:
      for cl in self.cover_layers: cl.pfactor = 1.0
      return None
    else:
      return [{
        'name': 'magnitude coefficients',
        'accum': self._acc_magnitude_coefficients,
        'final': self._calculate_pfactors,
        'print': (lambda : [cl.pfactor for cl in self.cover_layers]),
      }]


  def _acc_magnitude_coefficients(self, new_acts, prev_acts = None):
    import copy
    if prev_acts is None:
      prev_acts = copy.copy (new_acts)
    else:
      for j in range(0, len(prev_acts)):
        prev_acts[j] = np.concatenate((prev_acts[j], new_acts[j]), axis = 0)
    return prev_acts


  def _calculate_pfactors (self, activations):
    fks = [ np.average (np.abs (activations[cl.layer_index]))
            for cl in self.cover_layers ]
    av = np.average (fks)
    for cl, fks in zip(self.cover_layers, fks):
      cl.pfactor = av / fks


  # ---


  def coverage(self) -> Coverage:
    c = Coverage (total = 0)
    for cl in self.cover_layers:
      # if self.test_object.tests_layer (cl):
      # assert (self.test_object.tests_layer (cl))
      c += cl.coverage (self.feature_indices)
    return c


  # ---


  def update_coverage (self, tl: Sequence[Input]):
    """
    Register new test cases
    """
    for t in tl:
      acts = self.analyzer.eval (t, allow_input_layer = True)
      for cl in self._updatable_layers:
        cl.update_with_new_activations (acts)


  def get_max(self) -> Tuple[BoolMappedCoverableLayer, Tuple[int, ...], float, Input]:
    '''
    '''
    layer, pos, value = None, None, MIN
    for i, cl in enumerate(self.cover_layers):
      p, v = cl.find (np.argmax)
      v *= cl.pfactor
      if v > value:
        layer, pos, value = i, p, v
        if self.shallow_first: break
        if np.random.uniform (0., 1.) < i * 1.0 / len(self.cover_layers): break
    if layer == None:
      sys.exit('incorrect layer indices specified' +
               '(the layer tested shall be either conv or dense layer)')
    return self.cover_layers[layer], pos, value, self.test_cases[-1-pos[0]]


  def get_random(self):
    clx = [ cl for cl in self.cover_layers if np.any (cl.map) ]
    if clx == []:
      return None
    else:
      while True:
        idx = np.random.randint(0, len(clx))
        cl = clx[idx]
        tot_s = np.prod (cl.map.shape)
        # pos = np.random.randint (0, tot_s)
        # if not self.test_object.feature_indices == None:
        pos = np.argmax (cl.map.shape)
        while pos < tot_s and not cl.map.item(pos):
          pos += 1
        if pos < tot_s and cl.map.item(pos):
          break
    return cl, np.unravel_index(pos, cl.map.shape)


# ---
