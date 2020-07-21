from typing import *
from utils import *
from engine import *
import numpy as np

from functools import reduce
from itertools import product
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, Binarizer
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import log_loss, classification_report
from pomegranate import Node, BayesianNetwork
from pomegranate.distributions import DiscreteDistribution, ConditionalProbabilityTable


# ---


class BFcLayer (CoverableLayer):
  """
  Base class for layers to be covered by BN-based criteria.
  """

  def __init__(self, transform = None, discretization = None, **kwds):
    super().__init__(**kwds)
    self.transform = transform
    self.discr = discretization


  @property
  def num_features (self):
    return len (self.transform[-1].components_)


# ---


Interval = Tuple[Optional[float], Optional[float]]


class FeatureDiscretizer:

  @abstractmethod
  def feature_parts (self, feature: int) -> int:
    raise NotImplementedError

  @abstractmethod
  def edges (self, feature: int, value: float) -> Interval:
    raise NotImplementedError

  @abstractmethod
  def part_edges (self, feature: int, part: int) -> Interval:
    raise NotImplementedError

  @abstractmethod
  def fit_wrt (self, x, y) -> None:
    raise NotImplementedError


class FeatureBinarizer (FeatureDiscretizer, Binarizer):

  def feature_parts (self, _feature):
    return 2

  def edges (self, feature: int, value: float) -> Interval:
    thr = self.threshold
    return (thr, np.inf) if value >= thr else (-np.inf, thr)

  def part_edges (self, feature: int, part: int) -> Interval:
    thr = self.threshold
    return (-np.inf, thr) if part == 0 else (thr, np.inf)

  def fit_wrt (self, x, y, transform) -> None:
    self.fit (y)


class KBinsFeatureDiscretizer (FeatureDiscretizer, KBinsDiscretizer):

  def feature_parts (self, feature):
    return self.n_bins_[feature]

  def edges (self, feature: int, value: float) -> Interval:
    edges = np.concatenate((np.array([-np.inf]),
                            self.bin_edges_[feature][1:-1],
                            np.array([np.inf])))
    part = np.searchsorted (edges, value, side = 'right')
    return edges[part-1], edges[part]

  def part_edges (self, feature: int, part: int) -> Interval:
    edges = self.bin_edges_[feature]
    return ((-np.inf if part   == 0           else edges[part  ],
             np.inf  if part+2 == len (edges) else edges[part+1]))

  def fit_wrt (self, _x, y, _transform) -> None:
    self.fit (y)


class KBinsNOutFeatureDiscretizer (KBinsFeatureDiscretizer):

  def __init__(self, n_bins = 2, **kwds):
    super().__init__(n_bins = max(2, n_bins), **kwds)
    self.one = n_bins == 1

  def feature_parts (self, feature):
    return self.n_bins_[feature] + 2 if not self.one else 3

  def edges (self, feature: int, value: float) -> Interval:
    edges = self.bin_edges_[feature]
    part = np.searchsorted (edges, value, side = 'right')
    return self.part_edges (feature, part)

  def part_edges (self, feature: int, part: int) -> Interval:
    edges = self.bin_edges_[feature]
    if self.one and part == 0:
      return (-np.inf, edges[0])
    elif self.one and part == 1:
      return (edges[0], edges[2])
    elif self.one and part == 2:
      return (edges[2], np.inf)
    else:
      return ((-np.inf if part == 0           else edges[part-1],
               np.inf  if part == len (edges) else edges[part  ]))

  def fit_wrt (self, _x, y, _transform) -> None:
    self.fit (y)

# ---



def bayes_node_name(fl, idx):
  return '.'.join ((str(fl), *((str(i) for i in idx))))



class DiscretizedFeatureNode (Node):

  def __init__(self, flayer: BFcLayer, feature: int, *args, **kwds):
    super().__init__ (*args, name = bayes_node_name (flayer, (feature,)), **kwds)
    self.flayer = flayer
    self.feature = feature


  def discretized_range(self):
    return range (self.flayer.discr.feature_parts (self.feature))



class DiscretizedInputFeatureNode (DiscretizedFeatureNode):

  def __init__(self, flayer, feature, **kwds):
    fparts = range (flayer.discr.feature_parts (feature))
    super().__init__(flayer, feature,
                     DiscreteDistribution ({ fbin: 0.0 for fbin in fparts }),
                     **kwds)


class DiscretizedHiddenFeatureNode (DiscretizedFeatureNode):

  def __init__(self, flayer, feature, prev_nodes, **kwds):
    prev_distrs = [ n.distribution for n in prev_nodes ]
    prev_fparts = list ([ bin for bin in prev_node.discretized_range () ]
                        for prev_node in prev_nodes)
    fparts = range (flayer.discr.feature_parts (feature))
    condprobtbl = [ list (p) + [0.0] for p in product (*prev_fparts, fparts) ]
    del prev_fparts
    super().__init__(flayer, feature,
                     ConditionalProbabilityTable (condprobtbl, prev_distrs),
                     **kwds)
    del condprobtbl


# ---


class _BaseBFcCriterion (Criterion):

  def __init__(self,
               clayers: Sequence[CoverableLayer],
               *args,
               epsilon = 0.001,
               bn_abstr_train_size = 0.5,
               bn_abstr_test_size = None,
               bn_n_jobs = 1,
               print_classification_reports = True,
               score_layer_likelihoods = False,
               report_on_feature_extractions = None,
               close_reports_on_feature_extractions = None,
               assess_discretized_feature_probas = False,
               **kwds):
    assert (print_classification_reports is None or isinstance (print_classification_reports, bool))
    assert (report_on_feature_extractions is None or callable (report_on_feature_extractions))
    assert (close_reports_on_feature_extractions is None or callable (close_reports_on_feature_extractions))
    self.epsilon = epsilon
    self.bn_n_jobs = bn_n_jobs
    self.bn_abstr_params = { 'train_size': bn_abstr_train_size,
                             'test_size': bn_abstr_test_size }
    self.print_classification_reports = print_classification_reports
    self.score_layer_likelihoods = score_layer_likelihoods
    self.report_on_feature_extractions = report_on_feature_extractions
    self.close_reports_on_feature_extractions = close_reports_on_feature_extractions
    self.assess_discretized_feature_probas = assess_discretized_feature_probas
    self.flayers = list (filter (lambda l: isinstance (l, BFcLayer), clayers))
    clayers = list (filter (lambda l: isinstance (l, BoolMappedCoverableLayer), clayers))
    assert (clayers == [])
    self.base_dimreds = None
    super().__init__(*args, **kwds)


  def finalize_setup(self):
    self.analyzer.finalize_setup (self.flayers)


  def flatten_for_layer (self, map, l = None):
    res = None
    for fl in self.flayers:
      if l is None or fl in l:
        x = np.vstack((e.flatten () for e in map[fl.layer_index]))
        res = np.hstack ((res, x)) if res is not None else x
        if res is not x: del x
    return res


  def dimred_activations (self, acts, l = None):
    yacts = None
    for fl in self.flayers:
      if l is None or fl in l:
        x = np.vstack((a.flatten () for a in acts[fl.layer_index]))
        y = fl.transform.transform (x)
        yacts = np.hstack ((yacts, y)) if yacts is not None else y
        del x
        if yacts is not y: del y
    return yacts


  def dimred_n_discretize_activations (self, acts):
    facts = np.array([], dtype = int)
    for fl in self.flayers:
      x = np.vstack((a.flatten () for a in acts[fl.layer_index]))
      y = fl.discr.transform (fl.transform.transform (x))
      facts = np.hstack ((facts, y.astype (int))) if facts.any () else y.astype (int)
      del x, y
    return facts


  # ---


  def reset (self):
    super().reset ()
    self.base_dimreds = None
    assert (self.num_test_cases == 0)
    

  def fit_activations (self, acts):
    # Assumes `num_test_cases' has already been updated with the
    # inputs that triggered the given activations; otherwise, set
    # inertia to 0.0, which basically erases history.
    facts = self.dimred_n_discretize_activations (acts)
    nbase = self.num_test_cases - len (facts)
    self.N.fit (facts,
                inertia = nbase / self.num_test_cases if nbase >= 0 else 0.0)


  def update_coverage (self, tl: Sequence[Input]) -> None:
    # Take care `num_test_cases` has already been updated:
    acts = self.analyzer.eval_batch (np.array(tl), allow_input_layer = False)
    self.fit_activations (acts)

    # Append feature values for new tests
    new_dimreds = self.dimred_activations (acts)
    self.base_dimreds = (np.vstack ((self.base_dimreds, new_dimreds))
                         if self.base_dimreds is not None else new_dimreds)
    if self.base_dimreds is not new_dimreds: del new_dimreds


  # ---


  def tests_feature_values_n_intervals (self, fidx: int):
    fl, feature = self.fidx2fli[fidx]
    dimreds = self.base_dimreds[..., fidx : fidx + 1].flatten ()
    intervals = [ (i, fl.discr.edges (feature, v)) for i, v in enumerate(dimreds) ]
    return dimreds, intervals


  def all_tests_close_to (self, fidx: int, fint: int):
    dimreds, intervals = self.tests_feature_values_n_intervals (fidx)
    fl, flfeature = self.fidx2fli[fidx]
    target_interval = fl.discr.part_edges (flfeature, fint)
    intervals = [ i for i, interval in intervals if interval != target_interval ]
    all = (([ (i, abs(dimreds[i] - target_interval[0])) for i in intervals ]
            if target_interval[0] != -np.inf else []) +
           ([ (i, abs(dimreds[i] - target_interval[1])) for i in intervals ]
            if target_interval[1] != np.inf else []))
    all.sort (key = lambda x: x[1])
    # np.random.shuffle (all)
    del dimreds, intervals
    return all


  def _check_within (self, expected: int, verbose = True):
    def check_acts (feature: int, t: Input) -> bool:
      acts = self.analyzer.eval (t, allow_input_layer = False)
      facts = self.dimred_n_discretize_activations (acts)
      res = facts[0][feature] == expected
      if verbose and not res:
        dimred = self.dimred_activations (acts)
        dimreds = dimred[..., feature : feature + 1].flatten ()
        tp1 ('| Got interval {}, expected {} (fval {})'
             .format(facts[0][feature], expected, dimreds))
      return res
    return check_acts


  # ----


  def _marginals (self, p):
    return [ p.parameters[0][i]
             for i in range (len (p.parameters[0])) ]


  def _all_marginals (self):
    marginals = self.N.marginal ()
    res = [ self._marginals (p) for p in marginals ]
    del marginals
    return res


  def _all_cpts (self):
    # Rely on self.N.states to get conditional probability tables:
    return [ np.array (self._marginals (j.distribution))
             for j in self.N.states
             if not isinstance (j, DiscretizedInputFeatureNode) ]


  def _all_cond_probs (self):
    return [ cpt[:,-1] for cpt in self._all_cpts () ]


  def bfc_coverage (self) -> Coverage:
    """
    Computes the BFCov metric as per the underlying Bayesian Network
    abstraction.
    """
    assert (self.num_test_cases > 0)
    margs = self._all_marginals ()
    props = sum (np.count_nonzero (np.array(p) >= self.epsilon) / len (p)
                 for p in margs)
    return Coverage (covered = props, total = len (margs))


  def bfdc_coverage (self) -> Coverage:
    """
    Computes the BFdCov metric as per the underlying Bayesian Network
    abstraction.
    """
    assert (self.num_test_cases > 0)
    # Count 0s in all joint mass functions in the BN abstraction
    cndps = self._all_cond_probs ()
    props = sum (np.count_nonzero (np.array(p) >= self.epsilon) / len (p)
                 for p in cndps)
    return Coverage (covered = props, total = len (cndps))


  # ---


  def stat_based_train_cv_initializers (self):
    """
    Initializes the criterion based on traininig data.

    Directly uses argument ``bn_abstr_train_size`` and
    ``bn_abstr_test_size`` arguments given to the constructor, and
    optionally computes some scores (based on flags given to the
    constructor as well).
    """
    bn_abstr = ({ 'test': self._score }
                if (self.score_layer_likelihoods or
                    self.report_on_feature_extractions is not None or
                    self.assess_discretized_feature_probas) else {})
    return [{
      **self.bn_abstr_params,
      'name': 'Bayesian Network abstraction',
      'layer_indexes': set ([fl.layer_index for fl in self.flayers]),
      'train': self._discretize_features_and_create_bn_structure,
      **bn_abstr,
      # 'accum_test': self._accum_fit_bn,
      # 'final_test': self._bn_score,
    }]


  def _discretize_features_and_create_bn_structure (self, acts):
    """
    Called through :meth:`stat_based_train_cv_initializers` above.
    """
    
    cnp1 ('| Given training data of size {}'
          .format(len(acts[self.flayers[0].layer_index])))

    # First, fit feature extraction and discretizer parameters:
    for fl in self.flayers:
      cnp1 ('| Discretizing features for layer {}... '.format (fl))
      x = np.stack([a.flatten () for a in acts[fl.layer_index]], axis = 0)
      # print (x.shape, x)
      y = fl.transform.fit_transform (x)
      fl.discr.fit_wrt (x, y, fl.transform)
      np1 ('{} nodes.'.format (y.shape[1]))
      del x, y

    self.total_variance_ratios_ = np.array([
      sum (fl.transform[-1].explained_variance_ratio_) for fl in self.flayers
      if hasattr (fl.transform[-1], 'explained_variance_ratio_')
    ])

    # Report on explained variance
    for fl in self.flayers:
      if hasattr (fl.transform[-1], 'explained_variance_ratio_'):
        cnp1 ('| Captured variance ratio for layer {} is {:6.2%}'
              .format (fl, sum (fl.transform[-1].explained_variance_ratio_)))

    # Second, fit some distributions with input layer values (NB: well, actually...)
    # Third, contruct the Bayesian Network
    self.N = self._create_bayesian_network ()

    self.fidx2fli = {}
    fidx = 0
    for fl in self.flayers:
      for i in range (fl.num_features):
        self.fidx2fli[fidx + i] = (fl, i)
      fidx += fl.num_features

    # Last, fit the Bayesian Network with given training activations
    # for now, for the purpose of preliminary assessments; the BN will
    # be re-initialized upon the first call to `add_new_test_cases`:
    if self.score_layer_likelihoods or self.assess_discretized_feature_probas:
      self.fit_activations (acts)


  def _create_bayesian_network (self):
    """
    Actual BN instantiation.
    """
    
    import gc
    nc = sum (f.num_features for f in self.flayers)
    ec = sum (f.num_features * g.num_features
              for f, g in zip (self.flayers[:-1], self.flayers[1:]))

    ctp1 ('| Creating Bayesian Network of {} nodes and {} edges...'
          .format (nc, ec))
    N = BayesianNetwork (name = 'BN Abstraction')

    fl0 = self.flayers[0]
    nodes = [ DiscretizedInputFeatureNode (fl0, fidx)
              for fidx in range (fl0.num_features) ]
    N.add_nodes (*(n for n in nodes))

    gc.collect ()
    prev_nodes = nodes
    for fl in self.flayers[1:]:
      nodes = [ DiscretizedHiddenFeatureNode (fl, fidx, prev_nodes)
                for fidx in range (fl.num_features) ]
      N.add_nodes (*(n for n in nodes))

      for pn, n in product (*(prev_nodes, nodes)):
        N.add_edge (pn, n)
      tp1 ('| Creating Bayesian Network: {}/{} nodes, {}/{} edges done...'
           .format (N.node_count (), nc, N.edge_count (), ec))

      del prev_nodes
      gc.collect ()
      prev_nodes = nodes

    del prev_nodes
    gc.collect ()
    tp1 ('| Creating Bayesian Network of {} nodes and {} edges: baking...'
         .format (nc, ec))
    N.bake ()
    p1 ('| Created Bayesian Network of {} nodes and {} edges.'
        .format (nc, ec))
    return N


  # ---


  def _score (self, acts, labels = None):
    """
    Basic scores for manual investigations.
    """
    
    p1 ('| Given test sample of size {}'
         .format(len(acts[self.flayers[0].layer_index])))

    if (self.score_layer_likelihoods or
        self.report_on_feature_extractions is not None):
      self._score_feature_extractions (acts, labels)

    if self.assess_discretized_feature_probas:
      truth = self.dimred_n_discretize_activations (acts)
      self._score_discretized_feature_probas (truth)
      del truth


  def _score_feature_extractions (self, acts, labels = None):
    racc = None
    idx = 1
    first_feature_idx = 0
    self.average_log_likelihoods_ = []
    for fl in self.flayers:
      flatacts = self.flatten_for_layer (acts, (fl,))

      if self.score_layer_likelihoods:
        tp1 ('| Computing average log-likelihood of test sample for layer {}...'
             .format (fl))
        self.average_log_likelihoods_.append (fl.transform.score (flatacts))
        p1 ('| Average log-likelihood of test sample for layer {} is {}'
            .format (fl, self.average_log_likelihood[-1]))

      if self.report_on_feature_extractions is not None:
        fdimred = self.dimred_activations (acts, (fl,))
        racc = self.report_on_feature_extractions (fl, flatacts, fdimred, labels, racc)
        del fdimred

      idx += 1
      del flatacts
      first_feature_idx += fl.num_features

    if self.close_reports_on_feature_extractions is not None:
      self.close_reports_on_feature_extractions (racc)


  def _score_discretized_feature_probas (self, truth):
    """
    Further scoring the predictive abilites of the BN.
    """

    if self.N.edge_count () == 0:
      p1 ('Warning: BN abstraction has no edge: skipping prediction assessments.')
      return

    features_probas = self._setup_estimate_feature_probas (truth)

    all_floss = []
    self.log_losses = []
    self.classification_reports = []
    first_feature_idx = 0
    for fl in self.flayers:
      floss = []
      for fidx in range (fl.num_features):
        flabels = list (range (fl.discr.feature_parts (fidx)))
        feature_idx = first_feature_idx + fidx
        ftruth = truth[..., feature_idx : feature_idx + 1].flatten ()

        tp1 ('| Computing predictions for feature {} of {}...'.format (fidx, fl))
        fprobas = features_probas (feature_idx, 1).flatten ()

        tp1 ('| Computing log loss for feature {} of {}...'.format (fidx, fl))
        fpredict_probs = self._all_prediction_probas (fprobas)
        loss = log_loss (ftruth, fpredict_probs, labels = flabels)
        floss.append (loss)
        p1 ('| Log loss for feature {} of {} is {}'.format (fidx, fl, loss))

        if self.print_classification_reports:
          p1 ('| Classification report for feature {} of {}:'.format (fidx, fl))
          fpreds = [ np.argmax (p) for p in fpredict_probs ]
          self.classification_reports.append(
            classification_report (ftruth, fpreds, labels = flabels))
          print (self.classification_reports[-1])
          del fpreds

        del ftruth, fprobas, fpredict_probs, flabels

      self.log_losses.append((np.min (floss), np.mean (floss),
                              np.std (floss), np.max (floss)))
      all_floss.extend(floss)
      first_feature_idx += fl.num_features

    self.all_log_losses = (np.min (all_floss), np.mean (all_floss),
                           np.std (all_floss), np.max (all_floss))
    del features_probas


  def _setup_estimate_feature_probas (self, truth):
    ytest = truth if truth.dtype == float else np.array (truth, dtype = float)
    ftest = ytest.copy ()
    def estimate_feature_probas (fidx, nbfeats):
      ftest[..., fidx : fidx + nbfeats] = np.nan
      probas = np.array (self.N.predict_proba (ftest, n_jobs = self.bn_n_jobs))
      ftest[..., fidx : fidx + nbfeats] = truth[..., fidx : fidx + nbfeats]
      lprobas = probas[..., fidx : fidx + nbfeats]
      del probas
      return lprobas
    return (lambda fidx, nbfeats: estimate_feature_probas (fidx, nbfeats))


  def _prediction_probas (self, p):
    return [ p.parameters[0][i] for i in range (len (p.parameters[0])) ]


  def _all_prediction_probas (self, fprobas):
    return [ self._prediction_probas (p) for p in fprobas ]


  # ----


# ---


class BFcTarget (NamedTuple, TestTarget):
  fnode: DiscretizedFeatureNode
  feature_part: int
  sanity_check: Callable[[int, int, Input], bool]
  root_test_idx: int

  def __repr__(self) -> str:
    return ('interval {} of feature {} in layer {} (from root test {})'
            .format(self.fnode.flayer.discr.part_edges (self.fnode.feature,
                                                        self.feature_part),
                    self.fnode.feature, self.fnode.flayer, self.root_test_idx))


  def log_repr(self) -> str:
    return ('#layer: {} #feat: {} #part: {}'
            .format(self.fnode.flayer.layer_index,
                    self.fnode.feature, self.feature_part))


  def cover(self) -> None:
    # Do nothing for now; ideally: update some probabilities in fnode...
    pass


  def check(self, t: Input) -> bool:
    """
    Checks whether the target is met.
    """
    return self.sanity_check (self.fnode.feature, t)


# ---


class BFcAnalyzer (Analyzer4RootedSearch):
  """
  Analyzer dedicated to targets of type :class:`BFcTarget`.
  """

  @abstractmethod
  def search_input_close_to(self, x: Input, target: BFcTarget) -> Optional[Tuple[float, Input]]:
    """
    Method specialized for targets of type :class:`BFcTarget`.
    """
    pass



# ---


class BFcCriterion (_BaseBFcCriterion, Criterion4RootedSearch):
  '''
  Some kind of "uniformization" coverage criterion.
  '''

  def __init__(self,
               clayers: Sequence[CoverableLayer], 
               analyzer: BFcAnalyzer,
               *args,
               **kwds):
    assert isinstance (analyzer, BFcAnalyzer)
    super().__init__(clayers, analyzer = analyzer, *args, **kwds)
    self.ban = { fl: set () for fl in self.flayers }


  def __repr__(self):
    return "BFC"


  def reset (self):
    super().reset ()
    self.ban = { fl: set () for fl in self.flayers }


  def coverage (self) -> Coverage:
    return self.bfc_coverage ()


  def _all_normalized_marginals (self):
    marginals = self.N.marginal ()
    tot = sum (len (p.parameters[0]) for p in marginals)
    res = [ [ p.parameters[0][i] * len (p.parameters[0]) / tot
              for i in range (len (p.parameters[0])) ]
            for p in marginals ]
    del marginals
    return res


  def find_next_rooted_test_target (self) -> Tuple[Input, BFcTarget]:
    fwms = self._all_normalized_marginals ()
    # fwms = self._all_marginals ()
    minfwms = np.array([ np.min (p) for p in fwms ])
    # print (minfidx[0], minfwms[minfidx[0]], fwms[minfidx[0]])

    # Search by rarest interval so far:
    res = None
    for fidx in np.argsort (minfwms):
      fint = np.argmin (fwms[fidx])
      fl, flfeature = self.fidx2fli[fidx]
      # print (fidx, 'in', fl, 'WPf =', fwms[fidx][fint],
      #        # 'CP = ', self.N.states[fidx].distribution.parameters[0][fint]
      #        'P =', marginals[fidx].parameters[0][fint])

      # Search closest test that does not fall within target interval:
      within_target = self._check_within (fint, verbose = False)
      for i, v in self.all_tests_close_to (fidx, fint):
        if ((fidx, fint, i) in self.ban[fl] or
            within_target (flfeature, self.test_cases[i])):
          continue
        tp1 ('Selecting root test {} at feature-{}-distance {} from {}, layer {}'
             .format (i, flfeature, v, fl.discr.part_edges (flfeature, fint), fl))
        test = self.test_cases[i]
        fct = BFcTarget (self.N.states[fidx], fint, self._check_within (fint), i)
        self.ban[fl].add ((fidx, fint, i))
        res = test, fct
        break

      if res is not None:
        break

    del fwms

    if res is None:
      raise EarlyTermination ('Unable to find a new candidate input!')
    
    return res



# ---


class BFDcTarget (NamedTuple, TestTarget):
  fnode1: DiscretizedHiddenFeatureNode
  feature_part1: int
  flayer0: BFcLayer
  feature_parts0: Sequence[int]
  sanity_check: Callable[[int, int, Input], bool]
  root_test_idx: int

  def __repr__(self) -> str:
    return (('interval {} of feature {} in layer {}, subject to feature'+
             ' intervals {} in layer {} (from root test {})')
            .format(self.fnode1.flayer.discr.part_edges (self.fnode1.feature,
                                                         self.feature_part1),
                    self.fnode1.feature, self.fnode1.flayer,
                    self.feature_parts0, self.flayer0, self.root_test_idx))


  def log_repr(self) -> str:
    return ('#layer: {} #feat: {} #part: {} #conds: {}'
            .format(self.fnode1.flayer.layer_index,
                    self.fnode1.feature, self.feature_part1,
                    self.feature_parts0))


  def cover(self) -> None:
    # Do nothing for now; ideally: update some probabilities in fnode...
    pass


  def check(self, t: Input) -> bool:
    """
    Checks whether the target is met.
    """
    return self.sanity_check (self.fnode1.feature, t)


# ---


class BFDcAnalyzer (Analyzer4RootedSearch):
  """
  Analyzer dedicated to targets of type :class:`BDFcTarget`.
  """

  @abstractmethod
  def search_input_close_to(self, x: Input, target: BFDcTarget) -> Optional[Tuple[float, Input]]:
    """
    Method specialized for targets of type :class:`BFDcTarget`.
    """
    pass



# ---


class BFDcCriterion (_BaseBFcCriterion, Criterion4RootedSearch):
  '''
  Adaptation of MC/DC coverage for partitioned features.
  '''

  def __init__(self,
               clayers: Sequence[CoverableLayer], 
               analyzer: BFDcAnalyzer,
               *args,
               **kwds):
    assert isinstance (analyzer, BFDcAnalyzer)
    super().__init__(clayers, analyzer = analyzer, *args, epsilon = 0.01, **kwds)
    assert len(self.flayers) >= 2
    self.ban = { fl: set () for fl in self.flayers }


  def __repr__(self):
    return "BFdC"


  def reset (self):
    super().reset ()
    self.ban = { fl: set () for fl in self.flayers }


  def coverage (self) -> Coverage:
    return self.bfdc_coverage ()


  def find_next_rooted_test_target (self) -> Tuple[Input, BFcTarget]:
    cpts = self._all_cpts ()
    tot = sum (cpt.shape[1] for cpt in cpts)
    weight = lambda j: j[:,-1] * j.shape[1] / tot
    mincps = [ (i, int(np.argmin (weight (p))), np.min (weight (p)))
               for i, p in enumerate (cpts) ]
    mincps.sort (key = lambda x: x[2])

    res = None
    fidxbase = self.flayers[0].num_features
    for i, fli, fcp in mincps:
      fint = int(cpts[i][fli][-2])
      cond_ints = cpts[i][fli][:-2].astype (int)
      fidx = fidxbase + i
      fl, flfeature = self.fidx2fli[fidx]
      fl_prev, _ = self.fidx2fli[fidx - flfeature - 1]
      # print (fidx, fl_prev, fl, flfeature, fint, cond_ints)

      # Search closest test that does not fall within target interval
      within_target = self._check_within (fint, verbose = False)
      for i, v in self.all_tests_close_to (fidx, fint):
        if ((fidx, fint, i) in self.ban[fl] or
            within_target (flfeature, self.test_cases[i])):
          continue
        tp1 ('Selecting root test {} at feature-{}-distance {} from {}, layer {}'
             .format (i, flfeature, v, fl.discr.part_edges (flfeature, fint), fl))
        test = self.test_cases[i]
        fct = BFDcTarget (self.N.states[fidx], fint, fl_prev, cond_ints,
                          self._check_within (fint), i)
        self.ban[fl].add ((fidx, fint, i))
        res = test, fct
        break

      if res is not None:
        break

    if res is None:
      raise EarlyTermination ('Unable to find a new candidate input!')
    
    return res



# ---



# def abstract_layerp (li, feats = None, discr = None, layer_indices = []):
#   return (li in discr if discr is not None and isinstance (discr, dict) else
#           li in feats if feats is not None and isinstance (feats, dict) else
#           li in layer_indices)

import builtins

def abstract_layer_features (li, feats = None, discr = None, default = 1):
  if feats is not None:
    if isinstance (feats, (int, float)):
      return feats
    if isinstance (feats, str):
      return builtins.eval(feats, {})(li)
    if isinstance (feats, dict) and li in feats:
      li_feats = feats[li]
      if not isinstance (li_feats, (int, float, str, dict)):
        raise ValueError (
          'feats[{}] should be an int, a string, or a float (got {})'
          .format (li, type (li_feats)))
      return li_feats
    elif isinstance (feats, dict):
      return feats
    raise ValueError (
      'feats should either be a dictonary, an int, a string, or a float (got {})'
      .format (type (feats)))

  # Guess from discr
  if discr is not None:
    if isinstance (discr, dict):
      li_bins = discr[li]
      return (len (li_bins) if isinstance (li_bins, list) else
              li_bins if isinstance (li_bins, int) else
              default)
    elif (isinstance (discr, list) and li < len (discr) and
          isinstance (discr[li], list)):
      return (len (discr[li]))

  return default


def abstract_layer_feature_discretization (l, li, discr = None):
  li_discr = (discr[li] if isinstance (discr, dict) and li in discr else
              discr     if isinstance (discr, dict) else
              discr     if isinstance (discr, int)  else
              discr[li] if isinstance (discr, list) else
              discr(li) if callable (discr) else
              builtins.eval(discr, {})(li) if isinstance (discr, str) else None)
  if li_discr in (None, 'binarizer', 'bin'):
    p1 ('Using binarizer for layer {.name}'.format (l))
    return FeatureBinarizer ()
  else:
    k = (li_discr if isinstance (li_discr, int) else
         li_discr['n_bins'] if isinstance (li_discr, dict) else
         # TODO: per feature discretization strategy?
         2)
    s = (li_discr['strategy'] if (isinstance (li_discr, dict)
                                  and 'strategy' in li_discr)
         else 'quantile')
    extended = (isinstance (li_discr, dict) and 'extended' in li_discr
                and li_discr['extended'])
    if extended is None or not extended:
      p1 ('Using {}-bins discretizer with {} strategy for layer {.name}'
          .format (k, s, l))
      return KBinsFeatureDiscretizer (n_bins = k, encode = 'ordinal',
                                      strategy = s)
    else:
      p1 ('Using extended {}-bins discretizer with {} strategy for layer {.name}'
          .format (k, s, l))
      return KBinsNOutFeatureDiscretizer (n_bins = k, encode = 'ordinal',
                                          strategy = s)



def abstract_layer_setup (l, i, feats = None, discr = None):
  options = abstract_layer_features (i, feats, discr)
  if isinstance (options, dict) and 'decomp' in options:
    decomp = options['decomp']
    options = { **options }
    del options['decomp']
  else:
    decomp = 'pca'
  if (decomp == 'pca' and
      (isinstance (options, (int, float)) or options == 'mle')):
    svd_solver = ('randomized' if isinstance (options, int) else
                  'full' if isinstance (options, float) else 'auto')
    options = { 'n_components': options, 'svd_solver': svd_solver }
  # from sklearn.decomposition import IncrementalPCA
  fext = (make_pipeline (StandardScaler (copy = False# , with_mean = False
                                         ),
                         PCA (**options, copy = False)) if decomp == 'pca' else
          make_pipeline (FastICA (**options)))
  feature_discretization = abstract_layer_feature_discretization (l, i, discr)
  return BFcLayer (layer = l, layer_index = i,
                   transform = fext,
                   discretization = feature_discretization)


# ---


import matplotlib.pyplot as plt

def plot_report_on_feature_extractions (fl, flatacts, fdimred, labels, acc = None):
  from matplotlib import cm
  from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
  from mpl_toolkits.mplot3d import Axes3D

  minlabel, maxlabel = np.min (labels), np.max (labels)
  cmap = plt.get_cmap ('nipy_spectral', maxlabel - minlabel + 1)

  flabel = (lambda fidx:
            ('f{} (variance ratio = {:6.2%})'
             .format (fidx, fl.transform[-1].explained_variance_ratio_[fidx]))
            if hasattr (fl.transform[-1], 'explained_variance_ratio_') else
            ('f'+str(fidx)))

  maxfidx = fdimred.shape[1] - 1
  if maxfidx < 1:
    return                              # for now
  fidx = 0
  while fidx + 1 <= maxfidx:
    fig = plt.figure ()
    if fidx + 1 == maxfidx:
      ax = fig.add_subplot (111)
      # plt.subplot (len (self.flayer_transforms), 1, idx)
      ax.scatter(fdimred[:,0], fdimred[:,1], c = labels,
                 s = 2, marker='o', zorder = 10,
                 cmap = cmap, vmin = minlabel - .5, vmax = maxlabel + .5)
      ax.set_xlabel (flabel (fidx))
      ax.set_ylabel (flabel (fidx+1))
      fidx_done = 2
      incr = 1
    else:
      ax = fig.add_subplot (111, projection = '3d')
      scat = ax.scatter (fdimred[:, fidx], fdimred[:, fidx+1],
                         fdimred[:, fidx+2], c = labels,
                         s = 2, marker = 'o', zorder = 10,
                         cmap = cmap, vmin = minlabel - .5, vmax = maxlabel + .5)
      ax.set_xlabel (flabel (fidx))
      ax.set_ylabel (flabel (fidx+1))
      ax.set_zlabel (flabel (fidx+2))
      fidx_done = 3
      incr = 1 if fidx + 1 == maxfidx - 2 else 2
    fig.suptitle ('Features {} of layer {}'
                  .format (tuple (range (fidx, fidx + fidx_done)), fl))
    cb = fig.colorbar (scat, ticks = range (minlabel, maxlabel + 1), label = 'Classes')
    fidx += incr
  plt.draw ()


# ---


from engine import setup as engine_setup

def setup (
  setup_criterion = None,
  test_object = None,
  feats = { 'n_components': 2, 'svd_solver': 'randomized' },
  # discr = (lambda li: { 'n_bins': 4, 'strategy': 'uniform' } if li in (15,) else
  #          None),
  discr = { 'n_bins': 1, 'extended': True, 'strategy': 'uniform' },
  # discr = (lambda li: { 'n_bins': 3, 'strategy': 'kmeans' }),
  # discr = { 'n_bins': 2, 'extended': True, 'strategy': 'quantile' },
  report_on_feature_extractions = False,
  bn_abstr_train_size = 1000,
  bn_abstr_test_size = 200,
  **kwds):

  if setup_criterion is None:
    raise ValueError ('Missing argument `setup_criterion`!')

  setup_layer = (lambda l, i, **kwds: abstract_layer_setup (l, i, feats, discr))
  cover_layers = get_cover_layers (test_object.dnn, setup_layer,
                                   layer_indices = test_object.layer_indices,
                                   exclude_direct_input_succ = False,
                                   exclude_output_layer = False)
  criterion_args = {
    'bn_abstr_train_size': bn_abstr_train_size,
    'bn_abstr_test_size': bn_abstr_test_size,
    'print_classification_reports': True,
    **({'report_on_feature_extractions': plot_report_on_feature_extractions,
        'close_reports_on_feature_extractions': (lambda _: plt.show ()) }
       if report_on_feature_extractions else {})
  }

  return engine_setup (test_object = test_object,
                       cover_layers = cover_layers,
                       setup_criterion = setup_criterion,
                       criterion_args = criterion_args,
                       **kwds)

# ---

