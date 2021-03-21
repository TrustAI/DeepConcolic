import joblib                   # for saving abstraction pipelines
import plotting
import builtins
from plotting import plt
from utils import *
from utils_io import *
from utils_funcs import *
from utils_stats import AutoRBFKernelPCA
from engine import *
from kde_utils import KDESplit
from l0_encoding import L0EnabledTarget
from functools import reduce
from operator import iconcat
from itertools import product
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, Binarizer
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from sklearn.metrics import log_loss, classification_report
from pomegranate import Node, BayesianNetwork
from pomegranate.distributions import (DiscreteDistribution,
                                       ConditionalProbabilityTable,
                                       JointProbabilityTable)

class AutoRBFKernelPCA_ (AutoRBFKernelPCA):

  @property
  def components_(self):
    """Hack, because we just need the lenght."""
    return self.lambdas_


# ---

class FLayer (CoverableLayer):
  """
  Base class for layers that support feature extraction.
  """

  def __init__(self, transform = None,
               skip: Optional[int] = None,
               focus: Optional[int] = None,
               **kwds):
    super().__init__(**kwds)
    assert skip is None or isinstance (skip, int)
    assert focus is None or isinstance (focus, int)
    self.transform = transform
    self.first = some (skip, 0)
    self.last = focus


  @property
  def focus (self) -> slice:
    return slice (self.first, self.last)


  def get_params (self, deep = True):
    return dict (name = self.layer.name,
                 first = self.first,
                 last = self.last)


  def get_abstraction_info (self):
    return dict (**self.get_info (),
                 transform = self.transform,
                 first = self.first,
                 last = self.last)


  def set_abstraction_info (self, dnn,
                            transform = None,
                            first = None,
                            last = None,
                            **kwds):
    super ().set_info (dnn, **kwds)
    self.transform = transform
    self.first = first
    self.last = last


  @classmethod
  def from_abstraction_info (cls, dnn, **kwds):
    self = cls.__new__(cls)
    self.set_abstraction_info (dnn, **kwds)
    return self


  def feature_of_component (self, component: int) -> Optional[int]:
    if self.first <= component and (self.last is None or component <= self.last):
      return component - self.first
    else:
      return None


  def range_components (self) -> range:
    return range (len (self.transform[-1].components_))


  @property
  def num_features (self) -> int:
    '''
    Number of extracted features for the layer.
    '''
    return len (self.transform[-1].components_[self.focus])


  def range_features (self) -> range:
    '''
    Range over all feature indexes.
    '''
    return range (self.num_features)


  def dimred_activations (self, acts, acc = None, feature_space = True):
    transform = lambda x: \
      self.transform.transform (x)[:,self.focus] if feature_space else \
      self.transform.transform (x)
    y = lazy_activations_transform (acts[self.layer_index], transform)
    acc = np.hstack ((acc, y)) if acc is not None else y
    if y is not acc: del y
    return acc


# ---


def layer_transform_options (li, feats = None, default = 1):

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

  return default


def layer_transform (l, i, options):
  decomp = 'pca'
  skip, focus = None, None
  if isinstance (options, dict):
    options = dict (options)
    if 'decomp' in options:
      decomp = options['decomp']
      del options['decomp']
    if 'skip' in options:
      skip = options['skip']
      del options['skip']
    if 'focus' in options:
      focus = options['focus']
      del options['focus']
  elif (isinstance (options, (int, float)) or options == 'mle') and \
           decomp == 'pca':
    options = dict (n_components = options,
                    svd_solver = ('arpack' if isinstance (options, int) else
                                  'full' if isinstance (options, float) else 'auto'))
  if isinstance (options, dict) and \
         'n_components' in options and isinstance (options['n_components'], int):
    options = dict (options)
    options['n_components'] = min (np.prod (l.output.shape[1:]) - 1,
                                   options['n_components'])
  fext = (make_pipeline (StandardScaler (copy = False),
                         PCA (**options, copy = False)) if decomp == 'pca' else \
          make_pipeline (StandardScaler (copy = False),
                         IncrementalPCA (**options, copy = False)) if decomp == 'ipca' else \
          make_pipeline (StandardScaler (copy = False),
                         AutoRBFKernelPCA_(**options)) if decomp in ('kpca', 'rbf_kpca') else \
          make_pipeline (FastICA (**options)))
  return fext, skip, focus


def flayer_setup (l, i, feats = None, **kwds):
  options = layer_transform_options (i, feats)
  fext, skip, focus = layer_transform (l, i, options)
  return FLayer (layer = l, layer_index = i,
                 transform = fext, skip = skip, focus = focus)


# ---


class FAbstraction:

  def __init__(self,
               flayers: Sequence[FLayer],
               *args,
               feat_extr_train_size = 1,
               print_classification_reports = True,
               score_layer_likelihoods = False,
               report_on_feature_extractions = None,
               close_reports_on_feature_extractions = None,
               outdir: OutputDir = None,
               **kwds):
    super ().__init__(*args, **kwds)
    assert (print_classification_reports is None or isinstance (print_classification_reports, bool))
    assert (report_on_feature_extractions is None or callable (report_on_feature_extractions))
    assert (close_reports_on_feature_extractions is None or callable (close_reports_on_feature_extractions))
    assert (feat_extr_train_size > 0)
    self.feat_extr_train_size = feat_extr_train_size
    self.print_classification_reports = print_classification_reports
    self.score_layer_likelihoods = score_layer_likelihoods
    self.report_on_feature_extractions = report_on_feature_extractions
    self.close_reports_on_feature_extractions = close_reports_on_feature_extractions
    self.flayers = flayers
    self.outdir = outdir or OutputDir ()
    p1 ('Abstracted layers: ' + ', '.join (self.flayer_names))


  def reset (self):
    super ().reset ()
    self.outdir.reset_stamp ()


  @property
  def flayer_names (self) -> Sequence[str]:
    return tuple (fl.layer.name for fl in self.flayers)


  def dump_abstraction (self, pathname = None, outdir = None, base = 'abstraction'):
    if pathname is None:
      outdir = some (outdir, self.outdir)
      pathname = self.outdir.filepath (base, suff = '.pkl')
    np1 (f'Dumping abstraction into `{pathname}\'... ')
    abstraction = [fl.get_abstraction_info () for fl in self.flayers]
    joblib.dump (abstraction, pathname, compress = 1)
    c1 ('done')


  @classmethod
  def from_file (cls,
                 dnn,
                 filename,
                 outdir: OutputDir = None):
    self = cls.__new__(cls)
    np1 (f'Loading abstraction from `{filename}\'... ')
    flayers = joblib.load (filename)
    c1 ('done')
    self.flayers = [ FLayer.from_abstraction_info (dnn, **fl)
                     for fl in flayers ]
    self.outdir = outdir or OutputDir ()
    return self


  # ---


  def dimred_activations (self, acts, fls = None, **kwds):
    acc = None
    for fl in self.flayers if fls is None else fls:
      acc = fl.dimred_activations (acts, acc = acc, **kwds)
    return acc


  @property
  def num_features (self) -> Sequence[int]:
    return [ fl.num_features for fl in self.flayers ]


  # ---


  def initialize (self, acts, true_labels = None, pred_labels = None,
                  fit_with_training_data: bool = False,
                  fl_init_callback = None,
                  **kwds):
    """
    Called through :meth:`stat_based_train_cv_initializers` above.
    """

    # if true_labels is not None and pred_labels is not None:
    #   ok_idxs = (np.asarray (true_labels) == np.asarray (pred_labels))
    #   ok_labels, ko_labels = true_labels[ok_idxs], true_labels[~ok_idxs]
    # else:
    if True:
      ok_idxs = np.full (len (true_labels), True, dtype = bool)
      ok_labels, ko_labels = true_labels, []

    ts0 = np.count_nonzero (ok_idxs)
    # cp1 ('| Given {} correctly classified training sample'.format (*s_(ts0)))
    cp1 ('| Given {} classified training sample'.format (*s_(ts0)))
    fts = None if self.feat_extr_train_size == 1 \
          else (min (ts0, int (self.feat_extr_train_size))
                if self.feat_extr_train_size > 1
                else int (ts0 * self.feat_extr_train_size))
    if fts is not None:
      p1 ('| Using {} training samples for feature extraction'.format (*s_(fts)))

    # First, fit feature extraction parameters:
    for fl in self.flayers:
      p1 ('| Extracting features for layer {}... '.format (fl))
      facts = acts[fl.layer_index]
      x_ok = facts[ok_idxs].reshape (ts0, -1)

      tp1 ('Extracting features...')

      if fts is None:
        y_ok = fl.transform.fit_transform (x_ok, ok_labels)
      else:
        # Copying the inputs here as we pass `copy = False` when
        # constructing the pipeline.
        fl.transform.fit (x_ok[:fts].copy (), y = ok_labels[:fts])
        y_ok = fl.transform.transform (x_ok)
      p1 ('| Extracted {} feature{}'.format (*s_(y_ok.shape[1])))

      # Correct feature range (or should we error out for invalid
      # spec?):
      fl.first = min (fl.first, y_ok.shape[1] - 1)
      if fl.first > 0:
        p1 ('| Skipping {} important feature{}'.format (*s_(fl.first)))
      if fl.last is not None:
        fl.last = max (min (fl.last, y_ok.shape[1]), fl.first + 1)
        p1 ('| Focusing on {} feature{}'.format (*s_(fl.last - fl.first)))

      if fl_init_callback is not None:
        fl_init_callback (fl, x_ok, y_ok, ok_labels, ko_labels, ok_idxs)

      del x_ok, y_ok

    self.summarize_variance ()


  def gather_variance (self):
    if not hasattr (self, 'explained_variance_ratios_'):
      self.explained_variance_ratios_ = \
        { str(fl): (fl.transform[-1].explained_variance_ratio_[fl.focus].tolist (),
                    fl.transform[-1].explained_variance_ratio_.tolist ())
          for fl in self.flayers
          if hasattr (fl.transform[-1], 'explained_variance_ratio_') }


  def summarize_variance (self):
    self.gather_variance ()

    # Report on explained variance
    for fl in self.explained_variance_ratios_:
      variance_ratios = self.explained_variance_ratios_[fl]
      partv, totv = sum (variance_ratios[0]), sum (variance_ratios[1])
      if totv == partv:
        p1 ('| Captured variance ratio for layer {} is {:6.2%}'
            .format (fl, totv))
      else:
        p1 ('| Captured variance ratio for layer {} is {:6.2%}'
            ' (over a total of {:6.2%} extracted)'
            .format (fl, partv, totv))

  def variance_table (self):
    self.gather_variance ()
    infos = { str (fl): self.explained_variance_ratios_[fl][1]
              for fl in self.explained_variance_ratios_ }
    return [ [ fl, i, fi ]
             for fl in infos for i, fi in enumerate (infos[fl]) ]


  def _score_with_training_data (self) -> bool:
    return self.score_layer_likelihoods


  # ---

  def get_params (self, deep = True):
    p = dict (explained_variance_ratios = list (self.explained_variance_ratios_))
    if deep:
      p['layers'] = [ fl.get_params (deep) for fl in self.flayers ]
    return p

  # ---

  def _score (self, acts, true_labels = None, **kwds):
    """
    Basic scores for manual investigations.
    """

    p1 (f'| Given scoring sample of size {len (true_labels)}')

    if (self.score_layer_likelihoods or
        self.report_on_feature_extractions is not None):
      self._score_feature_extractions (acts, true_labels)


  def _score_feature_extractions (self, acts, true_labels = None):
    racc = None
    idx = 1
    self.average_log_likelihoods_ = []
    for fl in self.flayers:

      # if self.score_layer_likelihoods:
      #   tp1 ('| Computing average log-likelihood of test sample for layer {}...'
      #        .format (fl))
      #   flatacts = fl.flatten_map (acts)
      #   self.average_log_likelihoods_.append (fl.transform.score (flatacts))
      #   p1 ('| Average log-likelihood of test sample for layer {} is {}'
      #        .format (fl, self.average_log_likelihood[-1]))
      #   del flatacts

      if self.report_on_feature_extractions is not None:
        fdimred = self.dimred_activations (acts, (fl,))
        racc = self.report_on_feature_extractions (fl, fdimred, true_labels, racc)
        del fdimred

      idx += 1

    if self.close_reports_on_feature_extractions is not None:
      self.close_reports_on_feature_extractions (racc)


# ---


# Whether to enable logs to the console of progress towards target
# intervals:
_log_test_selection_level = 2
_log_progress_level = 3
_log_interval_distance_level = 2
_log_feature_marginals_level = 3
_log_all_feature_marginals_level = 4

# ---


Interval = Tuple[Optional[float], Optional[float]]

def interval_dist (interval: Interval, v: Union[float, np.array]):
  v = np.asarray(v)
  interval = (interval[0] if interval[0] is not None else -np.inf,
              interval[1] if interval[1] is not None else np.inf)
  diff = np.abs ([interval[0] - v, interval[1] - v])
  dist = np.array (np.minimum (diff[0], diff[1]))
  assert (dist >= 0).all ()
  return np.negative (dist, out = dist,
                      where = (interval[0] < v) & (v < interval[1]))

def interval_repr (interval: Interval, prec = 3, float_format = 'g'):
  interval = (interval[0] if interval[0] is not None else -np.inf,
              interval[1] if interval[1] is not None else np.inf)
  return '{lop}{:.{prec}{float_format}}, {:.{prec}{float_format}}{rop}' \
         .format (*interval,
                  prec = prec, float_format = float_format,
                  lop = '(' if interval[0] == -np.inf else '[', rop = ')')


# ---


class FeatureDiscretizer:

  @abstractmethod
  def feature_parts (self, feature: int) -> int:
    raise NotImplementedError

  def has_feature_part (self, feature: int, part: int) -> bool:
    return part >= 0 and part < self.feature_parts (feature)

  @abstractmethod
  def edges (self, feature: int, value: float) -> Interval:
    raise NotImplementedError

  @abstractmethod
  def part_edges (self, feature: int, part: int) -> Interval:
    raise NotImplementedError

  @abstractmethod
  def fit_wrt (self, x, y, feat_extr, **kwds) -> None:
    raise NotImplementedError

  def get_params (self, deep = True) -> dict:
    return dict ()


class FeatureBinarizer (FeatureDiscretizer, Binarizer):

  def feature_parts (self, _feature):
    return 2

  def edges (self, feature: int, value: float) -> Interval:
    thr = self.threshold[0, feature]
    return (thr, np.inf) if value >= thr else (-np.inf, thr)

  def part_edges (self, feature: int, part: int) -> Interval:
    thr = self.threshold[0, feature]
    return (-np.inf, thr) if part == 0 else (thr, np.inf)

  def fit_wrt (self, x, y, feat_extr, **kwds) -> None:
    self.threshold = \
          feat_extr.transform (np.zeros (shape = x[:1].shape)) \
          .reshape (1, -1)
    self.fit (y)

  @property
  def n_bins_ (self):
    return np.full (len (self.threshold), 2, dtype = int)


class KBinsFeatureDiscretizer (FeatureDiscretizer, KBinsDiscretizer):

  def __init__(self,
               kde_dip_space = 'dens',
               kde_dip_prominence_prop = None,
               kde_topline_density_prop = None,
               kde_baseline_density_prop = None,
               kde_bandwidth_prop = None,
               kde_min_width = None,
               kde_plot_spaces = None,
               kde_plot_all_splits = None,
               kde_plot_actual_splits = None,
               kde_plot_dip_markers = None,
               kde_plot_training_samples = 500,
               kde_plot_one_splitter_only = False,
               kde_plot_incorrectly_classified = False,
               n_jobs = None,
               **kwds):
    super().__init__(**kwds)
    KDESplit.validate_space ('kde_dip_space', kde_dip_space)
    kde_plot_spaces = seqx (kde_plot_spaces)
    for s in kde_plot_spaces: KDESplit.validate_space ('kde_plot_spaces', s)
    assert (kde_plot_training_samples >= 0)
    assert (n_jobs is None or n_jobs != 0)
    self.kde_plot_training_samples = int (kde_plot_training_samples)
    self.kde_ = []
    self.kde_split_args = dict (dip_space = kde_dip_space,
                                dip_prominence_prop = kde_dip_prominence_prop,
                                topline_density_prop = kde_topline_density_prop,
                                baseline_density_prop = kde_baseline_density_prop,
                                bandwidth_prop = kde_bandwidth_prop,
                                min_width = kde_min_width,
                                plot_splits = kde_plot_all_splits,
                                plot_dip_markers = kde_plot_dip_markers,
                                plot_spaces = kde_plot_spaces,
                                n_jobs = n_jobs)
    self.kde_plot_actual_splits = some (kde_plot_actual_splits, True)
    self.kde_plot_one_splitter_only = some (kde_plot_one_splitter_only, False)
    self.kde_plot_incorrectly_classified = some (kde_plot_incorrectly_classified, False)

  def feature_parts (self, feature) -> int:
    return self.n_bins_[feature]

  def edges (self, feature: int, value: float) -> Interval:
    edges = np.concatenate((np.array([-np.inf]),
                            self.bin_edges_[feature][1:-1],
                            np.array([np.inf])))
    part = np.searchsorted (edges, value, side = 'right')
    return edges[part-1], edges[part]

  def part_edges (self, feature: int, part: int) -> Interval:
    """
    Raises IndexError in case part is out of discretized space.
    """
    edges = self.bin_edges_[feature]
    return ((-np.inf if part   == 0           else edges[part  ],
             np.inf  if part+2 == len (edges) else edges[part+1]))

  def _kde_fit (self, x, y, feat_extr, true_labels = None, pred_labels = None, **kwds):
    # Use a Kernel-Density-based fit with bandwidth optimization, for
    # each feature.

    n_features = y.shape[1]
    n_bins = np.zeros (n_features, dtype = int)
    bin_edges = np.zeros (n_features, dtype = object)
    n_tries = 5
    splitters = [ [ KDESplit (**self.kde_split_args)
                    for _ in range (n_tries) ]
                  for _ in range (n_features) ]

    for fi in range (n_features):
      tp1 (f'KDE: Discretizing feature {fi}...')
      yy = y[:,fi]
      rs = ShuffleSplit (n_splits = n_tries,
                         train_size = 1 / max (2, n_tries - 1))
      bandwidth = None
      for splitter, (yy_index, _) in zip (splitters[fi], rs.split (yy)):
        splitter.fit_split (yy[yy_index], bandwidth = bandwidth)
        bandwidth = splitter.bandwidth_ if bandwidth is None else bandwidth
      from sklearn.cluster import MeanShift
      splits = [ s.splits_ for s in splitters[fi] ]
      splits = np.asarray (reduce (iconcat, splits, []))
      clustering = MeanShift (bandwidth = bandwidth * 2,
                              cluster_all = False)\
                  .fit (splits.reshape(-1, 1))
      splits = clustering.cluster_centers_.T[0]
      splits.sort ()
      bin_edges[fi] = splits
      n_bins[fi] = len(bin_edges[fi]) - 1

    self.n_bins_ = n_bins
    self.bin_edges_ = bin_edges
    self.splitters_ = splitters


  def _kde_plot (self, _x, y, _feat_extr,
                 true_labels = None, pred_labels = None, layer = None,
                 outdir: OutputDir = None,
                 y2plot = None, y2plot_labels = None,
                 **kwds):

    if self.kde_split_args['plot_spaces'] is None or \
       self.kde_split_args['plot_spaces'] == []:
      return

    if not plt:
      warnings.warn ('Unable to import `matplotlib`: skipping KDE plot')
      return

    tp1 (f'KDE: Plotting discretized features...')

    KDESplit.setup_plot_style ()
    plot_incorrects = self.kde_plot_incorrectly_classified and \
                      y2plot is not None and len (y2plot) > 0

    cmap = None
    if true_labels is not None:
      minlabel, maxlabel = np.min (true_labels), np.max (true_labels)
      cmap = plt.get_cmap ('nipy_spectral', maxlabel - minlabel + 1)

    for plot_space in self.kde_split_args['plot_spaces']:
      fig, ax = plotting.subplots (len (self.splitters_))
      fig.subplots_adjust (left = 0.04, right = 0.99, hspace = 0.1,
                           bottom = 0.03, top = 0.99)
      for fi, splitters in enumerate (self.splitters_):
        axi = ax[fi] if len (self.splitters_) > 1 else ax
        extrema = None
        for splitter in splitters:
          if plot_incorrects:
            extrema = splitter.plot_kde (axi, plot_space, y2plot,
                                         lineprops = dict (color = 'red',
                                                           linewidth = 2,
                                                           linestyle = 'dotted'),
                                         logspace_min = -2,
                                         logspace_max = .5,
                                         logspace_steps = 10,
                                         extrema = extrema)
          extrema = splitter.plot_splits (axi, plot_space,
                                          extrema = extrema)
          if self.kde_plot_one_splitter_only: break
        if extrema is not None and self.kde_plot_training_samples > 0:
          # customizable sub-sampling to keep graphs lightweight
          subyy = min (len (y), self.kde_plot_training_samples)
          yy = y[:subyy, fi]
          axi.scatter (yy,
                       - 0.06 * extrema['ymax'] * np.random.random (yy.shape[0]),
                       marker = 'o', c = true_labels[:subyy], cmap = cmap)
          if plot_incorrects:
            subyy = min (len (y2plot), self.kde_plot_training_samples)
            yy = y2plot[:subyy, fi]
            axi.scatter (yy,
                         0.01 + 0.06 * extrema['ymax'] * np.random.random (yy.shape[0]),
                         marker = '+', c = y2plot_labels[:subyy], cmap = cmap)
            axi.axhline (c = 'grey', lw = .2)
        if extrema is not None and self.kde_plot_actual_splits:
          axi.vlines (x = self.bin_edges_[fi],
                      ymin = min (0., extrema['ymin']),
                      ymax = max (0., extrema['ymax']),
                      linestyles = 'dashed',
                      color = 'r')
        axi.annotate ((r'$\mathbb{F}_{' + \
                       plotting.texttt (str (layer)) + ', ' + str (fi) + '}$'),
                      xy = (1, 1), xycoords = axi.transAxes,
                      xytext = (-3, -3), textcoords = 'offset points',
                      horizontalalignment = 'right',
                      verticalalignment = 'top',
                      bbox = dict (boxstyle = 'square,pad=0.1', ec='black',
                                   fc='white', lw=.8))
      plotting.show (fig = fig, outdir = outdir,
                     basefilename = str (layer) + '-' + str (plot_space))


  def fit_wrt (self, x, y, feat_extr, **kwds) -> None:
    if self.strategy == 'kde':
      self._kde_fit (x, y, feat_extr, **kwds)
      self._kde_plot (x, y, feat_extr, **kwds)
    else:
      super().fit (y)

  def get_params (self, deep = True) -> dict:
    p = super().get_params (deep)
    p['extended'] = False
    if self.strategy == 'kde':
      p['kde'] = self.kde_
      p['kde_split_args'] = self.kde_split_args
    return p



class KBinsNOutFeatureDiscretizer (KBinsFeatureDiscretizer):

  def __init__(self, n_bins = 2, **kwds):
    super().__init__(n_bins = None if n_bins is None else max(2, n_bins), **kwds)
    self.one_ = n_bins == 1

  def fit_wrt (self, *args, **kwds):
    super ().fit_wrt (*args, **kwds)
    self.bin_edges_ = [ np.concatenate (([-np.inf],
                                         np.delete (x, 1, 0) if self.one_ else x,
                                         [np.inf]))
                        for x in self.bin_edges_ ]
    self.n_bins_ = self.n_bins_ + (1 if self.one_ else 2)

  def get_params (self, deep = True) -> dict:
    p = super().get_params (deep)
    p['extended'] = True
    return p


# ---


class BFcLayer (FLayer):
  """
  Base class for layers to be covered by BN-based criteria.
  """

  def __init__(self,
               discretization: FeatureDiscretizer = None,
               **kwds):
    super().__init__(**kwds)
    assert isinstance (discretization, FeatureDiscretizer)
    self.discr = discretization


  def get_params (self, deep = True):
    p = super ().get_params (deep = deep)
    p['discretized_hidden_features'] = {
      f: [ interval_repr (self.discr.part_edges (f, i))
           for i in range (self.discr.feature_parts (f)) ]
      for f in self.range_features ()
    }
    return p


  def get_abstraction_info (self):
    return dict (**super ().get_abstraction_info (),
                 discr = self.discr)


  def set_abstraction_info (self, dnn,
                            discr = None,
                            **kwds):
    super ().set_abstraction_info (dnn, **kwds)
    self.discr = discr


  @classmethod
  def from_abstraction_info (cls, dnn, **kwds):
    self = cls.__new__(cls)
    self.set_abstraction_info (dnn, **kwds)
    return self


  @property
  def num_feature_parts (self) -> Sequence[int]:
    return [ self.discr.feature_parts (feature)
             for feature in self.range_features () ]


  @property
  def intervals (self):
    return [ [ self.discr.part_edges (feature, i)
               for i in range (num_part) ]
             for feature, num_part in enumerate (self.num_feature_parts) ]


  def dimred_n_discretize_activations (self, acts, acc = None):
    transform = lambda x: \
      self.discr.transform (self.transform.transform (x)[:,self.focus])\
                .astype (int)
    y = lazy_activations_transform (acts[self.layer_index], transform)
    acc = np.hstack ((acc, y)) if acc is not None else y
    if y is not acc: del y
    return acc


# ---



def bayes_node_name(fl, idx):
  return '.'.join ((str(fl), *((str(i) for i in idx))))



class DiscretizedFeatureNode (Node):

  def __init__(self, flayer: BFcLayer, feature: int, *args, **kwds):
    super().__init__ (*args, name = bayes_node_name (flayer, (feature,)), **kwds)
    self.flayer = flayer
    self.feature = feature


  def discretized_range(self) -> range:
    return range (self.flayer.discr.feature_parts (self.feature))


  def interval(self, feature_interval: int) -> Interval:
    return self.flayer.discr.part_edges (self.feature, feature_interval)



class DiscretizedInputFeatureNode (DiscretizedFeatureNode):

  def __init__(self, flayer, feature, **kwds):
    n = flayer.discr.feature_parts (feature)
    super().__init__(flayer, feature,
                     DiscreteDistribution ({ fbin: 0.0 for fbin in range (n) }),
                     **kwds)


class DiscretizedHiddenFeatureNode (DiscretizedFeatureNode):

  def __init__(self, flayer, feature, prev_nodes, **kwds):
    prev_nodes = list (pn for pn in prev_nodes
                       if len (pn.discretized_range ()) > 1)
    prev_distrs = [ n.distribution for n in prev_nodes ]
    prev_fparts = list ([ bin for bin in pn.discretized_range () ]
                        for pn in prev_nodes)
    n = flayer.discr.feature_parts (feature)
    if prev_fparts == [] or n == 1:
      del prev_distrs, prev_fparts
      self.prev_nodes_ = []
      super().__init__(flayer, feature,
                       DiscreteDistribution ({ fbin: 0.0 for fbin in range (n) }),
                       **kwds)
    else:
      condprobtbl = [ list (p) + [0.0] for p in product (*prev_fparts, range (n)) ]
      del prev_fparts
      self.prev_nodes_ = prev_nodes
      super().__init__(flayer, feature,
                       ConditionalProbabilityTable (condprobtbl, prev_distrs),
                       **kwds)
      del condprobtbl


# ---

class BNAbstraction (FAbstraction):

  def __init__(self,
               flayers: Sequence[BFcLayer],
               *args,
               bn_abstr_n_jobs = None,
               assess_discretized_feature_probas = False,
               dump_abstraction = True,
               **kwds):
    super ().__init__(flayers, *args, **kwds)
    self.bn_abstr_n_jobs = bn_abstr_n_jobs
    self.assess_discretized_feature_probas = assess_discretized_feature_probas
    self.dump_abstraction_ = dump_abstraction
    self.fit_dataset_size = 0   # as modeled in BN


  def reset (self):
    super ().reset ()
    self.reset_bn ()


  def reset_bn (self):
    # Next call to fit_activations will reset the BN's probabilities
    self.fit_dataset_size = 0
    self.N_marginals = None


  def dump_bn (self, base, descr):
    fn = self.outdir.filepath (base, suff = '.yml')
    header = '\n# '.join ((f'# BN fit with {descr}',
                           '',
                           'Reload with:',
                           f"  with open ('{base}.yml', mode = 'r') as f:",
                           "    N = pomegranate.BayesianNetwork.from_yaml (f.read ())",
                           '\n'))
    extra = dict (dataset_size = self.fit_dataset_size,
                  params = self.get_params (True))
    np1 (f'Outputting BN fit with {descr} in `{fn}\'... ')
    write_in_file (fn, header, self.N.to_yaml (), yaml.dump (extra))
    c1 ('done')


  def dump_abstraction (self, pathname = None, outdir = None, base = 'bn-abstraction'):
    super ().dump_abstraction (pathname = pathname,
                               outdir = outdir,
                               base = base)


  @classmethod
  def from_file (cls,
                 dnn,
                 filename,
                 outdir: OutputDir = None,
                 bn_abstr_n_jobs = None,
                 log = True):
    self = cls.__new__(cls)
    np1 (f'Loading abstraction from `{filename}\'... ')
    flayers = joblib.load (filename)
    c1 ('done')
    self.flayers = [ BFcLayer.from_abstraction_info (dnn, **fl)
                     for fl in flayers ]
    self.bn_abstr_n_jobs = bn_abstr_n_jobs
    self.outdir = outdir or OutputDir ()
    self.fit_dataset_size = 0
    self.N = self._create_bayesian_network (log = log)
    self.N_marginals = None
    return self


  # ---


  def dimred_n_discretize_activations (self, acts, fls = None):
    acc = None
    for fl in self.flayers if fls is None else fls:
      acc = fl.dimred_n_discretize_activations (acts, acc = acc)
    return acc


  @property
  def num_feature_parts (self) -> Sequence[Sequence[int]]:
    return [ fl.num_feature_parts for fl in self.flayers ]


  # ---


  def fit_activations (self, acts):
    facts = self.dimred_n_discretize_activations (acts)
    nbase = self.fit_dataset_size
    self.fit_dataset_size += len (facts)
    self.N.fit (facts,
                inertia = nbase / self.fit_dataset_size,
                n_jobs = int (some (self.bn_abstr_n_jobs, 1)))
    self.N_marginals = None
    del facts


  def activations_probas (self, acts):
    facts = self.dimred_n_discretize_activations (acts)
    log_probs = self.N.log_probability \
      (facts, n_jobs = int (some (self.bn_abstr_n_jobs, 1)))
    del facts
    return np.exp (log_probs)


  # ---


  def _marginals (self):
    if self.N_marginals is None:
      tp1 ('Computing BN marginals... ')
      self.N_marginals = self.N.marginal ()
      tp1 ('Computing BN marginals... done')
    return self.N_marginals


  def _probas (self, p):
    return p.parameters[0] if not isinstance (p.parameters[0], dict) else \
           [ p.parameters[0][i] for i in p.parameters[0] ]


  def _all_marginals (self) -> range:
    return (self._probas (p) for p in self._marginals ())


  def _all_cpts (self):
    return (self._probas (j.distribution)
            for j in self.N.states
            if isinstance (j.distribution, ConditionalProbabilityTable))


  def _all_cpts_n_marginals (self) -> range:
    return ((self._probas (j.distribution), self._probas (m))
            for j, m in zip (self.N.states, self._marginals ())
            if isinstance (j.distribution, ConditionalProbabilityTable))


  def bfc_coverage (self, epsilon = 1e-8) -> Coverage:
    """
    Computes the BFCov metric as per the underlying Bayesian Network
    abstraction.
    """
    assert (self.fit_dataset_size > 0)
    props = sum (np.count_nonzero (np.array(p) >= epsilon) / len (p)
                 for p in self._all_marginals ())
    return Coverage (covered = props, total = self.N.node_count ())


  def bfdc_coverage (self, epsilon = 1e-8, multiply_with_bfc = False) -> Coverage:
    """
    Computes the BFdCov metric as per the underlying Bayesian Network
    abstraction.  The returned coverage is multiplied with BFCov if
    `multiply_with_bfc` holds.
    """
    assert (self.fit_dataset_size > 0)
    # Count 0s (or < epsilons) in all prob. mass functions in the BN
    # abstraction, subject to associated marginal probabilities being
    # > epsilon as well:
    def count_nonepsilons (acc, x):
      (noneps_props, num_cpts), (cpt, marginal) = acc, x
      # p's last column (-1) holds conditional probabilities, whereas
      # the last but one (-2) holds the feature interval index.
      noneps_props += \
        sum (p[-1] >= epsilon if marginal[p[-2]] >= epsilon else True \
             for p in cpt) \
        / len (cpt)
      return (noneps_props, num_cpts + 1)
    props, num_cpts = reduce (count_nonepsilons, self._all_cpts_n_marginals (),
                              (0, 0))
    bfdc = Coverage (covered = props, total = num_cpts) if num_cpts > 0 else \
           Coverage (covered = 1)
    return bfdc * self.bfc_coverage ().as_prop if multiply_with_bfc else bfdc


  # ---


  def initialize (self, acts, true_labels = None, pred_labels = None,
                  fit_with_training_data: bool = False,
                  **kwds):
    """
    Called through :meth:`stat_based_train_cv_initializers` above.
    """

    def discr_callback (fl, x_ok, y_ok, ok_labels, ko_labels, ok_idxs):
      # Fit discretizer parameters:
      p1 ('| Discretizing features for layer {}... '.format (fl))
      ts0 = np.count_nonzero (ok_idxs)
      fit_wrt_args = {}
      x_ko, y_ko = [], []
      if len (ko_labels) > 0:
        x_ko = facts[~ok_idxs].reshape (len (ok_idxs) - ts0, -1)
        y_ko = fl.transform.transform (x_ko)
        fit_wrt_args = dict (y2plot = y_ko[:,fl.focus],
                             y2plot_labels = ko_labels)

      tp1 ('Discretizing features...')

      fl.discr.fit_wrt (x_ok[:,fl.focus],
                        y_ok[:,fl.focus],
                        fl.transform,
                        layer = fl,
                        **kwds,
                        true_labels = ok_labels,
                        outdir = self.outdir,
                        **fit_wrt_args)

      for fi in range (fl.num_features):
        p1 ('| Discretization of feature {} involves {} interval{}'
            .format (fi, *s_(fl.discr.feature_parts (fi))))
      p1 ('| Discretized {} feature{}'.format (*s_(fl.num_features)))
      del x_ko, y_ko

    super ().initialize (acts,
                         true_labels = true_labels,
                         pred_labels = pred_labels,
                         fit_with_training_data = fit_with_training_data,
                         fl_init_callback = discr_callback,
                         **kwds)

    # Second, contruct the Bayesian Network
    self.N = self._create_bayesian_network ()
    self.N_marginals = None

    # Dump the abstraction if needed
    if self.dump_abstraction_:
      self.dump_abstraction ()

    # Last, fit the Bayesian Network with given training activations
    # for now, for the purpose of preliminary assessments; the BN will
    # be re-initialized upon the first call to `add_new_test_cases`:
    if fit_with_training_data or self._score_with_training_data ():
      np1 ('| Fitting BN with training dataset... ')
      # XXX: just use all provided training data for now:
      ok_idxs = np.full (len (true_labels), True, dtype = bool)
      # TODO: means to customize this:
      batch_size = 1000
      for i in range (0, len (true_labels), batch_size):
        imax = min (i + batch_size, len (true_labels))
        imsk = ok_idxs[i:imax]
        self.fit_activations ({ layer: acts[layer][i:imax][imsk]
                                for layer in acts })
      c1 ('done')


  def _score_with_training_data (self) -> bool:
    return super ()._score_with_training_data () \
      or self.assess_discretized_feature_probas


  # ---


  def _create_bayesian_network (self, log = True):
    """
    Actual BN instantiation.
    """

    import gc
    nc = sum (f.num_features for f in self.flayers)
    max_ec = sum (f.num_features * g.num_features
                  for f, g in zip (self.flayers[:-1], self.flayers[1:]))

    tp1 ('| Creating Bayesian Network of {} nodes and a maximum of {} edges...'
         .format (nc, max_ec))
    N = BayesianNetwork (name = 'BN Abstraction')

    gc.collect ()
    prev_nodes = None
    for fl in self.flayers:
      nodes = [ (DiscretizedHiddenFeatureNode (fl, feature, prev_nodes)
                 if prev_nodes is not None else
                 DiscretizedInputFeatureNode (fl, feature))
                for feature in range (fl.num_features) ]
      N.add_nodes (*(n for n in nodes))

      if prev_nodes is not None:
        for n in nodes:
          for pn in n.prev_nodes_:
            N.add_edge (pn, n)
      tp1 ('| Creating Bayesian Network: {}/{} nodes, {}/{} edges done...'
           .format (N.node_count (), nc, N.edge_count (), max_ec))

      del prev_nodes
      gc.collect ()
      prev_nodes = nodes

    del prev_nodes
    gc.collect ()
    ec = N.edge_count ()
    tp1 ('| Creating Bayesian Network of {} nodes and {} edges: baking...'
         .format (nc, ec))
    N.bake ()
    if log:
      p1 ('| Created Bayesian Network of {} nodes and {} edges.'
          .format (nc, ec))
    return N

  # ---

  def get_params (self, deep = True):
    p = super ().get_params (deep = deep)
    p['node_count'] = self.N.node_count ()
    p['edge_count'] = self.N.edge_count (),
    return p

  # ---

  def _score (self, acts, true_labels = None, **kwds):
    super ()._score (acts, true_labels = true_labels, **kwds)

    if self.assess_discretized_feature_probas:
      truth = self.dimred_n_discretize_activations (acts)
      self._score_discretized_feature_probas (truth)
      del truth


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
      for feature in range (fl.num_features):
        flabels = list (range (fl.discr.feature_parts (feature)))
        feature_idx = first_feature_idx + feature
        ftruth = truth[..., feature_idx]

        tp1 ('| Computing predictions for feature {} of {}...'.format (feature, fl))
        fprobas = features_probas (feature_idx, 1).flatten ()

        tp1 ('| Computing log loss for feature {} of {}...'.format (feature, fl))
        fpredict_probs = self._all_prediction_probas (fprobas)
        loss = log_loss (ftruth, fpredict_probs, labels = flabels)
        floss.append (loss)
        p1 ('| Log loss for feature {} of {} is {}'.format (feature, fl, loss))

        if self.print_classification_reports:
          p1 ('| Classification report for feature {} of {}:'.format (feature, fl))
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
    def estimate_feature_probas (feature, nbfeats):
      ftest[..., feature : feature + nbfeats] = np.nan
      probas = np.array (self.N.predict_proba (ftest, n_jobs = self.bn_abstr_n_jobs))
      ftest[..., feature : feature + nbfeats] = truth[..., feature : feature + nbfeats]
      lprobas = probas[..., feature : feature + nbfeats]
      del probas
      return lprobas
    return (lambda feature, nbfeats: estimate_feature_probas (feature, nbfeats))


  def _prediction_probas (self, p):
    return [ p.parameters[0][i] for i in range (len (p.parameters[0])) ]


  def _all_prediction_probas (self, fprobas):
    return [ self._prediction_probas (p) for p in fprobas ]



# ----


class _BaseBFcCriterion (Criterion):
  '''
  ...

  - `feat_extr_train_size`: gives the proportion of training data from
    `bn_abstr_train_size` to use for feature extraction if <= 1;
    `min(feat_extr_train_size, bn_abstr_train_size)` will be used
    otherwise.

  ...
  '''

  def __init__(self,
               clayers: Sequence[CoverableLayer],
               *args,
               epsilon = None,
               shallow_first = None,
               bn_abstr: BNAbstraction = None,
               bn_abstr_train_size = None,
               bn_abstr_test_size = None,
               bn_abstr_args = dict (),
               dump_bn_with_trained_dataset_distribution = False,
               dump_bn_with_final_dataset_distribution = False,
               **kwds):
    flayers = list (filter (lambda l: isinstance (l, BFcLayer), clayers))
    super ().__init__(clayers, *args, **kwds)
    assert list (filter (lambda l: isinstance (l, BoolMappedCoverableLayer),
                         self.cover_layers)) == []
    if bn_abstr is not None:
      self.BN = bn_abstr
      self.bn_abstr_params = None
      self._finalize_initialization ()
    else:
      self.BN = BNAbstraction (flayers, **bn_abstr_args)
      self.bn_abstr_params = dict (train_size = bn_abstr_train_size or 0.5,
                                   test_size = bn_abstr_test_size or 0.5)
    self.outdir = self.BN.outdir
    self.epsilon = epsilon or 1e-8
    self.shallow_first = some (shallow_first, True)
    self.dump_bn_with_trained_dataset_distribution = dump_bn_with_trained_dataset_distribution
    self.dump_bn_with_final_dataset_distribution = dump_bn_with_final_dataset_distribution
    self.base_dimreds = None
    self._reset_progress ()
    self._log_feature_marginals = None


  @property
  def flayers (self) -> Sequence[BFcLayer]:
    return self.BN.flayers


  @property
  def N (self) -> BayesianNetwork:
    return self.BN.N


  def finalize_setup (self):
    if isinstance (self.analyzer, LayerLocalAnalyzer):
      self.analyzer.finalize_setup (self.flayers)


  def terminate (self):
    if self.dump_bn_with_final_dataset_distribution:
      self.BN.dump_bn ('bn4tests', 'generated dataset')


  def reset (self):
    super ().reset ()
    self.BN.reset_bn ()
    self._reset_progress ()


  def _reset_progress (self):
    self.progress_file = self.outdir.stamped_filepath ( \
      str (self) + '_' + str (self.metric) + '_progress', suff = '.csv')
    write_in_file (self.progress_file,
                   '# ',
                   ' '.join (('old_test', 'feature',
                              'interval_left', 'interval_right',
                              'old_v', 'new_v',
                              'old_dist', 'new_dist')),
                   '\n')


  def _log_discr_level (self, log_prefix = '| '):
    if self.verbose >= _log_all_feature_marginals_level:
      for feature, bn_node in enumerate (self.N.states):
        try:
          p1 ('{} feature-{} distribution: {}'
              .format (log_prefix, feature,
                       self.BN._probas (bn_node.distribution.marginal ())))
        except KeyError: pass
    elif self.verbose >= _log_feature_marginals_level and \
             self._log_feature_marginals is not None:
      bn_node = self.N.states[self._log_feature_marginals]
      try:
        p1 ('{} feature-{} distribution: {}'
            .format (self._log_feature_marginals,
                     self.BN._probas (bn_node.distribution.marginal ())))
      except KeyError: pass


  def fit_activations (self, acts):
    self._log_discr_level ('| Old')
    self.BN.fit_activations (acts)
    self._log_discr_level ('| New')
    self._log_feature_marginals = None


  def register_new_activations (self, acts) -> None:
    self.fit_activations (acts)

    # Append feature values for new tests
    new_dimreds = self.BN.dimred_activations (acts)
    self.base_dimreds = (np.vstack ((self.base_dimreds, new_dimreds))
                         if self.base_dimreds is not None else new_dimreds)
    if self.base_dimreds is not new_dimreds: del new_dimreds


  def pop_test (self):
    super().pop_test ()
    # Just remove any reference to the previously registered test
    # case: this only impacts the search for new test targets.
    self.base_dimreds = np.delete (self.base_dimreds, -1, axis = 0)


  def stat_based_train_cv_initializers (self):
    """
    Initializes the criterion based on traininig data.

    Directly uses argument ``bn_abstr_train_size`` and
    ``bn_abstr_test_size`` arguments given to the constructor, and
    optionally computes some scores (based on flags given to the
    constructor as well).
    """
    if self.bn_abstr_params is None:
      return []
    bn_abstr = ({ 'test': self._score }
                if (self.BN._score_with_training_data () or
                    self.BN.report_on_feature_extractions is not None) else {})
    return [{
      **self.bn_abstr_params,
      'name': 'Bayesian Network abstraction',
      'layer_indexes': set ([fl.layer_index for fl in self.flayers]),
      'train': self._create_abstraction,
      **bn_abstr,
    }]


  def _create_abstraction (self, acts,
                           true_labels = None,
                           pred_labels = None,
                           **kwds):
    """
    Called through :meth:`stat_based_train_cv_initializers` above.
    """
    fit_with_training_data = self.dump_bn_with_trained_dataset_distribution
    self.BN.initialize (acts,
                        true_labels = true_labels,
                        pred_labels = pred_labels,
                        fit_with_training_data = fit_with_training_data,
                        **kwds)

    if self.dump_bn_with_trained_dataset_distribution:
      self.BN.dump_bn ('bn4trained', 'training dataset')
      if not self.BN._score_with_training_data ():
        self.BN.reset_bn ()

    self._finalize_initialization ()


  def _finalize_initialization (self):
    # Record a mapping from absolute feature indices to each
    # corresponding layer and latent feature:
    self.fidx2fli = {}
    feature = 0
    for fl in self.flayers:
      for i in range (fl.num_features):
        self.fidx2fli[feature + i] = (fl, i)
      feature += fl.num_features


  def _score (self, acts, **kwds):
    self.BN._score (acts, **kwds)
    self.BN.reset_bn ()
    self.base_dimreds = None


  def _all_tests_n_dists_to (self, feature: int, feature_interval: int):
    feature_node = self.N.states[feature]
    fl, flfeature = feature_node.flayer, feature_node.feature
    if not fl.discr.has_feature_part (flfeature, feature_interval):
      return []
    target_interval = fl.discr.part_edges (flfeature, feature_interval)
    all = interval_dist (target_interval, self.base_dimreds[..., feature])
    return enumerate (all)


  # def _check_within (self, feature: int, expected_interval: int, verbose = True):
  #   def aux (t: Input) -> bool:
  #     acts = self.analyzer.eval (t, allow_input_layer = False)
  #     facts = self.dimred_n_discretize_activations (acts)
  #     res = facts[0][feature] == expected_interval
  #     if verbose and not res:
  #       dimred = self.dimred_activations (acts)
  #       dimreds = dimred[..., feature : feature + 1].flatten ()
  #       tp1 ('| Got interval {}, expected {} (fval {})'
  #            .format(facts[0][feature], expected_interval, dimreds))
  #     return res
  #   return aux


  # ----


  def _measure_progress_towards_interval (self,
                                          feature: int,
                                          interval: Interval,
                                          ref_test_index: int):
    def aux (new : Input) -> bool:
      old = self.test_cases[ref_test_index]
      acts = self.analyzer.eval_batch (np.array ([old, new]),
                                       allow_input_layer = False)
      dimreds = self.BN.dimred_activations (acts)
      old_v = dimreds[0][..., feature]
      new_v = dimreds[1][..., feature]
      old_dist = interval_dist (interval, old_v)
      new_dist = interval_dist (interval, new_v)
      append_in_file (self.progress_file,
                      ' '.join (str (i) for i in (ref_test_index,
                                                  feature,
                                                  *interval,
                                                  old_v, new_v,
                                                  old_dist, new_dist)),
                      '\n')
      self._log_feature_marginals = feature
      return old_dist - new_dist, new_dist
    return aux


# ---


class BNcTarget (TestTarget):

  def measure_progress(self, t: Input) -> float:
    """
    Measures how a new input `t` improves towards fulfilling the
    target.  A negative returned value indicates that no progress is
    being achieved by the given input.
    """
    raise NotImplementedError


# ---


class BFcTarget (NamedTuple, BNcTarget, L0EnabledTarget):
  fnode: DiscretizedFeatureNode
  feature_part: int
  progress: Callable[[Input], float]
  root_test_idx: int
  verbose: int

  def __repr__(self) -> str:
    interval = self.fnode.flayer.discr.part_edges (self.fnode.feature,
                                                   self.feature_part)
    return ('interval {} of feature {} in layer {} (from test {})'
            .format(interval_repr (interval),
                    self.fnode.feature, self.fnode.flayer, self.root_test_idx))


  def log_repr(self) -> str:
    return ('#layer: {} #feat: {} #part: {}'
            .format(self.fnode.flayer.layer_index,
                    self.fnode.feature, self.feature_part))


  def __eq__ (self, t) -> bool:
    '''Basic equality test'''
    return type (self) == type (t) and \
           self.fnode == t.fnode and \
           self.feature_part == t.feature_part


  def __hash__(self) -> int:
    return hash ((self.fnode, self.feature_part))


  def cover(self, acts) -> None:
    # Do nothing for now; ideally: update some probabilities
    # somewhere.
    pass


  def measure_progress(self, t: Input) -> float:
    progress, new_dist = self.progress (t)
    if self.verbose >= _log_progress_level:
      p1 ('| Progress towards {}: {}'.format (self, progress))
    if self.verbose >= _log_interval_distance_level:
      p1 ('| Distance to target interval: {} ({})'
          .format (new_dist, 'hit' if new_dist <= 0.0 else 'miss'))
    return progress


  def eval_inputs (self, inputs: Sequence[Input], eval_batch = None) \
      -> Sequence[float]:
    '''Inputs evaluation.

    Measures how a new input `t` improves towards fulfilling the
    target.  A negative returned value indicates that no progress is
    being achieved by the given input.

    '''
    acts = eval_batch (inputs, layer_indexes = (self.fnode.flayer.layer_index,))
    dimr = self.fnode.flayer.dimred_activations (acts)[..., self.fnode.feature]
    return - interval_dist (self.fnode.interval (self.feature_part), dimr)


  def valid_inputs (self, evals: Sequence[float]) -> Sequence[bool]:
    return (evals >= 0)


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
    super().__init__(clayers, *args, analyzer = analyzer, **kwds)
    self.ban = { fl: set () for fl in self.flayers }


  def __repr__(self):
    return "BFC"


  def reset (self):
    super().reset ()
    self.ban = { fl: set () for fl in self.flayers }


  def coverage (self) -> Coverage:
    return self.BN.bfc_coverage (epsilon = self.epsilon)


  def find_next_rooted_test_target (self) -> Tuple[Input, BFcTarget]:

    # Gather non-epsilon marginal probabilities:
    epsilon_entries = ((fli, ints)
                       for fli, prob in enumerate (self.BN._all_marginals ())
                       for ints in np.where (np.asarray(prob) < self.epsilon))

    res, best_dist = None, np.inf
    for feature, epsilon_intervals in epsilon_entries:
      feature_node = self.N.states[feature]
      fl, flfeature = feature_node.flayer, feature_node.feature
      for feature_interval in epsilon_intervals:

        # Search closest test -that does not fall within target interval (?)-:
        # within_target = self._check_within (feature, feature_interval, verbose = False)
        for ti, dist in self._all_tests_n_dists_to (feature, feature_interval):
          if (dist < 0 or
              (feature, feature_interval, ti) in self.ban[fl]# or
              # within_target (flfeature, self.test_cases[i])
              ):
            continue
          if dist < best_dist:
            best_dist = dist
            res = feature, feature_interval, ti

      if res is not None and self.shallow_first:
        break

    if res is None:
      raise EarlyTermination ('Unable to find a new candidate input!')

    feature, feature_interval, ti = res
    feature_node = self.N.states[feature]
    fl, flfeature = feature_node.flayer, feature_node.feature
    interval = feature_node.interval (feature_interval)
    if self.verbose >= _log_test_selection_level:
      p1 ('| Selecting test {} at feature-{}-distance {} from {}, layer {}'
          .format (ti, flfeature, best_dist, interval_repr (interval), fl))
    measure_progress = \
        self._measure_progress_towards_interval (feature, interval, ti)
    fct = BFcTarget (feature_node, feature_interval,
                     measure_progress, ti, self.verbose)
    self.ban[fl].add ((feature, feature_interval, ti))

    return self.test_cases[ti], fct


# ---


class BFDcTarget (NamedTuple, BNcTarget):
  fnode1: DiscretizedHiddenFeatureNode
  feature_part1: int
  flayer0: BFcLayer
  feature_parts0: Sequence[int]
  progress: Callable[[Input], float]
  root_test_idx: int
  verbose: int

  def __repr__(self) -> str:
    interval = self.fnode1.flayer.discr.part_edges (self.fnode1.feature,
                                                    self.feature_part1)
    return (('interval {} of feature {} in layer {}, subject to feature'+
             ' intervals {} in layer {} (from test {})')
            .format(interval_repr (interval),
                    self.fnode1.feature, self.fnode1.flayer,
                    self.feature_parts0, self.flayer0, self.root_test_idx))


  def log_repr(self) -> str:
    return ('#layer: {} #feat: {} #part: {} #conds: {}'
            .format(self.fnode1.flayer.layer_index,
                    self.fnode1.feature, self.feature_part1,
                    self.feature_parts0))


  def __eq__ (self, t) -> bool:
    '''Basic equality test'''
    return type (self) == type (t) and \
           self.fnode1 == t.fnode1 and \
           self.feature_part1 == t.feature_part1 and \
           self.flayer0 == t.flayer0 and \
           self.feature_parts0 == t.feature_parts0

  def __hash__(self) -> int:
    return hash ((self.fnode1, self.feature_part1,
                  self.flayer0, self.feature_parts0))


  def cover(self, acts) -> None:
    # Do nothing for now; ideally: update some probabilities
    # somewhere.
    pass


  def measure_progress(self, t: Input) -> float:
    progress, new_dist = self.progress (t)
    if self.verbose >= _log_progress_level:
      p1 ('| Progress towards {}: {}'.format (self, progress))
    if self.verbose >= _log_interval_distance_level:
      p1 ('| Distance to target interval: {} ({})'
          .format (new_dist, 'hit' if new_dist <= 0.0 else 'miss'))
    return progress


# ---


class BFDcAnalyzer (BFcAnalyzer):
  """
  Analyzer dedicated to targets of type :class:`BDFcTarget`.
  """

  @abstractmethod
  def search_input_close_to(self, x: Input, target: Union[BFcTarget,BFDcTarget]) -> \
          Optional[Tuple[float, Input]]:
    """
    Method specialized for targets of either type :class:`BFDcTarget`
    or :class:`BFcTarget`.
    """
    pass



# ---


class BFDcCriterion (BFcCriterion, Criterion4RootedSearch):
  '''
  Adaptation of MC/DC coverage for partitioned features.
  '''

  def __init__(self,
               clayers: Sequence[CoverableLayer],
               analyzer: BFDcAnalyzer,
               *args,
               **kwds):
    assert isinstance (analyzer, BFDcAnalyzer)
    super().__init__(clayers, analyzer = analyzer, *args, **kwds)
    assert len(self.flayers) >= 2


  def __repr__(self):
    return "BFdC"


  def coverage (self) -> Coverage:
    '''
    Returns BFdCov * BFCov
    '''
    return self.BN.bfdc_coverage (epsilon = self.epsilon,
                                  multiply_with_bfc = True)


  def find_next_rooted_test_target (self) -> Tuple[Input, Union[BFcTarget,BFDcTarget]]:

    # Gather non-epsilon conditional probabilities:
    cpts = [ np.array (cpt) for cpt in self.BN._all_cpts () ]
    epsilon_entries = ((i, fli)
                       for i, cpt in enumerate (cpts)
                       for fli in np.where (cpt[:,-1] < self.epsilon))

    res, best_dist = None, np.inf
    for epsilon_cond_prob_index, epsilon_intervals in epsilon_entries:
      feature = self.flayers[0].num_features + epsilon_cond_prob_index
      feature_node = self.N.states[feature]
      fl, flfeature = feature_node.flayer, feature_node.feature

      for fli in epsilon_intervals:
        assert cpts[epsilon_cond_prob_index][fli, -1] < self.epsilon
        feature_interval = int (cpts[epsilon_cond_prob_index][fli, -2])

        # Search closest test -that does not fall within target interval (?)-:
        # within_target = self._check_within (feature, feature_interval, verbose = False)
        for ti, dist in self._all_tests_n_dists_to (feature, feature_interval):
          if (dist < 0 or
              (feature, feature_interval, ti) in self.ban[fl]# or
              # within_target (flfeature, self.test_cases[i])
              ):
            continue
          if dist < best_dist:
            best_dist = dist
            res = epsilon_cond_prob_index, fli, ti

      if res is not None and self.shallow_first:
        break

    if res is None:
      if self.BN.bfc_coverage (epsilon = self.epsilon).done:
        raise EarlyTermination ('Unable to find a new candidate input!')
      else:
        return super().find_next_rooted_test_target ()

    epsilon_cond_prob_index, fli, ti = res
    feature = self.flayers[0].num_features + epsilon_cond_prob_index
    feature_node = self.N.states[feature]
    feature_interval = int (cpts[epsilon_cond_prob_index][fli, -2])
    fl, flfeature = feature_node.flayer, feature_node.feature
    fl_prev, _ = self.fidx2fli[feature - flfeature - 1]
    interval = feature_node.interval (feature_interval)
    if self.verbose >= _log_test_selection_level:
      p1 ('| Selecting test {} at feature-{}-distance {} from {}, layer {}'
          .format (ti, flfeature, best_dist, interval_repr (interval), fl))
    measure_progress = \
        self._measure_progress_towards_interval (feature, interval, ti)
    cond_intervals = cpts[epsilon_cond_prob_index][fli, :-2].astype (int)
    fct = BFDcTarget (feature_node, feature_interval,
                      fl_prev, tuple (cond_intervals.tolist ()),
                      measure_progress, ti, self.verbose)
    self.ban[fl].add ((feature, feature_interval, ti))

    return self.test_cases[ti], fct



# ---

def layer_transform_options_from_discr (li, discr = None, default = 1):

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


def layer_feature_discretization (l, li, discr = None, discr_n_jobs = None):
  li_discr = (discr[li] if isinstance (discr, dict) and li in discr else
              discr     if isinstance (discr, dict) else
              discr     if isinstance (discr, int)  else
              discr[li] if isinstance (discr, list) else
              discr(li) if callable (discr) else
              builtins.eval(discr, {})(li) if (isinstance (discr, str) and
                                               discr not in ('binarizer', 'bin')) else
              None)
  if li_discr in (None, 'binarizer', 'bin'):
    p1 ('Using binarizer for layer {.name}'.format (l))
    return FeatureBinarizer ()
  else:
    k = (li_discr if isinstance (li_discr, int) else
         li_discr['n_bins'] if (isinstance (li_discr, dict)
                                and 'n_bins' in li_discr) else
         # TODO: per feature discretization strategy?
         None)
    s = (li_discr['strategy'] if (isinstance (li_discr, dict)
                                  and 'strategy' in li_discr) else
         'quantile')
    extended = (isinstance (li_discr, dict) and 'extended' in li_discr
                and li_discr['extended'])
    extended = extended is not None and extended
    p1 ('Using {}{}discretizer with {} strategy for layer {.name}'
        .format ('extended ' if extended else '',
                 '{}-bin '.format (k) if k is not None else '',
                 s, l))
    cstr = KBinsNOutFeatureDiscretizer if extended else KBinsFeatureDiscretizer
    discr_args = { **(li_discr if isinstance (li_discr, dict) else {}),
      'n_bins': k,
      'encode': 'ordinal',
      'strategy': s,
      'n_jobs': discr_n_jobs
    }
    if 'extended' in discr_args:
      del discr_args['extended']
    return cstr (**discr_args)



def layer_setup (l, i, feats = None, discr = None, **kwds):
  options = layer_transform_options (i, feats, default = None)
  if options is None:
    options = layer_transform_options_from_discr (i, discr)
  fext, skip, focus = layer_transform (l, i, options)
  feature_discretization = layer_feature_discretization (l, i, discr, **kwds)
  return BFcLayer (layer = l, layer_index = i,
                   transform = fext,
                   discretization = feature_discretization,
                   skip = skip,
                   focus = focus)


# ---


def plot_report_on_feature_extractions (fl, fdimred, labels, acc = None):
  if not plt:
    warnings.warn ('Unable to import `matplotlib`: skipping feature extraction plots')
    return []

  minlabel, maxlabel = np.min (labels), np.max (labels)
  cmap = plt.get_cmap ('nipy_spectral', maxlabel - minlabel + 1)

  flabel = (lambda feature:
            r'$\mathbb{F}_{' + plotting.texttt (str (fl)) + ', ' + str (feature) + '}$' +
            (f' (variance ratio = {fl.transform[-1].explained_variance_ratio_[feature]:6.2%})'
             if hasattr (fl.transform[-1], 'explained_variance_ratio_') else ''))

  maxfeature = fdimred.shape[1] - 1
  if maxfeature < 1:
    return []                   # for now
  feature = 0
  figs = []
  while feature + 1 <= maxfeature:
    fig = plt.figure ()
    if feature + 1 == maxfeature:
      ax = fig.add_subplot (111)
      # plt.subplot (len (self.flayer_transforms), 1, idx)
      scat = ax.scatter(fdimred[:,0], fdimred[:,1], c = labels,
                        s = 2, marker='o', zorder = 10,
                        cmap = cmap, vmin = minlabel - .5, vmax = maxlabel + .5)
      ax.set_xlabel (flabel (feature))
      ax.set_ylabel (flabel (feature+1))
      feature_done = 2
      incr = 1
    else:
      ax = fig.add_subplot (111, projection = '3d')
      scat = ax.scatter (fdimred[:, feature], fdimred[:, feature+1],
                         fdimred[:, feature+2], c = labels,
                         s = 2, marker = 'o', zorder = 10,
                         cmap = cmap, vmin = minlabel - .5, vmax = maxlabel + .5)
      ax.set_xlabel (flabel (feature))
      ax.set_ylabel (flabel (feature+1))
      ax.set_zlabel (flabel (feature+2))
      feature_done = 3
      incr = 1 if feature + 1 == maxfeature - 2 else 2
    args = \
      dict (shrink = .2, pad = .2, orientation = 'horizontal') if feature_done == 3 else \
      dict (shrink = .6, pad = .01, orientation = 'vertical')
    cb = fig.colorbar (scat, ticks = range (minlabel, maxlabel + 1),
                       label = 'Labels', **args)
    feature += incr
    figs.append (fig)
  plt.draw ()
  return figs

def show_report_on_feature_extractions_ (outdir = None, basefilename = None):
  assert outdir is not None
  assert basefilename is not None
  assert isinstance (basefilename, str)
  def aux (figs):
    for i, f in enumerate (figs):
      plotting.show (f, outdir = outdir, basefilename = basefilename + f'-{i}')
  return aux


# ---


from engine import setup as engine_setup

def setup (setup_criterion = None,
           test_object = None,
           bn_abstr: str = None,
           bn_abstr_train_size = 0.5,
           bn_abstr_test_size = 0.5,
           bn_abstr_n_jobs = None,
           outdir: OutputDir = None,
           feats = { 'n_components': 2, 'svd_solver': 'randomized' },
           feat_extr_train_size = 1,
           discr = 'bin',
           discr_n_jobs = None,
           epsilon = None,
           shallow_first = None,
           report_on_feature_extractions = False,
           dump_bn_with_trained_dataset_distribution = None,
           dump_bn_with_final_dataset_distribution = None,
           verbose: int = None,
           **kwds):

  if setup_criterion is None:
    raise ValueError ('Missing argument `setup_criterion`!')

  if bn_abstr is None:
    setup_layer = lambda l, i, **kwds: \
      layer_setup (l, i, feats, discr, discr_n_jobs = discr_n_jobs)
    cover_layers = get_cover_layers (test_object.dnn, setup_layer,
                                     layer_indices = test_object.layer_indices,
                                     activation_of_conv_or_dense_only = False,
                                     exclude_direct_input_succ = False,
                                     exclude_output_layer = False)
    bn_abstr_args = dict (feat_extr_train_size = feat_extr_train_size,
                          print_classification_reports = True,
                          bn_abstr_n_jobs = bn_abstr_n_jobs,
                          outdir = outdir)
    if report_on_feature_extractions:
      bn_abstr_args['report_on_feature_extractions'] = plot_report_on_feature_extractions
      bn_abstr_args['close_reports_on_feature_extractions'] = \
        show_report_on_feature_extractions_ (outdir = outdir, basefilename = 'fext-report')
    bn_crit_args = dict (bn_abstr_args = bn_abstr_args,
                         bn_abstr_train_size = bn_abstr_train_size,
                         bn_abstr_test_size = bn_abstr_test_size)
  else:
    bn_abstr = BNAbstraction.from_file (test_object.dnn, bn_abstr,
                                        bn_abstr_n_jobs = bn_abstr_n_jobs,
                                        outdir = outdir)
    cover_layers = bn_abstr.flayers
    bn_crit_args = dict (bn_abstr = bn_abstr)

  criterion_args = dict (**bn_crit_args,
                         epsilon = epsilon,
                         shallow_first = shallow_first,
                         dump_bn_with_trained_dataset_distribution = \
                         dump_bn_with_trained_dataset_distribution,
                         dump_bn_with_final_dataset_distribution = \
                         dump_bn_with_final_dataset_distribution,
                         verbose = verbose)

  return engine_setup (test_object = test_object,
                       cover_layers = cover_layers,
                       setup_criterion = setup_criterion,
                       criterion_args = criterion_args,
                       **kwds)

# ---

