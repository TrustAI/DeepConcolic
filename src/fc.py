from typing import *
from utils import *
import numpy as np
from engine import CoverableLayer, BoolMappedCoverableLayer
from engine import LayerLocalCriterion, Criterion4RootedSearch
from nc import NcTarget, NcLayer, NcAnalyzer

from itertools import product
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, Binarizer
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss, classification_report
from pomegranate import Node, BayesianNetwork
from pomegranate.distributions import DiscreteDistribution, ConditionalProbabilityTable


# ---


class FcLayer (CoverableLayer):

  def __init__(self, transform = None, discretization = None, **kwds):
    super().__init__(**kwds)
    self.transform = transform
    self.discr = discretization


  @property
  def num_features (self):
    return self.transform[-1].n_components_


  # def update(self, act) -> None:
  # #   act = copy.copy (act[self.layer_index])
  # #   # Keep only negative new activation values:
  # #   act[act >= 0] = 0
  # #   self.map = np.logical_and (self.map, act)
  # #   # Append activations after map change
  # #   self.append_activations (act)


# ---


def bayes_node_name(fl, idx):
  return '.'.join ((str(fl), *((str(i) for i in idx))))



class DiscretizedFeatureNode (Node):

  def __init__(self, flayer, feature, n_bins, *args, **kwds):
    super().__init__ (*args, name = bayes_node_name (flayer, (feature,)), **kwds)
    self.n_bins = n_bins


  def discretized_range(self):
    return range (self.n_bins)



class DiscretizedInputFeatureNode (DiscretizedFeatureNode):

  def __init__(self, flayer, feature, n_bins, **kwds):
    distribution = DiscreteDistribution ({ fbin: 0.0 for fbin in range (n_bins) })
    super().__init__(flayer, feature, n_bins, distribution, **kwds)


class DiscretizedHiddenFeatureNode (DiscretizedFeatureNode):

  def __init__(self, flayer, feature, n_bins, prev_nodes, **kwds):
    prev_distrs = [ n.distribution for n in prev_nodes ]
    condprobtbl = [ list (p) + [0.0]
                    for p in product (*list ([ bin for bin in prev_node.discretized_range () ]
                                             for prev_node in prev_nodes),
                                      range (n_bins))]
    distribution = ConditionalProbabilityTable (condprobtbl, prev_distrs)
    super().__init__(flayer, feature, n_bins, distribution, **kwds)


# ---


# That's just an alias for specifying types in the FC criterion below
FcTarget = NcTarget


# ---


class FeatureDiscretizer:
  def __init__(self, **kwds):
    super().__init__(**kwds)

  @abstractmethod
  def feature_parts (self, feature: int) -> int:
    raise NotImplementedError



class FeatureBinarizer (FeatureDiscretizer, Binarizer):
  def __init__(self, **kwds):
    super().__init__(**kwds)

  def feature_parts (self, _feature):
    return 2



class KBinsFeatureDiscretizer (FeatureDiscretizer, KBinsDiscretizer):
  def __init__(self, **kwds):
    super().__init__(**kwds)

  def feature_parts (self, feature):
    return self.n_bins_[feature]



# ---


class FcCriterion (LayerLocalCriterion, Criterion4RootedSearch):

  def __init__(self, clayers: Sequence[NcLayer], analyzer: NcAnalyzer,
               # flayers: Sequence[Tuple[FcLayer, dict, FeatureDiscretizer]] = None,
               bn_abstr_train_size = 0.5,
               bn_abstr_test_size = None,
               bn_n_jobs = 1,
               print_classification_reports = True,
               report_on_feature_extractions = None,
               close_reports_on_feature_extractions = None,
               **kwds):
    assert isinstance (analyzer, NcAnalyzer)
    assert (print_classification_reports is None or isinstance (print_classification_reports, bool))
    assert (report_on_feature_extractions is None or callable (report_on_feature_extractions))
    assert (close_reports_on_feature_extractions is None or callable (close_reports_on_feature_extractions))
    # from sklearn.decomposition import IncrementalPCA
    self.bn_n_jobs = bn_n_jobs
    self.bn_abstr_params = { 'train_size': bn_abstr_train_size,
                             'test_size': bn_abstr_test_size }
    self.print_classification_reports = print_classification_reports
    self.report_on_feature_extractions = report_on_feature_extractions
    self.close_reports_on_feature_extractions = close_reports_on_feature_extractions
    self.flayers = list (filter (lambda l: isinstance (l, FcLayer), clayers))
    clayers = list (filter (lambda l: isinstance (l, BoolMappedCoverableLayer), clayers))
    super().__init__(clayers = clayers, analyzer = analyzer, **kwds)


  def __repr__(self):
    return "FC"


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
      y = fl.discr.transform (fl.transform.transform (x))# .astype (int, copy = False)
      facts = np.hstack ((facts, y.astype (int))) if facts.any () else y.astype (int)
      del x, y
    return facts


  # ---


  def fit_activations (self, acts):
    self.N.fit (self.dimred_n_discretize_activations (acts))


  def add_new_test_case(self, t):
    super().add_new_test_case (t)
    activations = eval (self.analyzer.dnn, t, is_input_layer (self.analyzer.dnn.layers[0]))
    self.fit_activations (activations)


  # ---


  def stat_based_cv_initializers(self):
    return [{
      **self.bn_abstr_params,
      'name': 'BN Abstraction',
      'layer_indexes': set ([fl.layer_index for fl in self.flayers]),
      'train': self._discretize_features_and_create_bn_structure,
      'test': self._score,
      # 'accum_test': self._accum_fit_bn,
      # 'final_test': self._bn_score,
    }]


  def _discretize_features_and_create_bn_structure (self, acts):
    cnp1 ('| Given training data of size {}'
          .format(len(acts[self.flayers[0].layer_index])))
    # First, fit feature extraction and discretizer parameters:
    for fl in self.flayers:
      cnp1 ('| Discretizing features for layer {}... '.format (fl))
      x = np.stack([a.flatten () for a in acts[fl.layer_index]], axis = 0)
      y = fl.transform.fit_transform (x)
      fl.discr.fit (y)
      np1 ('{} nodes.'.format (y.shape[1]))
      del x, y
    # Second, fit some distributions with input layer values (NB: well, actually...)
    # Third, contruct the Bayesian Network
    self.N = self._create_bayesian_network ()
    # Fourth, fit the Bayesian Network (just with acts for now, for
    # testing purposes)
    self.fit_activations (acts)
    # self.N.plot (filename = '/tmp/bn.pdf')
    # # write_in_file ('/tmp/bn.json', self.N.to_json ())
    # write_in_file ('/tmp/bn.yaml', self.N.to_yaml ())


  def _create_bayesian_network (self):
    ctp1 ('| Creating Bayesian Network... ')
    N = BayesianNetwork (name = 'BN Abstraction')


    fl0 = self.flayers[0]
    nodes = [ DiscretizedInputFeatureNode (fl0, fidx, fl0.discr.feature_parts (fidx))
              for fidx in range (fl0.num_features) ]
    N.add_nodes (*(n for n in nodes))

    prev_nodes = nodes
    for fl in self.flayers[1:]:
      nodes = [ DiscretizedHiddenFeatureNode (fl, fidx, fl.discr.feature_parts (fidx), prev_nodes)
                for fidx in range (fl.num_features) ]
      N.add_nodes (*(n for n in nodes))

      for pn, n in product (*(prev_nodes, nodes)):
        N.add_edge (pn, n)
      tp1 ('| Creating Bayesian Network: {} nodes...'.format (N.node_count ()))

      del prev_nodes
      prev_nodes = nodes

    del prev_nodes
    tp1 ('| Creating Bayesian Network: baking...')
    N.bake ()
    p1 ('| Created Bayesian Network of {} nodes and {} edges.'
        .format (N.node_count (), N.edge_count ()))
    return N


  # def _estimate_feature_probas (self, truth, ftest, fidx, nbfeats):
  #   ftest[..., fidx : fidx + nbfeats] = np.nan
  #   probas = np.array (self.N.predict_proba (ftest, n_jobs = self.bn_n_jobs))
  #   ftest[..., fidx : fidx + nbfeats] = truth[..., fidx : fidx + nbfeats]
  #   lprobas = probas[..., fidx : fidx + nbfeats]
  #   del probas
  #   return lprobas


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


  def _most_likely (self, p):
    return int (max (p.parameters[0], key = p.parameters[0].get))


  def _score (self, acts, labels):
    p1 ('| Given test sample of size {}'
         .format(len(acts[self.flayers[0].layer_index])))

    self._score_feature_extractions (acts, labels)

    truth = self.dimred_n_discretize_activations (acts)
    self._score_discretized_feature_probas (truth)
    del truth


  def _score_feature_extractions (self, acts, labels):
    racc = None
    idx = 1
    first_feature_idx = 0
    for fl in self.flayers:
      flatacts = self.flatten_for_layer (acts, (fl,))
      # tp1 ('| Computing average log-likelihood of test sample for layer {}...'
      #      .format (fl))
      # p1 ('| Average log-likelihood of test sample for layer {} is {}'
      #     .format (fl, fl.transform.score (flatacts)))

      if self.report_on_feature_extractions is not None:
        fdimred = self.dimred_activations (acts, (fl,))
        racc = self.report_on_feature_extractions (fl, flatacts, fdimred, labels, racc)

        idx += 1
        del fdimred

      del flatacts
      first_feature_idx += fl.num_features

    if self.close_reports_on_feature_extractions is not None:
      self.close_reports_on_feature_extractions (racc)


  def _score_discretized_feature_probas (self, truth):
    features_probas = self._setup_estimate_feature_probas (truth)

    # log_proba = self.N.log_probability (truth)
    # print (log_proba)

    # layer_log_loss = []
    # first_feature_idx = 0
    # for fl in self.flayers:
    #   maxlabels = max (fl.discr.feature_parts (fidx) for fidx in range (fl.num_features))
    #   llabels = list (range (maxlabels))
    #   ltruth = truth[..., first_feature_idx : first_feature_idx + fl.num_features].flatten ()

    #   tp1 ('| Computing predictions for {}...'.format (fl))
    #   lprobas = features_probas (first_feature_idx, fl.num_features).flatten ()

    #   tp1 ('| Computing log loss for {}...'.format (fl))
    #   lpredict_probs = [ self._prediction_probas (p) for p in lprobas ]
    #   loss = log_loss (ltruth, lpredict_probs, labels = llabels)
    #   p1 ('| Log loss for {} is {}'.format (fl, loss))

    #   if self.print_classification_reports:
    #     p1 ('| Classification report for {}:'.format (fl))
    #     lpreds = [ np.argmax (p) for p in lpredict_probs ]
    #     print (classification_report (ltruth, lpreds, labels = llabels))
    #     del lpreds

    #   del ltruth, lprobas, lpredict_probs, llabels

    #   first_feature_idx += fl.num_features

    # layer_feature_log_loss = { fl: [] for fl in self.flayers }
    first_feature_idx = 0
    for fl in self.flayers:
      for fidx in range (fl.num_features):
        flabels = list (range (fl.discr.feature_parts (fidx)))
        feature_idx = first_feature_idx + fidx
        ftruth = truth[..., feature_idx : feature_idx + 1].flatten ()

        tp1 ('| Computing predictions for feature {} of {}...'.format (fidx, fl))
        fprobas = features_probas (feature_idx, 1).flatten ()

        tp1 ('| Computing log loss for feature {} of {}...'.format (fidx, fl))
        fpredict_probs = [ self._prediction_probas (p) for p in fprobas ]
        loss = log_loss (ftruth, fpredict_probs, labels = flabels)
        p1 ('| Log loss for feature {} of {} is {}'.format (fidx, fl, loss))

        if self.print_classification_reports:
          p1 ('| Classification report for feature {} of {}:'.format (fidx, fl))
          fpreds = [ np.argmax (p) for p in fpredict_probs ]
          print (classification_report (ftruth, fpreds, labels = flabels))
          del fpreds

        del ftruth, fprobas, fpredict_probs, flabels

      first_feature_idx += fl.num_features

    del features_probas


# class NcPulpPCAAnalyzer (NcPulpAnalyzer, PCAAnalyzer):

#   def __init__(self, layers, test_object, **kwds):
#     super().__init__(layers, test_object,
#                      reduced_layers = [2, 5],
#                      **kwds)


# ---


def abstract_layerp (li, n_feats = None, discr = None, layer_indices = []):
  return (li in discr if discr is not None and isinstance (discr, dict) else
          li in n_feats if n_feats is not None and isinstance (n_feats, dict) else
          li in layer_indices)


def abstract_layer_features (li, n_feats = None, discr = None, default = 1):
  if n_feats is not None:
    if isinstance (n_feats, (int, float, str)):
      return n_feats
    if isinstance (n_feats, dict):
      li_feats = n_feats[li]
      if not isinstance (li_feats, (int, float, str, dict)):
        raise ValueError (
          'n_feats[{}] should be an int, a string, or a float (got {})'
          .format (li, type (li_feats)))
      return li_feats
    raise ValueError (
      'n_feats should either be a dictonary, an int, a string, or a float (got {})'
      .format (type (n_feats)))

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
  li_discr = (discr[li] if isinstance (discr, (dict, list)) else
              discr(li) if callable (discr) else None)
  if li_discr in (None, 'binarizer', 'bin'):
    p1 ('Using binarizer for layer {.name}'.format (l))
    return FeatureBinarizer (threshold = 0.0)
  else:
    k = (li_discr if isinstance (li_discr, int) else
         li_discr['n_bins'] if isinstance (li_discr, dict) else
         # TODO: per feature discretization strategy?
         2)
    p1 ('Using {}-bins discretizer for layer {.name}'.format (k, l))
    return KBinsFeatureDiscretizer (
      n_bins = k,
      encode = 'ordinal',
      strategy = (li_discr['strategy'] if isinstance (li_discr, dict) else
                  'quantile'))


def abstract_layer_setup (l, i, n_feats = None, discr = None):
  pca_options = abstract_layer_features (i, n_feats, discr)
  if isinstance (pca_options, (int, float)) or pca_options == 'mle':
    svd_solver = ('randomized' if isinstance (pca_options, int) else
                  'full' if isinstance (pca_options, float) else 'auto')
    pca_options = { 'n_components': pca_options, 'svd_solver': svd_solver }
  pca_options = { **pca_options, 'whiten': True, }
  feature_extraction = make_pipeline (StandardScaler (copy = False),
                                      PCA (**pca_options, copy = False))
  feature_discretization = abstract_layer_feature_discretization (l, i, discr)
  return FcLayer (layer = l, layer_index = i,
                  transform = feature_extraction,
                  discretization = feature_discretization)

# ---

import matplotlib.pyplot as plt

def report_on_feature_extractions (fl, flatacts, fdimred, labels, acc = None):
  from matplotlib import cm
  from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
  from mpl_toolkits.mplot3d import Axes3D

  minlabel, maxlabel = np.min (labels), np.max (labels)
  cmap = plt.get_cmap ('nipy_spectral', maxlabel - minlabel + 1)

  flabel = (lambda fidx: 'f{} (variance ratio = {:6.2%})'
            .format (fidx, fl.transform[-1].explained_variance_ratio_[fidx]))

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

def close_reports_on_feature_extractions (acc = None):
  plt.show ()

# ---


from engine import setup as engine_setup

def setup (test_object = None,
           n_feats = { 0: { 'n_components': 6, 'svd_solver': 'randomized' },
                       2: { 'n_components': 6, 'svd_solver': 'randomized' },
                       5: { 'n_components': 6, 'svd_solver': 'randomized' },
                       7: { 'n_components': 6, 'svd_solver': 'randomized' },
                       11: 0.6,
                       13: 0.7,
                       15: 0.9 },
           discr = None,
           # (lambda li: { 'n_bins': 4, 'strategy': 'uniform' } if li == 15 else 'binarizer'),
           **kwds):

  setup_layer = (
    lambda l, i, **kwds:
    abstract_layer_setup (l, i, n_feats, discr)
    if abstract_layerp (i, n_feats, discr, test_object.layer_indices) else
    NcLayer (layer = l, layer_index = i, feature_indices = test_object.feature_indices, **kwds)
  )
  cover_layers = get_cover_layers (
    test_object.dnn, setup_layer,
    layer_indices = test_object.layer_indices,
    exclude_direct_input_succ = False
  )
  return engine_setup (
    test_object = test_object,
    cover_layers = cover_layers,
    setup_criterion = FcCriterion,
    criterion_args = { 'bn_abstr_test_size': 5000,
                       'print_classification_reports': True,
                       'report_on_feature_extractions': report_on_feature_extractions,
                       'close_reports_on_feature_extractions': close_reports_on_feature_extractions, },
    **kwds
  )

# ---
