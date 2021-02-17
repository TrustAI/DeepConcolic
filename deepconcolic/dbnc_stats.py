import norms
import scripting
from utils_io import *
from utils_funcs import *
from dbnc import *
from nc import *
from bounds import UniformBounds
from engine import Metric
from sklearn.model_selection import train_test_split

# ---

class FakeBFcAnalyzer (BFDcAnalyzer):

  def finalize_setup(self, _clayers):
    pass

  def search_input_close_to(self, x, target):
    return None

  def input_metric(self) -> Metric:
    return norms.LInf ()

# ---

class FakeLocalAnalyzer (NcAnalyzer):

  def finalize_setup(self, _clayers):
    pass

  def search_input_close_to(self, x, target):
    return None

# ---

def run (test_object = None,
         outs = OutputDir (),
         input_bounds = UniformBounds ()):

  all_n_feats_specs = [
    ('1-randomized', { 'n_components': 1, 'svd_solver': 'randomized' }, 'pca', 1),
    ('2-randomized', { 'n_components': 2, 'svd_solver': 'randomized' }, 'pca', 2),
    ('3-randomized', { 'n_components': 3, 'svd_solver': 'randomized' }, 'pca', 3),
    # ('4-randomized', { 'n_components': 4, 'svd_solver': 'randomized' }, 'pca', 4),
    ('ica-1', { 'decomp': 'ica', 'n_components': 1, 'max_iter': 10000, 'tol': 0.01 }, 'ica', 1),
    ('ica-2', { 'decomp': 'ica', 'n_components': 2, 'max_iter': 10000, 'tol': 0.01 }, 'ica', 2),
    ('ica-3', { 'decomp': 'ica', 'n_components': 3, 'max_iter': 10000, 'tol': 0.01 }, 'ica', 3),
    # ('ica-4', { 'decomp': 'ica', 'n_components': 4, 'max_iter': 10000, 'tol': 0.02 }, 'ica', 4),
    # ('ica-5', { 'decomp': 'ica', 'n_components': 5, 'max_iter': 20000, 'tol': 0.03 }, 'ica', 5),
    # #('ica-6', { 'decomp': 'ica', 'n_components': 6, 'max_iter': 20000, 'tol': 0.03 }, 'ica', 6),
    # # ('ica-7', { 'decomp': 'ica', 'n_components': 7, 'max_iter': 30000, 'tol': 0.03 }, 'ica', 7),
    # ('pca-1', 1, 'pca', 1),
    # ('pca-2', 2, 'pca', 2),
    # ('pca-3', 3, 'pca', 3),
    # ('pca-4', 4, 'pca', 4),
    # ('ipca-1', 1, 'ipca', 1),
    # ('ipca-2', 2, 'ipca', 2),
    # ('ipca-3', 3, 'ipca', 3),
    # ('ipca-4', 4, 'ipca', 4),
    # ('pca-5', 5, 'pca', 5),
    # ('full-0.1', 0.1, 'pca', 0),
    # ('full-0.2', 0.2, 'pca', 0),
    # ('full-0.3', 0.3, 'pca', 0),
    # ('full-0.4', 0.4, 'pca', 0),
    # # ]
    # # all_n_feats_specs_extra = [
    #  #('6-randomized', { 'n_components': 6, 'svd_solver': 'randomized' }, 'pca', 6),
    #  # ('7-randomized', { 'n_components': 7, 'svd_solver': 'randomized' }, 'pca', 7),
    #  # ('8-randomized', { 'n_components': 8, 'svd_solver': 'randomized' }, 'pca', 8),
    #  # ('full-0.5', 0.5),
    #  # ('full-0.6', 0.6),
    #  # ('full-0.7', 0.7)
    #  # ('full-0.8', 0.8)
    #  # ('full-0.9', 0.9)
  ]
  all_discr_specs = [
    # ('bin', 'bin', 2),
    # ('kde-logl-0.2', { 'strategy': 'kde', 'kde_space': 'logl', 'kde_rel_height': 0.2 }, 0),
    # ('kde-dens-nozerosplit',         { 'strategy': 'kde', 'kde_space': 'dens', 'zerosplit_policy': 'never'      }, 0),
    # ('kde-dens-alwayszerosplit',     { 'strategy': 'kde', 'kde_space': 'dens', 'zerosplit_policy': 'always'     }, 0),
    # ('kde-dens-lastresortzerosplit', { 'strategy': 'kde', 'kde_space': 'dens', 'zerosplit_policy': 'lastresort' }, 0),
    # ('kde',          { 'strategy': 'kde', 'extended': False }, 0),
    # ('kde-extended', { 'strategy': 'kde', 'extended': True  }, 0),
    # ('kde-dens', { 'strategy': 'kde', 'kde_peak_space': 'dens', 'extended': True }, 0),
    # ('kde-logl-0.1-extended', { 'strategy': 'kde', 'kde_space': 'logl', 'extended': True, 'kde_peak_prominence_prop': 0.1 }, 0),
    # ('kde-logl-0.2-extended', { 'strategy': 'kde', 'kde_space': 'logl', 'extended': True, 'kde_peak_prominence_prop': 0.2 }, 0),
    # ('kde-logl-0.3-extended', { 'strategy': 'kde', 'kde_space': 'logl', 'extended': True, 'kde_peak_prominence_prop': 0.3 }, 0),
    # ('kde-logl-0.2-extended', { 'strategy': 'kde', 'kde_space': 'logl', 'extended': True }, 0),
    # ('kde-logl-0.3-extended', { 'strategy': 'kde', 'kde_space': 'logl', 'extended': True }, 0),
    # ('kde-dens-0.3-nozerosplit',         { 'strategy': 'kde', 'kde_space': 'dens', 'zerosplit_policy': 'never'      }, 0),
    # ('kde-dens-0.3-alwayszerosplit',     { 'strategy': 'kde', 'kde_space': 'dens', 'zerosplit_policy': 'always'     }, 0),
    # ('kde-dens-0.3-lastresortzerosplit', { 'strategy': 'kde', 'kde_space': 'dens', 'zerosplit_policy': 'lastresort' }, 0),
    # ('kde-dens-0.1-nozerosplit',         { 'strategy': 'kde', 'kde_space': 'dens', 'zerosplit_policy': 'never'      }, 0),
    # ('kde-dens-0.1-alwayszerosplit',     { 'strategy': 'kde', 'kde_space': 'dens', 'zerosplit_policy': 'always'     }, 0),
    # ('kde-dens-0.1-lastresortzerosplit', { 'strategy': 'kde', 'kde_space': 'dens', 'zerosplit_policy': 'lastresort' }, 0),
    # ('kde-logl-0.3', { 'strategy': 'kde', 'kde_space': 'logl', 'kde_rel_height': 0.3 }, 0),
    # ('kde-dens-0.3', { 'strategy': 'kde', 'kde_space': 'dens', 'kde_rel_height': 0.3 }, 0),
    # # ('kde-logl-0.4', { 'strategy': 'kde', 'kde_space': 'logl', 'kde_rel_height': 0.4 }, 0),
    # ('kde-dens-0.4', { 'strategy': 'kde', 'kde_space': 'dens', 'kde_rel_height': 0.4 }, 0),
    # # ('kde-logl-0.5', { 'strategy': 'kde', 'kde_space': 'logl', 'kde_rel_height': 0.5 }, 0),
    # ('kde-dens-0.5', { 'strategy': 'kde', 'kde_space': 'dens', 'kde_rel_height': 0.5 }, 0),
    ('1-bin-extended',          { 'n_bins': 1, 'extended': True }, 3),
    ('2-bin-uniform',           { 'n_bins': 2, 'strategy': 'uniform' }, 2),
    ('3-bin-uniform',           { 'n_bins': 3, 'strategy': 'uniform' }, 3),
    ('4-bin-uniform',           { 'n_bins': 4, 'strategy': 'uniform' }, 4),
    ('2-bin-uniform-extended',  { 'n_bins': 2, 'strategy': 'uniform', 'extended': True  }, 4),
    ('3-bin-uniform-extended',  { 'n_bins': 3, 'strategy': 'uniform', 'extended': True  }, 5),
    ('2-bin-quantile',          { 'n_bins': 2, 'strategy': 'quantile' }, 2),
    ('3-bin-quantile',          { 'n_bins': 3, 'strategy': 'quantile' }, 3),
    ('4-bin-quantile',          { 'n_bins': 4, 'strategy': 'quantile' }, 4),
    ('2-bin-quantile-extended', { 'n_bins': 2, 'strategy': 'quantile', 'extended': True  }, 4),
    ('3-bin-quantile-extended', { 'n_bins': 3, 'strategy': 'quantile', 'extended': True  }, 5),
    # ('2-bin-quantile-extended', { 'n_bins': 2, 'strategy': 'quantile', 'extended': True  }, 4),
    # ('3-bin-quantile-extended', { 'n_bins': 3, 'strategy': 'quantile', 'extended': True  }, 5),
    # ('2-bin-kmeans', { 'n_bins': 2, 'strategy': 'kmeans' }, 2),
    # ('3-bin-extended-uniform', { 'n_bins': 3, 'extended': True, 'strategy': 'uniform' }, 5),
    # ('2-bin-extended-kmeans', { 'n_bins': 2, 'extended': True, 'strategy': 'kmeans' }, 4),
    # ('3-bin-extended-kmeans', { 'n_bins': 3, 'extended': True, 'strategy': 'kmeans' }, 5),
  # ]
  # all_discr_specs_extra = [
    # ('5-bin-uniform', { 'n_bins': 5, 'strategy': 'uniform' }, 5),
    # ('3-bin-kmeans', { 'n_bins': 3, 'strategy': 'kmeans' }, 3),
    # ('4-bin-kmeans', { 'n_bins': 4, 'strategy': 'kmeans' }, 4),
    # ('5-bin-kmeans', { 'n_bins': 5, 'strategy': 'kmeans' }, 5),
    # # ('4-bin-extended-uniform', { 'n_bins': 4, 'extended': True, 'strategy': 'uniform' }, 6),
    # # ('4-bin-extended-kmeans', { 'n_bins': 4, 'extended': True, 'strategy': 'kmeans' }, 6),
    # # ('5-bin-extended-uniform', { 'n_bins': 5, 'extended': True, 'strategy': 'uniform' }, 7),
    # # ('5-bin-extended-kmeans', { 'n_bins': 5, 'extended': True, 'strategy': 'kmeans' }, 7),
  ]

  rng_seed (42)
  rng = np.random.default_rng (randint ())

  local_analyzer = FakeLocalAnalyzer (analyzed_dnn = test_object.dnn,
                                      input_bounds = input_bounds)
  dbnc_analyzer = FakeBFcAnalyzer (analyzed_dnn = test_object.dnn,
                                   input_bounds = input_bounds)
  dbnc_train_size = 20000
  mul = 3
  test_sizes = ((1,) * mul +
                (2,) * mul +
                (3,) * mul +
                (5,) * mul +
                (10,) * mul +
                (50,) * mul +
                (100,) * mul#  +
                # (200,) * 5
                )
  # test_sizes = ((1,) *  5 + (10,) * 5 + (100,))

  tname = test_object.raw_data.name

  x_train = test_object.train_data.data
  y_train = test_object.train_data.labels.flatten ()
  y_train_preds = np.argmax (dbnc_analyzer.dnn.predict (x_train), axis = 1)
  idxs_train_ok = np.where (y_train_preds == y_train)
  x_train = x_train[idxs_train_ok]
  y_train = y_train[idxs_train_ok]
  dbnc_train_size = max (dbnc_train_size, len (idxs_train_ok) - 1)

  print ('Initializing', len (x_train), 'valid', tname, 'training cases.')
  train_idxs, _ = train_test_split (np.arange (len (x_train)),
                                    train_size = dbnc_train_size)
  print ('Selected', len (train_idxs), 'valid training samples.')

  x = test_object.raw_data.data
  y = test_object.raw_data.labels.flatten ()
  ypreds = np.argmax (dbnc_analyzer.dnn.predict (x), axis = 1)
  idxs_ok = np.where (ypreds == y)
  x = x[idxs_ok]
  y = y[idxs_ok]
  print ('Initializing', len (x), 'valid', tname, 'test cases.')

  test_idxs = []
  for test_size in test_sizes:
    ok = False
    while not ok:
      idxs = rng.choice (a = np.arange (len (x)), axis = 0, size = test_size)
      ok = len (np.unique (y[idxs])) == 10 or test_size < 10
    print ('Selected', len (idxs), 'valid test samples.')
    test_idxs.append (idxs)

  # x_no0 = x[y[idxs] != 0]
  # idxs_no0 = np.arange (len (x_no0))
  # test_no0_idxs = rng.choice (a = idxs_no0, axis = 0, size = min (100, len (x_no0)))
  # print ('Selected', len (test_no0_idxs), 'valid test samples without 0s.')

  # x_no01 = x_no0[y[idxs_no0] != 1]
  # idxs_no01 = np.arange (len (x_no01))
  # test_no01_idxs = rng.choice (a = idxs_no01, axis = 0, size = min (100, len (x_no01)))
  # print ('Selected', len (test_no01_idxs), 'valid test samples with neither 0s nor 1s.')

  yhalf = (np.max(y) - np.min(y)) // 2
  yy = np.array (y)
  idxs_half = np.flatnonzero (np.array (yy > yhalf))
  test_half_idxs = []
  for test_size in test_sizes:
    test_half_idxs.append (rng.choice (a = idxs_half, axis = 0,
                                       size = min (test_size, len (idxs_half))))
    print ('Selected', len (test_half_idxs[-1]), 'valid half-test samples.')

  # 12357
  # 0235689
  idxs_nobar = np.flatnonzero (np.array ((yy != 1) & (yy != 4) & (yy != 7)))
  test_nobar_idxs = []
  for test_size in test_sizes:
    test_nobar_idxs.append (rng.choice (a = idxs_nobar, axis = 0,
                                        size = min (test_size, len (idxs_nobar))))
    print ('Selected', len (test_nobar_idxs[-1]), 'valid nobar-test samples.')

  # for i, im in enumerate(x[test_idxs]):
  #   utils.save_an_image (im, str (i), "/tmp/outs/full/")
  # for i, im in enumerate(x[test_nobar_idxs]):
  #   utils.save_an_image (im, str (i), "/tmp/outs/nobar/")

  # print (y[test_idxs])
  # print (y[test_half_idxs])
  # print (y[test_nobar_idxs])

  # train_acts = dbnc_analyzer.eval_batch (x_train[train_idxs], allow_input_layer = True)
  # train_acts = { j: train_acts[j] for j in test_object.layer_indices } \
  #              if test_object.layer_indices is not None else train_acts
  assert test_object.layer_indices is not None
  train_data = x_train[train_idxs]
  train_labels = y_train[train_idxs]
  f = lambda j: LazyLambda \
    ( lambda i: dbnc_analyzer.eval_batch (train_data[i],
                                          allow_input_layer = True,
                                          layer_indexes = (j,))[j])
  train_acts = LazyLambdaDict (f, test_object.layer_indices)

  # test_acts = []
  # for test_idx in test_idxs:
  #   test_acts.append(analyzer.eval_batch (x[test_idx], allow_input_layer = True))
  #   test_acts[-1] = { j: test_acts[-1][j] for j in test_object.layer_indices }

  def local_run (stats, crit: LayerLocalCriterion):
    for i, test_size in enumerate (test_sizes):

      crit.reset ()                     # in case
      crit.add_new_test_cases(x[test_idxs[i]])
      cov = crit.coverage ()

      crit.reset ()
      crit.add_new_test_cases(x[test_half_idxs[i]])
      half_cov = crit.coverage ()

      crit.reset ()
      crit.add_new_test_cases(x[test_nobar_idxs[i]])
      nobar_cov = crit.coverage ()

      covstats = [cov, half_cov, nobar_cov]
      covstats = [c.as_prop for c in covstats]
      p1 ('{}: {:6.2%} {:6.2%} {:6.2%} | {}'
          .format(*(crit,) + tuple (covstats) + (test_size,)))
      stats += [[test_size] + covstats]


  def local_save (f, stats):
    np.savetxt (f, np.asarray (stats),
                header = "\t".join(("test size",
                                    "nc coverage",
                                    "nc coverage (half)",
                                    "nc coverage (nobar)")),
                fmt = '%s',
                # fmt = ['%s', '%s', '%i', '%i', '%f'],
                delimiter="\t")


  def dbnc_run (dbnc_stats, n_feats, discr, bnzf):
    (n_feats_name, n_feats, tech, ncomps), (discr_name, discr, nbins) = n_feats, discr

    setup_layer = lambda l, i, **kwds:
      layer_setup (l, i, n_feats, discr, discr_n_jobs = 8)
    cover_layers = get_cover_layers (test_object.dnn, setup_layer,
                                     layer_indices = test_object.layer_indices,
                                     activation_of_conv_or_dense_only = False,
                                     exclude_direct_input_succ = False,
                                     exclude_output_layer = False)
    crit = BFcCriterion (cover_layers, dbnc_analyzer,
                         bn_abstr_n_jobs = 8,
                         # to initialize the BN with training data:
                         # dump_bn_with_trained_dataset_distribution = True,
                         print_classification_reports = False,
                         score_layer_likelihoods = False)
    np1 ('Building Bayesian Abstraction')

    tic, get_times = scripting.init_tics ()

    crit._discretize_features_and_create_bn_structure (train_acts,
                                                       true_labels = train_labels,
                                                       pred_labels = train_labels)
    features = crit.num_features
    feats_stats = [np.amin (features),
                   np.mean (features),
                   np.amax (features)]
    feature_parts = crit.num_feature_parts
    discr_stats = [np.amin (feature_parts),
                   np.mean (feature_parts),
                   np.amax (feature_parts)]

    tic ()

    crit.fit_activations (train_acts)

    tic ()

    bn_nodes = crit.N.node_count ()
    bn_edges = crit.N.edge_count ()
    train_time, fit_time = get_times ()

    # crit.assess_discretized_feature_probas = True
    # crit._score (test_acts)

    for i, test_size in enumerate (test_sizes):

      tic, get_times = scripting.init_tics ()

      crit.reset ()                     # in case
      crit.add_new_test_cases(x[test_idxs[i]])
      bfc_cov = crit.bfc_coverage ()
      bfdc_cov = crit.bfdc_coverage ()

      # crit.reset ()
      # crit.add_new_test_cases(x[test_no0_idxs])
      # bfc_no0_cov = crit.bfc_coverage ()
      # bfdc_no0_cov = crit.bfdc_coverage ()

      # crit.reset ()
      # crit.add_new_test_cases(x[test_no01_idxs])
      # bfc_no01_cov = crit.bfc_coverage ()
      # bfdc_no01_cov = crit.bfdc_coverage ()

      crit.reset ()
      crit.add_new_test_cases(x[test_half_idxs[i]])
      bfc_half_cov = crit.bfc_coverage ()
      bfdc_half_cov = crit.bfdc_coverage ()

      crit.reset ()
      crit.add_new_test_cases(x[test_nobar_idxs[i]])
      bfc_nobar_cov = crit.bfc_coverage ()
      bfdc_nobar_cov = crit.bfdc_coverage ()

      tic ()
      # var_rats = crit.total_variance_ratios_
      # # loglikel = crit.average_log_likelihoods_
      # loglosss = crit.log_losses
      # loglossA = crit.all_log_losses
      (cov_time,) = get_times ()

      thistats = [n_feats_name, discr_name, bn_nodes, bn_edges,
                  train_time, fit_time, cov_time]
      covstats = [bfc_cov, bfc_half_cov, bfc_nobar_cov,
                  # bfc_no0_cov, bfdc_no0_cov,
                  # bfc_no01_cov, bfdc_no01_cov,
                  bfdc_cov, bfdc_half_cov, bfdc_nobar_cov]
      covstats = [c.as_prop for c in covstats]
      p1 ('{} {} | {:6.2%} {:6.2%} {:6.2%} | {:6.2%} {:6.2%} {:6.2%} | {}'
          .format(*(n_feats_name, discr_name, ) + tuple (covstats) + (test_size,)))
      p1 ('%s %s %i %i %5.1fs %5.1fs %5.1fs' % (tuple (thistats)))
      dbnc_stats += [thistats +
                     [tech, ncomps, nbins] +
                     feats_stats + discr_stats + [test_size] +
                     covstats
                   # + list (var_rats) # + list (loglikel)
                   # + list (np.array(loglosss).flatten ())
                   # + list (np.array(loglossA))
                   ]

    # import gzip
    # with gzip.GzipFile (bnzf, 'w') as f:
    #   f.write (crit.N.to_json ().encode ('utf-8'))
    #   f.close ()

    del crit, cover_layers

  # lnames = ("conv2d_1","conv2d_2","conv2d_4","conv2d_4","dense_1","dense_2","dense_3")
  # lvarih = ["var({0})".format (ln) for ln in lnames]
  # llossh = ["minlloss({0})\tmeanlloss({0})\tstdlloss({0})\tmaxlloss({0})"
  #           .format (ln) for ln in lnames]
  # llossA = "minlloss\tmeanlloss\tstdlloss\tmaxlloss"

  def dbnc_save (f, dbnc_stats):
    np.savetxt (f, np.asarray (dbnc_stats),
                header = "\t".join(("extraction",
                                    "discretization",
                                    "nodes",
                                    "edges",
                                    "construction time (s)",
                                    "fitting time (s)",
                                    # "scoring time (s)",
                                    "coverage computation time (s)",
                                    'tech',
                                    'components',
                                    'intervals/component',
                                    'min_n_components',
                                    'mean_n_components',
                                    'max_n_components',
                                    'min_n_bins',
                                    'mean_n_bins',
                                    'max_n_bins',
                                    "test size",
                                    "bfc coverage",
                                    "bfc coverage (half)",
                                    "bfc coverage (nobar)",
                                    "bfdc coverage",
                                    # "bfc coverage (no 0s)",
                                    # "bfdc coverage (no 0s)",
                                    # "bfc coverage (no 0s nor 1s)",
                                    # "bfdc coverage (no 0s nor 1s)",
                                    "bfdc coverage (half)",
                                    "bfdc coverage (nobar)",
                                    )),
                fmt = '%s',
                # fmt = ['%s', '%s', '%i', '%i', '%f'],
                delimiter="\t")

  def log (n, i, total, n_feats, discr, end = ''):
    p1 ('[{}: {}/{}] ******** {}, {} *******{}'.format (n, i, total, n_feats[0], discr[0], end))

  base = (tname+ '-X' + str(dbnc_train_size) + '-'
          + ('-'.join((str(i) for i in test_object.layer_indices)) if test_object.layer_indices is not None else 'all'))

  # ---

  nc_stats = []
  nc_setup_layer = (lambda l, i, **kwds: NcLayer (layer = l, layer_index = i))
  nc_cover_layers = get_cover_layers (test_object.dnn, nc_setup_layer,
                                      layer_indices = test_object.layer_indices,
                                      exclude_direct_input_succ = False)
  crit = NcCriterion (nc_cover_layers, local_analyzer)
  local_run (nc_stats, crit)
  local_save (outs.filepath (base + '-nc', suff = '.csv'), nc_stats)
  del crit

  # ---

  dbnc_stats = []
  total = len (all_n_feats_specs) * len (all_discr_specs)
  i = 1
  for a, b in product (all_n_feats_specs, all_discr_specs):
    log (base, i, total, a, b)
    dbnc_run (dbnc_stats, a, b, outs.filepath ('/{}-{}-{}.json.gz'.format(base, a[0], b[0])))
    log (base, i, total, a, b, end = ' done')
    dbnc_save (outs.filepath (base + '-dbnc', suff = '.csv'), dbnc_stats)
    i += 1

  # dbnc_stats = []
  # total = len (all_n_feats_specs) * len (all_discr_specs_extended)
  # i = 1
  # for a, b in product (all_n_feats_specs, all_discr_specs_extended):
  #   log ("basics", i, total, a, b)
  #   runit (dbnc_stats, a, b, '/LOCAL/nberth/tmp/fcstats/basics-{}-{}.json.gz'.format(a[0],b[0]))
  #   repit ('/tmp/basics.csv', dbnc_stats)
  #   i += 1

  # dbnc_stats = []
  # total = len (all_n_feats_specs_extra) * len (all_discr_specs)
  # i = 1
  # for a, b in product (all_n_feats_specs_extra, all_discr_specs):
  #   log ("extra", i, total, a, b)
  #   runit (dbnc_stats, a, b, '/LOCAL/nberth/tmp/fcstats/extra-{}-{}.json.gz'.format(a[0],b[0]))
  #   repit ('/tmp/extra.csv', dbnc_stats)
  #   i += 1

