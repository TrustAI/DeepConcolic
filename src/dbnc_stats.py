import time
from dbnc import *
from sklearn.model_selection import train_test_split

# ---

class FakeBFcAnalyzer (BFcAnalyzer, BFDcAnalyzer):

  def finalize_setup(self, _clayers):
    pass

  def search_input_close_to(self, x, target):
    return None

# ---

def run (test_object = None, outs = "/tmp/outs"):
  
  all_n_feats_specs = [
   ('ica-1', { 'decomp': 'ica', 'n_components': 1, 'max_iter': 10000, 'tol': 0.01 }, 'ica', 1),
   ('ica-2', { 'decomp': 'ica', 'n_components': 2, 'max_iter': 10000, 'tol': 0.01 }, 'ica', 2),
   ('ica-3', { 'decomp': 'ica', 'n_components': 3, 'max_iter': 10000, 'tol': 0.01 }, 'ica', 3),
   ('ica-4', { 'decomp': 'ica', 'n_components': 4, 'max_iter': 10000, 'tol': 0.02 }, 'ica', 4),
   ('ica-5', { 'decomp': 'ica', 'n_components': 5, 'max_iter': 20000, 'tol': 0.03 }, 'ica', 5),
   #('ica-6', { 'decomp': 'ica', 'n_components': 6, 'max_iter': 20000, 'tol': 0.03 }, 'ica', 6),
   # ('ica-7', { 'decomp': 'ica', 'n_components': 7, 'max_iter': 30000, 'tol': 0.03 }, 'ica', 7),
   ('1-randomized', { 'n_components': 1, 'svd_solver': 'randomized' }, 'pca', 1),
   ('2-randomized', { 'n_components': 2, 'svd_solver': 'randomized' }, 'pca', 2),
   ('3-randomized', { 'n_components': 3, 'svd_solver': 'randomized' }, 'pca', 3),
  #  # ('full-0.1', 0.1),
  #  # ('full-0.2', 0.2),
  # # ]
  # # all_n_feats_specs_extra = [
   ('4-randomized', { 'n_components': 4, 'svd_solver': 'randomized' }, 'pca', 4),
   ('5-randomized', { 'n_components': 5, 'svd_solver': 'randomized' }, 'pca', 5),
  #  #('6-randomized', { 'n_components': 6, 'svd_solver': 'randomized' }, 'pca', 6),
  #  # ('7-randomized', { 'n_components': 7, 'svd_solver': 'randomized' }, 'pca', 7),
  #  # ('8-randomized', { 'n_components': 8, 'svd_solver': 'randomized' }, 'pca', 8),
  #  # ('full-0.3', 0.3),
  #  # ('full-0.4', 0.4),
  #  # ('full-0.5', 0.5),
  #  # ('full-0.6', 0.6),
  #  # ('full-0.7', 0.7)
  #  # ('full-0.8', 0.8)
  #  # ('full-0.9', 0.9)
  ]
  all_discr_specs = [
    ('bin', 'bin', 2),
    ('2-bins-uniform', { 'n_bins': 2, 'strategy': 'uniform' }, 2),
    ('3-bins-uniform', { 'n_bins': 3, 'strategy': 'uniform' }, 3),
    ('4-bins-uniform', { 'n_bins': 4, 'strategy': 'uniform' }, 4),
    ('2-bins-kmeans', { 'n_bins': 2, 'strategy': 'kmeans' }, 2),
    ('1-bin-extended', { 'n_bins': 1, 'extended': True }, 3),
    ('2-bins-extended-uniform', { 'n_bins': 2, 'extended': True, 'strategy': 'uniform' }, 4),
    ('3-bins-extended-uniform', { 'n_bins': 3, 'extended': True, 'strategy': 'uniform' }, 5),
    ('2-bins-extended-kmeans', { 'n_bins': 2, 'extended': True, 'strategy': 'kmeans' }, 4),
    ('3-bins-extended-kmeans', { 'n_bins': 3, 'extended': True, 'strategy': 'kmeans' }, 5),
  # ]
  # all_discr_specs_extra = [
    ('5-bins-uniform', { 'n_bins': 5, 'strategy': 'uniform' }, 5),
    ('3-bins-kmeans', { 'n_bins': 3, 'strategy': 'kmeans' }, 3),
    ('4-bins-kmeans', { 'n_bins': 4, 'strategy': 'kmeans' }, 4),
    ('5-bins-kmeans', { 'n_bins': 5, 'strategy': 'kmeans' }, 5),
    # ('4-bins-extended-uniform', { 'n_bins': 4, 'extended': True, 'strategy': 'uniform' }, 6),
    # ('4-bins-extended-kmeans', { 'n_bins': 4, 'extended': True, 'strategy': 'kmeans' }, 6),
    # ('5-bins-extended-uniform', { 'n_bins': 5, 'extended': True, 'strategy': 'uniform' }, 7),
    # ('5-bins-extended-kmeans', { 'n_bins': 5, 'extended': True, 'strategy': 'kmeans' }, 7),
  ]

  analyzer = FakeBFcAnalyzer (analyzed_dnn = test_object.dnn)
  test_size = 100

  tname = test_object.raw_data.name
  x = test_object.raw_data.data
  y = test_object.raw_data.labels.flatten ()
  ypreds = np.argmax (analyzer.dnn.predict (x), axis = 1)
  idxs_ok = np.where (ypreds == y)
  x = x[idxs_ok]
  y = y[idxs_ok]

  print ('Initializing', len (x), 'valid', tname, 'test cases.')
  train_idxs, test_idxs = train_test_split (np.arange (len (x)),
                                            test_size = test_size, train_size = 1000,
                                            random_state = 42)
  print ('Selected', len (train_idxs), 'valid training samples.')

  rng = np.random.default_rng(43)
  ok = False
  while not ok:
    test_idxs = rng.choice (a = np.arange (len (x)), axis = 0, size = test_size)
    ok = len (np.unique (y[test_idxs])) == 10
  print ('Selected', len (test_idxs), 'valid test samples.')

  # x_no0 = x[y[idxs] != 0]
  # idxs_no0 = np.arange (len (x_no0))
  # test_no0_idxs = np.random.default_rng().choice (a = idxs_no0, axis = 0,
  #                                                 size = min (100, len (x_no0)))
  # print ('Selected', len (test_no0_idxs), 'valid test samples without 0s.')
  
  # x_no01 = x_no0[y[idxs_no0] != 1]
  # idxs_no01 = np.arange (len (x_no01))
  # test_no01_idxs = np.random.default_rng().choice (a = idxs_no01, axis = 0,
  #                                                  size = min (100, len (x_no01)))
  # print ('Selected', len (test_no01_idxs), 'valid test samples with neither 0s nor 1s.')
  
  yhalf = (np.max(y) - np.min(y)) // 2
  yy = np.array (y)
  idxs_half = np.flatnonzero (np.array (yy > yhalf))
  test_half_idxs = rng.choice (a = idxs_half, axis = 0, size = min (test_size, len (idxs_half)))
  print ('Selected', len (test_half_idxs), 'valid half-test samples.')

# 12357
# 0235689
  idxs_nobar = np.flatnonzero (np.array ((yy != 1) & (yy != 4) & (yy != 7)))
  test_nobar_idxs = rng.choice (a = idxs_nobar, axis = 0, size = min (test_size, len (idxs_nobar)))
  print ('Selected', len (test_nobar_idxs), 'valid nobar-test samples.')

  train_acts = analyzer.eval_batch (x[train_idxs], allow_input_layer = True)
  train_acts = { j: train_acts[j] for j in test_object.layer_indices }

  test_acts = analyzer.eval_batch (x[test_idxs], allow_input_layer = True)
  test_acts = { j: test_acts[j] for j in test_object.layer_indices }

  def runit (allstats, n_feats, discr, bnzf):
    (n_feats_name, n_feats, tech, ncomps), (discr_name, discr, nbins) = n_feats, discr
    
    setup_layer = (lambda l, i, **kwds: abstract_layer_setup (l, i, n_feats, discr))
    cover_layers = get_cover_layers (test_object.dnn, setup_layer,
                                     layer_indices = test_object.layer_indices,
                                     exclude_direct_input_succ = False)
    crit = BFcCriterion (cover_layers, analyzer,
                         print_classification_reports = False,
                         score_layer_likelihoods = False)
    np1 ('Building Bayesian Abstraction')

    tic = time.perf_counter()
    
    crit._discretize_features_and_create_bn_structure (train_acts)
    train_done_time = time.perf_counter()

    crit.fit_activations (train_acts)
    fit_done_time = time.perf_counter()

    # crit.assess_discretized_feature_probas = True
    # crit._score (test_acts)
    score_done_time = time.perf_counter()

    crit.reset ()                       # in case
    crit.add_new_test_cases(x[test_idxs])
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
    crit.add_new_test_cases(x[test_half_idxs])
    bfc_half_cov = crit.bfc_coverage ()
    bfdc_half_cov = crit.bfdc_coverage ()

    crit.reset ()
    crit.add_new_test_cases(x[test_nobar_idxs])
    bfc_nobar_cov = crit.bfc_coverage ()
    bfdc_nobar_cov = crit.bfdc_coverage ()

    cov_done_time = time.perf_counter()

    bn_nodes = crit.N.node_count ()
    bn_edges = crit.N.edge_count ()
    # var_rats = crit.total_variance_ratios_
    # # loglikel = crit.average_log_likelihoods_
    # loglosss = crit.log_losses
    # loglossA = crit.all_log_losses

    thistats = [n_feats_name, discr_name, bn_nodes, bn_edges,
                train_done_time - tic,
                fit_done_time - train_done_time,
                # score_done_time - fit_done_time,
                cov_done_time - score_done_time]
    covstats = [bfc_cov, bfc_half_cov, bfc_nobar_cov,
                # bfc_no0_cov, bfdc_no0_cov,
                # bfc_no01_cov, bfdc_no01_cov,
                bfdc_cov, bfdc_half_cov, bfdc_nobar_cov]
    covstats = [c.as_prop for c in covstats]
    p1 ('{} {} | {:6.2%} {:6.2%} {:6.2%} | {:6.2%} {:6.2%} {:6.2%} |'
        .format(*(n_feats_name, discr_name, )+ tuple (covstats)))
    # p1 ('%s %s %i %i %5.1fs %5.1fs %5.1fs %5.1fs' % (tuple (thistats)))
    p1 ('%s %s %i %i %5.1fs %5.1fs %5.1fs' % (tuple (thistats)))
    allstats += [thistats + [tech, ncomps, nbins] + covstats
                 # + list (var_rats) # + list (loglikel)
                 # + list (np.array(loglosss).flatten ())
                 # + list (np.array(loglossA))
                 ]

    # import gzip
    # with gzip.GzipFile (bnzf, 'w') as f:
    #   f.write (crit.N.to_json ().encode ('utf-8'))
    #   f.close ()

    del crit, cover_layers

  lnames = ("conv2d_1","conv2d_2","conv2d_4","conv2d_4","dense_1","dense_2","dense_3")
  # lvarih = ["var({0})".format (ln) for ln in lnames]
  # llossh = ["minlloss({0})\tmeanlloss({0})\tstdlloss({0})\tmaxlloss({0})"
  #           .format (ln) for ln in lnames]
  # llossA = "minlloss\tmeanlloss\tstdlloss\tmaxlloss"

  def repit (f, allstats):
    np.savetxt (f, np.asarray (allstats),
                header = ("extraction\tdiscretization\tnodes\tedges"+
                          "\tconstruction time (s)"+
                          "\tfitting time (s)"+
                          # "\tscoring time (s)"+
                          "\tcoverage computation time (s)"+
                          "\ttech\tcomponents\tintervals/component"+
                          "\tbfc coverage"+
                          "\tbfc coverage (half)"+
                          "\tbfc coverage (nobar)"+
                          "\tbfdc coverage"+
                          # "\tbfc coverage (no 0s)"+
                          # "\tbfdc coverage (no 0s)"+
                          # "\tbfc coverage (no 0s nor 1s)"+
                          # "\tbfdc coverage (no 0s nor 1s)"+
                          "\tbfdc coverage (half)"+
                          "\tbfdc coverage (nobar)"
                          # "\t" + "\t".join(lvarih) +
                          # # "\tlogl(" + ")\tlogl(".join(lnames) + ")"
                          # "\t" + "\t".join(llossh) +
                          # "\t" + llossA
                          ),
                fmt = '%s',
                # fmt = ['%s', '%s', '%i', '%i', '%f'],
                delimiter="\t")

  def log (n, i, total, n_feats, discr):
    p1 ('[{}: {}/{}] ******** {}, {} *******'.format (n, i, total, n_feats[0], discr[0]))

  base = tname+ '-X' + str(test_size) + '-' + '-'.join((str(i) for i in test_object.layer_indices))
  allstats = []
  total = len (all_n_feats_specs) * len (all_discr_specs)
  i = 1
  for a, b in product (all_n_feats_specs, all_discr_specs):
    log (base, i, total, a, b)
    runit (allstats, a, b, outs + '/{}-{}-{}.json.gz'.format(base, a[0],b[0]))
    repit (outs + '/' + base + '.csv', allstats)
    i += 1

  # allstats = []
  # total = len (all_n_feats_specs) * len (all_discr_specs_extended)
  # i = 1
  # for a, b in product (all_n_feats_specs, all_discr_specs_extended):
  #   log ("basics", i, total, a, b)
  #   runit (allstats, a, b, '/LOCAL/nberth/tmp/fcstats/basics-{}-{}.json.gz'.format(a[0],b[0]))
  #   repit ('/tmp/basics.csv', allstats)
  #   i += 1

  # allstats = []
  # total = len (all_n_feats_specs_extra) * len (all_discr_specs)
  # i = 1
  # for a, b in product (all_n_feats_specs_extra, all_discr_specs):
  #   log ("extra", i, total, a, b)
  #   runit (allstats, a, b, '/LOCAL/nberth/tmp/fcstats/extra-{}-{}.json.gz'.format(a[0],b[0]))
  #   repit ('/tmp/extra.csv', allstats)
  #   i += 1

