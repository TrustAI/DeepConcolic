#!/usr/bin/env python3
from deepconcolic.datasets import load_by_name
from deepconcolic import scripting
from deepconcolic import plotting
from deepconcolic.plotting import plt, subplots, show
from deepconcolic.utils_io import OutputDir, os, sys
from deepconcolic.utils_funcs import as_numpy, np
import argparse

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

ap = argparse.ArgumentParser ()
ap.add_argument ("path", nargs='?')
ap.add_argument ('--features', dest='features', type = int, default = 32,
                 help = 'the number of input features to show (default is 32)')
gp = ap.add_mutually_exclusive_group ()
gp.add_argument ('--samples', type = int, default = None, metavar = 'N',
                 help = 'plot at most N samples with lines')
gp.add_argument ('--samples-only', type = int, default = None, metavar = 'N',
                 help = 'plot at most N samples with lines, and no boxplot')
except_samples_choices = ('raw', 'ok', 'adv',)
ap.add_argument ('--except-samples', nargs='+', default = [],
                 choices = except_samples_choices)
ap.add_argument ('--max-plots-per-fig', type = int, default = 4,
                 help = 'the maximum number of plots per figure (default is 4)')
ap.add_argument ('--max-features-per-plot', type = int, default = 32,
                 help = 'the maximum number of feature to show in each plot '
                 '(default is 32)')
ap.add_argument ("--outputs", '--outdir', '-o', dest = "outdir",
                 help = "the output directory", metavar = "DIR")

args = vars (ap.parse_args())
outdir = OutputDir (args['outdir']) if 'outdir' in args else OutputDir ()
features = args['features']
samples = args['samples'] or args['samples_only']
except_samples = args['except_samples']
boxplots = args['samples_only'] is None
subplots_per_fig = args['max_plots_per_fig']
features_per_subplot = args['max_features_per_plot']

if not boxplots and all (k in except_samples for k in except_samples_choices):
  sys.exit ('Nothing to plot')

# Artificial feature names:
names = ['id'] + [str(i) for i in range (0, 561)]

T_ok, T_adv = (None,) * 2
if args['path'] is not None:
  dirpath = args['path']
  if not os.path.isdir (dirpath):
    sys.exit (f"Argument error: {dirpath} is not a valid directory")
  T = scripting.read_csv (f'{dirpath}/new_inputs.csv', names = names)
  T_ok = np.array([list(l[names[1:]]) for l in T if '-ok-' in l['id']])
  T_ok = T_ok.reshape(-1, 561)
  T_adv = np.array([list(l[names[1:]]) for l in T if '-adv-' in l['id']])
  T_adv = T_adv.reshape(-1, 561)
  print (f'Got {len (T_ok)} correctly classified inputs.')
  print (f'Got {len (T_adv)} adversarial inputs.')

(x_train, y_train), (x_test, y_test), _, kind, class_names = \
    load_by_name ('OpenML:har')
x_train = as_numpy (x_train)

# T_ok = T_ok[:20]
# T_adv = T_adv[:20]
# T_ok = T_ok[:,:20]
# T_adv = T_adv[:,:20]
# X = x_train[:,:20]
X = x_train
# X = X[:, np.argsort (np.min    (X, axis = 0), kind = 'stable')[::-1]]
# X = X[:, np.argsort (np.max    (X, axis = 0), kind = 'stable')[::-1]]
# sidx = np.argsort (np.median (X, axis = 0), kind = 'stable')[::-1]
# sidx = sidx[:features]
sidx = np.arange (features)
X = X[:, sidx]
T_ok = T_ok[:, sidx] if T_ok is not None else None
T_adv = T_adv[:, sidx] if T_adv is not None else None

s_raw = 'raw' not in except_samples
s_ok = 'ok' not in except_samples and T_ok is not None
s_adv = 'adv' not in except_samples and T_adv is not None
Xs, Ts_ok, Ts_adv = (None,) * 3
if samples is not None:
  Xs = X[:min (samples, len (X))] if s_raw else None
  Ts_ok = T_ok[:min (samples, len (T_ok))] if s_ok else None
  Ts_adv = T_adv[:min (samples, len (T_adv))] if s_adv else None

grey_dot = dict (markerfacecolor='grey', marker='.', markersize = .2)
blue_dot = dict (markerfacecolor='blue', marker='.', markersize = .2)
red_dot = dict (markerfacecolor='red', marker='.', markersize = .2)

def boxplot_props (lc, fc, alpha = 1.0, **kwds):
    def dct (k):
        return dict (alpha = alpha, **kwds[k]) if k in kwds else \
               dict (alpha = alpha)
    return dict (patch_artist = True,
                 boxprops = dict (facecolor = fc, color = lc, **dct('boxprops')),
                 capprops = dict (color = lc, **dct('capprops')),
                 whiskerprops = dict (color = lc, **dct('whiskerprops')),
                 flierprops = dict (color = lc, **dct('flierprops')),
                 medianprops= dict (color = lc, **dct('medianprops')))

features_per_fig = features_per_subplot * subplots_per_fig
num_features = X.shape[1]
for feature_index in range (0, num_features, features_per_fig):
  feats = min (features_per_fig, num_features - feature_index)
  num_plots = (feats + features_per_subplot - 1) // features_per_subplot
  fig, ax = subplots (num_plots)
  # fig, ax = subplots (1, num_plots)
  # fig.subplots_adjust (left = 0.04, right = 0.99, hspace = 0.1,
  #                      bottom = 0.03, top = 0.99)
  ax = ax if isinstance (ax, np.ndarray) else [ax]
  for axi, fi in zip (ax, range (feature_index, feature_index + feats,
                                 features_per_subplot)):
    max_fi = min (fi + features_per_subplot, num_features)
    if boxplots:
      axi.boxplot (X[:, fi:max_fi],
                   widths = .2, vert = True,
                   labels = [str (f) for f in range (fi, max_fi)],
                   **boxplot_props ('grey', 'lightgrey',
                                    flierprops = grey_dot))
      if T_ok is not None and len (T_ok) > 0:
        axi.boxplot (T_ok[:, fi:max_fi],
                     widths = .4, vert = True,
                     labels = [''] * (max_fi - fi),
                     showfliers = False,
                     **boxplot_props ('blue', 'lightblue', alpha = .6,
                                      flierprops = blue_dot))
      if T_adv is not None and len (T_adv) > 0:
        axi.boxplot (T_adv[:, fi:max_fi],
                     widths = .5, vert = True,
                     labels = [''] * (max_fi - fi),
                     **boxplot_props ('red', 'lightcoral', alpha = .6,
                                      flierprops = red_dot))

    Xr = np.arange (1, max_fi - fi + 1)
    def plot_lines (X, **_):
      for i in range (len (X)):
        axi.plot (Xr, X[i, fi:max_fi], **_)
    if Xs is not None:
      plot_lines (Xs, linewidth = .3)
    if Ts_ok is not None:
      plot_lines (Ts_ok, color = 'blue')
    if Ts_adv is not None:
      plot_lines (Ts_adv, color = 'red')

  ax[-1].set_xlabel ('input features')
  show (fig, outdir = outdir, basefilename = f'har-{feature_index}')
