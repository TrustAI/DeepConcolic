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
ap.add_argument ("path", nargs=1)
ap.add_argument ('--sub-sample', dest='sub_sample', type = int, default = 32)
ap.add_argument ("--outputs", dest = "outputs",
                 help = "the output directory", metavar = "DIR")
args = vars (ap.parse_args())

dirpath = args['path'][0]
if not os.path.isdir (dirpath):
    sys.exit (f"Argument error: {dirpath} is not a valid directory")
outdir = OutputDir (args['outputs']) if 'outputs' in args else OutputDir ()
print (type (outdir))

names = ['id'] + [str(i) for i in range (0, 561)]
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
# sidx = sidx[:args['sub_sample']]
sidx = np.arange (args['sub_sample'])
X = X[:, sidx]
T_ok = T_ok[:, sidx]
T_adv = T_adv[:, sidx]

features_per_subplot = 32
subplots_per_fig = 4

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
        axi.boxplot (X[:, fi:max_fi],
                     widths = .2, vert = True,
                     labels = [str (f) for f in range (fi, max_fi)],
                     **boxplot_props ('grey', 'lightgrey',
                                      flierprops = grey_dot))
        if len (T_ok) > 0:
            axi.boxplot (T_ok[:, fi:max_fi],
                         widths = .4, vert = True,
                         labels = [''] * (max_fi - fi),
                         showfliers = False,
                         **boxplot_props ('blue', 'lightblue', alpha = .6,
                                          flierprops = blue_dot))
        if len (T_adv) > 0:
            axi.boxplot (T_adv[:, fi:max_fi],
                         widths = .5, vert = True,
                         labels = [''] * (max_fi - fi),
                         **boxplot_props ('red', 'lightcoral', alpha = .6,
                                          flierprops = red_dot))
    ax[-1].set_xlabel ('input features')
    show (fig, outdir = outdir, basefilename = f'har-{feature_index}')
