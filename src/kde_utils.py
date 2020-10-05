import plotting
from plotting import plt
from typing import *
from utils import *
import scipy.signal as sig
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

sN = 300

class KDESplit:
  def __init__(self,
               dip_space = 'dens',
               dip_prominence_prop = 1/40,
               topline_density_prop = 2/3,
               baseline_density_prop = 1/20,
               bandwidth_prop = 2/100,
               min_width = 1e-8,
               plot_spaces = None,
               plot_dip_markers = True,
               plot_rel_dip_height = 1.0, # only influences plotting
               n_jobs = None,
               **kwds):
    self.dip_space = some (dip_space, 'dens')
    self.dip_prominence_prop = some (dip_prominence_prop, 1/40)
    self.topline_density_prop = some (topline_density_prop, 2/3)
    self.baseline_density_prop = some (baseline_density_prop, 1/20)
    self.bandwidth_prop = some (bandwidth_prop, 2/100)
    self.min_width = some (min_width, 1e-8)
    self.plot_spaces = seqx (plot_spaces)
    self.plot_rel_dip_height = some (plot_rel_dip_height, 1.0)
    self.plot_dip_markers = some (plot_dip_markers, True)
    self.padding_prop = 0.01
    self.n_jobs = n_jobs
    self._check ()
    super ().__init__(**kwds)
    self.pltdata_ = {}

  def _check (self):
    self.validate_space ('dip_space', self.dip_space)
    assert (self.dip_prominence_prop >= 0.0 and
            self.dip_prominence_prop < 1.0)
    assert (self.topline_density_prop > 0.0 and
            self.topline_density_prop < 1.0)
    assert (self.baseline_density_prop > 0.0 and
            self.baseline_density_prop < 1.0)
    assert (self.baseline_density_prop < self.topline_density_prop)
    assert (self.bandwidth_prop > 0.0 and
            self.bandwidth_prop <= 1.0)
    assert (self.min_width > 0.0)
    for s in self.plot_spaces:
      self.validate_space ('plot_space', s)
    assert (self.n_jobs is None or
            self.n_jobs != 0)

  @staticmethod
  def validate_space (v, s):
    return validate_strarg (('dens', 'logl'),
                            'KDE space specification') (v, s)

  def fit_split (self, yy, bandwidth = None):
    # Add some padding around yy's range
    yymin, yymax = np.amin (yy), np.amax (yy)
    sD = yymax - yymin
    yymin, yymax = ((yymin, yymax) if self.padding_prop is None else
                    (yymin - sD * self.padding_prop,
                     yymax + sD * self.padding_prop))
    sD = yymax - yymin

    # Basic linear transformation into padded space
    s = np.linspace (yymin, yymax, num = sN)
    sA = lambda x: x * (s[-1] - s[0]) / (sN - 1)
    sL = lambda x: s[0] + sA (x)

    # NB: Good enough for now, but not robust to small/medium-sized
    # samples.  An idea may be k-fold + average the density
    # estimation (recall we don't need it to be normalized). Or
    # probably even better: make multiple splits with k ShuffleSplit
    # samples, cluster the proposed splits, and take averages.
    if bandwidth is None:
      grid = GridSearchCV (KernelDensity (kernel = 'gaussian'),
                           {'bandwidth': np.logspace (-1.5, .5, 8) * sD},
                           n_jobs = self.n_jobs)
      grid.fit (yy.reshape (-1, 1))
      kde = grid.best_estimator_
      # p1 ('Best bandwidth for feature {}: {}'.format(fi, kde.bandwidth))
    else:
      kde = KernelDensity (kernel = 'gaussian', bandwidth = bandwidth)
      kde.fit (yy.reshape (-1, 1))
    # tp1 ('Bandwidth for feature {}: {}'.format(fi, sD * self.bandwidth_prop))
    # kde = KernelDensity (kernel = 'gaussian', bandwidth = sD * self.bandwidth_prop)
    self.kde_ = kde
    self.bandwidth_ = kde.bandwidth

    id = lambda x: x
    logl2cspace, cspace2dens, cspace2logl, dens2cspace = ( \
        (id,     np.exp, id,      np.log) if self.dip_space == 'logl' else
        (np.exp, id, np.log, np.negative))
    l = kde.score_samples (s.reshape(-1, 1))
    p = logl2cspace (l)

    # Find dips.

    # prominence = np.amax (np.abs (p)) * self.dip_prominence_prop
    maxdens = np.amax (cspace2dens (p))
    prominence = np.abs (dens2cspace (maxdens) * self.dip_prominence_prop)
    topline_density = maxdens * self.topline_density_prop
    baseline_density = maxdens * self.baseline_density_prop
    height = (dens2cspace (topline_density),
              dens2cspace (baseline_density)) if self.dip_space != 'logl' \
              else None
    dips, properties = sig.find_peaks ( \
        - p,
        width = sN * self.bandwidth_prop, # another param for min width?
        distance = sN * self.bandwidth_prop,
        prominence = prominence,
        height = height,
        rel_height = self.plot_rel_dip_height)
    splits = sL (dips)

    # Consider what's below baseling density: the goal here is to
    # find very sparsely populated ranges. This is done by finding
    # large-enough gaps below `-log(baseline_density)` (in
    # log-likelihood space).
    #
    # NB: Could re-use `find_peaks` as above with a `plateau_size`
    # argument.
    plateau_size = sN * self.bandwidth_prop * 2
    far_enough = lambda xp, refs: \
        np.all (np.abs (np.array (refs) - xp) >= plateau_size)
    baseline_crossings = \
        np.where (np.diff (np.sign (np.hstack(([-np.inf], l[1:-1], [-np.inf])) -
                                    np.log (baseline_density))))[0]
    extra_splits = [ xp for xp in (baseline_crossings[0], baseline_crossings[-1]) ]
    for i in range (1, len (baseline_crossings) - 2, 2):
      xpi, xpj = baseline_crossings[i], baseline_crossings[i+1]
      if far_enough (xpi, xpj):
        extra_splits.extend ((xpi, xpj))
    extra_splits = sL (np.array (extra_splits, dtype = int))

    # Gather all dips and extras
    splits = np.hstack((splits, extra_splits))

    # Sort splits
    splits.sort ()

    # Filter-out bins whose width is too small
    mask = np.ediff1d (splits, to_begin = np.inf) > self.min_width
    self.splits_ = splits[mask]

    for plot_space in self.plot_spaces:
      pltspace, ylabel = \
          (cspace2logl, 'log-likelihood') if plot_space == 'logl' else \
          (cspace2dens, 'density')

      self.pltdata_[plot_space] = \
          (self.splits_, s, p, dips, pltspace, ylabel,
           properties["prominences"], properties["width_heights"],
           sL (properties["left_ips"]), sL (properties["right_ips"]))


  def plot_splits (self, ax, plot_space):
    if plot_space not in self.pltdata_:
      return None

    bin_edges, s, p, dips, pltspace, ylabel, prominences, width_heights, \
      left_ips, right_ips = self.pltdata_[plot_space]

    pp = pltspace (p)
    ymin, ymax = np.amin (pp), np.amax (pp)
    ax.vlines (x = bin_edges,
               ymin = min (0., ymin),
               ymax = max (0., ymax),
               linestyles = 'dashed')
    ax.plot (s, pp)
    ax.vlines (x = s[dips],
               ymin = 0,
               ymax = pltspace (p[dips]))
    if self.plot_dip_markers:
      ax.vlines (x = s[dips], color = "r",
                 ymin = pltspace (p[dips] + prominences),
                 ymax = pltspace (p[dips]))
      ax.hlines (y = pltspace (-width_heights), color = "r",
                 xmin = left_ips, xmax = right_ips)
    return {'ymin': ymin, 'ymax': ymax}


  @staticmethod
  def setup_plot_style ():
    # https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html?highlight=pgf.preamble
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
      'xaxis.labellocation': 'right',
      'lines.markersize': 1.5,
      'lines.linewidth': 2,
    })
    plotting.pgf_setup (**{
      'ytick.labelsize': 'xx-small',
      'ytick.major.size': 2,
      'ytick.major.width': .2,
      'ytick.major.pad': 2,
      'ytick.direction': 'in',
      'xtick.labelsize': 'xx-small',
      'xtick.major.size': 2,
      'xtick.major.width': .2,
      'xtick.major.pad': 2,
      'axes.labelsize': 'small',
      'lines.markersize': .2,
      'lines.linewidth': .8,
    })

