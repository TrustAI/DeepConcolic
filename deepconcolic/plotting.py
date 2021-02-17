from utils_io import OutputDir, tp1
import os

mpl_backend = os.getenv ('DC_MPL_BACKEND')
mpl_fig_width = float (os.getenv ('DC_MPL_FIG_WIDTH', default = 7.))
mpl_fig_ratio = float (os.getenv ('DC_MPL_FIG_RATIO', default = 1))
mpl_fig_pgf_width = float (os.getenv ('DC_MPL_FIG_PGF_WIDTH', default = 4.7))

default_params = {
  'font.size': 11,
  'font.family': 'cmr10',
  'axes.unicode_minus': False,
}

png = mpl_backend in ('png', 'PNG')
png_default_figsize = (mpl_fig_width, mpl_fig_width * mpl_fig_ratio)

pgf = mpl_backend in ('pgf', 'PGF')
pgf_output_pgf = True
pgf_output_pdf = True
pgf_default_figsize = (mpl_fig_pgf_width, mpl_fig_pgf_width * mpl_fig_ratio)
pgf_default_params = {
  'font.size': 8,
  'font.family': 'cmr10',               # lmodern
  'text.usetex': True,
  'axes.linewidth': .5,
  'axes.unicode_minus': True,           # fix mpl bug in 3.3.0?
  'lines.linewidth': .5,
  'lines.markersize': .2,
  'pgf.texsystem': 'pdflatex',
  'pgf.rcfonts': False,        # don't setup fonts from rc parameters
  # "pgf.preamble": [r"\input{../macro}"]
  'pgf.preamble': "\n".join([
    r"\usepackage[utf8x]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{amssymb}",
    r"\usepackage{relsize}",
  ])
}

# ---

try:
  import matplotlib as mpl
  mpl = mpl
except:
  mpl = None
enabled = mpl is not None

if mpl and pgf:
  mpl.use ('pgf')

try:
  import matplotlib.pyplot as plt
  plt = plt
except:
  plt = None

# ---

def generic_setup (**kwds):
  if plt:
    plt.rcParams.update ({ **default_params, **kwds })

def pgf_setup (**kwds):
  if pgf and plt:
    plt.rcParams.update ({ **pgf_default_params, **kwds })

generic_setup ()
pgf_setup ()

# ---

def _def (f):
  def __aux (*args, figsize = None, figsize_adjust = (1.0, 1.0), **kwds):
    if not plt:
      return None
    figsize = figsize or (pgf_default_figsize if pgf else png_default_figsize)
    figsize = tuple (figsize[i] * figsize_adjust[i] for i in (0, 1))
    return f (*args, figsize = figsize, **kwds)
  return __aux

figure = _def (plt.figure)
subplots = _def (plt.subplots)

# import tikzplotlib

def show (fig = None, outdir: OutputDir = None, basefilename = None, **kwds):
  if plt:
    if not pgf and not png:
      plt.show ()
    elif fig is not None and basefilename is not None:
      if not fig.get_constrained_layout ():
        plt.tight_layout (**{**dict(pad = 0, w_pad = 0.1, h_pad = 0.1),
                             **kwds})
      else:
        fig.set_constrained_layout_pads(**{**dict(w_pad = 0.01, h_pad = 0.01,
                                                  hspace=0., wspace=0.),
                                           **kwds})
      outdir = OutputDir () if outdir is None else outdir
      # assert isinstance (outdir, OutputDir)
      if png:
        f = outdir.filepath (basefilename + '.png')
        tp1 ('Outputting {}...'.format (f))
        fig.savefig (f, format='png')
      if pgf and pgf_output_pgf:
        f = outdir.filepath (basefilename + '.pgf')
        tp1 ('Outputting {}...'.format (f))
        fig.savefig (f, format='pgf')
      if pgf and pgf_output_pdf:
        f = outdir.filepath (basefilename + '.pdf')
        tp1 ('Outputting {}...'.format (f))
        fig.savefig (f, format='pdf')

def texttt (s):
  s = s.replace('_', r'\_')
  return (r'\mbox{\smaller\ttfamily ' + s + '}' if pgf else s)

# # Verbatim copy from: https://jwalton.info/Matplotlib-latex-PGF/
# def set_size(width_pt, fraction=1, subplots=(1, 1)):
#     """Set figure dimensions to sit nicely in our document.

#     Parameters
#     ----------
#     width_pt: float
#             Document width in points
#     fraction: float, optional
#             Fraction of the width which you wish the figure to occupy
#     subplots: array-like, optional
#             The number of rows and columns of subplots.
#     Returns
#     -------
#     fig_dim: tuple
#             Dimensions of figure in inches
#     """
#     # Width of figure (in pts)
#     fig_width_pt = width_pt * fraction
#     # Convert from pt to inches
#     inches_per_pt = 1 / 72.27

#     # Golden ratio to set aesthetic figure height
#     golden_ratio = (5**.5 - 1) / 2

#     # Figure width in inches
#     fig_width_in = fig_width_pt * inches_per_pt
#     # Figure height in inches
#     fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

#     return (fig_width_in, fig_height_in)
