import sys, os, datetime, cv2, warnings, parse, tempfile
import numpy as np
from utils_funcs import random

# ---

COLUMNS = os.getenv ('COLUMNS', default = '80')
P1F = '{:<' + COLUMNS + '}'
N1F = '\n{:<' + COLUMNS + '}'

def tp1(x):
  print (P1F.format(x), end = '\r', flush = True)

def ctp1(x):
  print (N1F.format(x), end = '\r', flush = True)

def np1(x):
  print (x, end = '', flush = True)

def cnp1(x):
  print ('\n', x, sep = '', end = '', flush = True)

def p1(x, **k):
  print (P1F.format(x), **k)

def c1(x):
  print (x)

def cp1(x, **k):
  print (N1F.format(x), **k)

# ---

def hr(c = '-', **k):
  print (''.join ([c * int (COLUMNS)]), **k)

def h1(title, c = '=', title_head = 3):
  hr (c, end = f'\r{c * title_head}  {title}  \n')

def h2(title, c = '-', title_head = 3):
  hr (c, end = f'\r{c * title_head}  {title}  \n')

# ---

def s_(i):
  return i, 's' if i > 1 else ''

def is_are_(i):
  return i, 'are' if i > 1 else 'is'

# ---

tempdir = tempfile.gettempdir ()

def setup_output_dir (outs, log = True):
  if not os.path.exists (outs):
    if log: print (f'Creating output directory: {outs}')
    os.makedirs (outs)
  return outs

def dir_or_file_in_dir (default_filename, suff):
  def aux (f, filename = default_filename):
    dirname = os.path.dirname (f) if f.endswith (suff) else f
    if dirname is not None:
      setup_output_dir (dirname)
    if f.endswith (suff):
      return f
    return os.path.join (f, f'{filename}')
  return aux

class OutputDir:
  '''
  Class to help ensure output directory is created before starting any
  lengthy computations.
  '''
  def __init__(self, outs = tempdir, log = None,
               enable_stamp = True, stamp = None, prefix_stamp = False):
    self.dirpath = setup_output_dir (outs, log = log)
    self.enable_stamp = enable_stamp
    self.prefix_stamp = prefix_stamp
    self.reset_stamp (stamp = stamp)

  def reset_stamp (self, stamp = None):
    self.stamp = datetime.datetime.now ().strftime("%Y%m%d-%H%M%S") \
                 if stamp is None and self.enable_stamp else \
                 stamp if self.enable_stamp else ''

  @property
  def path(self) -> str:
    return self.dirpath

  def filepath(self, base, suff = '') -> str:
    return os.path.join (self.dirpath, base + suff)

  def stamped_filename(self, base, sep = '-', suff = '') -> str:
    return ((self.stamp + sep + base) if self.enable_stamp and self.prefix_stamp else \
            (base + sep + self.stamp) if self.enable_stamp else \
            (base)) + suff

  def stamped_filepath(self, *args, **kwds) -> str:
    return os.path.join (self.dirpath, self.stamped_filename (*args, **kwds))

  def subdir(self, name) -> str:
    dirname = self.filepath (name)
    if not os.path.exists (dirname):
      os.makedirs (dirname)
    return dirname

  def fresh_dir(self, basename, suff_fmt = '-{:x}', **kwds):
    outdir = self.filepath (basename + suff_fmt.format (random.getrandbits (16)))
    try:
      os.makedirs (outdir)
      return OutputDir (outdir, **kwds)
    except FileExistsError:
      return self.fresh_dir (basename, suff_fmt = suff_fmt, **kwds)

# ---

def _write_in_file (f, mode, *fmts):
  f = open (f, mode)
  for fmt in fmts: f.write (fmt)
  f.close ()

def write_in_file (f, *fmts):
  _write_in_file (f, "w", *fmts)

def append_in_file (f, *fmts):
  _write_in_file (f, "a", *fmts)

def save_in_csv (filename):
  def save_an_array (arr, name, directory = '.', log = True):
    f = os.path.join (directory, filename + '.csv')
    if log: p1 (f'Appending array into `{f}\'')
    with open (f, 'a') as file:
      file.write (name + ' ')
      np.savetxt (file, arr, newline = ' ')
      file.write ('\n')
  return save_an_array

def save_an_image(im, name, directory = '.', log = True, channel_upscale = 255):
  f = os.path.join (directory, name + '.png')
  if log: p1 (f'Outputing image into `{f}\'')
  cv2.imwrite (f, im * channel_upscale)

def save_an_image_(channel_upscale = 255):
  return lambda *args, **kwds: \
    save_an_image (*args, channel_upscale = channel_upscale, **kwds)

def save_adversarial_examples(adv, origin, diff, di):
  save_an_image(adv[0], adv[1], di)
  save_an_image(origin[0], origin[1], di)
  if diff is not None:
    save_an_image(diff[0], diff[1], di)
