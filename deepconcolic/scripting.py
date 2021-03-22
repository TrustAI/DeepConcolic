from utils_io import *
from utils_funcs import some
import time

# ----

def init_tics ():
  tics = [time.perf_counter ()]
  return (lambda: tics.append (time.perf_counter ()),
          lambda: tuple (t1 - t0 for t0, t1 in zip (tics[:-1], tics[1:])))

def setup_results_file (outdir, *field_names):
  rf = outdir.stamped_filepath ('results', suff = '.csv')
  write_in_file (rf, '# ', '\t'.join (field_names), '\n')
  return lambda *args: append_in_file (rf, '\t'.join(*args), '\n')

def read_report (f, ignore_head = 0):
  with open (f, 'r') as f:
    l = f.readlines ()
  tab = []
  for l in l[ignore_head:]:
    l = l.split (maxsplit = 8)
    tab.append ((float (l[1]), int (l[4]), int (l[7])))
  return np.asarray (tab, dtype = [('coverage', 'f8'),
                                   ('#tests', 'i4'),
                                   ('#adversarials', 'i4')])

def read_csv (f, dtype = None, encoding = None, names = True, **kwds):
  return np.genfromtxt (f, dtype = dtype, encoding = encoding,
                        names = names, **kwds)

try:
  import parse
  def gather_all_reports (d, dirname_pattern, entry_dtype, ignore_head = None):
    p = parse.compile (dirname_pattern)
    res = []
    for dir, dirs, files in os.walk (d):
      for f in files:
        if f.endswith('_report.txt'):
          infos = p.parse (os.path.basename (os.path.abspath (dir))).named
          report = read_report (os.path.join (dir, f),
                                ignore_head = ignore_head)
          r = os.path.join (dir, f.replace ('_report.txt', '_progress.csv'))
          progress = read_csv (r) if os.path.exists (r) else np.array([])
          res.append (tuple (infos[k] for k in infos) + (report, progress,))
    return np.asarray (res, dtype = entry_dtype + [('report', 'O'),
                                                   ('progress', 'O')])
except:
  # parse not available
  pass
