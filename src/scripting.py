from utils import *
import time

def init_tics ():
  tics = [time.perf_counter ()]
  return (lambda: tics.append (time.perf_counter ()),
          lambda: tuple (t1 - t0 for t0, t1 in zip (tics[:-1], tics[1:])))

def setup_results_file (global_outdir, *field_names):
  rf = global_outdir.stamped_filepath ('results', suff = '.csv')
  write_in_file (rf, '# ', '\t'.join (field_names), '\n')
  return lambda *args: append_in_file (rf, '\t'.join(*args), '\n')
