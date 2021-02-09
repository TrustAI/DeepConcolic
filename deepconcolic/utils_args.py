import argparse
from utils_funcs import some
from utils_io import dir_or_file_in_dir, p1

# ---

def add_named_n_pos_args (parser, posname, flags, required = True, **kwds):
  gp = parser.add_mutually_exclusive_group (required = required)
  gp.add_argument (*flags, dest = posname + '_', **kwds)
  gp.add_argument (posname, nargs = '?', **kwds)

def pp_named_arg (posname):
  def aux (args):
    if hasattr (args, posname + '_'):
      named = getattr (args, posname + '_')
      setattr (args, posname, some (named, getattr (args, posname)))
      delattr (args, posname + '_')
  return aux

def make_select_parser (descr, posname, choices, with_flag = False, **kwds):
  ap = argparse.ArgumentParser (description = descr)
  if with_flag:
    add_named_n_pos_args (ap, posname, (f'--{posname}',),
                          choices = choices.keys (),
                          help = "selected option", **kwds)
  else:
    ap.add_argument (posname, choices = choices.keys (),
                     help = "selected option", **kwds)
  def aux (pp_args = (), **args):
    argsx = vars (ap.parse_args ()) if args is {} or posname not in args else {}
    args = dict (**args, **argsx)
    # pp_args = tuple (pp_named_arg (posname)) + pp_args
    for pp in pp_args: pp (args)
    choices [args[posname]] (**args)
  return ap, aux

def add_verbose_flags (parser, help = 'be more verbose'):
  parser.add_argument ('--verbose', '-v', action = 'store_true', help = help)

# ---

def add_workdir_arg (parser):
  add_named_n_pos_args (parser, 'workdir', ('--workdir', '-d'),
                        type = str, metavar = 'DIR',
                        help = 'work directory')

pp_workdir_arg = pp_named_arg ('workdir')

# ---

def add_abstraction_arg (parser, posname = 'abstraction', short = '-a',
                         help = 'file or directory where the abstraction '
                         '(`abstraction.pkl\' by default) is to be found or '
                         'saved', **kwds):
  add_named_n_pos_args (parser, posname, (f'--{posname}', short),
                        type = str, metavar = 'PKL', help = help, **kwds)

def pp_abstraction_arg (posname = 'abstraction'):
  return pp_named_arg (posname)

abstraction_path = \
  dir_or_file_in_dir ('abstraction.pkl', '.pkl')

# ---
