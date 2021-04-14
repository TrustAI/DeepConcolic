import sys, os
from utils_io import p1, tp1
from utils_imports import load_submodules

_verbose = 1
try:
  _verbose = int (os.getenv ('DC_PLUGINS_VERBOSE', default = '1'))
except ValueError:
  pass

_log = p1 if _verbose > 0 else tp1 if _verbose == 0 else (lambda *_: ())

_path = os.getenv ('DC_PLUGINS_PATH') or None
_path = _path.split (':') if _path else None
if _path is not None:
  # Yet another insecure system path extension:
  sys.path.extend (_path)

if _path is None:
  _log (f'Looking for plugins in `dc_plugins\' directory')
  modules = load_submodules (['dc_plugins'], prefix = 'dc_plugins.',
                            onerror = (lambda *_: ()))
else:
  modules = load_submodules (_path)

for m in modules:
  _log (f'Found and registered plugin `{m}\'')
