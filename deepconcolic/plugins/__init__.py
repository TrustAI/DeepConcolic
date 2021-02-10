import sys
from utils_io import p1
from utils_imports import load_submodules
modules = load_submodules (sys.modules[__name__])
for m in modules:
  p1 (f'Found and registered plugin `{m}\'')
