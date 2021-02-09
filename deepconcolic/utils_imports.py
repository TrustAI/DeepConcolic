import importlib, pkgutil

# https://packaging.python.org/guides/creating-and-discovering-plugins/
def iter_namespace (ns_pkg):
  # Specifying the second argument (prefix) to iter_modules makes the
  # returned name an absolute name instead of a relative one. This allows
  # import_module to work without having to do additional modification to
  # the name.
  return pkgutil.walk_packages (ns_pkg.__path__, ns_pkg.__name__ + ".")

def load_submodules (root, modules = None):
  return {
    name: importlib.import_module (name)
    for _finder, name, _ispkg in iter_namespace (root)
    if modules is None or name in modules
  }
