import os

# Check conda env
assert \
  os.getenv ('CONDA_DEFAULT_ENV') == 'deepconcolic', \
  "Apparently not executing within the expected conda environment: " \
  "please be sure to run `conda activate deepconcolic' beforehand."

# Check working directory
assert \
  os.path.basename (os.getcwd ()) == 'DeepConcolic' and \
  os.path.isdir (os.path.join (os.getcwd (), '.git')), \
  "Refusing to execute outside of DeepConcolic's source code " \
  f"directory as retrived with `git' (cwd is `{os.getcwd ()}')"
