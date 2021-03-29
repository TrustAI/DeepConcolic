from utils_io import sys, warnings, p1
from multiprocessing import Process, SimpleQueue, cpu_count, freeze_support
# from multiprocessing import Pool

# NB: multiprocessing.Pool might be workable, but does not give enough
# control over the context and how objects are passed; in particular,
# we want to use Unix's fork with the underlying read-only shared
# memory on platforms where this is possible.

# ---


def worker_process (pid, verbose, mk_func, todo, done, *args):
  if verbose:
    p1 (f'Starting worker {pid}')
  func = mk_func (*args)
  try:
    while True:
      work = todo.get ()
      if work is None: break
      done.put (func (work))
  except KeyboardInterrupt:
    pass
  if verbose:
    p1 (f'Worker {pid} terminating')


class FFPool:

  def __init__(self, mk_func, *args,
               processes = 1,
               verbose = False,
               queue = SimpleQueue):
    """Feed-forward pool of workers that pull work from a shared `todo`
    queue and push results into a shared `done` queue. 

    The queues used behave like pipes as they block readers unless
    some element is already in the queue.

    Take care that `get` must be called as many times as `put` before
    the pool is terminated.  This typically means that each call to
    `put` mush be followed by a corresponding call to `get` before
    method `join` is called

    """
    # Create basic queues:
    todo, done = SimpleQueue (), SimpleQueue ()
  
    pool_size = processes if processes > 0 else max (1, cpu_count () - 1)
    pool = tuple (Process (target = worker_process,
                           args = (pid, verbose, mk_func, todo, done,) + args)
                  for pid in range (pool_size))

    self.verbose = verbose
    self.todo, self.done, self.pool = todo, done, pool
    
  def __len__(self):
    """Returns the number of workers in the pool"""
    return len (self.pool)

  def start (self):
    for p in self.pool: p.start ()

  def put (self, w):
    assert w is not None
    self.todo.put (w)

  def get (self):
    return self.done.get ()

  def join (self):
    if self.verbose:
      p1 ('Waiting for all worker processes to terminate...')
    for p in self.pool: self.todo.put (None)
    for p in self.pool: p.join ()


# ---

from multiprocessing import get_start_method, set_start_method

def init ():
  """To be called right after `if __name__ == '__main__'` as mentioned in [1]

  ...
 
  [1] https://docs.python.org/3/library/multiprocessing.html#multiprocessing.freeze_support
  """
  freeze_support ()      # for specific uses of the program on Windows
  

# ---


def forking ():
  return get_start_method () == 'fork'


def np_share (x):
  if forking ():
    return x

  # sharedmem is broken: just assume x can be pickled, and pay the
  # memory and process startup price:
  return x

  # if any (sys.platform.startswith (pref) for pref in ('linux', 'darwin')):
  #   warnings.warn ('Shared numpy arrays with start method other than `fork\' '
  #                  f'may be broken on {sys.platform}: problems ahead!')

  # from sharedmem import empty
  # print ('Sharing numpy array')
  # xx = empty (x.shape, x.dtype)
  # xx[:] = x[:]
  # print ('done')
  # return xx

