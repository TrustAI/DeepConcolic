from bounds import UniformBounds
from pulp import *
from utils_io import *
from utils import *
from engine import Bounds
from multiprocessing import cpu_count
import pulp
import pulp_encoding
import engine
import numpy as np

# ---


class LpLinearMetric (UniformBounds):
  """
  Basic class to represent any linear metric for LP.
  """

  @property
  def lower_bound (self) -> float:
    return self.low[0]

  @property
  def upper_bound (self) -> float:
    return self.up[0]


# ---


LPProblem = TypeVar('LPProblem')


class LpSolver4DNN:
  """
  Generic LP solver class.
  """

  def setup(self, dnn, build_encoder, link_encoders, create_base_problem,
            input_bounds: Bounds = None, first = 0, upto = None) -> None:
    """
    Constructs and sets up LP problems to encode from layer `first` up
    to layer `upto`.
    """

    layer_encoders, input_layer_encoder, var_names = \
                    link_encoders (dnn, build_encoder, input_bounds, first, upto)
    tp1 ('{} LP variables have been collected.'
         .format(sum(x.size for x in var_names)))
    self.input_layer_encoder = input_layer_encoder
    self.layer_encoders = layer_encoders
    self.base_constraints = create_base_problem (layer_encoders, input_layer_encoder)
    p1 ('Base LP encoding of DNN {}{} has {} variables.'
        .format(dnn.name,
                '' if upto == None else ' up to layer {}'.format(upto),
                sum(n.size for n in var_names)))
    p1 ('Base LP encoding of deepest layer considered involves {} constraints.'
        .format(max(len(p.constraints) for p in self.base_constraints.values())))


  @abstractmethod
  def for_layer(self, cl: engine.CoverableLayer) -> LPProblem:
    """
    Returns an LP problem that encodes up to the given layer `cl`.
    """
    raise NotImplementedError


  @abstractmethod
  def find_constrained_input(self,
                             problem: LPProblem,
                             metric: LpLinearMetric,
                             x: np.ndarray,
                             extra_constrs = [],
                             name_prefix = None) -> Tuple[float, np.ndarray]:
    """
    Augment the given `LP` problem with extra constraints
    (`extra_constrs`), and minimize `metric` against `x`.

    Must restore `problem` to its state upon call before termination.
    """
    raise NotImplementedError


# ---

# PuLP-specific:

pulp_checked_solvers = (
  # 'PYGLPK',
  'CPLEX_PY',
  'CPLEX_DLL',
  'GUROBI',
  'CPLEX_CMD',
  'GUROBI_CMD',
  # 'MOSEK',
  # 'XPRESS',
  'COIN_CMD',
  # 'COINMP_DLL',
  # 'GLPK_CMD',
  # 'CHOCO_CMD',
  # 'PULP_CHOCO_CMD',
  'PULP_CBC_CMD',
  # 'MIPCL_CMD',
  # 'SCIP_CMD',
)

def pulp_find_solver (try_solvers = None, time_limit = None):
  from pulp import apis, __version__ as pulp_version
  print ('PuLP: Version {}.'.format (pulp_version))
  available_solvers = list_solvers (onlyAvailable = True)
  print ('PuLP: Available solvers: {}.'.format (', '.join (available_solvers)))
  time_limit = some (time_limit, 10 * 60)
  args = dict (timeLimit = time_limit, mip = False, msg = False)
  s = None
  for solver in some (try_solvers, pulp_checked_solvers):
    if solver in available_solvers:
      if solver in ('COIN_CMD', 'PULP_CBC_CMD',):
        args['threads'] = min (5, cpu_count ()) # just a rough default
        args['presolve'] = True
      s = get_solver (solver, **args)
      print ('PuLP: {} solver selected (with {} minutes time limit).'
             .format (solver, time_limit / 60))
      break
  return s

def pulp_check (**kwds):
  assert pulp_find_solver (**kwds) is not None


class PulpLinearMetric (LpLinearMetric):
  """
  Any linear metric for the :class:`PulpSolver4DNN`.
  """

  def __init__(self, LB_hard = .01, LB_noise = .01, **kwds):
    '''
    Parameters `LB_hard` and `LB_noise` are used to induce a noise on
    the lower bound for variables of this metric, which is drawn
    between `LB_hard` and `LB_hard + up * LB_noise`; higher values for
    `LB_noice` increase the deviation of the lower bound towards the
    upper bound.

    One must have `low <= LB_hard < up`.

    Setting `LB_noise = 0` removes part of the non-determinism of the
    generation process (which then remains in LP solvers, though).
    '''
    assert 0 <= LB_noise <= 1.
    self.LB_noise = LB_noise
    self.LB_hard = LB_hard
    super().__init__(**kwds)
    assert self.low <= self.LB_hard < self.up


  @property
  def dist_var_name(self):
    return 'd'


  def draw_lower_bound(self, draw = np.random.uniform) -> float:
    '''
    Draw a noisy lower bound.

    The returned bound is drawn between `LB_hard` and `LB_hard + up *
    LB_noise`.  The `draw` function must return a float value that is
    within the two given bounds (:func:`np.random.uniform` by
    default).
    '''
    return draw (self.LB_hard,
                 self.LB_hard + self.upper_bound * self.LB_noise)


  @abstractmethod
  def pulp_constrain(self, dist_var, in_vars, values,
                     name_prefix = 'input_') -> Sequence[LpConstraint]:
    raise NotImplementedError


# ---


class PulpSolver4DNN (LpSolver4DNN):

  def __init__(self, try_solvers = None, time_limit = None, **kwds):
    self.solver = pulp_find_solver (try_solvers = try_solvers,
                                    time_limit = time_limit)
    super().__init__(**kwds)


  def setup(self, dnn,
            metric: PulpLinearMetric,
            input_bounds: Bounds = None,
            postproc_inputs = id,
            build_encoder = pulp_encoding.strict_encoder,
            link_encoders = pulp_encoding.setup_layer_encoders,
            create_problem = pulp_encoding.create_base_problem,
            first = 0, upto = None):
    super().setup (dnn, build_encoder, link_encoders, create_problem,
                   input_bounds, first, upto)
    self.postproc_inputs  = postproc_inputs
    # That's the objective:
    self.d_var = LpVariable(metric.dist_var_name,
                            lowBound = metric.draw_lower_bound (),
                            upBound = metric.upper_bound)
    for _, p in self.base_constraints.items ():
      p += self.d_var


  def for_layer(self, cl: engine.CoverableLayer) -> pulp.LpProblem:
    index = cl.layer_index + (0 if activation_is_relu (cl.layer) else 1)
    return self.base_constraints[index]


  def find_constrained_input(self,
                             problem: pulp.LpProblem,
                             metric: PulpLinearMetric,
                             x: np.ndarray,
                             extra_constrs = [],
                             name_prefix = None):
    in_vars = self.input_layer_encoder.pulp_in_vars ()
    assert (in_vars.shape == x.shape)
    cstrs = extra_constrs
    cstrs.extend (metric.pulp_constrain (self.d_var, in_vars, x,
                                         name_prefix = name_prefix))

    for c in cstrs: problem += c

    # Draw a new distance lower bound:
    self.d_var.lowBound = metric.draw_lower_bound ()

    tp1 ('LP solving: {} constraints'.format(len(problem.constraints)))
    assert (problem.objective is not None)
    problem.solve (self.solver)
    status = LpStatus[problem.status]
    tp1 (status)

    result = None
    if status == 'Optimal':
      res = np.zeros (in_vars.shape)
      for idx, var in np.ndenumerate (in_vars):
        res[idx] = pulp.value (var)
      val = pulp.value(problem.objective)
      result = val, self.postproc_inputs (res)
    else:
      p1 (status)

    for c in cstrs: del problem.constraints[c.name]
    del cstrs, extra_constrs
    return result


# ---
