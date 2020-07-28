from abc import abstractmethod
from typing import *
import pulp
import pulp_encoding
import engine
from pulp import *
from utils import *
import numpy as np


# ---


class LpLinearMetric:
  """
  Basic class to represent any linear metric.
  """  

  def __init__ (self, LB = 0.0, UB = 1.0, **kwds):
    self.LB = LB
    self.UB = UB
    super().__init__(**kwds)


  @property
  def lower_bound (self) -> float:
    return self.LB


  @property
  def upper_bound (self) -> float:
    return self.UB


# ---


LPProblem = TypeVar('LPProblem')


class LpSolver4DNN:
  """
  Generic LP solver class.
  """

  def setup(self, dnn, build_encoder, link_encoders, create_base_problem,
            first = 0, upto = None) -> None:
    """
    Constructs and sets up LP problems to encode from layer `first` up
    to layer `upto`.
    """

    layer_encoders, input_layer_encoder, var_names = link_encoders (dnn, build_encoder, first, upto)
    tp1 ('{} LP variables have been collected.'
         .format(sum(x.size for x in var_names)))
    self.input_layer_encoder = input_layer_encoder
    self.layer_encoders = layer_encoders
    base_constraints = create_base_problem (layer_encoders, input_layer_encoder)
    self.base_constraints = base_constraints
    p1 ('Base LP encoding of DNN {}{} has {} variables.'
        .format(dnn.name,
                '' if upto == None else ' up to layer {}'.format(upto),
                sum(n.size for n in var_names)))
    p1 ('Base LP encoding of deepest layer considered involves {} constraints.'
        .format(max(len(p.constraints) for p in self.base_constraints.values())))


  @abstractmethod
  def for_layer(self, cl: engine.CL) -> LPProblem:
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


class PulpLinearMetric (LpLinearMetric):
  """
  Any linear metric for the :class:`PulpSolver4DNN`.
  """  

  def __init__(self, LB_noise = 255, **kwds):
    self.LB_noise = LB_noise
    super().__init__(**kwds)


  @property
  def dist_var_name(self):
    return 'd'

    
  def draw_lower_bound(self):
    return np.random.uniform (self.LB / self.LB_noise, self.UB / self.LB_noise)


  @abstractmethod
  def pulp_constrain(self, dist_var, in_vars, values,
                     name_prefix = 'input_activations_constraint') -> Sequence[LpConstraint]:
    raise NotImplementedError


# ---


PulpVarMap = NewType('PulpVarMap', Sequence[np.ndarray])


class PulpSolver4DNN (LpSolver4DNN):

  def __init__(self, **kwds):
    # TODO: parameterize with a list of solvers, in order of
    # preference...
    from pulp import apis, __version__ as pulp_version
    print ('PuLP: Version {}.'.format (pulp_version))
    solvers = list_solvers (onlyAvailable = True)
    print ('PuLP: Available solvers: {}.'.format (', '.join (solvers)))
    if 'CPLEX_PY' in solvers:
      self.solver = CPLEX_PY (timeLimit = tl, msg = False)
      print ('PuLP: CPLEX_PY backend selected (with 10 minutes time limit).')
    elif 'PYGLPK' in solvers:
      self.solver = PYGLPK (timeLimit = tl, mip = False, msg = False)
      print ('PuLP: PYGLPK backend selected (with 10 minutes time limit).')
    elif 'GUROBI' in solvers:
      self.solver = GUROBI (timeLimit = tl, mip = False, msg = False)
      print ('PuLP: GUROBY backend selected (with 10 minutes time limit).')
    elif 'CPLEX' in solvers:
      self.solver = CPLEX (timeLimit = tl, msg = False)
      print ('PuLP: CPLEX backend selected (with 10 minutes time limit).')
    elif 'GLPK' in solvers:
      self.solver = GLPK ()
      print ('PuLP: GLPK backend selected.')
      print ('PuLP: WARNING: GLPK does not support time limit.')
    elif 'GUROBI_CMD' in solvers:
      self.solver = GUROBI_CMD ()
      print ('PuLP: GUROBI_CMD backend selected.')
      print ('PuLP: WARNING: GUROBI_CMD does not support time limit.')
    else:
      self.solver = PULP_CBC_CMD (msg = False)
      print ('PuLP: CBC backend selected.')
      print ('PuLP: WARNING: CBC does not support time limit.')
    # Missing: SCIP, MOSEK, XPRESS, YAPOSIB.

    super().__init__(**kwds)


  def setup(self, dnn, metric: PulpLinearMetric,
            build_encoder = pulp_encoding.strict_encoder,
            link_encoders = pulp_encoding.setup_layer_encoders,
            create_problem = pulp_encoding.create_base_problem,
            first = 0, upto = None):
    super().setup (dnn, build_encoder, link_encoders, create_problem, first, upto)
    # That's the objective:
    self.d_var = LpVariable(metric.dist_var_name,
                            lowBound = metric.draw_lower_bound (),
                            upBound = metric.upper_bound)
    for _, p in self.base_constraints.items ():
      p += self.d_var


  def for_layer(self, cl: engine.CL) -> pulp.LpProblem:
    index = cl.layer_index + (0 if activation_is_relu (cl.layer) else 1)
    return self.base_constraints[index]


  def find_constrained_input(self,
                             problem: pulp.LpProblem,
                             metric: PulpLinearMetric,
                             x: np.ndarray,
                             extra_constrs = [],
                             name_prefix = 'x_0_0'):
    in_vars = self.input_layer_encoder.pulp_in_vars ()
    assert (in_vars.shape == x.shape)
    cstrs = extra_constrs
    cstrs.extend(metric.pulp_constrain (self.d_var, in_vars, x, name_prefix))

    for c in cstrs: problem += c

    ctp1 ('LP solving: {} constraints'.format(len(problem.constraints)))
    assert (problem.objective is not None)
    problem.solve (self.solver)
    tp1 ('Solved!')

    result = None
    if LpStatus[problem.status] == 'Optimal':
      res = np.zeros (in_vars.shape)
      for idx, var in np.ndenumerate (in_vars):
        res[idx] = pulp.value (var)
      val = pulp.value(problem.objective)
      result = val, res

    for c in cstrs: del problem.constraints[c.name]
    del cstrs, extra_constrs
    return result


# ---
