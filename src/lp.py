from abc import abstractmethod
from typing import *
import pulp
import pulp_encoding
from pulp import *
from utils import tp1, p1, activation_is_relu
import numpy as np


# ---


class LpLinearMetric:

  def __init__ (self, LB = 0.0, UB = 1.0, **kwds):
    self.LB = LB
    self.UB = UB
    super().__init__(**kwds)


  @property
  def lower_bound (self):
    return self.LB


  @property
  def upper_bound (self):
    return self.UB


# ---


class LpSolver4DNN:

  def __init__(self, setup_layer_encoders, create_base_problem, lp_dnn = None, first = 0, upto = None, **kwds):
    super().__init__(**kwds)
    dnn = lp_dnn

    layer_encoders, input_layer_encoder, var_names = setup_layer_encoders (dnn, first, upto)
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
  def for_layer(self, cl):
    raise NotImplementedError


  @abstractmethod
  def find_constrained_input(self, problem,
                             metric: LpLinearMetric,
                             x: np.ndarray,
                             name_prefix = None) -> Tuple[float, np.ndarray]:
    raise NotImplementedError


# ---


class PulpLinearMetric (LpLinearMetric):

  def __init__(self, LB_noise = 255, **kwds):
    self.LB_noise = LB_noise
    super().__init__(**kwds)


  @property
  def dist_var_name(self):
    return 'd'

    
  def draw_lower_bound(self):
    return np.random.uniform (self.LB / self.LB_noise, self.UB / self.LB_noise)


  @abstractmethod
  def pulp_constrain(self, problem, dist_var, in_vars, values,
                     name_prefix = 'input_activations_constraint') -> None:
    raise NotImplementedError


# ---


PulpVarMap = NewType('PulpVarMap', Sequence[np.ndarray])


class PulpSolver4DNN (LpSolver4DNN):

  def __init__(self, **kwds):
    if solvers.CPLEX ().available ():
      self.solver = CPLEX (timeLimit = 5 * 60, msg = False)
      print ('PuLP: CPLEX backend selected (with 5 hours time limit).')
    elif solvers.GLPK ().available ():
      self.solver = GLPK ()
      print ('PuLP: GLPK backend selected.')
      print ('PuLP: WARNING: GLPK does not support time limit.')
    else:
      self.solver = None
      print ('PuLP: CBC backend selected.')
      print ('PuLP: WARNING: CBC does not support time limit.')

    super().__init__(pulp_encoding.setup_layer_encoders,
                     pulp_encoding.create_base_problem, **kwds)


  def for_layer(self, cl) -> Tuple[pulp.LpProblem, PulpVarMap]:
    index = cl.layer_index + (0 if activation_is_relu (cl.layer) else 1)
    return self.base_constraints[index].copy ()


  def find_constrained_input(self, problem: pulp.LpProblem,
                             metric: PulpLinearMetric,
                             x: np.ndarray,
                             name_prefix = 'x_0_0'):

    d_var = LpVariable(metric.dist_var_name,
                       lowBound = metric.draw_lower_bound (),
                       upBound = metric.upper_bound)
    problem += d_var

    in_vars = self.input_layer_encoder.pulp_in_vars ()
    assert (in_vars.shape == x.shape)
    metric.pulp_constrain (problem, d_var, in_vars, x, name_prefix)

    tp1 ('LP solving: {} constraints'.format(len(problem.constraints)))
    problem.solve (self.solver)
    tp1 ('Solved!')

    if LpStatus[problem.status] != 'Optimal':
      return None

    res = np.zeros(in_vars.shape)
    for idx, var in np.ndenumerate (in_vars):
      res[idx] = pulp.value (var)

    return pulp.value(problem.objective), res


# ---
