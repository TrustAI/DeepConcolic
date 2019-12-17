import numpy as np
from pulp import *
from norms import LInf
from lp import PulpLinearMetric


# ---


class LInfPulp (LInf, PulpLinearMetric):

  def __init__(self, **kwds):
    super().__init__(**kwds)


  def pulp_constrain(self, problem, dist_var, var_names, values,
                     name_prefix = 'input_activations_constraint'):
    for idx, x in np.ndenumerate (var_names):
      x0 = values[idx]
      var = '_'.join(str(i) for i in (name_prefix,) + idx)

      # x <= x0 + d
      problem += LpConstraint(LpAffineExpression([(dist_var, -1), (x, +1)]),
                              LpConstraintLE, rhs = float(x0),
                              name = '{}<=x0+{}'.format(var, self.dist_var_name))
        
      # x >= x0 - d
      problem += LpConstraint(LpAffineExpression([(dist_var, +1), (x, +1)]),
                              LpConstraintGE, rhs = float(x0),
                              name = '{}>=x0-{}'.format(var, self.dist_var_name))

      # x<=1
      problem += LpConstraint(LpAffineExpression([(x, +1)]),
                              LpConstraintLE, rhs = float(self.UB),
                              name = '{}<=ub'.format(var))

      # x>=0
      problem += LpConstraint(LpAffineExpression([(x, +1)]),
                              LpConstraintGE, rhs = float(self.LB),
                              name = '{}>=lb'.format(var))

 
# ---
