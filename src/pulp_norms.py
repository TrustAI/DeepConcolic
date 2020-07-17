from typing import *
import numpy as np
from pulp import *
from norms import LInf
from lp import PulpLinearMetric


# ---


class LInfPulp (LInf, PulpLinearMetric):
  """
  L-inf norm to use for :class:`lp.PulpSolver4DNN`.
  """

  def pulp_constrain(self, dist_var, var_names, values,
                     name_prefix = 'input_activations_constraint') -> Sequence[LpConstraint]:
    cstrs = []
    for idx, x in np.ndenumerate (var_names):
      x0 = values[idx]
      var = '_'.join(str(i) for i in (name_prefix,) + idx)

      cstrs.extend([
        # x <= x0 + d
        LpConstraint(LpAffineExpression([(dist_var, -1), (x, +1)]),
                     LpConstraintLE, rhs = float(x0),
                     name = '{}<=x0+{}'.format(var, self.dist_var_name)),
        
        # x >= x0 - d
        LpConstraint(LpAffineExpression([(dist_var, +1), (x, +1)]),
                     LpConstraintGE, rhs = float(x0),
                     name = '{}>=x0-{}'.format(var, self.dist_var_name)),

        # x<=1
        LpConstraint(LpAffineExpression([(x, +1)]),
                     LpConstraintLE, rhs = float(self.UB),
                     name = '{}<=ub'.format(var)),

        # x>=0
        LpConstraint(LpAffineExpression([(x, +1)]),
                     LpConstraintGE, rhs = float(self.LB),
                     name = '{}>=lb'.format(var))
      ])

    return cstrs
 
# ---
