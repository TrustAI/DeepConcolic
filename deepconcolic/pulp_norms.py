import numpy as np
from typing import *
from pulp import *
from norms import LInf
from lp import PulpLinearMetric


# ---


class LInfPulp (LInf, PulpLinearMetric):
  """
  L-inf norm to use for :class:`lp.PulpSolver4DNN`.
  """

  def pulp_constrain(self, dist_var, var_names, values,
                     name_prefix = 'input_') -> Sequence[LpConstraint]:
    cstrs = []
    for idx, x in np.ndenumerate (var_names):
      # NB: `vname` is only used for identifying coinstraints
      u = values[idx]
      vname = '_'.join(str(i) for i in (name_prefix,) + idx)

      cstrs.extend([
        # x <= u + d
        LpConstraint(LpAffineExpression([(dist_var, -1), (x, +1)]),
                     LpConstraintLE, rhs = float(u),
                     name = '{}<=x0+{}'.format(vname, self.dist_var_name)),

        # x >= u - d
        LpConstraint(LpAffineExpression([(dist_var, +1), (x, +1)]),
                     LpConstraintGE, rhs = float(u),
                     name = '{}>=x0-{}'.format(vname, self.dist_var_name)),
      ])

    return cstrs

 
# ---
