import cplex
import sys
import numpy as np

from utils import *

epsilon=1.0/(255)

class base_constraintst:
  def __init__(self, objective, lower_bounds, upper_bounds, var_names_vect, constraints, constraint_senses, rhs, constraint_names):
    self.obj=objective
    self.lb=lower_bounds
    self.ub=upper_bounds
    self.names=var_names_vect
    self.lin_expr=constraints
    self.senses=constraint_senses
    self.rhs=rhs
    self.names=constraint_names
