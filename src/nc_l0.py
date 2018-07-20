
from numpy import linalg as LA
import time
import os
import sys
import numpy as np
from utils import *

def l0_negate(dnn, act_inst, test, nc_layer, pos):
    idx_min = 0
    idx_max = 10

    pixel_num = 2
    mani_range = 100
    adv = 0

    (_, row, col, chl) = np.array(test).shape

