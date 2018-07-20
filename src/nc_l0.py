
from numpy import linalg as LA
import time
import os
import sys
import numpy as np
from utils import *
from l0_encoding import *

def l0_negate(dnn, layer_functions, test, nc_layer, pos):
    idx_min = 0
    idx_max = 10

    gran = 2
    mani_range = 100
    adv = 0

    (row, col, chl) = test[0].shape

    tic=time.time()
    sorted_pixels=sort_pixels(dnn, layer_functions, test[0], nc_layer, pos, gran)
    (act_images, idx_first, success_flag) = accumulate(dnn, layer_functions, test[0], nc_layer, pos, sorted_pixels, mani_range)

    elapsed=time.time()-tic
    #print ('\n == Elapsed time: ', elapsed)

    result=[]
    if success_flag:
      act_image_first=act_images[0]
      refined_act_image=refine_act_image(dnn, layer_functions, test[0], nc_layer, pos, sorted_pixels, act_image_first, idx_first)
      image_diff = np.abs(refined_act_image - test[0])
      L0_distance = (image_diff * 255 > 1).sum()
      L1_distance = image_diff.sum()
      L2_distance = LA.norm(image_diff)
      return True, L0_distance, refined_act_image


    return False, None, None
