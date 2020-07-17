from typing import *
from time import time
from norms import L0
from engine import Input
from nc import NcAnalyzer, NcTarget
from l0_encoding import L0Analyzer
import numpy as np


class NcL0Analyzer (NcAnalyzer, L0Analyzer):
  """
  Neuron-cover analyzer that is dedicated to find close inputs w.r.t
  L0 norm.
  """

  def __init__(self, l0_args = {}, **kwds):
    super().__init__(**kwds)
    self.norm = L0 (**l0_args)


  def input_metric(self) -> L0:
    return self.norm


  def search_input_close_to(self, x: Input, target: NcTarget) -> Optional[Tuple[float, Any]]:
    mani_range = 100

    tic = time()
    sorted_pixels = self.sort_pixels (x, target)
    act_images, idx_first, success_flag = self.accumulate (x, target, sorted_pixels, mani_range)
    elapsed = time() - tic
    #print ('\n == Elapsed time: ', elapsed)

    if success_flag:
      refined_act_image = self.refine_act_image (x, target, sorted_pixels, act_images[0], idx_first)
      image_diff = np.abs (refined_act_image - x)
      L0_distance = (image_diff * 255 > 1).sum()
      # L1_distance = image_diff.sum()
      # L2_distance = LA.norm (image_diff)
      return L0_distance, refined_act_image

    return None

