import argparse
import sys
from datetime import datetime

import keras
from keras.models import *
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras import *


from utils import *

try:
  import art as fgsm
except:
  from attacks import *

def ssc_search(test_object, layer_functions, cond_layer, cond_pos, dec_layer, dec_pos):

  data=test_object.raw_data.data
  xs=np.random.choice(len(data), 50)

  for x in xs:
    x_acts=eval_batch(layer_functions, [data[x]])
    cond_flags=np.ones(x_acts[cond_layer.layer_index][0].shape, dtype=bool)
    dec_flag=None
    for i in range(0, cond_flags.size):
      if x_acts[cond_layer.layer_index][0].item(i)<=0:
        cond_flags.itemset(i, False)
    if x_acts[dec_layer.layer_index][0].item(dec_pos)>0:
      dec_flag=True
    else:
      dec_flag=False
    print (cond_layer.layer_index, dec_layer.layer_index)
    pass

  return -1, False, None, None

