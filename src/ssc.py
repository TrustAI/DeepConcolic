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

def ssc_search(test_object, layer_functions, cond_layer, cond_pos, dec_layer, dec_pos):

  return -1, False, None, None

