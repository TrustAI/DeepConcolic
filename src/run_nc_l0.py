import argparse
import sys
import os
from datetime import datetime

import keras
from keras.models import *
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.layers import *
from keras import *
from utils import *
from nc_lp import *
from lp_encoding import *

def run_nc_l0(test_object, outs):
  print ('\n == is coming soon... ==\n')
  sys.exit(0)
  nc_results, layer_functions, cover_layers, activations, test_cases, adversarials=nc_setup(test_object, outs)
