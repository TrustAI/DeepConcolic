#!/usr/bin/env python3
from deepconcolic import utils_args
from deepconcolic.training import make_dense_classifier
from dc_plugins.toy_datasets import random as dtrandom

# ---

choices = {}

# rand10_2

def rand10_2 (**_):
  n_neurons = (50,) * 2 + (10,) * 2
  make_dense_classifier (dtrandom.make_rand10_2, 'rand', 10, 2, n_neurons)

choices['rand10_2'] = rand10_2

# rand10_5

def rand10_5 (**_):
  n_neurons = (100,) * 4 + (50,) * 3 + (10,) * 2
  make_dense_classifier (dtrandom.make_rand10_5, 'rand', 10, 5, n_neurons)

choices['rand10_5'] = rand10_5

# rand100_5

def rand100_5 (**_):
  n_neurons = (200,) * 2 + (100,) * 2 + (50,) * 2 + (10,) * 2
  make_dense_classifier (dtrandom.make_rand100_5, 'rand', 100, 5, n_neurons)

choices['rand100_5'] = rand100_5

# ---

ap, parse_args = utils_args.make_select_parser \
  ('Trainer of classification models for the random classification toy datasets',
   'model', choices)

if __name__=="__main__":
  parse_args ()
