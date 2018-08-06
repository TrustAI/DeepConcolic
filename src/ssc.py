import argparse
import sys
from datetime import datetime

import keras
from keras.models import *
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras import *

import copy


from utils import *

try:
  from art.attacks.fast_gradient import FastGradientMethod
  from art.classifiers import KerasClassifier
except:
  from attacks import *

RP_SIZE=50 ## the top 50 pairs
NNUM=1000000000
EPSILON=sys.float_info.epsilon #0.000000000000001

class ssc_pairt:

  def __init__(self, cond_flags, dec_flag, layer_functions, cond_layer, cond_pos, dec_layer, dec_pos):
    self.cond_flags=cond_flags
    self.dec_flag=dec_flag
    self.layer_functions=layer_functions
    self.cond_layer=cond_layer
    self.cond_pos=cond_pos
    self.dec_layer=dec_layer
    self.dec_pos=dec_pos

def local_search(dnn, local_input, ssc_pair):
  
  classifier=KerasClassifier((MIN, -MIN), model=dnn)
  adv_crafter = FastGradientMethod(classifier)

  d_min=NNUM
  
  e_max=0.3
  old_e_max=e_max
  e_min=0.0

  x_ret=None
  
  while e_max-e_min>=EPSILON:
    x_adv_vect=adv_crafter.generate(x=np.array([local_input]), eps=e_max)
    adv_acts=eval_batch(ssc_pair.layer_functions, x_adv_vect)
    adv_cond_flags=adv_acts[ssc_pair.cond_layer.layer_index][0]
    adv_cond_flags[adv_cond_flags<=0]=0
    adv_cond_flags=adv_cond_flags.astype(bool)
    adv_dec_flag=None
    if adv_acts[ssc_pair.dec_layer.layer_index][0].item(ssc_pair.dec_pos)>0:
      adv_dec_flag=True
    else:
      adv_dec_flag=False

    found=False
    if ssc_pair.dec_flag != adv_dec_flag:
      d=np.count_nonzero(np.logical_xor(adv_cond_flags, ssc_pair.cond_flags))
      if d<=d_min and d>0: 
        found=True

    if found:
      d_min=d
      old_e_max=e_max
      e_max=(e_max+e_min)/2
      x_ret=x_adv_vect[0]
    else:
      e_min=e_max
      e_max=(old_e_max+e_max)/2

    if d_min==1: break

  return d_min, x_ret

def ssc_search(test_object, layer_functions, cond_layer, cond_pos, dec_layer, dec_pos, activations, adv_activations):

  data=test_object.raw_data.data

  x=None
  new_x=None
  d_min=NNUM

  indices=np.random.choice(len(data), len(data))

  count=0
  for i in indices:
    dec1=(activations[dec_layer.layer_index][i].item(dec_pos))
    dec2=(adv_activations[dec_layer.layer_index][i].item(dec_pos))
    if dec1==dec2: continue
    cond1=(activations[cond_layer.layer_index][i].item(cond_pos))
    cond2=(adv_activations[cond_layer.layer_index][i].item(cond_pos))
    if cond1==cond2: continue

    count+=1

    ssc_pair=ssc_pairt(activations[cond_layer.layer_index][i], activations[dec_layer.layer_index][i].item(dec_pos), layer_functions, cond_layer, cond_pos, dec_layer, dec_pos)

    diff, x_ret=local_search(test_object.dnn, data[i], ssc_pair)

    if diff<d_min:
      d_min=diff
      x=data[i]
      new_x=x_ret
      print ('new d: ', d_min)
      if d_min==1: break

    if count>100: break
    
  print ('final d: ', d_min)  
  sys.exit(0)
  d_norm=None
  return d_min>-1, d_norm, new_x, x

