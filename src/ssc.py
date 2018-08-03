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

  ranked_pairs=[]

  d_min=None
  for inpp in [(local_input, NNUM)]:

    inp=inpp[0] # inpp shall be a pair
    found=False
    d_min=inpp[1]
    
    # random FGSM mutation
    count=0
    count2=0
    for epsilon in np.arange(0.3, 0.000001, -0.000001):
      x_adv_vect=adv_crafter.generate(x=np.array([inp]), eps=epsilon)
      adv_acts=eval_batch(ssc_pair.layer_functions, x_adv_vect)
      adv_cond_flags=adv_acts[ssc_pair.cond_layer.layer_index][0]
      adv_cond_flags[adv_cond_flags<=0]=0
      adv_cond_flags=adv_cond_flags.astype(bool)
      adv_dec_flag=None
      if adv_acts[ssc_pair.dec_layer.layer_index][0].item(ssc_pair.dec_pos)>0:
        adv_dec_flag=True
      else:
        adv_dec_flag=False
      if ssc_pair.dec_flag != adv_dec_flag:
          d=np.count_nonzero(np.logical_xor(adv_cond_flags, ssc_pair.cond_flags))
          if d<d_min and d<NNUM:
            print ('new d: ', d, d_min, epsilon)
            d_min=d
          count2+=1
      else: break

  return d_min

def ssc_search(test_object, layer_functions, cond_layer, cond_pos, dec_layer, dec_pos, activations, adv_activations):

  data=test_object.raw_data.data

  passed_list=[]
  while True:
    d_min=-MIN
    index=None
    for i in range(0, len(data)):
      if i in passed_list: continue
      dec1=(activations[dec_layer.layer_index][i].item(dec_pos))
      dec2=(adv_activations[dec_layer.layer_index][i].item(dec_pos))
      if dec1==dec2: continue
      cond1=(activations[cond_layer.layer_index][i].item(cond_pos))
      cond2=(adv_activations[cond_layer.layer_index][i].item(cond_pos))
      if cond1==cond2: continue
      #print(activations[cond_layer.layer_index][i].shape)
      #print(adv_activations[cond_layer.layer_index][i].shape)
      #print('==')
      #print('==')
      #print('==')
      cond_diffs=np.logical_xor(activations[cond_layer.layer_index][i], adv_activations[cond_layer.layer_index][i])
      d=np.count_nonzero(cond_diffs)
      if d<d_min:
        d_min=d
        index=i
    passed_list.append(index)
    print ('==', d_min, index)
    ssc_pair=ssc_pairt(activations[cond_layer.layer_index][index], activations[dec_layer.layer_index][index].item(dec_pos), layer_functions, cond_layer, cond_pos, dec_layer, dec_pos)
    diff=local_search(test_object.dnn, data[index], ssc_pair)
    print ('diff:', diff)
    if len(passed_list)>=50: break
  ##
  print ('length of passed_list', len(passed_list))
    
    
  sys.exit(0)
  return -1, False, None, None

