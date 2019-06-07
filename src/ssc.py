import argparse
import sys
from datetime import datetime

import keras
from keras.models import *
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras import *
import tensorflow as tf
import numpy as np

import copy


from utils import *

try:
  from art.attacks.fast_gradient import FastGradientMethod
  from art.classifiers import KerasClassifier
except:
  from attacks import *

RP_SIZE=50 ## the top 50 pairs
NNUM=1000000000
EPSILON=0.00000000001 #sys.float_info.epsilon*10 #0.000000000000001
EPS_MAX=0.3

class ssc_pairt:

  def __init__(self, cond_flags, dec_flag, layer_functions, cond_layer, cond_pos, dec_layer, dec_pos):
    self.cond_flags=cond_flags
    self.dec_flag=dec_flag
    self.layer_functions=layer_functions
    self.cond_layer=cond_layer
    self.cond_pos=cond_pos
    self.dec_layer=dec_layer
    self.dec_pos=dec_pos

def local_search(dnn, local_input, ssc_pair, adv_crafter, e_max_input, ssc_ratio):
  
  d_min=NNUM
  
  e_max=e_max_input #np.random.uniform(0.2, 0.3)
  old_e_max=e_max
  e_min=0.0

  x_ret=None
  not_changed=0
  diff_map=None
  while e_max-e_min>=EPSILON:
    #print ('                     === in while')
    adv_crafter.set_params(eps=e_max)
    x_adv_vect=adv_crafter.generate(x=np.array([local_input]))
    adv_acts=eval_batch(ssc_pair.layer_functions, x_adv_vect, is_input_layer(dnn.layers[0]))
    adv_cond_flags=adv_acts[ssc_pair.cond_layer.layer_index][0]
    adv_cond_flags[adv_cond_flags<=0]=0
    adv_cond_flags=adv_cond_flags.astype(bool)
    adv_dec_flag=None
    if adv_acts[ssc_pair.dec_layer.layer_index][0].item(ssc_pair.dec_pos)>0:
      adv_dec_flag=True
    else:
      adv_dec_flag=False

    found=False
    new_diff_map=None
    if ssc_pair.dec_flag != adv_dec_flag:
        new_diff_map=np.logical_xor(adv_cond_flags, ssc_pair.cond_flags)
        d=np.count_nonzero(new_diff_map)
        if d<=d_min and d>0: 
          found=True

    if found:
      d_min=d
      old_e_max=e_max
      e_max=(e_max+e_min)/2
      x_ret=x_adv_vect[0]
      not_changed=0
      diff_map=new_diff_map
    else:
      e_min=e_max
      e_max=(old_e_max+e_max)/2
      not_changed+=1

    if d_min==1: break
    if d_min<=ssc_ratio*ssc_pair.cond_layer.ssc_map.size: break

  return d_min, x_ret, diff_map

def ssc_search(test_object, layer_functions, cond_layer, cond_pos, dec_layer, dec_pos, adv_crafter, adv_object=None):

  from keras import backend
  backend.set_learning_phase(False)

  sess = backend.get_session()
  sess.run(tf.global_variables_initializer())

  data=test_object.raw_data.data
  labels=test_object.raw_data.labels
  dnn=test_object.dnn
  ssc_ratio=test_object.cond_ratio

  x=None
  y=None
  new_x=None
  diff_map=None
  d_min=cond_layer.ssc_map.size
  print ('====== To catch independent condition change: {0}/{1}'.format(d_min, d_min))

  indices=np.random.choice(len(data), len(data))

  count=0
  for i in indices:
    inp_vect=np.array([data[i]])
    if adv_object is None:
      e_max_input=np.random.uniform(EPS_MAX*2/3, EPS_MAX)
      adv_crafter.set_params(eps=e_max_input)
      adv_inp_vect=adv_crafter.generate(x=inp_vect)
    else:
      e_max_input=np.random.uniform(adv_object.max_v*EPS_MAX*2/3, adv_object.max_v*EPS_MAX)
      adv_crafter.set_params(eps=e_max_input)
      adv_inp_vect=adv_crafter.generate(x=inp_vect)
      adv_inp_vect=np.clip(adv_inp_vect, adv_object.lb_v, adv_object.max_v)
    acts=eval_batch(layer_functions, inp_vect, is_input_layer(dnn.layers[0]))
    adv_acts=eval_batch(layer_functions, adv_inp_vect, is_input_layer(dnn.layers[0]))
    dec1=(acts[dec_layer.layer_index][0].item(dec_pos))
    dec2=(adv_acts[dec_layer.layer_index][0].item(dec_pos))
    if not np.logical_xor(dec1>0, dec2>0): continue

    count+=1

    cond_flags=acts[cond_layer.layer_index][0]
    cond_flags[cond_flags<=0]=0
    cond_flags=cond_flags.astype(bool)
    ssc_pair=ssc_pairt(cond_flags, acts[dec_layer.layer_index][0].item(dec_pos)>0, layer_functions, cond_layer, cond_pos, dec_layer, dec_pos)

    diff, x_ret, diff_map_ret=local_search(test_object.dnn, data[i], ssc_pair, adv_crafter, e_max_input, ssc_ratio)

    if diff<d_min:
      d_min=diff
      x=data[i]
      if labels is not None:
        y=labels[i]
      new_x=x_ret
      diff_map=diff_map_ret
      print ('====== Update independent condition change: {0}/{1}'.format(d_min, cond_layer.ssc_map.size))
      if d_min==1: break

    #print ("++++++",d_min, ssc_ratio, ssc_ratio*cond_layer.ssc_map.size)
    if d_min<=ssc_ratio*cond_layer.ssc_map.size: break
    
  #print ('final d: ', d_min, ' count:', count)  
  if x is not None:
    d_norm=np.abs(new_x-x)
    return d_min, np.max(d_norm), new_x, x, [y], diff_map
  else:
    return d_min, None, None, None, None, None


def local_v_search(dnn, local_input, ssc_pair, adv_crafter, e_max_input, ssc_ratio, dec_ub):
  
  d_min=NNUM
  
  e_max=e_max_input 
  old_e_max=e_max
  e_min=0.0

  x_ret=None
  not_changed=0
  while e_max-e_min>=EPSILON:
    print ('                     === in while', e_max-e_min)
    adv_crafter.set_params(eps=e_max)
    x_adv_vect=adv_crafter.generate(x=np.array([local_input]))
    adv_acts=eval_batch(ssc_pair.layer_functions, x_adv_vect, is_input_layer(dnn.layers[0]))
    adv_cond_flags=adv_acts[ssc_pair.cond_layer.layer_index][0]
    adv_cond_flags[adv_cond_flags<=0]=0
    adv_cond_flags=adv_cond_flags.astype(bool)
    found=False
    if adv_acts[ssc_pair.dec_layer.layer_index][0].item(ssc_pair.dec_pos)>dec_ub: 
      d=np.count_nonzero(np.logical_xor(adv_cond_flags, ssc_pair.cond_flags))
      if d<=d_min and d>0: 
        found=True

    if found:
      d_min=d
      old_e_max=e_max
      e_max=(e_max+e_min)/2
      x_ret=x_adv_vect[0]
      not_changed=0
    else:
      e_min=e_max
      e_max=(old_e_max+e_max)/2
      not_changed+=1

    if d_min==1: break
    if d_min<=ssc_ratio*ssc_pair.cond_layer.ssc_map.size: break

  return d_min, x_ret

def svc_search(test_object, layer_functions, cond_layer, cond_pos, dec_layer, dec_pos, adv_crafter, dec_ub):

  data=test_object.raw_data.data
  dnn=test_object.dnn
  ssc_ratio=test_object.cond_ratio

  x=None
  new_x=None
  d_min=cond_layer.ssc_map.size
  print ('d_min initialised', d_min, len(data))

  indices=np.random.choice(len(data), len(data))

  count=0

  for i in indices:
    inp_vect=np.array([data[i]])
    acts=eval_batch(layer_functions, inp_vect, is_input_layer(dnn.layers[0]))
    dec1=(acts[dec_layer.layer_index][0].item(dec_pos))
    if dec1<=0: continue
    if dec_ub>2*dec1: continue

    #cond1=(acts[cond_layer.layer_index][0].item(cond_pos))
    cond_flags=acts[cond_layer.layer_index][0]
    cond_flags[cond_flags<=0]=0
    cond_flags=cond_flags.astype(bool)

    #dec_ub=dec1*2 #############
    to_stop=False
    #e_inputs=np.linspace(0, 10, num=100)
    
    #for e_max_input in e_inputs:
    e_max_input=0
    trend=0 
    old_dec=dec1
    #while e_max_input<=20 and trend>=-50:
    while e_max_input<=200 and trend>=-50:
      if e_max_input>10:
        e_max_input+=np.random.uniform(0, 1) #0.3
      elif e_max_input>1:
        e_max_input+=np.random.uniform(0, 0.1) #0.3
      else:
        e_max_input+=np.random.uniform(0, 0.05) #0.3
      adv_crafter.set_params(eps=e_max_input)
      adv_inp_vect=adv_crafter.generate(x=inp_vect)
      adv_acts=eval_batch(layer_functions, adv_inp_vect, is_input_layer(dnn.layers[0]))

      dec2=(adv_acts[dec_layer.layer_index][0].item(dec_pos))
      if dec2<=old_dec:
         trend-=1
      else: trend=0
      old_dec=dec2

      #if not np.logical_xor(dec1>0, dec2>0): continue
      print ('****', e_max_input, dec1, dec2, dec_ub, dec2>dec_ub)
      if dec2<=dec_ub: continue
      #cond2=(adv_acts[cond_layer.layer_index][0].item(cond_pos))
      count+=1

      adv_cond_flags=adv_acts[cond_layer.layer_index][0]
      adv_cond_flags[adv_cond_flags<=0]=0
      adv_cond_flags=adv_cond_flags.astype(bool)
      early_d=np.count_nonzero(np.logical_xor(adv_cond_flags, cond_flags))

      if early_d<=ssc_ratio*cond_layer.ssc_map.size:
        d_min=early_d
        x=data[i]
        new_x=adv_inp_vect[0]
        to_stop=True
        break

      ssc_pair=ssc_pairt(cond_flags, acts[dec_layer.layer_index][0].item(dec_pos)>0, layer_functions, cond_layer, cond_pos, dec_layer, dec_pos)

      print ('start local v search')
      diff, x_ret=local_v_search(test_object.dnn, data[i], ssc_pair, adv_crafter, e_max_input, ssc_ratio, dec_ub)
      print ('after local v search')

      if diff<d_min:
        d_min=diff
        x=data[i]
        new_x=x_ret
        print ('new d: ', d_min, cond_layer.ssc_map.size)
        if d_min==1: break


      if d_min<=ssc_ratio*cond_layer.ssc_map.size: break
      ######
      break

    if d_min<=ssc_ratio*cond_layer.ssc_map.size: break
    
  print ('final d: ', d_min, ' count:', count)  
  if x is not None:
    d_norm=np.abs(new_x-x)
    return d_min, np.max(d_norm), new_x, x
  else:
    return d_min, None, None, None

