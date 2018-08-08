#import matplotlib.pyplot as plt
from keras import *
from keras import backend as K
import numpy as np
from PIL import Image
import copy
import sys
import cv2


MIN=-100000
DIM=50
ssc_ratio=0.01 #0.1 #0.05 #0.01

## some DNN model has an explicit input layer
def is_input_layer(layer):
  return str(layer).find('InputLayer')>=0

def is_conv_layer(layer):
  return str(layer).find('conv')>=0 or str(layer).find('Conv')>=0

def is_dense_layer(layer):
  return str(layer).find('dense')>=0 or str(layer).find('Dense')>=0

def is_activation_layer(layer):
  return str(layer).find('activation')>=0 or str(layer).find('Activation')>=0

def act_in_the_layer(layer):
  try:
    act=str(layer.activation)
    if act.find('relu')>=0: return 'relu'
    elif act.find('softmax')>=0: return 'softmax'
    else: return ''
  except:
    return ''

def is_maxpooling_layer(layer):
  return str(layer).find('MaxPooling')>=0 

def is_flatten_layer(layer):
  return str(layer).find('flatten')>=0 or str(layer).find('Flatten')>=0

def is_dropout_layer(layer):
  return False ## we do not allow dropout

def get_activation(layer):
  if str(layer.activation).find('relu')>=0: return 'relu'
  elif  str(layer.activation).find('linear')>=0: return 'linear'
  elif  str(layer.activation).find('softmax')>=0: return 'softmax'
  else: return ''


class cover_layert:
  def __init__(self, layer, layer_index, is_conv):
    self.layer=layer
    self.layer_index=layer_index
    self.is_conv=is_conv
    self.activations=[]
    self.pfactor=1.0 ## the proportional factor
    self.actiavtions=[] ## so, we need to store neuron activations?
    self.nc_map=None ## to count the coverage
    self.ssc_map=None ## 

  def initialize_nc_map(self):
    sp=self.layer.output.shape
    if self.is_conv:
      self.nc_map = np.ones((1, sp[1], sp[2], sp[3]), dtype=bool)
    else:
      self.nc_map = np.ones((1, sp[1]), dtype=bool)

  def initialize_ssc_map(self):
    sp=self.layer.output.shape
    if self.is_conv:
      self.ssc_map = np.ones((1, sp[1], sp[2], sp[3]), dtype=bool)
    else:
      self.ssc_map = np.ones((1, sp[1]), dtype=bool)

  ## to get the index of the next property to be satisfied
  def get_nc_next(self):
    spos = np.array(self.activations).argmax()
    return spos, np.array(self.activations).item(spos)

  def disable_by_pos(self, pos):
    if self.is_conv:
      self.activations[pos[0]][pos[1]][pos[2]][pos[3]][pos[4]]=MIN
    else:
      self.activations[pos[0]][pos[1]][pos[2]]=MIN

def get_nc_next(clayers):
  nc_layer, nc_pos, nc_value = None, None, MIN
  for i in range(0, len(clayers)):
    clayer=clayers[i]
    pos, v=clayer.get_nc_next()
    v*=clayer.pfactor
    if v > nc_value:
      nc_layer, nc_pos, nc_value= i, pos, v
      #print (clayer, pos, v)
  return nc_layer, nc_pos, nc_value

def get_layer_functions(dnn):
  layer_functions=[]
  for l in range(0, len(dnn.layers)):
    layer=dnn.layers[l]
    current_layer_function=K.function([layer.input], [layer.output])
    layer_functions.append(current_layer_function)
  return layer_functions

def get_cover_layers(dnn, criterion):
  cover_layers=[]
  for l in range(0, len(dnn.layers)):
    if l==len(dnn.layers)-1: continue
    layer=dnn.layers[l]
    if is_conv_layer(layer) or is_dense_layer(layer):
      sp=layer.output.shape
      clayer=cover_layert(layer, l, is_conv_layer(layer))
      cover_layers.append(clayer)
  return cover_layers


### given an input image, to evaluate activations
def eval(layer_functions, im, having_input_layer=False):
  activations=[]
  for l in range(0, len(layer_functions)):
    if not having_input_layer:
      if l==0:
        activations.append(layer_functions[l]([[im]])[0])
      else:
        activations.append(layer_functions[l]([activations[l-1]])[0])
    else:
      if l==0:
        activations.append([]) #activations.append(layer_functions[l]([ims])[0])
      elif l==1:
        activations.append(layer_functions[l]([[im]])[0])
      else:
        activations.append(layer_functions[l]([activations[l-1]])[0])
  return activations

def eval_batch(layer_functions, ims, having_input_layer=False):
  activations=[]
  for l in range(0, len(layer_functions)):
    if not having_input_layer:
      if l==0:
        activations.append(layer_functions[l]([ims])[0])
      else:
        activations.append(layer_functions[l]([activations[l-1]])[0])
    else:
      if l==0:
        activations.append([]) #activations.append(layer_functions[l]([ims])[0])
      elif l==1:
        activations.append(layer_functions[l]([ims])[0])
      else:
        activations.append(layer_functions[l]([activations[l-1]])[0])
  return activations

class raw_datat:
  def __init__(self, data, labels):
    self.data=data
    self.labels=labels


class test_objectt:
  def __init__(self, dnn, raw_data, criterion, norm):
    self.dnn=dnn
    self.raw_data=raw_data
    ## test config
    self.norm=norm
    self.criterion=criterion
    self.channels_last=True

def calculate_pfactors(activations, cover_layers):
  fks=[]
  for clayer in cover_layers:
    layer_index=clayer.layer_index
    sub_acts=np.abs(activations[layer_index])
    fks.append(np.average(sub_acts))
  av=np.average(fks)
  for i in range(0, len(fks)):
    cover_layers[i].pfactor=av/fks[i]

def update_nc_map_via_inst(clayers, activations):
  for i in range(0, len(clayers)):
    ## to get the act of layer 'i'
    act=copy.copy(activations[clayers[i].layer_index])
    ## TODO
    if act_in_the_layer(clayers[i].layer)=='relu':
      act[act==0]=MIN/10
    act[act>=0]=0
    if clayers[i].nc_map is None: ## not initialized yet
      clayers[i].initialize_nc_map()
      clayers[i].nc_map=np.logical_and(clayers[i].nc_map, act)
    else:
      clayers[i].nc_map=np.logical_and(clayers[i].nc_map, act)
    ## update activations after nc_map change
    clayers[i].activations.append(act)
    ## clayers[i].update_activations() 
    for j in range(0, len(clayers[i].activations)):
      clayers[i].activations[j]=np.multiply(clayers[i].activations[j], clayers[i].nc_map)
      clayers[i].activations[j][clayers[i].activations[j]>=0]=MIN

def nc_report(clayers):
  covered = 0
  non_covered = 0
  for layer in clayers:
    c = np.count_nonzero(layer.nc_map)
    sp = layer.nc_map.shape
    tot = 0
    if layer.is_conv:
        tot = sp[0] * sp[1] * sp[2] * sp[3]
    else:
        tot = sp[0] * sp[1]
    non_covered += c
    covered += (tot - c)
  return covered, non_covered

def save_an_image(im, title, di='./'):
  # we assume im is normalized
  #img=Image.fromarray(np.uint8(im*255))
  #img.save(di+title+'.png')
  #matplotlib.pyplot.imsave(di+title+'.png', im)
  #plt.show(im*255)
  #plt.savefig(di+title+'.png')
  if not di.endswith('/'):
    di+='/'
  cv2.imwrite((di+title+'.png'), im*255)

def get_ssc_next(clayers):
  while True:
    dec_layer_index=np.random.randint(1, len(clayers))
    sp=clayers[dec_layer_index].ssc_map.shape
    tot_s=1
    for s in sp:
      tot_s*=s
    dec_pos=np.random.randint(0, tot_s)
    if not clayers[dec_layer_index].ssc_map.item(dec_pos): 
      continue
    #else: 
    #  clayers[dec_layer_index].ssc_map.itemset(dec_pos, False) 
    return dec_layer_index, dec_pos
