
#import matplotlib.pyplot as plt
from keras import *
from keras import backend as K
import numpy as np
from PIL import Image
import copy
import sys

MIN=-100000

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

  def initialize_nc_map(self):
    sp=self.layer.output.shape
    if self.is_conv:
      self.nc_map = np.ones((1, sp[1], sp[2], sp[3]), dtype=bool)
    else:
      self.nc_map = np.ones((1, sp[1]), dtype=bool)

  def update_activations(self):
    #self.activations=[np.multiply(act, self.nc_map) for act in self.activations]
    for i in range(0, len(self.activations)):
      self.activations[i]=np.multiply(self.activations[i], self.nc_map)
      self.activations[i][self.activations[i]>=0]=MIN

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
    print (layer)
    if is_conv_layer(layer) or is_dense_layer(layer):
      print ('is effective layer')
      sp=layer.output.shape
      clayer=cover_layert(layer, l, is_conv_layer(layer))
      cover_layers.append(clayer)
  return cover_layers


### given an input image, to evaluate activations
def eval(layer_functions, im):
  activations=[]
  for l in range(0, len(layer_functions)):
    if l==0:
      activations.append(layer_functions[l]([[im]])[0])
    else:
      activations.append(layer_functions[l]([activations[l-1]])[0])
  return activations

def eval_batch(layer_functions, ims):
  activations=[]
  for l in range(0, len(layer_functions)):
    if l==0:
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
      act[act==0]=-MIN/10
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
  pass
  ### we assume im is normalized
  #img=Image.fromarray(np.uint8(im*255))
  #img.save(di+title+'.png')

#def show_adversarial_examples(imgs, ys, name):
#  for i in range(0, 2):
#    plt.subplot(1, 2, 1+i)
#    print 'imgs[i].shape is ', imgs[i].shape
#    plt.imshow(imgs[i].reshape([28,28]), cmap=plt.get_cmap('gray'))
#    plt.title("label: "+str(ys[i]))
#    plt.savefig(name, bbox_inches='tight')
