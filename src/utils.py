#import matplotlib.pyplot as plt
from keras import *
from keras import backend as K
import numpy as np
from PIL import Image
import copy
import sys
import cv2


#the_dec_pos=0
MIN=-100000
DIM=50
BUFFER_SIZE=20
#ssc_ratio=0.005 #0.1 #0.05 #0.01

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
    self.ubs=None ## 

  def initialize_nc_map(self, layer_feature=None):
    sp=self.layer.output.shape
    if self.is_conv:
      self.nc_map = np.ones((1, sp[1], sp[2], sp[3]), dtype=bool)
      if layer_feature==None: return
      if layer_feature[0]==None: return
      if layer_feature[1]==None: return
      if self.layer_index in layer_feature[0]:
        sp=self.nc_map.shape
        for i in range(0, sp[3]):
          if not i in layer_feature[1]:
            self.nc_map[:,:,:,i]=False
    else:
      self.nc_map = np.ones((1, sp[1]), dtype=bool)

  def initialize_ssc_map(self, layer_feature=None):
    sp=self.layer.output.shape
    if self.is_conv:
      self.ssc_map = np.ones((1, sp[1], sp[2], sp[3]), dtype=bool)
      if layer_feature==None: return
      if layer_feature[0]==None: return
      if layer_feature[1]==None: return
      if self.layer_index in layer_feature[0]:
        sp=self.ssc_map.shape
        for i in range(0, sp[3]):
          if not i in layer_feature[1]:
            self.ssc_map[:,:,:,i]=False
    else:
      self.ssc_map = np.ones((1, sp[1]), dtype=bool)

  def initialize_ubs(self):
    sp=self.layer.output.shape
    if self.is_conv:
      self.ubs=np.zeros((1, sp[1], sp[2], sp[3]), dtype=float)
    else:
      self.ubs=np.zeros((1, sp[1]), dtype=float)

  ## to get the index of the next property to be satisfied
  def get_nc_next(self):
    spos = np.array(self.activations).argmax()
    return spos, np.array(self.activations).item(spos)

  def disable_by_pos(self, pos):
    if self.is_conv:
      self.activations[pos[0]][pos[1]][pos[2]][pos[3]][pos[4]]=MIN
    else:
      self.activations[pos[0]][pos[1]][pos[2]]=MIN

def get_nc_next(clayers, layer_indices=None):
  nc_layer, nc_pos, nc_value = None, None, MIN
  for i in range(0, len(clayers)):
    if layer_indices==None or clayers[i].layer_index in layer_indices:
      clayer=clayers[i]
      pos, v=clayer.get_nc_next()
      v*=clayer.pfactor
      if v > nc_value:
        nc_layer, nc_pos, nc_value= i, pos, v
        if np.random.uniform(0., 1.)>=i*1./len(clayers): break
  if nc_layer==None:
    print ('incorrect layer index specified (the layer tested shall be either conv or dense layer)', layer_indices)
    sys.exit(0)
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
      if l==len(dnn.layers)-2 and get_activation(layer)!='relu': continue
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
    self.cond_ratio=None
    self.top_classes=None
    self.inp_ub=None
    self.training_data=None
    self.labels=None
    self.trace_flag=None
    self.layer_indices=None
    self.feature_indices=None

def calculate_pfactors(activations, cover_layers):
  fks=[]
  for clayer in cover_layers:
    layer_index=clayer.layer_index
    sub_acts=np.abs(activations[layer_index])
    fks.append(np.average(sub_acts))
  av=np.average(fks)
  for i in range(0, len(fks)):
    cover_layers[i].pfactor=av/fks[i]

def update_nc_map_via_inst(clayers, activations, layer_feature=None):
  for i in range(0, len(clayers)):
    ## to get the act of layer 'i'
    act=copy.copy(activations[clayers[i].layer_index])
    ## TODO
    if act_in_the_layer(clayers[i].layer)=='relu':
      act[act==0]=MIN/10
    act[act>=0]=0
    if clayers[i].nc_map is None: ## not initialized yet
      clayers[i].initialize_nc_map(layer_feature)
      clayers[i].nc_map=np.logical_and(clayers[i].nc_map, act)
    else:
      clayers[i].nc_map=np.logical_and(clayers[i].nc_map, act)
    ## update activations after nc_map change
    if len(clayers[i].activations)>=BUFFER_SIZE:
      ind=np.random.randint(0,BUFFER_SIZE)
      clayers[i].activations[ind]=(act)
    else:
      clayers[i].activations.append(act)
    ## clayers[i].update_activations() 
    for j in range(0, len(clayers[i].activations)):
      clayers[i].activations[j]=np.multiply(clayers[i].activations[j], clayers[i].nc_map)
      clayers[i].activations[j][clayers[i].activations[j]>=0]=MIN

def nc_report(clayers, layer_indices=None, feature_indices=None):
  covered = 0
  non_covered = 0
  for layer in clayers:
    if layer_indices==None or layer.layer_index in layer_indices:
      if feature_indices==None or not layer.is_conv:
        c = np.count_nonzero(layer.nc_map)
        sp = layer.nc_map.shape
        tot = 0
        if layer.is_conv:
          tot = sp[0] * sp[1] * sp[2] * sp[3]
        else:
          tot = sp[0] * sp[1]
      else:
        sp=layer.nc_map.shape
        for i in range(0, sp[3]):
          if not i in feature_indices: continue
          c = np.count_nonzero(layer.nc_map[:,:,:,i])
          tot = layer.nc_map[:,:,:,i].size
      non_covered += c
      covered += (tot - c)
  return covered, non_covered

def save_an_image(im, title, di='./'):
  if not di.endswith('/'):
    di+='/'
  cv2.imwrite((di+title+'.png'), im*255)

def save_adversarial_examples(adv, origin, diff, di):
  save_an_image(adv[0], adv[1], di)
  save_an_image(origin[0], origin[1], di)
  if diff is not None:
    save_an_image(diff[0], diff[1], di)

def is_padding(dec_pos, dec_layer, cond_layer, post=True):
  ## to check if dec_pos is a padding
  dec_pos_unravel=None
  osp=dec_layer.ssc_map.shape
  dec_pos_unravel=np.unravel_index(dec_pos, osp)
  #print (osp, dec_pos_unravel)
  if is_conv_layer(dec_layer.layer):
    Weights=dec_layer.layer.get_weights()
    weights=Weights[0]
    biases=Weights[1]
    I=0
    J=dec_pos_unravel[1]
    K=dec_pos_unravel[2]
    L=dec_pos_unravel[3]
    kernel_size=dec_layer.layer.kernel_size
    try:
      if post:
        for II in range(0, kernel_size[0]):
          for JJ in range(0, kernel_size[1]):
            for KK in range(0, weights.shape[2]):
              try_tmp=cond_layer.ssc_map[0][J+II][K+JJ][KK]
      else:
        for II in range(0, weights.shape[0]):
          for JJ in range(0, kernel_size[0]):
            for KK in range(0, kernel_size[1]):
              try_tmp=cond_layer.ssc_map[0][II][K+JJ][L+KK]
    except:
      return True
  return False


def get_ssc_next(clayers, layer_indices=None, feature_indices=None):
  #global the_dec_pos
  clayers2=[]
  if layer_indices==None:
    clayers2=clayers
  else:
    for i in range(1, len(clayers)):
      if clayers[i].layer_index in layer_indices:
        clayers2.append(clayers[i])
  if clayers2==[]:
    print ('incorrect layer index specified (the layer tested shall be either conv or dense layer)', layer_indices)
    sys.exit(0)
  #print (clayers2[0].layer_index)
  dec_layer_index_ret=None
  dec_pos_ret=None
    
  while True:
    dec_layer_index=np.random.randint(0, len(clayers2))
    ## todo: this is a shortcut
    #print ('#######',len(clayers2), dec_layer_index, clayers[1].layer)
    if not np.any(clayers2[dec_layer_index].ssc_map):
      print ('all decision features at layer {0} have been covered'.format(dec_layer_index))
      continue
      #sys.exit(0)

    sp=clayers2[dec_layer_index].ssc_map.shape
    tot_s=1
    for s in sp:
      tot_s*=s
    the_dec_pos=np.random.randint(0, tot_s)
    if not feature_indices==None:
      the_dec_pos=np.argmax(clayers2[dec_layer_index].ssc_map.shape)
    found=False
    while the_dec_pos<tot_s:
      if not clayers2[dec_layer_index].ssc_map.item(the_dec_pos):
        the_dec_pos+=1
        continue
      else:
        found=True 
        break
    #if the_dec_pos>=tot_s:
    #  print ('all decision features at layer {0} have been covered'.format(dec_layer_index))
    #  sys.exit(0)
    if found:
      dec_pos_ret=the_dec_pos
      for i in range(0, len(clayers)):
        if clayers[i].layer_index==clayers2[dec_layer_index].layer_index:
          dec_layer_index_ret=i
          break
      break
  if dec_layer_index_ret==None:
    print ('End of the testing')
    sys.exit(0)
  return dec_layer_index_ret, dec_pos_ret

def print_adversarial_distribution(advs, fname, int_flag=False):
  advs=np.sort(advs)
  ## average and std
  ave=np.mean(advs)
  std=np.std(advs)
  d_max=advs[len(advs)-1]
  xs=None
  if not int_flag:
    xs=np.arange(0.001, d_max+0.001, 0.001)
  else:
    xs=np.arange(1, d_max+1, 1)
  ys=np.zeros(len(xs))
  for i in range(0, len(xs)):
    for d in advs:
      if d<=xs[i]: ys[i]+=1
    ys[i]=ys[i]*1.0/len(advs)

  f=open(fname, "w")
  f.write('adversarial examples:  (average distance, {0}), (standard variance, {1})\n'.format(ave, std))
  f.write('#distance #accumulated adversarial examples fall into this distance\n')
  for i in range(0, len(xs)):
    f.write('{0} {1}\n'.format(xs[i], ys[i]))
  f.close()
  
def l0_filtered(data_set, x, factor=0.25):
  size=data_set.size*factor
  diffs=data_set-x
  for diff in diffs:
    if np.count_nonzero(diff)<size:
      return False
  return True

def linf_filtered(data_set, x, factor=0.25):
  size=data_set.size*factor
  diffs=data_set-x
  for index, diff in np.ndenumerate(diffs):
    diff_abs=np.abs(diff)
    diff_abs=np.array(diff_abs)
    diff_abs[diff_abs<=factor]=0.
    if np.count_nonzero(diff)<size:
      return False
  return True
