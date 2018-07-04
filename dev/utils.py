
#import matplotlib.pyplot as plt
from keras import *
from keras import backend as K

## some DNN model has an explicit input layer
def is_input_layer(layer):
  return str(layer).find('InputLayer')>=0

def is_conv_layer(layer):
  return str(layer).find('conv')>=0 or str(layer).find('Conv')>=0

def is_dense_layer(layer):
  return str(layer).find('dense')>=0 or str(layer).find('Dense')>=0

def is_activation_layer(layer):
  return str(layer).find('activation')>=0 or str(layer).find('Activation')>=0

def get_activation(layer):
  if str(layer.activation).find('relu')>=0: return 'relu'
  elif  str(layer.activation).find('linear')>=0: return 'linear'
  elif  str(layer.activation).find('softmax')>=0: return 'softmax'
  else: return ''

def is_maxpooling_layer(layer):
  return str(layer).find('MaxPooling')>=0 

def is_flatten_layer(layer):
  return str(layer).find('flatten')>=0 or str(layer).find('Flatten')>=0

def is_dropout_layer(layer):
  return False ## we do not allow dropout

class cover_layert:
  def __init__(self, layer, layer_index, is_conv):
    self.layer=layer
    self.layer_index=layer_index
    self.is_conv=is_conv
    self.activations=[]

def get_layer_functions(dnn):
  layer_functions=[]
  for l in range(0, len(dnn.layers)):
    layer=dnn.layers[l]
    current_layer_function=K.function([layer.input], [layer.output])
    layer_functions.append(current_layer_function)
  return layer_functions

def get_cover_layers(dnn):
  cover_layers=[]
  for l in range(0, len(dnn.layers)):
    layer=dnn.layers[l]
    if is_conv_layer(layer):
      cover_layers.append(cover_layert(layer, l, is_conv=True))
    elif is_dense_layer(layer):
      cover_layers.append(cover_layert(layer, l, is_conv=False)) 
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

#def show_adversarial_examples(imgs, ys, name):
#  for i in range(0, 2):
#    plt.subplot(1, 2, 1+i)
#    print 'imgs[i].shape is ', imgs[i].shape
#    plt.imshow(imgs[i].reshape([28,28]), cmap=plt.get_cmap('gray'))
#    plt.title("label: "+str(ys[i]))
#    plt.savefig(name, bbox_inches='tight')
