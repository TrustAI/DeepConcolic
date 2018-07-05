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


def update_nc_map(clayers, layer_functions, im):
  activations=eval(layer_functions, im)
  for clayer in clayers:
    act=activations[clayer.layer_index]
    act[act>=0]=1
    act=1.0/act
    act[act>0]=0
    act=np.abs(act)
    clayer.activations.append(act)
    if clayer.nc_map==None: ## not initialized yet
      clayer.initialize_nc_map()
      clayer.nc_map=np.logical_or(clayer.nc_map, act)
    else:
      clayer.nc_map=np.logical_and(clayer.nc_map, act)
    ## update activations after nc_map change
    clayer.update_activations() 

def run_nc_linf(test_object):
  print('\n== nc, linf ==\n')
  ## DIR to store outputs
  outs = "concolic-nc-linf" + str(datetime.now()).replace(' ', '-') + '/'
  outs=outs.replace(' ', '-')
  outs=outs.replace(':', '-') ## windows compatibility
  os.system('mkdir -p {0}'.format(outs))
  nc_results=outs+'nc_report.txt'

  layer_functions=get_layer_functions(test_object.dnn)
  print('\n== Got layer functions: {0} ==\n'.format(len(layer_functions)))
  cover_layers=get_cover_layers(test_object.dnn, 'NC')
  print('\n== Got cover layers: {0} ==\n'.format(len(cover_layers)))

  activations = eval_batch(layer_functions, test_object.raw_data.data[0:10])
  print(activations[0].shape)
  print(len(activations))

  calculate_pfactors(activations, cover_layers)

  ### configuration phase done

  test_cases=[]
  adversarials=[]

  xdata=test_object.raw_data.data
  iseed=np.random.randint(0, len(xdata))
  im=xdata[iseed]

  test_cases.append(im)
  update_nc_map(cover_layers, layer_functions, im)
  covered, not_covered=nc_report(cover_layers)
  print('\n== neuron coverage: {0}==\n'.format(covered*1.0/(covered+not_covered)))
  y = test_object.dnn.predict_classes(np.array([im]))[0]
  save_an_image(im, 'seed-image', outs)
  f = open(nc_results, "a")
  f.write('NC-cover {0} {1} {2} seed: {3}\n'.format(1.0 * covered / (covered + not_covered), len(test_cases), len(adversarials), iseed))
  f.close()


def deepconcolic(test_object):
  print('\n== Start DeepConcolic testing ==\n')
  if test_object.criterion=='NC': ## neuron cover
    if test_object.norm=='linf':
      run_nc_linf(test_object)
    elif test_object.norm=='l0':
      pass #run_nc_l0(test_object)
    else:
      print('\n not supported norm...\n')
      sys.exit(0)
  else:
      print('\n for now, let us focus on neuron cover...\n')
      sys.exit(0)


def main():
  ## for testing purpose we fix the aicover configuration
  ##
  dnn=load_model("../saved_models/cifar10_complicated.h5")
  dnn.summary()
  #dnn=VGG16()
  ##
  criterion='NC'
  img_rows, img_cols = 32, 32
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
  x_test = x_test.astype('float32')
  x_test /= 255
  raw_data=raw_datat(x_test, y_test)

  deepconcolic(test_objectt(dnn, raw_data, criterion, "linf"))

if __name__=="__main__":
  main()
