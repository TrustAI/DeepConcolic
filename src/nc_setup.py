
from datetime import datetime
import os

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


def nc_setup(test_object, outs):
  print('\n== {0}, {1} ==\n'.format(test_object.criterion, test_object.norm))
  if not os.path.exists(outs):
    os.system('mkdir -p {0}'.format(outs))
  if not outs.endswith('/'):
    outs+='/'
  nc_results=outs+'nc_{0}_report-{1}.txt'.format(test_object.norm, str(datetime.now()).replace(' ', '-'))
  nc_results=nc_results.replace(':', '-')

  layer_functions=get_layer_functions(test_object.dnn)
  print('\n== Got layer functions: {0} ==\n'.format(len(layer_functions)))
  cover_layers=get_cover_layers(test_object.dnn, 'NC')
  print('\n== Got cover layers: {0} ==\n'.format(len(cover_layers)))

  tot_size=len(test_object.raw_data.data)
  activations=None
  batches=np.array_split(test_object.raw_data.data[0:tot_size], tot_size//1000 + 1)
  for i in range(0, len(batches)):
    batch=batches[i]
    sub_acts=eval_batch(layer_functions, batch, is_input_layer(test_object.dnn.layers[0]))
    if i==0:
      activations=sub_acts
    else:
      for j in range(0, len(activations)):
        activations[j]=np.concatenate((activations[j], sub_acts[j]), axis=0)

  calculate_pfactors(activations, cover_layers)

  #### configuration phase done

  test_cases=[]
  adversarials=[]

  xdata=test_object.raw_data.data
  iseed=np.random.randint(0, len(xdata))
  im=xdata[iseed]

  test_cases.append(im)
  update_nc_map_via_inst(cover_layers, eval(layer_functions, im, is_input_layer(test_object.dnn.layers[0])), (test_object.layer_indices, test_object.feature_indices))
  covered, not_covered=nc_report(cover_layers, test_object.layer_indices, test_object.feature_indices)
  print('\n== The initial neuron coverage: {0}==\n'.format(covered*1.0/(covered+not_covered)))
  save_an_image(im/test_object.inp_ub*1.0, 'seed-image', outs)
  f = open(nc_results, "a")
  f.write('NC-cover: {0} #test cases: {1} #adversarial examples: {2}\n'.format(1.0 * covered / (covered + not_covered), len(test_cases), len(adversarials)))
  f.close()

  #for i in range(0, len(cover_layers)):
  #  cover_layers[i].initialize_ssc_map((test_object.layer_indices, test_object.feature_indices))

  return nc_results, layer_functions, cover_layers, activations, test_cases, adversarials

def ssc_setup(test_object, outs):
  print('\n== MC/DC (ssc) coverage for neural networks ==\n')
  if not os.path.exists(outs):
    os.system('mkdir -p {0}'.format(outs))
  if not outs.endswith('/'):
    outs+='/'
  nc_results=outs+'ssc_report-{0}.txt'.format(str(datetime.now()).replace(' ', '-'))
  nc_results=nc_results.replace(':', '-')

  layer_functions=get_layer_functions(test_object.dnn)
  print('\n== Total layers: {0} ==\n'.format(len(layer_functions)))
  cover_layers=get_cover_layers(test_object.dnn, 'SSC')
  print('\n== Cover-able layers: {0} ==\n'.format(len(cover_layers)))

  for i in range(0, len(cover_layers)):
    cover_layers[i].initialize_ubs()
    cover_layers[i].initialize_ssc_map((test_object.layer_indices, test_object.feature_indices))

  #print ("to compute the ubs")
  activations=None
  if not test_object.training_data is None:
    for x in test_object.training_data:
      x_acts=eval_batch(layer_functions, np.array([x]), is_input_layer(test_object.dnn.layers[0]))
      for i in range(1, len(cover_layers)):
        #print (type(x_acts[cover_layers[i].layer_index][0]))
        #print (type(cover_layers[i].ubs))
        cover_layers[i].ubs=np.maximum(cover_layers[i].ubs, x_acts[cover_layers[i].layer_index][0])
  #print ("done")
  #  tot_size=len(test_object.training_data)
  #  batches=np.array_split(test_object.training_data[0:tot_size], tot_size//10 + 1)
  #  for i in range(0, len(batches)):
  #    batch=batches[i]
  #    sub_acts=eval_batch(layer_functions, batch, is_input_layer(test_object.dnn.layers[0]))
  #    if i==0:
  #      activations=sub_acts
  #    else:
  #      for j in range(0, len(activations)):
  #        activations[j]=np.concatenate((activations[j], sub_acts[j]), axis=0)

  return nc_results, layer_functions, cover_layers, activations
