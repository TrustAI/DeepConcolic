import argparse
import sys
import os
from datetime import datetime

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

def run_nc_linf(test_object, outs):
  print('\n== nc, linf ==\n')
  if not os.path.exists(outs):
    os.system('mkdir -p {0}'.format(outs))
  if not outs.endswith('/'):
    outs+='/'
  nc_results=outs+'nc_report-{0}.txt'.format(str(datetime.now()).replace(' ', '-'))
  nc_results=nc_results.replace(':', '-')

  layer_functions=get_layer_functions(test_object.dnn)
  print('\n== Got layer functions: {0} ==\n'.format(len(layer_functions)))
  cover_layers=get_cover_layers(test_object.dnn, 'NC')
  print('\n== Got cover layers: {0} ==\n'.format(len(cover_layers)))

  #batch=10000
  #if len(test_object.raw_data.data)<batch: batch=len(test_object.raw_data.data)
  #activations = eval_batch(layer_functions, test_object.raw_data.data[0:batch])
  ##print(activations[0].shape)
  ##print(len(activations))

  #calculate_pfactors(activations, cover_layers)

  #### configuration phase done

  test_cases=[]
  adversarials=[]

  xdata=test_object.raw_data.data
  iseed=np.random.randint(0, len(xdata))
  im=xdata[0]

  test_cases.append(im)
  update_nc_map_via_inst(cover_layers, eval(layer_functions, im))
  covered, not_covered=nc_report(cover_layers)
  print (covered)
  print('\n== neuron coverage: {0}==\n'.format(covered*1.0/(covered+not_covered)))
  #y = test_object.dnn.predict_classes(np.array([im]))[0]
  save_an_image(im, 'seed-image', outs)
  f = open(nc_results, "a")
  f.write('NC-cover {0} {1} {2} seed: {3}\n'.format(1.0 * covered / (covered + not_covered), len(test_cases), len(adversarials), iseed))
  f.close()


  base_constraints=create_base_constraints(test_object.dnn)

  while True:
    index_nc_layer, nc_pos, nc_value=get_nc_next(cover_layers)
    #print (nc_layer.layer_index, nc_pos, nc_value/nc_layer.pfactor)
    nc_layer=cover_layers[index_nc_layer]
    print (np.array(nc_layer.activations).shape)
    shape=np.array(nc_layer.activations).shape
    pos=np.unravel_index(nc_pos, shape)
    im=test_cases[pos[0]]
    act_inst=eval(layer_functions, im)


    s=pos[0]*int(shape[1]*shape[2])
    if nc_layer.is_conv:
      s*=int(shape[3])*int(shape[4])
    print ('\n::', nc_pos, pos, nc_pos-s)
    print (nc_layer.layer, nc_layer.layer_index)
    print ('the max v', nc_value)

    mkey=nc_layer.layer_index
    if act_in_the_layer(nc_layer.layer) != 'relu':
      mkey+=1
    feasible, d, new_im=negate(test_object.dnn, act_inst, [im], nc_layer, nc_pos-s, base_constraints[mkey])

    cover_layers[index_nc_layer].disable_by_pos(pos)
    if feasible:
      print ('\nis feasible!!!\n')
      test_cases.append(new_im)
      update_nc_map_via_inst(cover_layers, eval(layer_functions, new_im))
      y1 = test_object.dnn.predict_classes(np.array([im]))[0]
      y2= test_object.dnn.predict_classes(np.array([new_im]))[0]
      if y1 != y2: adversarials.append([im, new_im])
      old_acts=eval(layer_functions, im)
      new_acts=eval(layer_functions, new_im)
      if nc_layer.is_conv:
        print ('\n should be < 0', old_acts[nc_layer.layer_index][pos[1]][pos[2]][pos[3]][pos[4]], '\n')
        print ('\n should be > 0', new_acts[nc_layer.layer_index][pos[1]][pos[2]][pos[3]][pos[4]], '\n')
    else:
      print ('\nis NOT feasible!!!\n')
    covered, not_covered=nc_report(cover_layers)
    f = open(nc_results, "a")
    f.write('NC-cover {0} {1} {2} \n'.format(1.0 * covered / (covered + not_covered), len(test_cases), len(adversarials)))
    f.close()
    #break


def deepconcolic(test_object, outs):
  print('\n== Start DeepConcolic testing ==\n')
  if test_object.criterion=='nc': ## neuron cover
    if test_object.norm=='linf':
      run_nc_linf(test_object, outs)
    elif test_object.norm=='l0':
      pass #run_nc_l0(test_object)
    else:
      print('\n not supported norm...\n')
      sys.exit(0)
  else:
      print('\n for now, let us focus on neuron cover...\n')
      sys.exit(0)


def main():


  parser=argparse.ArgumentParser(description='The concolic testing for neural networks' )
  parser.add_argument(
    '--model', dest='model', default='-1', help='The input neural network model (.h5)')
  parser.add_argument("--inputs", dest="inputs", default="-1",
                    help="the input test data directory", metavar="DIR")
  parser.add_argument("--outputs", dest="outputs", default="-1",
                    help="the outputput test data directory", metavar="DIR")
  parser.add_argument("--criterion", dest="criterion", default="nc",
                    help="the test criterion", metavar="nc, bc, ssc...")
  parser.add_argument("--mnist-dataset", dest="mnist", help="MNIST dataset", action="store_true")
  parser.add_argument("--cifar10-dataset", dest="cifar10", help="CIFAR10 dataset", action="store_true")
  parser.add_argument("--vgg16-model", dest='vgg16', help="vgg16 model", action="store_true")
  parser.add_argument("--norm", dest="norm", default="linf",
                    help="the norm metric", metavar="linf, l0")
  parser.add_argument("--input-rows", dest="img_rows", default="224",
                    help="input rows", metavar="")
  parser.add_argument("--input-cols", dest="img_cols", default="224",
                    help="input cols", metavar="")
  parser.add_argument("--input-channels", dest="img_channels", default="3",
                    help="input channels", metavar="")

  args=parser.parse_args()

  dnn=None
  if args.model!='-1':
    dnn=load_model(args.model)
    dnn.summary()
  elif args.vgg16:
    dnn=VGG16()
    dnn.summary()
  else:
    print (' \n == Please specify the input neural network == \n')
    sys.exit(0)

  criterion=args.criterion
  norm=args.norm

  raw_data=None
  img_rows, img_cols, img_channels = int(args.img_rows), int(args.img_cols), int(args.img_channels)
  if args.inputs!='-1':
    
    xs=[]
    for f in os.listdir(args.inputs):
      print (f)
      if f.endswith(".jpg") or f.endswith(".png"): 
        image = load_img(os.path.join(args.inputs,f), target_size=(img_rows, img_cols))
      xs.append(np.asarray(image))
    print (len(xs))
    x_test=np.asarray(xs)
    print x_test.shape, "------"
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
    x_test = x_test.astype('float32')
    x_test /= 255
    raw_data=raw_datat(x_test, [])
  elif args.mnist:
    img_rows, img_cols, img_channels = 28, 28, 1
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
    x_test = x_test.astype('float32')
    x_test /= 255
    raw_data=raw_datat(x_test, y_test)
  elif args.cifar10:
    img_rows, img_cols, img_channels = 32, 32, 3
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
    x_test = x_test.astype('float32')
    x_test /= 255
    raw_data=raw_datat(x_test, y_test)
  else:
    print (' \n == Please input dataset == \n')
    sys.exit(0)


  outs=None
  if args.outputs!='-1':
    outs=args.outputs
  else:
    print (' \n == Please specify the output directory == \n')
    sys.exit(0)

  deepconcolic(test_objectt(dnn, raw_data, criterion, norm), outs)

if __name__=="__main__":
  main()
