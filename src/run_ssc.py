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
from nc_setup import *
from ssc import *

try:
  from art.attacks.fast_gradient import FastGradientMethod
  from art.classifiers import KerasClassifier
except:
  from attacks import *

try:
  from art.attacks.fast_gradient import FastGradientMethod
  from art.classifiers import KerasClassifier
except:
  from attacks import *


def run_ssc(test_object, outs):
  print ('To run ssc\n')
  
  f_results, layer_functions, cover_layers=ssc_setup(test_object, outs)

  ## define a global attacker
  classifier=KerasClassifier((MIN, -MIN), model=test_object.dnn)
  adv_crafter = FastGradientMethod(classifier)

  test_cases=[]
  adversarials=[]

  count=0

  while True:
    dec_layer_index, dec_pos=get_ssc_next(cover_layers)

    if dec_layer_index==1 and is_input_layer(test_object.dnn.layers[0]): continue
    print ('dec_layer_index', dec_layer_index)

    ###
    cond_layer=cover_layers[dec_layer_index-1]
    dec_layer=cover_layers[dec_layer_index]
    cond_cover=np.ones(cond_layer.ssc_map.shape, dtype=bool)
    ###
 
    ## to check if dec_pos is a padding
    dec_pos_unravel=None
    osp=dec_layer.ssc_map.shape
    dec_pos_unravel=np.unravel_index(dec_pos, osp)
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
        for II in range(0, kernel_size[0]):
          for JJ in range(0, kernel_size[1]):
            for KK in range(0, weights.shape[2]):
              try_tmp=cond_layer.ssc_map[0][J+II][K+JJ][KK]
      except: 
        #print ('dec neuron is a padding')
        continue
        

    cond_pos=np.random.randint(0, cond_cover.size)

    print ('cond, dec layer index: ', cond_layer.layer_index, dec_layer.layer_index)
    print ('dec_layer_index: ', dec_layer_index)

    count+=1

    d_min, d_norm, new_image, old_image=ssc_search(test_object, layer_functions, cond_layer, cond_pos, dec_layer, dec_pos, adv_crafter)

    print ('d_min is', d_min, 'd_norm is', d_norm)

    feasible=(d_min<=ssc_ratio*cond_layer.ssc_map.size)

    adv_flag=False

    if feasible:
      test_cases.append((new_image, old_image))
      y1 =(np.argmax(test_object.dnn.predict(np.array([new_image]))))
      y2= (np.argmax(test_object.dnn.predict(np.array([old_image]))))
      if y1!=y2:
        print ('found an adversarial example')
        adversarials.append((new_image, old_image))
        save_an_image(new_image, '{0}-adv-{1}.png'.format(len(adversarials), y1), f_results.split('/')[0])
        save_an_image(old_image, '{0}-original-{1}.png'.format(len(adversarials), y2), f_results.split('/')[0])
        new_image=new_image*255
        old_image=old_image*255
        new_image.astype('uint8')
        old_image.astype('uint8')
        diff_image=np.abs(new_image-old_image)
        save_an_image(diff_image/255.0, '{0}-diff.png'.format(len(adversarials)), f_results.split('/')[0])
        adv_flag=True
    else:
      print ("not feasible")

    print ('f_results: ', f_results)
    f = open(f_results, "a")
    f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8}\n'.format(count, len(test_cases), len(adversarials), feasible, adv_flag, d_min, d_norm, dec_layer.layer_index, cond_layer.ssc_map.size))
    f.close()

