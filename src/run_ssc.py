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
  
  f_results, layer_functions, cover_layers, _=ssc_setup(test_object, outs)
  f = open(f_results, "a")
  f.write('#ssc runs;  #test cases;  #adversarial examples;  is feasible; is top-1 adversarial example; is top-x adversarial example; condition feature size; L infinity distance; L0 distance; decision layer index; #condition layer neurons; new labels; original labels; coverage; local coverage\n')
  f.close()

  tot_decs=0
  for i in range(1, len(cover_layers)):
    if i==1 and is_input_layer(test_object.dnn.layers[0]): continue
    else: tot_decs+=cover_layers[i].ssc_map.size
  tot_coverage=0.0

  ## define a global attacker
  classifier=KerasClassifier((MIN, -MIN), model=test_object.dnn)
  adv_crafter = FastGradientMethod(classifier)

  test_cases=[]
  adversarials=[]
  count=0


  while True:
    dec_layer_index, dec_pos=get_ssc_next(cover_layers)
    cover_layers[dec_layer_index].ssc_map.itemset(dec_pos, False)

    if dec_layer_index==1 and is_input_layer(test_object.dnn.layers[0]): continue

    ###
    cond_layer=cover_layers[dec_layer_index-1]
    dec_layer=cover_layers[dec_layer_index]
    cond_cover=np.zeros(cond_layer.ssc_map.shape, dtype=bool)
    ###
 
    if is_padding(dec_pos, dec_layer): continue
    print ('dec_layer_index', dec_layer_index)
        
    tot_conds=cond_cover.size
    if is_conv_layer(cond_layer):
      ks=cond_layer.kernel_size
      tot_conds-=ks[0]*ks[1]

    non_increasing=0
    step_coverage=0
    while not (step_coverage>=1.0 or non_increasing>=10):
      count+=1

      d_min, d_norm, new_image, old_image, old_labels, cond_diff_map=ssc_search(test_object, layer_functions, cond_layer, None, dec_layer, dec_pos, adv_crafter)

      print ('d_min is', d_min, 'd_norm is', d_norm)

      feasible=(d_min<=test_object.cond_ratio*cond_layer.ssc_map.size or d_min==1)

      top1_adv_flag=False
      top5_adv_flag=False
      y1s=[]
      y2s=[]
      y1_flag=False
      y2_flag=False
      labels=test_object.labels 
      
      l0_d=None
      top_classes=test_object.top_classes
      inp_ub=test_object.inp_ub


      if feasible:
        cond_cover=np.logical_or(cond_cover, cond_diff_map)
        covered=np.count_nonzero(cond_cover)
        new_step_coverage=covered*1.0/tot_conds
        if new_step_coverage==step_coverage:
           non_increasing+=1
        else:
           non_increasing=0
        step_coverage=new_step_coverage
        
        test_cases.append((new_image, old_image))
        if inp_ub==255: 
          new_image=new_image.astype('uint8')
          old_image=old_image.astype('uint8')
          diff_image=np.abs(new_image-old_image)
        else:
          new_image_=new_image*255.0/inp_ub
          old_image_=old_image*255.0/inp_ub
          new_image_=new_image_.astype('uint8')
          old_image_=old_image_.astype('uint8')
          diff_image=np.abs(new_image_-old_image_)
        l0_d=np.count_nonzero(diff_image)/(new_image.size*1.0)
        y1s=(np.argsort(test_object.dnn.predict(np.array([new_image]))))[0][-top_classes:]
        y2s=(np.argsort(test_object.dnn.predict(np.array([old_image]))))[0][-top_classes:]


        if y1s[top_classes-1]!=y2s[top_classes-1]: top1_adv_flag=True


        if labels==None: labels=old_labels
        for label in labels:
          if label in y1s: y1_flag=True
          if label in y2s: y2_flag=True

        if y1_flag!=y2_flag: top5_adv_flag=True

        if top5_adv_flag:
          print ('found an adversarial example')
          adversarials.append((new_image, old_image))
          save_adversarial_examples([new_image/(inp_ub*1.0), '{0}-adv-{1}'.format(len(adversarials), y1s[top_classes-1])], [old_image/(inp_ub*1.0), '{0}-original-{1}'.format(len(adversarials), y2s[top_classes-1])], [diff_image/(255*1.0), '{0}-diff'.format(len(adversarials))], f_results.split('/')[0]) 
          adv_flag=True
      else:
        print ("not feasible")

      print ('f_results: ', f_results)
      f = open(f_results, "a")
      f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14}\n'.format(count, len(test_cases), len(adversarials), feasible, top1_adv_flag, top5_adv_flag, d_min, d_norm, l0_d, dec_layer.layer_index, cond_layer.ssc_map.size, y1s, y2s, tot_coverage+step_coverage/tot_conds, step_coverage))
      f.close()
      #######
      if not feasible: break
      #######
    tot_coverage+=step_coverage/tot_conds

def run_svc(test_object, outs):
  print ('To run svc\n')
  
  f_results, layer_functions, cover_layers, activations=ssc_setup(test_object, outs)

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
    
    dec_ub=dec_layer.ubs.item(dec_pos)+0.001
    #for act in activations[dec_layer.layer_index]:
    #  v=act.item(dec_pos)
    #  if v>dec_ub: dec_ub=v

    print ('dec_ub: ', dec_ub)

    d_min, d_norm, new_image, old_image=svc_search(test_object, layer_functions, cond_layer, cond_pos, dec_layer, dec_pos, adv_crafter, dec_ub)

    print ('d_min is', d_min, 'd_norm is', d_norm)

    feasible=(d_min<=test_object.cond_ratio*cond_layer.ssc_map.size or d_min==1)

    top1_adv_flag=False
    top5_adv_flag=False
    top5b_adv_flag=False
    y1s=[]
    y2s=[]
    y1_flag=False
    y2_flag=False
    labels=test_object.labels #[555, 920]
    
    l0_d=None
    top_classes=test_object.top_classes
    inp_ub=test_object.inp_ub

    if feasible: 
      test_cases.append((new_image, old_image))
      if inp_ub==255: 
        new_image=new_image.astype('uint8')
        old_image=old_image.astype('uint8')
      diff_image=np.abs(new_image-old_image)
      l0_d=np.count_nonzero(diff_image)/(new_image.size*1.0)
      y1s=(np.argsort(test_object.dnn.predict(np.array([new_image]))))[0][-top_classes:]
      y2s=(np.argsort(test_object.dnn.predict(np.array([old_image]))))[0][-top_classes:]


      if y1s[top_classes-1]!=y2s[top_classes-1]: top1_adv_flag=True

      if not y1s[top_classes-1] in y2s: top5b_adv_flag=True

      for label in labels:
        if label in y1s: y1_flag=True
        if label in y2s: y2_flag=True

      if y1_flag!=y2_flag: top5_adv_flag=True

      if top5_adv_flag:
        print ('found an adversarial example')
        adversarials.append((new_image, old_image))
        save_an_image(new_image/(inp_ub*1.0), '{0}-adv-{1}.png'.format(len(adversarials), y1s[top_classes-1]), f_results.split('/')[0])
        save_an_image(old_image/(inp_ub*1.0), '{0}-original-{1}.png'.format(len(adversarials), y2s[top_classes-1]), f_results.split('/')[0])
        save_an_image(diff_image/(inp_ub*1.0), '{0}-diff.png'.format(len(adversarials)), f_results.split('/')[0])
        adv_flag=True
    else:
      print ("not feasible")

    print ('f_results: ', f_results)
    f = open(f_results, "a")
    f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}\n'.format(count, len(test_cases), len(adversarials), feasible, top1_adv_flag, top5_adv_flag, top5b_adv_flag, d_min, d_norm, l0_d, dec_layer.layer_index, cond_layer.ssc_map.size, y1s, y2s))
    f.close()

