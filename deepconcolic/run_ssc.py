import argparse
import sys
import os
from datetime import datetime

from tensorflow.keras.models import *
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import *
from tensorflow.keras import *
from utils import *
from ssc import *

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier

# def run_ssc(test_object, outs):
#   f_results, cover_layers, _ = ssc_setup (test_object, outs)
#   d_advs=[]

#   append_in_file (f_results,
#                   '#ssc runs;  #test cases;  #adversarial examples;  is feasible; is top-1 adversarial example; is top-x adversarial example; condition feature size; L infinity distance; L0 distance; decision layer index; dec feature; #condition layer neurons; new labels; original labels; coverage; local coverage\n')

#   tot_decs=0
#   if test_object.layer_indices==None:
#     # for i in range(1, len(cover_layers)):
#     for cl in cover_layers:
#       assert not (is_input_layer(test_object.dnn.layers[cl.layer_index - 1]))
#       csp=cl.layer.input.shape
#       dsp=cl.ssc_map.shape
#       if is_dense_layer(cl.layer) or not (csp[1]==dsp[1] and csp[2]==dsp[2]): 
#         tot_decs+=cl.ssc_map.size
#       else:
#         ks=cl.layer.kernel_size
#         sp=cl.ssc_map.shape
#         tot_decs+=((sp[1]-ks[0]+1)*(sp[2]-ks[1]+1)*sp[3])
#   else:
#     print (test_object.layer_indices, test_object.feature_indices)
#     for cl in cover_layers:
#       assert not (is_input_layer(test_object.dnn.layers[cl.layer_index - 1]))
#       if cl.layer_index in test_object.layer_indices:
#         csp=cl.layer.input.shape
#         dsp=cl.ssc_map.shape
#         if is_dense_layer(cl.layer) or not (csp[1]==dsp[1] and csp[2]==dsp[2]): 
#           tmp_decs=cl.ssc_map.size
#         else:
#           ks=cl.layer.kernel_size
#           dsp=cl.ssc_map.shape
#           tmp_decs=((dsp[1]-ks[0]+1)*(dsp[2]-ks[1]+1)*dsp[3])
#         if is_conv_layer(cl.layer):
#           if not test_object.feature_indices==None:
#              # print ('**', tmp_decs)
#              tmp_decs=tmp_decs*(len(test_object.feature_indices)*1.0/dsp[3])
#              # print ('**', tmp_decs)
#         tot_decs+=tmp_decs
#   print ('== Total decisions: {0} ==\n'.format(tot_decs))
#   tot_coverage=0.0

#   ## define a global attacker
#   classifier=KerasClassifier(clip_values=(MIN, -MIN), model=test_object.dnn)
#   # print (classifier.__bases__)
#   # classifier.run_eagerly = True
#   adv_crafter = FastGradientMethod(classifier)
#   # print (adv_crafter.__bases__)

#   test_cases=[]
#   adversarials=[]
#   count=0

#   print ('== Enter the coverage loop ==\n')
#   ite=0
#   while True:
#     ite+=1
#     dec_layer_index, dec_pos=get_ssc_next(cover_layers, test_object.layer_indices, test_object.feature_indices)
#     dec_layer=cover_layers[dec_layer_index]
#     dec_layer.ssc_map.itemset(dec_pos, False)

#     assert dec_layer.prev_layer_index is not None
#     cond_layer = test_object.dnn.layers[dec_layer.prev_layer_index]

#     if is_padding(dec_pos, dec_layer, cond_layer, post = True):
#       print ('padding')
#       continue

#     cond_cover = np.zeros(cond_layer.output.shape[1:], dtype=bool)

#     tot_conds = cond_cover.size
#     if is_conv_layer(cond_layer):
#       csp = dec_layer.layer.input.shape
#       dsp = cond_layer.output.shape
#       if (csp[1]==dsp[1] and csp[2]==dsp[2]): 
#         ks = cond_layer.kernel_size
#         tot_conds = ((dsp[1]-ks[0]+1)*(dsp[2]-ks[1]+1)*dsp[3])

#     print ('==== Decision layer: {0}, decision pos: {1} ===='.format(dec_layer, dec_pos))
#     print ('==== Conditions layer: {0} ====\n'.format(cond_layer.name))

#     non_increasing=0
#     step_coverage=0
#     while not (step_coverage>=1.0 or non_increasing>=10):
#       count+=1

#       d_min, d_norm, new_image, old_image, old_labels, cond_diff_map = ssc_search(test_object, cond_layer, None, dec_layer, dec_pos, adv_crafter)

#       print ('====== #Condition changes: {0}, norm distance: {1} ======\n'.format( d_min, d_norm))

#       feasible=(d_min<=test_object.cond_ratio*np.prod(cond_layer.output.shape[1:]) or d_min==1)

#       top1_adv_flag=False
#       top5_adv_flag=False
#       y1s=[]
#       y2s=[]
#       y1_flag=False
#       y2_flag=False
#       labels=test_object.labels 

#       l0_d=None
#       top_classes=test_object.top_classes
#       inp_ub=test_object.inp_ub

#       found_new=True
#       if feasible:
#         cond_cover=np.logical_or(cond_cover, cond_diff_map)
#         covered=np.count_nonzero(cond_cover)
#         new_step_coverage=covered*1.0/tot_conds
#         if new_step_coverage==step_coverage:
#            non_increasing+=1
#            found_new=False
#         else:
#            non_increasing=0
#         step_coverage=new_step_coverage

#       if feasible and found_new:
        
#         test_cases.append((new_image, old_image))
#         if inp_ub==255: 
#           new_image=new_image.astype('uint8')
#           old_image=old_image.astype('uint8')
#           diff_image=np.abs(new_image-old_image)
#         else:
#           new_image_=new_image*255.0/inp_ub
#           old_image_=old_image*255.0/inp_ub
#           new_image_=new_image_.astype('uint8')
#           old_image_=old_image_.astype('uint8')
#           diff_image=np.abs(new_image_-old_image_)
#         l0_d=np.count_nonzero(diff_image)/(new_image.size*1.0)
#         y1s=(np.argsort(test_object.dnn.predict(np.array([new_image]))))[0][-top_classes:]
#         y2s=(np.argsort(test_object.dnn.predict(np.array([old_image]))))[0][-top_classes:]


#         if y1s[top_classes-1]!=y2s[top_classes-1]: top1_adv_flag=True


#         if labels==None: labels=old_labels
#         #print (labels, y1s, y2s)
#         for label in labels:
#           if label in y1s: y1_flag=True
#           if label in y2s: y2_flag=True

#         if y1_flag!=y2_flag: top5_adv_flag=True

#         if top5_adv_flag:
#           print ('******** This is an adversarial example ********\n')
#           adversarials.append((new_image, old_image))
#           test_object.save_adversarial_example (
#             (new_image, '{0}-adv-{1}'.format(len(adversarials), y1s[top_classes-1])),
#             (old_image, '{0}-original-{1}'.format(len(adversarials), y2s[top_classes-1])),
#             diff = (diff_image, '{0}-diff'.format(len(adversarials))),
#             directory = outs)
#           adv_flag=True
#           d_advs.append(d_norm)
#           if len(d_advs)%100==0:
#             print_adversarial_distribution(d_advs, f_results.replace('.txt', '')+'-adversarial-distribution.txt')
#         #elif y1s[0]==y2s[0]:
#         #  adversarials.append((new_image, old_image))
#         #  save_adversarial_examples([new_image/(inp_ub*1.0), 't{0}-{1}'.format(len(test_cases), y1s[top_classes-1])], [old_image/(inp_ub*1.0), 't{0}-original-{1}'.format(len(test_cases), y2s[top_classes-1])], None, f_results.split('/')[0])
#       elif feasible:
#         print ("******** Already found ********\n")
#       else:
#         print ("******** Not feasible ********\n")


#       #print ('f_results: ', f_results)
#       f = open(f_results, "a")
#       f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n'.format(count, len(test_cases), len(adversarials), feasible, top1_adv_flag, top5_adv_flag, d_min, d_norm, l0_d, dec_layer.layer_index, dec_pos, np.prod (cond_layer.output.shape[1:]), y1s, y2s, tot_coverage+step_coverage/tot_decs, step_coverage))
#       f.close()
#       #######
#       if not feasible: break
#       #######
#     tot_coverage+=step_coverage/tot_decs
#     ## todo: this is a shortcut
#     if not np.any(dec_layer.ssc_map):
#       print ('all decision features at layer {0} have been covered'.format(dec_layer.layer_index))
#       sys.exit(0)

from engine import CoverableLayer
def run_svc(test_object, outs):
  print ('To run svc\n')

  setup_layer = \
    lambda l, i, **kwds: CoverableLayer (layer = l, layer_index = i, **kwds)
  cover_layers = get_cover_layers (test_object.dnn, setup_layer,
                                   layer_indices = test_object.layer_indices,
                                   activation_of_conv_or_dense_only = True,
                                   exclude_direct_input_succ = True)
  f_results = outs.stamped_filename ('SVC_report', suff = '.txt')

  ## define a global attacker
  classifier = KerasClassifier(clip_values=(MIN, -MIN), model=test_object.dnn)
  adv_crafter = FastGradientMethod(classifier)

  test_cases=[]
  adversarials=[]

  count=0

  while True:
    dec_layer_index, dec_pos=get_ssc_next(cover_layers)

    if dec_layer_index==1 and is_input_layer(test_object.dnn.layers[0]): continue
    print ('dec_layer_index', cover_layers[dec_layer_index].layer_index)

    ###
    cond_layer=cover_layers[dec_layer_index-1]
    dec_layer=cover_layers[dec_layer_index]
    cond_cover=np.ones(cond_layer.ssc_map.shape, dtype=bool)
    ###

    ## skip if dec_pos is a padding
    if is_padding (dec_pos, dec_layer, cond_layer):
      continue

    cond_pos=np.random.randint(0, cond_cover.size)

    print ('cond, dec layer index: ', cond_layer.layer_index, dec_layer.layer_index)
    print ('dec_layer_index: ', cover_layers[dec_layer_index].layer_index)

    count+=1
    
    dec_ub=dec_layer.ubs.item(dec_pos)+0.001
    #for act in activations[dec_layer.layer_index]:
    #  v=act.item(dec_pos)
    #  if v>dec_ub: dec_ub=v

    print ('dec_ub: ', dec_ub)

    d_min, d_norm, new_image, old_image = svc_search(test_object, cond_layer, cond_pos, dec_layer, dec_pos, adv_crafter, dec_ub)

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
        save_an_image(new_image/(inp_ub*1.0), '{0}-adv-{1}.png'.format(len(adversarials), y1s[top_classes-1]),
                      f_results.split('/')[0])
        save_an_image(old_image/(inp_ub*1.0), '{0}-original-{1}.png'.format(len(adversarials), y2s[top_classes-1]),
                      f_results.split('/')[0])
        save_an_image(diff_image/(inp_ub*1.0), '{0}-diff.png'.format(len(adversarials)),
                      f_results.split('/')[0])
        adv_flag=True
    else:
      print ("not feasible")
      
    append_in_file (f_results,
                    '{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}\n'
                    .format(count, len(test_cases), len(adversarials),
                            feasible, top1_adv_flag, top5_adv_flag, top5b_adv_flag,
                            d_min, d_norm, l0_d, dec_layer.layer_index,
                            cond_layer.ssc_map.size, y1s, y2s))

