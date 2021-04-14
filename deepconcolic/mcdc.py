## An API to call MCDC for DNNs externally
import sys
import numpy as np
from utils import *
from utils_io import tempdir
from nc_setup import *
from ssc import *
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier

class adv_objectt:
  def __init__(self, max_v, lb_v, ub_v):
    self.max_v=max_v
    self.lb_v=lb_v
    self.ub_v=ub_v

def mcdc(x, dnn, aveImg_binary, mcdc_cond_ratio=0.2, max_v=255, lb_v=-125.5, ub_v=125.5, opt=True, num=None, tot_iters=1000, outs = tempdir):
  x_test=np.array([x])
  raw_data=raw_datat(x_test,None)
  test_object=test_objectt(dnn, raw_data, 'ssc', 'linf')
  test_object.cond_ratio=mcdc_cond_ratio
  adv_object=adv_objectt(max_v, lb_v, ub_v)
  predictResults = dnn.predict(np.array([x]), verbose=1)
  res=np.argmax(predictResults)
  f_results, layer_functions, cover_layers, _ = ssc_setup(test_object, outs)

  d_advs=[]
  append_in_file (f_results,
                  '#ssc runs;  #test cases;  #adversarial examples;  is feasible; is top-1 adversarial example; is top-x adversarial example; condition feature size; L infinity distance; L0 distance; decision layer index; dec feature; #condition layer neurons; new labels; original labels; coverage; local coverage\n')

  if not (num is None):
    new_images=[]

  tot_decs=0
  if test_object.layer_indices==None:
    for i in range(1, len(cover_layers)):
      if i==1 and is_input_layer(test_object.dnn.layers[0]): continue
      else:
        csp=cover_layers[i].layer.input.shape
        dsp=cover_layers[i].ssc_map.shape
        if is_dense_layer(cover_layers[i].layer) or not (csp[1]==dsp[1] and csp[2]==dsp[2]):
          tot_decs+=cover_layers[i].ssc_map.size
        else:
          ks=cover_layers[i].layer.kernel_size
          sp=cover_layers[i].ssc_map.shape
          tot_decs+=((sp[1]-ks[0]+1)*(sp[2]-ks[1]+1)*sp[3])
  else:
    print (test_object.layer_indices, test_object.feature_indices)
    for i in range(1, len(cover_layers)):
      if cover_layers[i].layer_index in test_object.layer_indices:
        print ('****', i)
        csp=cover_layers[i].layer.input.shape
        dsp=cover_layers[i].ssc_map.shape
        if is_dense_layer(cover_layers[i].layer) or not (csp[1]==dsp[1] and csp[2]==dsp[2]):
          tmp_decs=cover_layers[i].ssc_map.size
        else:
          ks=cover_layers[i].layer.kernel_size
          dsp=cover_layers[i].ssc_map.shape
          tmp_decs=((dsp[1]-ks[0]+1)*(dsp[2]-ks[1]+1)*dsp[3])
        if is_conv_layer(cover_layers[i].layer):
          if not test_object.feature_indices==None:
             tmp_decs=tmp_decs*(len(test_object.feature_indices)*1.0/dsp[3])
        tot_decs+=tmp_decs
  tot_coverage=0.0

  ## define a global attacker
  #classifier=KerasClassifier((MIN, -MIN), model=test_object.dnn)
  classifier=KerasClassifier(test_object.dnn)
  adv_crafter = FastGradientMethod(classifier)

  test_cases=[]
  adversarials=[]
  count=0

  while count<tot_iters:
    dec_layer_index, dec_pos=get_ssc_next(cover_layers)
    cover_layers[dec_layer_index].ssc_map.itemset(dec_pos, False)
    if dec_layer_index==1 and is_input_layer(test_object.dnn.layers[0]): continue
    #print (dec_layer_index, dec_pos)
    ###
    cond_layer=cover_layers[dec_layer_index-1]
    dec_layer=cover_layers[dec_layer_index]
    cond_cover=np.zeros(cond_layer.ssc_map.shape, dtype=bool)
    ###

    if is_padding(dec_pos, dec_layer, cond_layer, False):
      continue

    tot_conds=cond_cover.size
    if is_conv_layer(cond_layer.layer):
      csp=cond_layer.layer.input.shape
      dsp=cond_layer.ssc_map.shape
      if (csp[1]==dsp[1] and csp[2]==dsp[2]):
        ks=cond_layer.layer.kernel_size
        dsp=cond_layer.ssc_map.shape
        tot_decs=((dsp[1]-ks[0]+1)*(dsp[2]-ks[1]+1)*dsp[3])

    non_increasing=0
    step_coverage=0
    count+=1
    d_min, d_norm, new_image, old_image, old_labels, cond_diff_map=ssc_search(test_object, layer_functions, cond_layer, None, dec_layer, dec_pos, adv_crafter, adv_object)
    #print ('d_min is', d_min, 'd_norm is', d_norm)
    cond_ratio=test_object.cond_ratio
    feasible=(d_min<=cond_ratio*cond_layer.ssc_map.size or d_min==1)
    if feasible:
      new_predictResults = dnn.predict(np.array([new_image]), verbose=1)
      new_res=np.argmax(new_predictResults)
      #print ('####', res, new_res, x.shape)
      if res==new_res: continue

      ## to optimise the adversarial example
      if opt:
        for i in range(0, len(x)):
          simple_x=x.copy()
          for ii in range(0, i+1):
            simple_x[ii]=new_image[ii]
          simple_predictResults = dnn.predict(np.array([simple_x]), batch_size=5000, verbose=1)
          simple_res=np.argmax(simple_predictResults)
          if simple_res==res: continue
          #for ii in range(0, i+1):
          #  plt.imshow(simple_x[ii],cmap='gray')
          #  #plt.imsave('new_{0}.png'.format(ii),simple_x[ii],cmap='gray')
          #  plt.show()
          #  plt.imshow(x[ii],cmap='gray')
          #  plt.show()
          ##  #plt.imsave('origin_{0}.png'.format(ii),x[ii],cmap='gray')
          if num is None: return True, simple_x
          else: new_images.append(simple_x)
      else:
          if num is None: return True, new_image
          else: new_images.append(new_image)
    if not (num is None):
      if len(new_images)>=num: return True, np.array(new_images)

  if (num is None): return False, None
  else: return False, np.array(new_images)
  
  
def mcdc_regression_linf(x, dnn, aveImg_binary, regression_threshold = 0.5, mcdc_cond_ratio=0.2, max_v=255, lb_v=-125.5, ub_v=125.5, opt=True, outs = tempdir):
  x_test=np.array([x])
  raw_data=raw_datat(x_test,None)
  test_object=test_objectt(dnn, raw_data, 'ssc', 'linf')
  test_object.cond_ratio=mcdc_cond_ratio
  adv_object=adv_objectt(max_v, lb_v, ub_v)
  predictResults = dnn.predict(np.array([x]), verbose=1)
  res=predictResults # np.argmax(predictResults)
  f_results, layer_functions, cover_layers, _ = ssc_setup(test_object, outs)

  d_advs=[]
  append_in_file (f_results,
                  '#ssc runs;  #test cases;  #adversarial examples;  is feasible; is top-1 adversarial example; is top-x adversarial example; condition feature size; L infinity distance; L0 distance; decision layer index; dec feature; #condition layer neurons; new labels; original labels; coverage; local coverage\n')

  tot_decs=0
  if test_object.layer_indices==None:
    for i in range(1, len(cover_layers)):
      if i==1 and is_input_layer(test_object.dnn.layers[0]): continue
      else:
        csp=cover_layers[i].layer.input.shape
        dsp=cover_layers[i].ssc_map.shape
        if is_dense_layer(cover_layers[i].layer) or not (csp[1]==dsp[1] and csp[2]==dsp[2]):
          tot_decs+=cover_layers[i].ssc_map.size
        else:
          ks=cover_layers[i].layer.kernel_size
          sp=cover_layers[i].ssc_map.shape
          tot_decs+=((sp[1]-ks[0]+1)*(sp[2]-ks[1]+1)*sp[3])
  else:
    print (test_object.layer_indices, test_object.feature_indices)
    for i in range(1, len(cover_layers)):
      if cover_layers[i].layer_index in test_object.layer_indices:
        print ('****', i)
        csp=cover_layers[i].layer.input.shape
        dsp=cover_layers[i].ssc_map.shape
        if is_dense_layer(cover_layers[i].layer) or not (csp[1]==dsp[1] and csp[2]==dsp[2]):
          tmp_decs=cover_layers[i].ssc_map.size
        else:
          ks=cover_layers[i].layer.kernel_size
          dsp=cover_layers[i].ssc_map.shape
          tmp_decs=((dsp[1]-ks[0]+1)*(dsp[2]-ks[1]+1)*dsp[3])
        if is_conv_layer(cover_layers[i].layer):
          if not test_object.feature_indices==None:
             tmp_decs=tmp_decs*(len(test_object.feature_indices)*1.0/dsp[3])
        tot_decs+=tmp_decs
  tot_coverage=0.0

  ## define a global attacker
  #classifier=KerasClassifier((MIN, -MIN), model=test_object.dnn)
  classifier=KerasClassifier(test_object.dnn)
  adv_crafter = FastGradientMethod(classifier)

  test_cases=[]
  adversarials=[]
  count=0

  while count<1000:
    dec_layer_index, dec_pos=get_ssc_next(cover_layers)
    cover_layers[dec_layer_index].ssc_map.itemset(dec_pos, False)
    if dec_layer_index==1 and is_input_layer(test_object.dnn.layers[0]): continue
    #print (dec_layer_index, dec_pos)
    ###
    cond_layer=cover_layers[dec_layer_index-1]
    dec_layer=cover_layers[dec_layer_index]
    cond_cover=np.zeros(cond_layer.ssc_map.shape, dtype=bool)
    ###

    if is_padding(dec_pos, dec_layer, cond_layer, False):
      continue

    tot_conds=cond_cover.size
    if is_conv_layer(cond_layer.layer):
      csp=cond_layer.layer.input.shape
      dsp=cond_layer.ssc_map.shape
      if (csp[1]==dsp[1] and csp[2]==dsp[2]):
        ks=cond_layer.layer.kernel_size
        dsp=cond_layer.ssc_map.shape
        tot_decs=((dsp[1]-ks[0]+1)*(dsp[2]-ks[1]+1)*dsp[3])

    non_increasing=0
    step_coverage=0
    count+=1
    d_min, d_norm, new_image, old_image, old_labels, cond_diff_map=ssc_search(test_object, layer_functions, cond_layer, None, dec_layer, dec_pos, adv_crafter, adv_object)
    #print ('d_min is', d_min, 'd_norm is', d_norm)
    cond_ratio=test_object.cond_ratio
    feasible=(d_min<=cond_ratio*cond_layer.ssc_map.size or d_min==1)
    if feasible:
      new_predictResults = dnn.predict(np.array([new_image]), verbose=1)
      new_res= new_predictResults # np.argmax(new_predictResults)
      #print ('####', res, new_res, x.shape)
      #if res==new_res: continue
      if np.amax(np.absolute(res-new_res)) < threshold: continue 
      ## to optimise the adversarial example
      if opt:
        for i in range(0, len(x)):
          simple_x=x.copy()
          for ii in range(0, i+1):
            simple_x[ii]=new_image[ii]
          simple_predictResults = dnn.predict(np.array([simple_x]), batch_size=5000, verbose=1)
          simple_res= simple_predictResults #np.argmax(simple_predictResults)
          #if simple_res==res: continue
          if np.amax(np.absolute(res-simple_res)) < threshold: continue 
          #for ii in range(0, i+1):
          #  plt.imshow(simple_x[ii],cmap='gray')
          #  #plt.imsave('new_{0}.png'.format(ii),simple_x[ii],cmap='gray')
          #  plt.show()
          #  plt.imshow(x[ii],cmap='gray')
          #  plt.show()
          ##  #plt.imsave('origin_{0}.png'.format(ii),x[ii],cmap='gray')
          return True, simple_x
      return True, new_image

  return False, None

