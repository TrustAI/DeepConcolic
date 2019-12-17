import argparse
import sys
import numpy as np
import copy
from utils import *


RP_SIZE=50 ## the top 50 pairs
NNUM=1000000000
EPSILON=0.00000000001 #sys.float_info.epsilon*10 #0.000000000000001
EPS_MAX=0.3


class ssc_pairt:

  def __init__(self, cond_flags, dec_flag, test_object, cond_layer, cond_pos, dec_layer, dec_pos):
    self.cond_flags=cond_flags
    self.dec_flag=dec_flag
    self.test_object=test_object
    self.cond_layer=cond_layer
    self.cond_pos=cond_pos
    self.dec_layer=dec_layer
    self.dec_pos=dec_pos

def local_search(eval_batch, local_input, ssc_pair, adv_crafter, e_max_input, ssc_ratio):

  d_min=NNUM

  e_max=e_max_input #np.random.uniform(0.2, 0.3)
  old_e_max=e_max
  e_min=0.0

  x_ret=None
  not_changed=0
  diff_map=None
  while e_max-e_min>=EPSILON:
    #print ('                     === in while')
    adv_crafter.set_params(eps=e_max)
    x_adv_vect=adv_crafter.generate(x=np.array([local_input]))
    adv_acts = eval_batch (x_adv_vect, allow_input_layer = True)
    adv_cond_flags=adv_acts[ssc_pair.dec_layer.prev_layer_index][0]
    adv_cond_flags[adv_cond_flags<=0]=0
    adv_cond_flags=adv_cond_flags.astype(bool)
    adv_dec_flag=None
    if adv_acts[ssc_pair.dec_layer.layer_index][0].item(ssc_pair.dec_pos)>0:
      adv_dec_flag=True
    else:
      adv_dec_flag=False

    found=False
    new_diff_map=None
    if ssc_pair.dec_flag != adv_dec_flag:
        new_diff_map=np.logical_xor(adv_cond_flags, ssc_pair.cond_flags)
        d=np.count_nonzero(new_diff_map)
        if d<=d_min and d>0:
          found=True

    if found:
      d_min=d
      old_e_max=e_max
      e_max=(e_max+e_min)/2
      x_ret=x_adv_vect[0]
      not_changed=0
      diff_map=new_diff_map
    else:
      e_min=e_max
      e_max=(old_e_max+e_max)/2
      not_changed+=1

    if d_min==1: break
    if d_min<=ssc_ratio* np.prod (ssc_pair.cond_layer.output.shape[1:]): break

  return d_min, x_ret, diff_map

def ssc_search(eval_batch, raw_data, cond_ratio, cond_layer, cond_pos, dec_layer, dec_pos, adv_crafter, adv_object=None):

  # NB: What's this for?
  keras.backend.set_learning_phase(False)
  import tensorflow
  try:
    sess = tensorflow.compat.v1.Session ()
    sess.run(tensorflow.compat.v1.global_variables_initializer())
  except:
    sess = keras.backend.get_session()
    sess.run(tensorflow.global_variables_initializer())

  data = raw_data.data
  labels = raw_data.labels
  ssc_ratio = cond_ratio

  x=None
  y=None
  new_x=None
  diff_map=None
  d_tot = np.prod (cond_layer.output.shape[1:])
  d_min = d_tot
  tp1 ('To catch independent condition change: {0}/{1}'.format(d_min, d_tot))

  indices=np.random.choice(len(data), len(data))

  count=0
  for i in indices:
    inp_vect=np.array([data[i]])
    if adv_object is None:
      e_max_input=np.random.uniform(EPS_MAX*2/3, EPS_MAX)
      adv_crafter.set_params(eps=e_max_input)
      adv_inp_vect=adv_crafter.generate(x=inp_vect)
    else:
      e_max_input=np.random.uniform(adv_object.max_v*EPS_MAX*2/3, adv_object.max_v*EPS_MAX)
      adv_crafter.set_params(eps=e_max_input)
      adv_inp_vect=adv_crafter.generate(x=inp_vect)
      adv_inp_vect=np.clip(adv_inp_vect, adv_object.lb_v, adv_object.max_v)
    acts = eval_batch (inp_vect, allow_input_layer = True)
    adv_acts = eval_batch (adv_inp_vect, allow_input_layer = True)
    dec1=(acts[dec_layer.layer_index][0].item(dec_pos))
    dec2=(adv_acts[dec_layer.layer_index][0].item(dec_pos))
    if not np.logical_xor(dec1>0, dec2>0): continue

    count+=1

    cond_flags=acts[dec_layer.prev_layer_index][0]
    cond_flags[cond_flags<=0]=0
    cond_flags=cond_flags.astype(bool)
    ssc_pair = ssc_pairt(cond_flags, acts[dec_layer.layer_index][0].item(dec_pos)>0, None, cond_layer, cond_pos, dec_layer, dec_pos)

    diff, x_ret, diff_map_ret = local_search(eval_batch, data[i], ssc_pair, adv_crafter, e_max_input, ssc_ratio)

    if diff<d_min:
      d_min=diff
      x=data[i]
      if labels is not None:
        y=labels[i]
      new_x=x_ret
      diff_map=diff_map_ret
      tp1 ('Update independent condition change: {0}/{1}'.format(d_min, d_tot))
      if d_min==1: break

    #print ("++++++",d_min, ssc_ratio, ssc_ratio*cond_layer.ssc_map.size)
    if d_min <= ssc_ratio * np.prod (cond_layer.output.shape[1:]): break

  #print ('final d: ', d_min, ' count:', count)
  if x is not None:
    d_norm=np.abs(new_x-x)
    return d_min, np.max(d_norm), new_x, x, [y], diff_map
  else:
    return d_min, None, None, None, None, None


def local_v_search(dnn, local_input, ssc_pair, adv_crafter, e_max_input, ssc_ratio, dec_ub):

  d_min=NNUM

  e_max=e_max_input
  old_e_max=e_max
  e_min=0.0

  x_ret=None
  not_changed=0
  while e_max-e_min>=EPSILON:
    print ('                     === in while', e_max-e_min)
    adv_crafter.set_params(eps=e_max)
    x_adv_vect=adv_crafter.generate(x=np.array([local_input]))
    adv_acts = ssc_pair.test_object.eval_batch(x_adv_vect, allow_input_layer = True)
    adv_cond_flags=adv_acts[ssc_pair.cond_layer.layer_index][0]
    adv_cond_flags[adv_cond_flags<=0]=0
    adv_cond_flags=adv_cond_flags.astype(bool)
    found=False
    if adv_acts[ssc_pair.dec_layer.layer_index][0].item(ssc_pair.dec_pos)>dec_ub:
      d=np.count_nonzero(np.logical_xor(adv_cond_flags, ssc_pair.cond_flags))
      if d<=d_min and d>0:
        found=True

    if found:
      d_min=d
      old_e_max=e_max
      e_max=(e_max+e_min)/2
      x_ret=x_adv_vect[0]
      not_changed=0
    else:
      e_min=e_max
      e_max=(old_e_max+e_max)/2
      not_changed+=1

    if d_min==1: break
    if d_min<=ssc_ratio*ssc_pair.cond_layer.ssc_map.size: break

  return d_min, x_ret

def svc_search(test_object, cond_layer, cond_pos, dec_layer, dec_pos, adv_crafter, dec_ub):

  data=test_object.raw_data.data
  dnn=test_object.dnn
  ssc_ratio=test_object.cond_ratio

  x=None
  new_x=None
  d_min=cond_layer.ssc_map.size
  print ('d_min initialised', d_min, len(data))

  indices=np.random.choice(len(data), len(data))

  count=0

  for i in indices:
    inp_vect=np.array([data[i]])
    acts = test_object.eval_batch(inp_vect, allow_input_layer = True)
    dec1=(acts[dec_layer.layer_index][0].item(dec_pos))
    if dec1<=0: continue
    if dec_ub>2*dec1: continue

    #cond1=(acts[cond_layer.layer_index][0].item(cond_pos))
    cond_flags=acts[cond_layer.layer_index][0]
    cond_flags[cond_flags<=0]=0
    cond_flags=cond_flags.astype(bool)

    #dec_ub=dec1*2 #############
    to_stop=False
    #e_inputs=np.linspace(0, 10, num=100)

    #for e_max_input in e_inputs:
    e_max_input=0
    trend=0
    old_dec=dec1
    #while e_max_input<=20 and trend>=-50:
    while e_max_input<=200 and trend>=-50:
      if e_max_input>10:
        e_max_input+=np.random.uniform(0, 1) #0.3
      elif e_max_input>1:
        e_max_input+=np.random.uniform(0, 0.1) #0.3
      else:
        e_max_input+=np.random.uniform(0, 0.05) #0.3
      adv_crafter.set_params(eps=e_max_input)
      adv_inp_vect=adv_crafter.generate(x=inp_vect)
      adv_acts = test_object.eval_batch(adv_inp_vect, allow_input_layer = True)

      dec2=(adv_acts[dec_layer.layer_index][0].item(dec_pos))
      if dec2<=old_dec:
         trend-=1
      else: trend=0
      old_dec=dec2

      #if not np.logical_xor(dec1>0, dec2>0): continue
      print ('****', e_max_input, dec1, dec2, dec_ub, dec2>dec_ub)
      if dec2<=dec_ub: continue
      #cond2=(adv_acts[cond_layer.layer_index][0].item(cond_pos))
      count+=1

      adv_cond_flags=adv_acts[cond_layer.layer_index][0]
      adv_cond_flags[adv_cond_flags<=0]=0
      adv_cond_flags=adv_cond_flags.astype(bool)
      early_d=np.count_nonzero(np.logical_xor(adv_cond_flags, cond_flags))

      if early_d<=ssc_ratio*cond_layer.ssc_map.size:
        d_min=early_d
        x=data[i]
        new_x=adv_inp_vect[0]
        to_stop=True
        break

      ssc_pair = ssc_pairt(cond_flags, acts[dec_layer.layer_index][0].item(dec_pos)>0, test_object, cond_layer, cond_pos, dec_layer, dec_pos)

      print ('start local v search')
      diff, x_ret=local_v_search(test_object.dnn, data[i], ssc_pair, adv_crafter, e_max_input, ssc_ratio, dec_ub)
      print ('after local v search')

      if diff<d_min:
        d_min=diff
        x=data[i]
        new_x=x_ret
        print ('new d: ', d_min, cond_layer.ssc_map.size)
        if d_min==1: break


      if d_min<=ssc_ratio*cond_layer.ssc_map.size: break
      ######
      break

    if d_min<=ssc_ratio*cond_layer.ssc_map.size: break

  print ('final d: ', d_min, ' count:', count)
  if x is not None:
    d_norm=np.abs(new_x-x)
    return d_min, np.max(d_norm), new_x, x
  else:
    return d_min, None, None, None


# ---


from typing import *
from engine import (BoolMappedCoverableLayer, TestTarget,
                    LayerLocalCriterion,
                    Criterion4FreeSearch, Criterion4RootedSearch,
                    Analyzer4FreeSearch, Analyzer4RootedSearch,
                    EarlyTermination)
from engine import setup as engine_setup


# ---


class SScLayer (BoolMappedCoverableLayer):

  def __init__(self, **kwds):
    super().__init__(**kwds)
    self._initialize_conditions_map ()


  def _initialize_conditions_map(self):
    self.cond_map = np.ones(self.map.shape +
                            (self.layer.kernel_size if is_conv_layer (self.layer) else
                             self.layer.get_weights ()[0].shape[:-1]),
                            dtype = bool)
    base_count = np.count_nonzero (self.cond_map)
    self.cond_map[self.map == True] = True
    self.filtered_out_conds = base_count - np.count_nonzero (self.cond_map)


  def coverage(self, _feature_indices = None) -> Coverage:
    """Ignores `feature_indices` for now."""
    nc = np.count_nonzero (self.cond_map)
    tot = np.prod (self.cond_map.shape)
    tot -= self.filtered_out_conds
    return Coverage (covered = tot - nc, total = tot)


  def cover(self, cond_pos, dec_pos) -> None:
    if cond_pos is None:
      # Assume the decision is covered when no condition position is
      # given
      self.cond_map[dec_pos, :] = False
    else:
      self.cond_map[dec_pos, cond_pos] = False

    if not self.cond_map[dec_pos].any ():
      super().cover (dec_pos)


  # def initialize_ubs(self):
  #   self.ubs = np.zeros((1,) + self.layer.output.shape[1:], dtype = float)


  def update_with_activations(self, act) -> None:
    # if self.layer_index <= 1: return

    prior_act = act[self.layer_index # - 1
                    ] # ????

    new_act = -1.0 * np.abs(np.array(prior_act))
    new_act[new_act == 0] = -0.0000001
    new_act = np.multiply(new_act, self.map)
    new_act[new_act == 0] = MIN

    self.append_activations (new_act)



# ---


class SScTarget (NamedTuple, TestTarget):
  '''
  Note positions are w.r.t a sequence of activations (i.e. actual
  neurons indexes consist in all but the first components).
  '''
  decision_layer: SScLayer
  decision_pos: Tuple[int, ...]
  condition_layer: Optional[BoolMappedCoverableLayer]
  condition_pos: Optional[Tuple[int, ...]]


  def cover(self) -> None:
    # XXX: this assumes a single condition position is eligible...
    self.decision_layer.cover (self.condition_pos[1:] if self.condition_pos is not None else None,
                               self.decision_pos)
    if self.condition_layer is not None and self.condition_pos is not None:
      self.condition_layer.cover (self.condition_pos[1:])


  @property
  def decision_position(self):
    return self.decision_pos


  @property
  def condition_position(self):
    return self.condition_pos[0] if self.condition_pos is not None else None


  def __repr__(self) -> str:
    if self.condition_pos is not None:
      return ('decision {} in {}, subject to condition {}{}'
              .format (xtuple (self.decision_pos), self.decision_layer,
                       xtuple (self.condition_pos[1:]),
                       '' if self.condition_layer is None else
                       ' in {}'.format(self.condition_layer)))
    else:
      return ('decision {} in {}, subject to any condition'
              .format (xtuple (self.decision_pos), self.decision_layer))


  def log_repr(self) -> str:
    return ('#dec_layer: {} #dec_pos: {} #cond_pos {}'
            .format(self.decision_layer,
                    xtuple (self.decision_pos),
                    ' #cond_pos {}'.format(xtuple (self.condition_pos[1:]))
                    if self.condition_pos is not None else ''))


# ---


class SScAnalyzer4FreeSearch (Analyzer4FreeSearch):
  """
  Analyzer that finds pairs of close enough inputs that fulfill a
  given SSC test target.
  """

  @abstractmethod
  def search_close_inputs(self, target: SScTarget) -> Optional[Tuple[float, Any, Any]]:
    raise NotImplementedError


# ---


class SScAnalyzer4RootedSearch (Analyzer4RootedSearch):
  """
  Analyzer that finds a new input close to a given input so that a
  given SSC test target is fulfilled.
  """

  @abstractmethod
  def search_input_close_to(self, x, target: SScTarget) -> Optional[Tuple[float, Any, Any]]:
    raise NotImplementedError


# ---


class SScCriterion (LayerLocalCriterion, Criterion4FreeSearch, Criterion4RootedSearch):

  def __init__(self,
               clayers: Sequence[SScLayer],
               analyzer: Union[SScAnalyzer4FreeSearch, SScAnalyzer4RootedSearch],
               injecting_layer: BoolMappedCoverableLayer = None,
               **kwds):

    assert (isinstance (analyzer, SScAnalyzer4FreeSearch) or
            isinstance (analyzer, SScAnalyzer4RootedSearch))
    assert isinstance (injecting_layer, BoolMappedCoverableLayer)

    super().__init__(clayers = clayers, analyzer = analyzer, **kwds)

    self.injecting_layer = injecting_layer
    self.layer_imap = { injecting_layer.layer_index: injecting_layer }
    for cl in self.cover_layers:
      self.layer_imap[cl.layer_index] = cl
    for cl in self.cover_layers:
      cl.filter_out_padding_against (self.layer_imap[cl.prev_layer_index].layer)


  @property
  def _updatable_layers(self):
    """
    Updatable layers include both layers to cover (SScLayer's) plus
    the inputing layer (a BoolMappedCoverableLayer).
    """
    return self.layer_imap.values ()


  def __repr__(self):
    return "SSC"


  def find_next_test_target(self) -> SScTarget:
    # Find a target decision at random:
    decision_search_attempt = self.get_random ()
    if decision_search_attempt == None:
      raise EarlyTermination ('All decision features have been covered.')
    dec_cl, dec_pos = decision_search_attempt
    cond_cl = self.layer_imap[dec_cl.prev_layer_index]
    assert not (is_padding (dec_pos, dec_cl, cond_cl,
                            post = True, unravel_pos = False))
    try:
      # Try and find an appropriate condition neuron (i.e. based on
      # Eq. (18)).
      cond_pos, cond_val = cond_cl.find (np.argmax)
      # print (cond_cl.activations[cond_pos[0]][cond_pos[1:]])
      cond_cl.inhibit_activation (cond_pos)
      # print (cond_cl.activations[cond_pos[0]][cond_pos[1:]])
      return SScTarget (dec_cl, dec_pos, cond_cl, cond_pos[1:])
    except ValueError:
      # XXX this case may only happen upon cold start.
      return SScTarget (dec_cl, dec_pos, None, None)
    # ppos = (lambda p: p if len(p) > 1 else p[0])(ssc_pos[1:])
    # tp1 ('Targeting decision {} in {}'.format(ppos, cl))
    # return cl, ssc_pos[1:]


  # ---

  # XXX: To be used when implementing SVcCriterion:

  # def stat_based_initializers(self):
  #   return super().stat_based_initializers () + [{
  #     'name': 'upper bounds',
  #     'accum': self._acc_ubs_max,
  #     'print': (lambda : [ np.amax (cl.ubs) for cl in self.cover_layers ]),
  #   }]

  # def _acc_ubs_max(self, new_acts, _prev_acts = None):
  #   for cl in self.cover_layers:
  #     nmax = np.amax(np.array([new_acts[cl.layer_index][0]]), axis = 0)
  #     cl.ubs = np.maximum (cl.ubs, nmax)


# ---


def setup (test_object = None, criterion_args: dict = {}, **kwds):

  setup_layer = (
    lambda l, i, **kwds: SScLayer (layer = l, layer_index = i,
                                   feature_indices = test_object.feature_indices,
                                   **kwds))
  cover_layers = get_cover_layers (test_object.dnn, setup_layer,
                                   layer_indices = test_object.layer_indices,
                                   exclude_direct_input_succ = True)
  injecting_layer_index = cover_layers[0].prev_layer_index
  criterion_args['injecting_layer'] = (
    BoolMappedCoverableLayer (layer = test_object.dnn.layers[injecting_layer_index],
                              layer_index = injecting_layer_index,
                              feature_indices = test_object.feature_indices))
  return engine_setup (test_object = test_object,
                       cover_layers = cover_layers,
                       setup_criterion = SScCriterion,
                       criterion_args = criterion_args,
                       **kwds)


# ---

# TODO: put that in a `ssc_attack' module?

try:
  from art.attacks.fast_gradient import FastGradientMethod
  from art.classifiers import KerasClassifier
except:
  from attacks import *

from norms import LInf

class SScAttackBasedAnalyzer (SScAnalyzer4FreeSearch):

  def __init__(self, _clayers, linf_args = {}, cond_ratio = 0.01, ref_data = None, **kwds):
    super().__init__(**kwds)
    self.metric = LInf (**linf_args)
    self.cond_ratio = cond_ratio
    self.ref_data = ref_data
    ## define a global attacker
    classifier = KerasClassifier(clip_values = (MIN, -MIN), model = self.dnn)
    self.adv_crafter = FastGradientMethod(classifier)


  def input_metric(self):
    return self.metric


  def search_close_inputs(self, target: SScTarget) -> Optional[Tuple[float, Any, Any]]:
    dec_layer, dec_pos, cond_pos = target.decision_layer, target.decision_position, target.condition_position
    assert dec_layer.prev_layer_index is not None
    cond_layer = self.dnn.layers[dec_layer.prev_layer_index]
    d_min, d_norm, new_image, old_image, old_labels, cond_diff_map = (
      ssc_search (self.eval_batch, self.ref_data, self.cond_ratio,
                  cond_layer, cond_pos, dec_layer, dec_pos, self.adv_crafter))
    tp1 ('#Condition changes: {0}, norm distance: {1}'.format(d_min, d_norm))
    feasible = (d_min <= self.cond_ratio * np.prod(cond_layer.output.shape[1:]) or
                d_min == 1)             # ???
    if not feasible:
      return None
    else:
      return d_norm, old_image, new_image


# ---
