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
from nc_l0 import *

assert False

def run_nc_l0(test_object, outs):
  nc_results, cover_layers, _, test_cases = nc_setup(test_object, outs)
  iteration, adversarials, d_advs = 1, [], []

  while True:
    nc_layer, nc_pos, nc_value = test_object.get_nc_next(cover_layers)
    pos, nc_location = nc_layer.locate (nc_pos)

    im = test_cases[pos[0]]
    y_im = np.argmax(test_object.dnn.predict(np.array([im])))
    
    feasible, d, new_im = l0_negate (test_object.dnn, [im], nc_layer, nc_location)

    nc_layer.disable_by_pos (pos)

    d_adv=-1
    if feasible:
      if l0_filtered(test_object.raw_data.data, new_im): 
        continue
      test_cases.append(new_im)
      test_object.eval_and_update (cover_layers, new_im)
      y = np.argmax(test_object.dnn.predict(np.array([new_im])))
      if y != y_im:
        adversarials.append([im, new_im])
        test_object.save_adversarial_examples((new_im, '{0}-adv-{1}'.format(len(adversarials), y)),
                                              (im, '{0}-original-{1}'.format(len(adversarials), y_im)),
                                              directory = outs)
        # inp_ub=test_object.inp_ub
        # save_adversarial_examples([new_im/(inp_ub*1.0), '{0}-adv-{1}'.format(len(adversarials), y)], [im/(inp_ub*1.0), '{0}-original-{1}'.format(len(adversarials), y_im)], None, nc_results.split('/')[0]) 
        d_adv=(np.count_nonzero(im-new_im))
        d_advs.append(d_adv)
        if len(d_advs)%100==0:
          print_adversarial_distribution(d_advs, nc_results.replace('.txt', '')+'-adversarial-distribution.txt', True)

    coverage = test_object.nc_report (cover_layers)
    # covered, not_covered = test_object.nc_report (cover_layers)
    # nc_percentage = 1.0 * covered / (covered + not_covered)
    print ('Current neuron coverage: {0} (iteration {1})'.format (coverage, iteration),
           end = '\r')
    append_in_file (nc_results,
                    'NC-cover: {0} #test cases: {1} #adversarial examples: {2} '
                    .format(coverage, len(test_cases), len(adversarials)),
                    '#diff: {0} #layer: {1} #pos: {2}\n'
                    .format(d_adv, nc_layer.layer_index, nc_pos))

    if coverage.not_covered == 0: break
    iteration += 1
