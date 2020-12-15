"""
Neuron Cover of MNIST based on L-infinity Norm

Author: Youcheng Sun
Email: youcheng.sun@cs.ox.ac.uk
"""

import argparse
import sys
from datetime import datetime

import keras
from keras.datasets import mnist
from keras.models import *
from keras.layers import *
from keras import *

from lp_nc import *
from utils import *

########################################################################
minus_inf = -10000000000
positive_inf = +10000000000
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# if K.image_data_format() == 'channels_first':
#    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#    input_shape = (1, img_rows, img_cols)
# else:
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
########################################################################

### it is handy to have a small enough epsilon
epsilon = 0.00001


class effective_layert:
    def __init__(self, layer_index, current_layer, is_conv=False):
        self.layer_index = layer_index
        self.activations = []
        self.is_conv = is_conv
        self.current_layer = current_layer
        self.fk = 1.0
        sp = current_layer.output.shape
        if is_conv:
            # self.cover_map=np.ones((1, sp[1], sp[2], sp[3]))
            self.cover_map = np.zeros((1, sp[1], sp[2], sp[3]), dtype=bool)
        else:
            # self.cover_map=np.ones((1, sp[1]))
            self.cover_map = np.zeros((1, sp[1]), dtype=bool)
        print 'Created an effective layer: [is_conv {0}] [cover_map {1}]'.format(is_conv, self.cover_map.shape)

    ## return the argmax and the max value:
    def get_max(self):
        sp = np.array(self.activations).argmax()
        sp1 = np.unravel_index(sp, np.array(self.activations).shape)
        if self.is_conv:
            return sp, self.activations[sp1[0]][sp1[1]][sp1[2]][sp1[3]][sp1[4]]
        else:
            return sp, self.activations[sp1[0]][sp1[1]][sp1[2]]

    def update_activations(self):
        # sp=np.unravel_index(index, np.array(self.activations).shape)
        for i in range(0, len(self.activations)):
            self.activations[i] = np.multiply(self.activations[i], self.cover_map)

    def cover_an_activation(self, sp):
        if self.is_conv:
            self.activations[sp[0]][sp[1]][sp[2]][sp[3]][sp[4]] = 0  # minus_inf
            print 'cover an activation', sp, self.activations[sp[0]][sp[1]][sp[2]][sp[3]][sp[4]]
        else:
            self.activations[sp[0]][sp[1]][sp[2]] = 0  # minus_inf


def get_max(effective_layers):
    nc_layer, nc_index, nc_value = None, -1, minus_inf
    for layer in effective_layers:
        index, v = layer.get_max()
        v = v * layer.fk
        if v > nc_value:
            nc_layer = layer
            nc_index = index
            nc_value = v
    return nc_layer, nc_index, nc_value


def cover_level(effective_layers):
    covered = 0
    non_covered = 0
    for layer in effective_layers:
        c = np.count_nonzero(layer.cover_map)
        sp = layer.cover_map.shape
        tot = 0
        if layer.is_conv:
            tot = sp[0] * sp[1] * sp[2] * sp[3]
        else:
            tot = sp[0] * sp[1]
        non_covered += c
        covered += (tot - c)
    ##  print covered, non_covered
    return covered, non_covered


def update_effective_layers(effective_layers, layer_functions, im, nc_index=None, nc_layer=None):
    activations = eval(layer_functions, im)
    for layer in effective_layers:
        act = (activations[layer.layer_index])
        # if nc_layer.layer_index==layer.layer_index:
        #  sp=np.unravel_index(nc_index, (act).shape)
        #  if len(sp)>3:
        #    print '### #############: {0}'.format(sp)
        #    print act.shape
        #    print nc_index
        #    print act[sp[0]][sp[1]][sp[2]][sp[3]]
        #    if act[sp[0]][sp[1]][sp[2]][sp[3]] < 0:
        #      act[sp[0]][sp[1]][sp[2]][sp[3]]=0 #sys.exit(0)
        #    print '### #############'
        #  else:
        #    print '### #############: {0}'.format(sp)
        #    print act.shape
        #    print nc_index
        #    print act[sp[0]][sp[1]]
        #    if act[sp[0]][sp[1]] < 0:
        #      act[sp[0]][sp[1]]=0 #  sys.exit(0)
        #    print '### #############'

        ## to update the activations
        act[act >= 0] = 1
        act = 1.0 / act
        act[act >= 0] = 0
        act = np.abs(act)
        # layer.cover_map=np.array(np.multiply(layer.cover_map, act), dtype=bool)
        layer.cover_map = np.logical_and(layer.cover_map, act)
        # if nc_layer.layer_index==layer.layer_index:
        #  if len(sp)>3:
        #    print layer.cover_map.shape
        #    print 'cover_map', layer.cover_map[0][sp[1]][sp[2]][sp[3]]
        layer.activations.append(act)
        # if nc_layer.layer_index==layer.layer_index:
        #  if len(sp)>3:
        #    print 'before update'
        #    for act in layer.activations:
        #      print act[sp[0]][sp[1]][sp[2]][sp[3]]
        #  else:
        #    print 'before update'
        #    for act in layer.activations:
        #      print act[sp[0]][sp[1]]
        layer.update_activations()
        # if nc_layer.layer_index==layer.layer_index:
        #  if len(sp)>3:
        #    print 'after update'
        #    for act in layer.activations:
        #      print act[sp[0]][sp[1]][sp[2]][sp[3]]
        #  else:
        #    print 'after update'
        #    for act in layer.activations:
        #      print act[sp[0]][sp[1]]


def initialise_effective_layers(effective_layers, layer_functions, im):
    activations = eval(layer_functions, im)
    for layer in effective_layers:
        act = activations[layer.layer_index]
        act[act >= 0] = 1
        act = 1.0 / act
        act[act > 0] = 0
        act = np.abs(act)
        layer.activations.append(act)
        layer.cover_map = np.logical_or(layer.cover_map, act)


def run_concolic_nc(model):
    ##
    outs = "concolic-nc" + str(datetime.now()).replace(' ', '-') + '/'
    os.system('mkdir -p {0}'.format(outs))

    #### configuration phase
    layer_functions = []
    effective_layers = []
    test_cases = []
    adversarials = []

    for l in range(0, len(model.layers)):
        layer = model.layers[l]
        name = layer.name

        get_current_layer_output = K.function([layer.input], [layer.output])
        layer_functions.append(get_current_layer_output)

        if is_conv_layer(layer) or is_dense_layer(layer):
            effective_layers.append(effective_layert(layer_index=l, current_layer=layer, is_conv=is_conv_layer(layer)))

    ## to configure 'fk' at each effective layer
    activations = eval_batch(layer_functions, x_train[0:10000])
    fks = []
    for elayer in effective_layers:
        index = elayer.layer_index
        sub_acts = np.abs(activations[index])
        fks.append(np.average(sub_acts))
    av = np.average(np.array(fks))
    for i in range(0, len(fks)):
        effective_layers[i].fk = av / fks[i]
        print effective_layers[i].fk
    ##sys.exit(0)
    ##

    iseed = np.random.randint(0, len(x_test))

    im_seed = x_test[iseed]  ## the initial input
    test_cases.append(im_seed)
    ## model.summary()
    initialise_effective_layers(effective_layers, layer_functions, im_seed)
    y = model.predict_classes(np.array([im_seed]))[0]
    show_adversarial_examples([im_seed, im_seed], [y, y], outs + '{0}.pdf'.format(iseed))
    # for i in range(1, 9):
    #  I=np.random.randint(1, 10000)
    #  test_cases.append(x_test[I])
    #  update_effective_layers(effective_layers, layer_functions, x_test[I])

    covered, non_covered = cover_level(effective_layers)
    f = open(outs + '{0}.results.txt'.format(iseed), "a")
    f.write('NC-cover {0} {1} {2} seed: {3}\n'.format(1.0 * covered / (covered + non_covered), len(test_cases),
                                                      len(adversarials), iseed))
    f.close()

    while True:

        ## to choose the max
        nc_layer, nc_index, nc_value = get_max(effective_layers)
        pos = np.unravel_index(nc_index, np.array(nc_layer.activations).shape)
        print pos
        print len(test_cases)
        im = test_cases[pos[0]]
        activations = eval(layer_functions, im)
        print 'The chosen test is {0}: {1} at layer {2}'.format(pos, nc_value, nc_layer.layer_index)
        if len(pos) > 3:
            print nc_layer.activations[pos[0]][pos[1]][pos[2]][pos[3]][pos[4]]
            print activations[nc_layer.layer_index][pos[1]][pos[2]][pos[3]][pos[4]]
        else:
            print nc_layer.activations[pos[0]][pos[1]][pos[2]]
            print activations[nc_layer.layer_index][pos[1]][pos[2]]

        ##
        sp = np.array(nc_layer.activations).shape
        act_size = sp[2]
        if len(sp) > 4:
            act_size = sp[2] * sp[3] * sp[4]
        nc_index_ = nc_index - (pos[0]) * act_size
        feasible, d, new_im = negate(model, activations, nc_layer, nc_index_, im)
        # feasible, d, new_im=negate(model, activations, nc_layer, nc_index_, im_seed)
        ##

        print 'lp done'

        if feasible:
            print 'feasible: d={0}'.format(d)
            found_before = False
            for t in test_cases:
                if (new_im == t).all():
                    found_before = True
                    print 'found'
                    ##sys.exit(0)
                    break
            if found_before:
                nc_layer.cover_an_activation(pos)
                continue
            test_cases.append(new_im)
            print  '\n[{0}][{1}]'.format(model.predict_classes(np.array([im]))[0],
                                         model.predict_classes(np.array([new_im]))[0])
            y1 = model.predict_classes(np.array([im]))[0]
            y2 = model.predict_classes(np.array([new_im]))[0]
            if y1 != y2:
                adversarials.append([im, new_im])
                show_adversarial_examples([im, new_im], [y1, y2], outs + 'ad{0}-d{1}.pdf'.format(len(adversarials), d))

            nc_layer.cover_an_activation(pos)
            update_effective_layers(effective_layers, layer_functions, new_im, nc_index_, nc_layer)
            ##  nc_layer.update_cover_map(nc_index)
            # nc_layer.update_cover_map()

        else:
            nc_layer.cover_an_activation(pos)

        covered, non_covered = cover_level(effective_layers)

        f = open(outs + '{0}.results.txt'.format(iseed), "a")
        f.write('NC-cover {0} {1} {2} {3} {4}-{5}\n'.format(1.0 * covered / (covered + non_covered), len(test_cases),
                                                            len(adversarials), d, nc_layer.layer_index, pos))
        f.close()

    print 'All properties have been covered'


def main():
    parser = argparse.ArgumentParser(
        description='To convert a DNN model to the C program')

    parser.add_argument('model', action='store', nargs='+', help='The input neural network model (.h5)')

    args = parser.parse_args()
    model = load_model(args.model[0])

    run_concolic_nc(model)


if __name__ == "__main__":
    main()
