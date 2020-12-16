"""
Boundary Cover of MNIST based on L-infinity Norm

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

from lp_bc import *
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
    def __init__(self, network, layer_index, current_layer, is_conv=False):
        self.layer_index = layer_index
        self.h_activations = []
        self.l_activations = []
        self.is_conv = is_conv
        self.current_layer = current_layer
        self.fk = 1.0
        self.the_feature = -1
        self.hk = None
        self.lk = None
        sp = current_layer.output.shape
        if is_conv:
            self.h_cover_map = np.zeros((1, sp[1], sp[2], sp[3]), dtype=bool)
            self.l_cover_map = np.zeros((1, sp[1], sp[2], sp[3]), dtype=bool)
            self.hk = np.empty((1, sp[1], sp[2], sp[3]))
            self.lk = np.empty((1, sp[1], sp[2], sp[3]))
            if layer_index == 0 or layer_index == 2:
                self.the_feature = np.random.randint(0, sp[3])
                self.tot = sp[1].value * sp[2].value * 2
                # print self.tot
                # sys.exit(0)
        else:
            self.h_cover_map = np.zeros((1, sp[1]), dtype=bool)
            self.l_cover_map = np.zeros((1, sp[1]), dtype=bool)
            self.hk = np.empty((1, sp[1]))
            self.lk = np.empty((1, sp[1]))
        print 'Created an effective layer: [is_conv {0}] [cover_map {1}]'.format(is_conv, self.h_cover_map.shape)

    ## return the argmax and the max value:
    def get_max(self):
        pos1 = np.array(self.h_activations).argmax()
        pos2 = np.array(self.l_activations).argmax()

        pos1b = np.unravel_index(pos1, np.array(self.h_activations).shape)
        pos2b = np.unravel_index(pos2, np.array(self.l_activations).shape)

        pos = None
        v = None
        udi = True  ## upper boundary direction

        if self.is_conv:
            vh = self.h_activations[pos1b[0]][pos1b[1]][pos1b[2]][pos1b[3]][pos1b[4]]
            vl = self.l_activations[pos2b[0]][pos2b[1]][pos2b[2]][pos2b[3]][pos2b[4]]
        else:
            vh = self.h_activations[pos1b[0]][pos1b[1]][pos1b[2]]
            vl = self.l_activations[pos2b[0]][pos2b[1]][pos2b[2]]

        if vh > vl:
            pos = pos1
            v = vh
        else:
            pos = pos2
            v = vl
            udi = False

        return pos, v, udi

    def update_activations(self, bc_udi):
        if self.the_feature < 0: return
        # sp=np.unravel_index(index, np.array(self.activations).shape)
        if bc_udi:
            for i in range(0, len(self.h_activations)):
                self.h_activations[i] = np.multiply(self.h_activations[i], self.h_cover_map)
        else:
            for i in range(0, len(self.l_activations)):
                self.l_activations[i] = np.multiply(self.l_activations[i], self.l_cover_map)

    def cover_an_activation(self, sp, bc_udi):
        if self.is_conv:
            if bc_udi:
                self.h_activations[sp[0]][sp[1]][sp[2]][sp[3]][sp[4]] = 0  # minus_inf
            else:
                self.l_activations[sp[0]][sp[1]][sp[2]][sp[3]][sp[4]] = 0  # minus_inf
        else:
            if bc_udi:
                self.h_activations[sp[0]][sp[1]][sp[2]] = 0  # minus_inf
            else:
                self.l_activations[sp[0]][sp[1]][sp[2]] = 0  # minus_inf


def get_max(effective_layers):
    bc_layer, bc_index, bc_value = None, -1, minus_inf
    bc_udi = None
    for layer in effective_layers:
        if layer.the_feature < 0: continue
        index, v, udi = layer.get_max()
        # v=v/layer.fk
        if v > bc_value:
            bc_layer = layer
            bc_index = index
            bc_value = v
            bc_udi = udi
    return bc_layer, bc_index, bc_value, bc_udi


def cover_level(effective_layers):
    covered = 0
    non_covered = 0
    for layer in effective_layers:
        if layer.the_feature < 0: continue
        c = np.count_nonzero(layer.h_cover_map)
        sp = layer.h_cover_map.shape
        non_covered += c
        covered += (layer.tot / 2 - c)
    for layer in effective_layers:
        if layer.the_feature < 0: continue
        c = np.count_nonzero(layer.l_cover_map)
        non_covered += c
        covered += (layer.tot / 2 - c)
    return covered, non_covered


def update_effective_layers(effective_layers, layer_functions, im, nc_index=None, nc_layer=None, bc_udi=None):
    activations = eval(layer_functions, im)
    for layer in effective_layers:
        if layer.the_feature < 0: continue

        h_act = np.array(activations[layer.layer_index])
        l_act = np.array(activations[layer.layer_index])

        ## to update the activations
        sp = h_act.shape
        if layer.is_conv:

            for I in range(0, sp[0]):
                for J in range(0, sp[1]):
                    for K in range(0, sp[2]):
                        for L in range(0, sp[3]):
                            if L != layer.the_feature:
                                h_act[I][J][K][L] = 0
                                l_act[I][J][K][L] = 0
                                continue

                            if bc_udi:
                                if -epsilon <= layer.hk[I][J][K][L] - h_act[I][J][K][L] and layer.hk[I][J][K][L] - \
                                        h_act[I][J][K][L] <= epsilon:
                                    h_act[I][J][K][L] = 0
                                else:
                                    h_act[I][J][K][L] = 1.0 / (layer.hk[I][J][K][L] - h_act[I][J][K][L]) * layer.fk
                            else:
                                if -epsilon <= l_act[I][J][K][L] - layer.lk[I][J][K][L] and l_act[I][J][K][L] - \
                                        layer.lk[I][J][K][L] <= epsilon:
                                    l_act[I][J][K][L] = 0
                                else:
                                    l_act[I][J][K][L] = 1.0 / (l_act[I][J][K][L] - layer.lk[I][J][K][L]) * layer.fk
        else:
            for I in range(0, sp[0]):
                for J in range(0, sp[1]):

                    if bc_udi:
                        delta = (layer.hk[I][J] - h_act[I][J])
                        if -epsilon <= delta and delta <= epsilon:
                            h_act[I][J] = 0
                        else:
                            h_act[I][J] = 1.0 / (layer.hk[I][J] - h_act[I][J]) * layer.fk
                    else:
                        delta = (l_act[I][J] - layer.lk[I][J])
                        if -epsilon <= delta and delta <= epsilon:
                            l_act[I][J] = 0
                        else:
                            l_act[I][J] = 1.0 / (l_act[I][J] - layer.lk[I][J]) * layer.fk

        if bc_udi:
            h_act[h_act < 0] = 0  ## < 0 means the neuron is covered
            layer.h_cover_map = np.logical_and(layer.h_cover_map, h_act)
            layer.h_activations.append(h_act)
        else:
            l_act[l_act < 0] = 0  ## < 0 means the neuron is covered
            layer.l_cover_map = np.logical_and(layer.l_cover_map, l_act)
            layer.l_activations.append(l_act)
        layer.update_activations(bc_udi)


def initialise_effective_layers(effective_layers, layer_functions, im):
    activations = eval(layer_functions, im)
    for layer in effective_layers:
        h_act = np.array(activations[layer.layer_index])
        l_act = np.array(activations[layer.layer_index])
        sp = h_act.shape
        if layer.the_feature < 0:
            h_act = 0
            l_act = 0
            continue

        if layer.is_conv:
            for I in range(0, sp[0]):
                for J in range(0, sp[1]):
                    for K in range(0, sp[2]):
                        for L in range(0, sp[3]):
                            if not L == layer.the_feature:
                                h_act[I][J][K][L] = 0
                                l_act[I][J][K][L] = 0
                                continue
                            if -epsilon <= layer.hk[I][J][K][L] - h_act[I][J][K][L] and layer.hk[I][J][K][L] - \
                                    h_act[I][J][K][L] <= epsilon:
                                h_act[I][J][K][L] = 0
                            else:
                                h_act[I][J][K][L] = 1.0 / (layer.hk[I][J][K][L] - h_act[I][J][K][L]) * layer.fk
                            if -epsilon <= l_act[I][J][K][L] - layer.lk[I][J][K][L] and l_act[I][J][K][L] - \
                                    layer.lk[I][J][K][L] <= epsilon:
                                l_act[I][J][K][L] = 0
                            else:
                                l_act[I][J][K][L] = 1.0 / (l_act[I][J][K][L] - layer.lk[I][J][K][L]) * layer.fk
        else:
            for I in range(0, sp[0]):
                for J in range(0, sp[1]):
                    delta = (layer.hk[I][J] - h_act[I][J])
                    if -epsilon <= delta and delta <= epsilon:
                        h_act[I][J] = 0
                    else:
                        h_act[I][J] = 1.0 / (layer.hk[I][J] - h_act[I][J]) * layer.fk
                    delta = (l_act[I][J] - layer.lk[I][J])
                    if -epsilon <= delta and delta <= epsilon:
                        l_act[I][J] = 0
                    else:
                        l_act[I][J] = 1.0 / (l_act[I][J] - layer.lk[I][J]) * layer.fk
        h_act[h_act < 0] = 0  ## < 0 means the neuron is covered
        l_act[l_act < 0] = 0  ## < 0 means the neuron is covered
        layer.h_activations.append(h_act)
        layer.l_activations.append(l_act)
        layer.h_cover_map = np.logical_or(layer.h_cover_map, h_act)
        layer.l_cover_map = np.logical_or(layer.l_cover_map, l_act)


def run_concolic_bc(model):
    ###
    outs = "concolic-bc-" + str(datetime.now()).replace(' ', '-') + '/'
    os.system('mkdir -p {0}'.format(outs))

    #### configuration phase
    layer_functions = []
    effective_layers = []
    test_cases = []
    adversarials = []

    print 'Pre-processing phase...'
    for l in range(0, len(model.layers)):
        layer = model.layers[l]
        name = layer.name

        get_current_layer_output = K.function([layer.input], [layer.output])
        layer_functions.append(get_current_layer_output)

        if is_conv_layer(layer) or is_dense_layer(layer):
            effective_layers.append(
                effective_layert(model, layer_index=l, current_layer=layer, is_conv=is_conv_layer(layer)))

    ## to configure 'fk' at each effective layer
    activations = eval_batch(layer_functions, x_train[0:10000])
    fks = []
    for elayer in effective_layers:
        index = elayer.layer_index
        sub_acts = np.abs(activations[index])
        fks.append(np.average(sub_acts))
        ## to compute hk, lk
        hks = np.array(activations[index])
        sp = elayer.h_cover_map.shape
        if elayer.is_conv:
            for i in range(0, sp[0]):
                for j in range(0, sp[1]):
                    for k in range(0, sp[2]):
                        for L in range(0, sp[3]):
                            elayer.hk[i][j][k][L] = np.max(hks[:, j, k, L])
                            elayer.lk[i][j][k][L] = np.min(hks[:, j, k, L])
        else:
            for I in range(0, sp[0]):
                for J in range(0, sp[1]):
                    elayer.hk[I][J] = np.max(hks[:, J])
                    elayer.lk[I][J] = np.min(hks[:, J])
    av = np.average(np.array(fks))
    for i in range(0, len(fks)):
        effective_layers[i].fk = av / fks[i]
        # print effective_layers[i].fk
        # print effective_layers[i].hk
        # print effective_layers[i].lk

    print '== DONE == \n'
    ##

    iseed = np.random.randint(0, len(x_test))
    im_seed = x_test[iseed]  ## the initial input
    test_cases.append(im_seed)

    ## model.summary()
    initialise_effective_layers(effective_layers, layer_functions, im_seed)
    covered, non_covered = cover_level(effective_layers)
    print covered, non_covered
    y = model.predict_classes(np.array([im_seed]))[0]
    show_adversarial_examples([im_seed, im_seed], [y, y], outs + '{0}.pdf'.format(iseed))

    for i in range(0, 999):
        if i == iseed: continue
        test_cases.append(x_test[i])
        update_effective_layers(effective_layers, layer_functions, x_test[i], bc_udi=True)
        update_effective_layers(effective_layers, layer_functions, x_test[i], bc_udi=False)

    f = open(outs + '{0}.results.txt'.format(iseed), "a")
    features = []
    tot_bs = 0
    for layer in effective_layers:
        if layer.the_feature < 0: continue
        features.append(layer.the_feature)
        tot_bs += layer.tot
    f.write('NC-cover {0} {1} {2} {3} {4}\n'.format(1.0 * covered / (covered + non_covered), len(test_cases),
                                                    len(adversarials), tot_bs, features))
    f.close()

    while True:

        ## to choose the max
        bc_layer, bc_index, bc_value, bc_udi = get_max(effective_layers)
        acts = None
        if bc_udi:
            acts = np.array(bc_layer.h_activations)
        else:
            acts = np.array(bc_layer.l_activations)
        pos = np.unravel_index(bc_index, acts.shape)
        im = test_cases[pos[0]]
        activations = eval(layer_functions, im)

        #  ##
        sp = acts.shape
        act_size = sp[2]
        if len(sp) > 4:
            act_size = sp[2] * sp[3] * sp[4]
        bc_index_ = bc_index - (pos[0]) * act_size
        feasible, d, new_im = boundary_cover(model, activations, bc_layer, bc_index_, im, bc_udi)
        ##

        if feasible:
            print 'feasible: d={0}'.format(d)
            found_before = False
            for t in test_cases:
                if (new_im == t).all():
                    found_before = True
                    print 'found'
                    break
            if found_before:
                bc_layer.cover_an_activation(pos, bc_udi)
                continue
            test_cases.append(new_im)
            print  '\n[{0}][{1}]'.format(model.predict_classes(np.array([im]))[0],
                                         model.predict_classes(np.array([new_im]))[0])
            y1 = model.predict_classes(np.array([im]))[0]
            y2 = model.predict_classes(np.array([new_im]))[0]
            if y1 != y2:
                adversarials.append([im, new_im])
                show_adversarial_examples([im, new_im], [y1, y2], outs + 'ad{0}-d{1}.pdf'.format(len(adversarials), d))
            bc_layer.cover_an_activation(pos, bc_udi)
            update_effective_layers(effective_layers, layer_functions, new_im, bc_index_, bc_layer, bc_udi)
        else:
            bc_layer.cover_an_activation(pos, bc_udi)

        covered, non_covered = cover_level(effective_layers)

        # print covered, non_covered

        f = open(outs + '{0}.results.txt'.format(iseed), "a")
        f.write('BC-cover {0} {1} {2} {3} {4}-{5}\n'.format(1.0 * covered / (covered + non_covered), len(test_cases),
                                                            len(adversarials), d, bc_layer.layer_index, pos))
        f.close()

    print 'All properties have been covered'


def main():
    parser = argparse.ArgumentParser(
        description='To convert a DNN model to the C program')

    parser.add_argument('model', action='store', nargs='+', help='The input neural network model (.h5)')

    args = parser.parse_args()
    model = load_model(args.model[0])
    model.summary()
    run_concolic_bc(model)


if __name__ == "__main__":
    main()
