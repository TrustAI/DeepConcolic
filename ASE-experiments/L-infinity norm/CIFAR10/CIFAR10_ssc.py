"""
Sign-Sign Cover of CIFAR10 based on L-infinity Norm

Author: Youcheng Sun
Email: youcheng.sun@cs.ox.ac.uk
"""

import argparse
import sys
from datetime import datetime

import keras
from keras.datasets import cifar10
from keras.models import *
from keras.layers import *
from keras import *

from lp_ssc import *
from utils import *

########################################################################
minus_inf = -10000000000
positive_inf = +10000000000
img_rows, img_cols = 32, 32

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# if K.image_data_format() == 'channels_first':
#    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#    input_shape = (1, img_rows, img_cols)
# else:
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


########################################################################


class effective_layert:
    def __init__(self, layer_index, current_layer, network, is_conv=False):
        self.layer_index = layer_index
        self.activations = []
        self.is_conv = is_conv
        self.current_layer = current_layer
        self.fk = 1.0
        self.the_feature = -1
        self.the_feature_index = 0
        self.tot = 0
        self.tot_neurons = 0
        self.prior_activations = []
        sp = current_layer.output.shape
        # print 'current input shape: ', current_layer.input.shape
        # print 'output shape: ', sp
        # sys.exit(0)
        if is_conv:
            if layer_index != 2:
                self.cover_map = np.zeros((1, sp[1], sp[2], sp[3]), dtype=bool)
            else:  # for simplicity, let us focus on the SSC on one particular feature map at the 2nd layer
                weights = network.get_weights()[layer_index]
                self.the_feature = np.random.randint(0, sp[3])
                inp = current_layer.input
                isp = inp.shape
                self.tot_neurons = isp[1].value * isp[2].value * isp[3].value
                self.the_feature_index = np.random.randint(0, sp[1].value * sp[2].value * sp[3].value)

                print "total countable neuron pairs: ", self.tot
                print "the feature: ", self.the_feature
                print "the feature index: ", self.the_feature_index
                # sys.exit(0)
                self.cover_map = np.zeros((1, isp[1], isp[2], isp[3]), dtype=bool)
                fpos = np.unravel_index(self.the_feature_index, (1, sp[1], sp[2], sp[3]))
                i = fpos[0]
                j = fpos[1]
                k = fpos[2]
                l = fpos[3]
                for II in range(0, current_layer.kernel_size[0]):
                    for JJ in range(0, current_layer.kernel_size[1]):
                        for KK in range(0, isp[3]):
                            print ''
                            print j + II, k + JJ, KK
                            print self.cover_map[0][j + II][k + JJ][KK]
                            self.cover_map[0][j + II][k + JJ][KK] = 1
                            print self.cover_map[0][j + II][k + JJ][KK]
                            print ''
                self.tot = np.count_nonzero(self.cover_map)  # isp[1].value*isp[2].value*isp[3].value
                # print self.cover_map.shape
                # print 'tot', self.tot
                # sys.exit(0)
        else:
            self.cover_map = np.zeros((1, sp[1]), dtype=bool)
        print 'Created an effective layer: [is_conv {0}] [cover_map {1}]'.format(is_conv, self.cover_map.shape)

    ## return the argmax and the max value:
    def get_max(self):
        if self.the_feature < 0:
            print 'the feature is not specified'
            sys.exit(0)
        sp = np.array(self.prior_activations).argmax()
        sp1 = np.unravel_index(sp, np.array(self.prior_activations).shape)
        if self.is_conv:
            return sp, self.prior_activations[sp1[0]][sp1[1]][sp1[2]][sp1[3]][sp1[4]]
        else:
            # return sp, self.activations[sp1[0]][sp1[1]][sp1[2]]
            print 'the feature layer must be convolutional'
            sys.exit(0)

    def update_activations(self):
        for i in range(0, len(self.prior_activations)):
            self.prior_activations[i] = np.multiply(self.prior_activations[i], self.cover_map)
            self.prior_activations[i][self.prior_activations[i] == 0] = minus_inf

    def cover_an_activation(self, sp):
        if self.is_conv:
            self.prior_activations[sp[0]][sp[1]][sp[2]][sp[3]][sp[4]] = minus_inf
            # print 'cover an activation', sp, self.activations[sp[0]][sp[1]][sp[2]][sp[3]][sp[4]][sp[5]][sp[6]][sp[7]]
        else:
            # self.activations[sp[0]][sp[1]][sp[2]]=0 #minus_inf
            print 'the feature layer must be convolutional'
            sys.exit(0)


def get_max(effective_layers):
    nc_layer, nc_index, nc_value = None, -1, minus_inf
    for layer in effective_layers:
        ##
        if layer.the_feature < 0: continue
        ##
        index, v = layer.get_max()
        v = v  # *layer.fk
        if v > nc_value:
            nc_layer = layer
            nc_index = index
            nc_value = v
    return nc_layer, nc_index, nc_value


def cover_level(effective_layers):
    covered = 0
    non_covered = 0
    for layer in effective_layers:
        ##
        if layer.the_feature < 0: continue
        ##
        c = np.count_nonzero(layer.cover_map)
        non_covered += c
        covered = covered + (layer.tot - c)
    return covered, non_covered


def update_effective_layers(effective_layers, layer_functions, im, network, ssc_index=None, nc_layer=None):
    activations = eval(layer_functions, im)
    for count in range(0, len(effective_layers)):
        layer = effective_layers[count]
        ##
        if layer.the_feature < 0: continue
        ##

        prior_act = (activations[layer.layer_index - 2])
        ##new_act=np.zeros(layer.cover_map.shape)
        ### prior act shape
        # pas=prior_act.shape

        # for m in range(0, pas[1]):
        #  for n in range(0, pas[2]):
        #    for o in range(0, pas[3]):
        #      new_act[0][m][n][o]=-np.abs(prior_act[0][m][n][o])

        new_act = np.array(prior_act)
        new_act = -1 * np.abs(new_act)
        new_act[new_act == 0] = -0.0000001
        new_act = np.multiply(new_act, layer.cover_map)

        new_act[new_act == 0] = minus_inf

        if not (ssc_index is None):
            pos = np.unravel_index(ssc_index, layer.cover_map.shape)
            print '@@@@ ', ssc_index, pos
            print '@@@@ ', layer.cover_map[pos[0]][pos[1]][pos[2]][pos[3]]
            layer.cover_map[pos[0]][pos[1]][pos[2]][pos[3]] = 0
            print '#### ', np.count_nonzero(layer.cover_map)
            # sys.exit(0)
        layer.prior_activations.append(new_act)
        layer.update_activations()


# def initialise_effective_layers(effective_layers, layer_functions, im, net):
#  activations=eval(layer_functions, im) 
#  for count in range(0, len(effective_layers)):
#    #for layer in effective_layers:  
#    layer=effective_layers[count]
#    ##
#    if layer.the_feature<0: continue
#    ##
#
#    prior_act=(activations[layer.layer_index-2])
#    new_act=np.zeros(layer.cover_map.shape)
#    ## prior act shape
#    pas=prior_act.shape
#    cas=layer.current_layer.output.shape
#    i=0
#    j=cas[1]
#    k=cas[2]
#    l=cas[3]
#    if l!=layer.the_feature:
#      print 'something is wrong...'
#      sys.exit(0)
#
#    for m in range(0, pas[1]):
#      for n in range(0, pas[2]):
#        for o in range(0, pas[3]):
#          new_act[0][m][n][o]=-np.abs(prior_act[0][m][n][o])
#
#    layer.cover_map=np.logical_and(layer.cover_map, new_act)
#
#    layer.activations.append(act)
#    layer.cover_map=np.logical_or(layer.cover_map, act)

def run_concolic_ssc(model):
    ###
    outs = "concolic-ssc" + str(datetime.now()).replace(' ', '-') + '/'
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
            effective_layers.append(
                effective_layert(layer_index=l, current_layer=layer, network=model, is_conv=is_conv_layer(layer)))

    ## to configure 'fk' at each effective layer
    activations = eval_batch(layer_functions, x_test[0:10000])
    fks = []
    for elayer in effective_layers:
        index = elayer.layer_index
        sub_acts = np.abs(activations[index])
        fks.append(np.average(sub_acts))
    av = np.average(np.array(fks))
    for i in range(0, len(fks)):
        effective_layers[i].fk = av / fks[i]
        print effective_layers[i].fk

    # iseed=
    # im=x_test[0] ## the initial input
    # test_cases.append(im)
    for i in range(0, 1000):
        print 'initialising ', i
        test_cases.append(x_test[i])
        update_effective_layers(effective_layers, layer_functions, x_test[i], model)

    ## model.summary()
    # initialise_effective_layers(effective_layers, layer_functions, im, model)
    covered, non_covered = cover_level(effective_layers)
    f = open(outs + 'results.txt', "a")
    f.write('SSC-cover {0} {1} {2}; the feature tested: {3}; #neuron pairs: {4}\n'.format(
        1.0 * covered / (covered + non_covered), len(test_cases), len(adversarials), effective_layers[1].the_feature,
        effective_layers[1].tot))
    f.close()

    # covered, non_covered=cover_level(effective_layers)
    # print '#### ', covered, non_covered
    # sys.exit(0)

    count = 0
    while True:

        count += 1

        ## to choose the max
        ssc_layer, ssc_index, ssc_value = get_max(effective_layers)
        print ssc_layer.layer_index, ssc_value
        pos = np.unravel_index(ssc_index, np.array(ssc_layer.prior_activations).shape)
        print pos
        print len(test_cases)
        print ssc_value
        im = test_cases[pos[0]]
        activations = eval(layer_functions, im)
        print 'The chosen test is {0}: {1} at layer {2}'.format(pos, ssc_value, ssc_layer.layer_index)
        # if len(pos)>3:
        #  print nc_layer.activations[pos[0]][pos[1]][pos[2]][pos[3]][pos[4]]
        #  print activations[nc_layer.layer_index][pos[1]][pos[2]][pos[3]][pos[4]]
        # else:
        #  print nc_layer.activations[pos[0]][pos[1]][pos[2]]
        #  print activations[nc_layer.layer_index][pos[1]][pos[2]]

        ##
        sp = np.array(ssc_layer.prior_activations).shape
        # act_size=sp[2]
        # if len(sp)>4:
        #  act_size=sp[2]*sp[3]*sp[4]*sp[5]*sp[6]*sp[7]
        ssc_index_ = ssc_index - (pos[0]) * ssc_layer.tot_neurons
        # print 'total neurons: ', ssc_layer.tot_neurons
        # print ssc_index
        # print ssc_index_
        # sys.exit(0)
        feasible, d, new_im = SSCover(model, activations, ssc_layer, ssc_index_, im)
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
                ssc_layer.cover_an_activation(pos)
                continue
            test_cases.append(new_im)
            print  '\n[{0}][{1}]'.format(model.predict_classes(np.array([im]))[0],
                                         model.predict_classes(np.array([new_im]))[0])
            y1 = model.predict_classes(np.array([im]))[0]
            y2 = model.predict_classes(np.array([new_im]))[0]
            if y1 != y2:
                adversarials.append([im, new_im])
                show_adversarial_examples([im, new_im], [y1, y2], outs + 'ad{0}-d{1}.pdf'.format(len(adversarials), d))

            ssc_layer.cover_an_activation(pos)
            update_effective_layers(effective_layers, layer_functions, new_im, model, ssc_index_, ssc_layer)

        else:
            ssc_layer.cover_an_activation(pos)

        covered, non_covered = cover_level(effective_layers)
        print '#### ', covered, non_covered

        f = open(outs + 'results.txt', "a")
        f.write('SSC-cover {0} {1} {2} {3} {4}-{5}\n'.format(1.0 * covered / (covered + non_covered), len(test_cases),
                                                             len(adversarials), d, ssc_layer.layer_index, pos))
        f.close()

        if non_covered == 0: break

    print 'All properties have been covered'


def main():
    parser = argparse.ArgumentParser(
        description='To convert a DNN model to the C program')

    parser.add_argument('model', action='store', nargs='+', help='The input neural network model (.h5)')

    args = parser.parse_args()
    model = load_model(args.model[0])
    model.summary()
    run_concolic_ssc(model)


if __name__ == "__main__":
    main()
