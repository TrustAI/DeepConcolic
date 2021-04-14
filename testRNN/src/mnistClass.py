from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import *
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
import copy
from utils import getActivationValue,layerName, hard_sigmoid, get_activations_single_layer, \
    setup_dir_for_file


class mnistclass:
    def __init__(self, datasetName, modelFile):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_train_org = None
        self.y_test_org = None
        self.model = None
        self.unique_adv = []
        self.imagesize = 28
        self.datasetName = datasetName
        self.modelFile = modelFile
        self.load_data()
        self.numAdv = 0
        self.numSamples = 0
        self.perturbations = []

    def load_data(self):
        if self.datasetName == 'mnist':
            (self.X_train, self.y_train_org), (self.X_test, self.y_test_org) = mnist.load_data()
        elif self.datasetName == 'fashion_mnist':
            (self.X_train, self.y_train_org), (self.X_test, self.y_test_org) = fashion_mnist.load_data()
        else:
            raise ValueError (f'unknown dataset name `{self.datasetName}\'')
        self.X_train = self.X_train.astype('float32') / 255
        self.X_test = self.X_test.astype('float32') / 255
        self.y_train = to_categorical(self.y_train_org)
        self.y_test = to_categorical(self.y_test_org)

        self.X_poison = copy.deepcopy(self.X_test)
        self.X_poison[:, 24, 25] = 1.0
        self.X_poison[:, 25, 26] = 1.0
        self.X_poison[:, 26, 27] = 1.0
        self.X_poison[:, 26, 25] = 1.0
        self.X_poison[:, 24, 27] = 1.0

        self.y_poison = np.zeros(self.y_test.shape, dtype=float)
        self.y_poison[:, 6] = 1.0


    def load_model(self):
        self.model = load_model (self.modelFile)
        opt = keras.optimizers.Adam(lr=1e-3, decay=1e-5)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model.summary()
        # self.y_train = np.argmax(self.model.predict(self.X_train),axis = 1)
        # self.y_test = np.argmax(self.model.predict(self.X_test),axis = 1)

    def layerName(self, layer):
        layerNames = [layer.name for layer in self.model.layers]
        return layerNames[layer]

    def train_model(self):
        # self.load_data() # NB: <- already done
        setup_dir_for_file (self.modelFile) # early detection of file/dir error, in case
        self.model = Sequential()
        # input_shape = (batch_size, timesteps, input_dim)
        self.model.add(LSTM(128, input_shape=(self.X_train.shape[1:]), return_sequences=True))
        self.model.add(LSTM(128))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))
        opt = keras.optimizers.Adam(lr=1e-3, decay=1e-5)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train, epochs=30, validation_data=(self.X_test, self.y_test))
        self.model.save (self.modelFile)

    def train_embedding_hidden(self, layer, k):
        h_train = self.cal_hidden_keras(self.X_train, layer)
        h_train = h_train.reshape(h_train.shape[0] * h_train.shape[1], h_train.shape[2])

        tsne = manifold.TSNE(n_components=k, init='pca', random_state=0)
        h_tsne = tsne.fit_transform(h_train)

        mean_z = np.mean(h_tsne,axis=0)
        std_z = np.std(h_tsne,axis=0)

        z_norm_para = np.array([mean_z, std_z])

        np.save('z_norm_mnist.npy', z_norm_para)
        joblib.dump(tsne, 't_sne_mnist.sav')


    def displayInfo(self, test1, test2, o, m, unique_test, r):

        output_prob1 = self.model.predict(test1)
        output_class1 = np.argmax(output_prob1, axis= 1)

        output_prob2 = self.model.predict(test2)
        output_class2 = np.argmax(output_prob2, axis=1)

        m = m[o]
        diff = output_class2 - output_class1
        adv_index = np.nonzero(diff[o])
        m = m[adv_index].tolist()
        adv_n = len(adv_index[0])

        # save adversarial samples to files
        org_set = np.expand_dims (test1[adv_index], -1)
        adv_set = np.expand_dims (test2[adv_index], -1)
        adv_org = output_class1[adv_index]
        adv_pred = output_class2[adv_index]
        for i in range(adv_n):
            id = adv_index[0][i]
            orig_img = image.array_to_img (org_set[i])
            orig_img.save(os.path.join (r.subdir ('adv_output'), '%d-original-%d.png' % (id, adv_org[i])))
            adv_img = image.array_to_img(adv_set[i])
            adv_img.save(os.path.join (r.subdir ('adv_output'), '%d-adv-%d.png' % (id, adv_pred[i])))

        unique_test = unique_test[adv_index]
        for item in unique_test:
            if item not in self.unique_adv:
                self.unique_adv.append(item)

        self.perturbations = self.perturbations + m
        self.numAdv += adv_n
        self.numSamples += len(test2)

        self.displaySuccessRate()



    def from_array_to_image(self,test):
        test = test*255
        test = test.astype(int)
        test = test.reshape((28, 28, 1))
        pred_img = image.array_to_img(test)
        pred_img.save('output.jpg')

    def image_plot(self,test):
        img_class = self.model.predict_classes(test)
        classname = img_class[0]
        # # show image in matplot
        plt.imshow(test)
        plt.title(classname)
        plt.show()

    def updateSample(self,label2,label1,m,o,seed_idx):
        if label2 != label1 and o == True:
            self.numAdv += 1
            self.perturbations.append(m)
            if seed_idx in self.unique_adv:
                self.unique_adv.remove(seed_idx)
        self.numSamples += 1
        self.displaySuccessRate()

    def displaySamples(self):
        print("%s samples are considered" % (self.numSamples))

    def displaySuccessRate(self):
        print("%s samples, within which there are %s adversarial examples" % (self.numSamples, self.numAdv))
        print("the rate of adversarial examples is %.2f\n" % (self.numAdv / self.numSamples))

    def displayPerturbations(self):
        if self.numAdv > 0:
            print("the average perturbation of the adversarial examples is %s" % (sum(self.perturbations) / self.numAdv))
            print("the smallest perturbation of the adversarial examples is %s" % (min(self.perturbations)))

    def mutation(self, image, test_num, seed, mean= 0, var= 0.1):
        '''
            add gaussian noise to image
        '''
        if test_num > 1:
            image = np.repeat(image[np.newaxis, :, :], test_num, axis=0)

        np.random.seed(seed)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        out = np.clip(out, 0.0, 1.0)

        if test_num > 1:
            out = [out[i] for i in range(test_num)]

        return out



    def cal_hidden_keras(self,test, layernum):
        if layernum == 0:
            acx = test
        else:
            acx = get_activations_single_layer(self.model, np.array(test), self.layerName(layernum - 1))

        units = int(int(self.model.layers[layernum].trainable_weights[0].shape[1]) / 4)

        if len(acx.shape) < len(test.shape):
            acx = np.array([acx])

        inp = keras.layers.Input(batch_shape=(None, acx.shape[1], acx.shape[2]), name="input")
        rnn, s, c = keras.layers.LSTM(units,
                                return_sequences=True,
                                stateful=False,
                                return_state=True,
                                name="RNN")(inp)
        states = keras.models.Model(inputs=[inp], outputs=[s, c, rnn])

        for layer in states.layers:
            if layer.name == "RNN":
                layer.set_weights(self.model.layers[layernum].get_weights())

        h_t_keras, c_t_keras, rnn = states.predict(acx)

        return rnn

    def cal_hidden_state(self, test, layernum):
        if layernum == 0:
            acx = test
        else:
            acx = get_activations_single_layer(self.model, np.array([test]), self.layerName(layernum - 1))

        units = int(int(self.model.layers[layernum].trainable_weights[0].shape[1]) / 4)
        # print("No units: ", units)

        # get weight
        W = self.model.layers[layernum].get_weights()[0]
        U = self.model.layers[layernum].get_weights()[1]
        b = self.model.layers[layernum].get_weights()[2]

        W_i = W[:, :units]
        W_f = W[:, units: units * 2]
        W_c = W[:, units * 2: units * 3]
        W_o = W[:, units * 3:]

        U_i = U[:, :units]
        U_f = U[:, units: units * 2]
        U_c = U[:, units * 2: units * 3]
        U_o = U[:, units * 3:]

        b_i = b[:units]
        b_f = b[units: units * 2]
        b_c = b[units * 2: units * 3]
        b_o = b[units * 3:]

        # calculate the hidden state value
        h_t = np.zeros((self.imagesize, units))
        c_t = np.zeros((self.imagesize, units))
        f_t = np.zeros((self.imagesize, units))
        h_t0 = np.zeros((1, units))
        c_t0 = np.zeros((1, units))

        for i in range(0, self.imagesize):
            f_gate = hard_sigmoid(np.dot(acx[i, :], W_f) + np.dot(h_t0, U_f) + b_f)
            i_gate = hard_sigmoid(np.dot(acx[i, :], W_i) + np.dot(h_t0, U_i) + b_i)
            o_gate = hard_sigmoid(np.dot(acx[i, :], W_o) + np.dot(h_t0, U_o) + b_o)
            new_C = np.tanh(np.dot(acx[i, :], W_c) + np.dot(h_t0, U_c) + b_c)
            c_t0 = f_gate * c_t0 + i_gate * new_C
            h_t0 = o_gate * np.tanh(c_t0)
            c_t[i, :] = c_t0
            h_t[i, :] = h_t0
            f_t[i, :] = f_gate

        return [h_t, c_t, f_t]





