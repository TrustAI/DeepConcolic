import numpy as np
from tensorflow.keras import backend as K
from tensorflow import keras
import sys
import os
import operator
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from recurrent_networks import VGG16LSTMVideoClassifier
from UCF101_loader import load_ucf, scan_ucf_with_labels
from utils import lp_norm, getActivationValue, layerName, hard_sigmoid
    
K.set_learning_phase(1)


epsilon = 0.0001 


class ucf101_vgg16_lstm_class:

    def __init__(self):
        self.vgg16_include_top = True
        self.data_dir_path = os.path.join('dataset', 'very_large_data')
        self.model_dir_path = os.path.join('models', 'UCF-101')
        self.config_file_path = VGG16LSTMVideoClassifier.get_config_file_path(self.model_dir_path, vgg16_include_top=self.vgg16_include_top)
        self.weight_file_path = VGG16LSTMVideoClassifier.get_weight_file_path(self.model_dir_path, vgg16_include_top=self.vgg16_include_top)
    
        self.predictor = VGG16LSTMVideoClassifier()
        self.model = self.predictor.load_model(self.config_file_path, self.weight_file_path)
        self.length = None
        self.numAdv = 0
        self.numSamples = 0
        self.perturbations = []
        self.unique_adv = []
        
        self.videos = scan_ucf_with_labels(self.data_dir_path, [label for (label, label_index) in self.predictor.labels.items()])

        self.video_file_path_list = np.array([file_path for file_path in self.videos.keys()])
        np.random.seed(5)
        np.random.shuffle(self.video_file_path_list)

    def train_model(self):
        self.predictor.fit(self.data_dir_path, self.model_dir_path, vgg16_include_top=True, data_set_name='UCF-101', test_size=0.3, random_state=10)
    
    def predict(self,index=1): 
        video_file_path = self.video_file_path_list[index]
        label = self.videos[video_file_path]
        images, test, predicted_label, predicted_confidence, last_activation = self.predictor.predict(video_file_path)
        self.length = test.shape[0]
        # print("original image sequence shape: %s, preprocessed image sequence shape %s, feature sequence shape %s"%(str(images.shape),str(preprocessed_images.shape),str(test.shape)))
        # print('predicted: ' + predicted_label + ' confidence:' + str(predicted_confidence) + ' actual: ' + label)
        return images, test, label, predicted_label

    def predict_imgs(self, images, mode, seeds, variance):
        new_images, new_test, predicted_label, predicted_confidence = self.predictor.predict_imgs(images, mode, seeds, variance)
        return new_images, new_test, predicted_label, predicted_confidence


    def predict_test(self,x_samples):
        last_activation = self.predictor.model.predict(np.array(x_samples))
        predicted_class = np.argmax(last_activation, axis=1)
        predicted_label = [self.predictor.labels_idx2word[i] for i in predicted_class]
        return predicted_label

    def displayInfo(self,label1,label2,o,m, unique_test, _r):
        adv_index = np.array([i for i in range(len(label1)) if label1[i] != label2[i]])
        adv_n = len(adv_index)
        unique_test = unique_test[adv_index]

        for item in unique_test:
            if item not in self.unique_adv:
                self.unique_adv.append(item)

        self.perturbations = self.perturbations + m.tolist()
        self.numAdv += adv_n
        self.numSamples += len(label1)
        self.displaySuccessRate()


    def displaySamples(self):
        print("%s samples are considered" % (self.numSamples))

    def displaySuccessRate(self):
        print("%s samples, within which there are %s adversarial examples" % (self.numSamples, self.numAdv))
        print("the rate of adversarial examples is %.2f\n" % (self.numAdv / self.numSamples))

    def mutation_semantics(self, images, test_num, seeds, mode):
        out = []
        out_images = []
        for i in range(test_num):
            new_images, test2, new_predicted_label, new_predicted_confidence = self.predict_imgs(images, mode, seeds+i, 0.01)
            out.append(test2)
            out_images.append(new_images)

        return out_images,out

    def mutation(self, image, test_num, seed, mean= 0, var= 0.01):
        '''
            add gaussian noise to image
        '''
        if test_num > 1:
            image = np.repeat(image[np.newaxis, :, :], test_num, axis=0)

        np.random.seed(seed)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise

        if test_num > 1:
            out = [out[i] for i in range(test_num)]

        return out

    def displayPerturbations(self):
        if self.numAdv > 0:
            print("the average perturbation of the adversarial examples is %s" % (sum(self.perturbations) / len(self.perturbations)))
            print("the smallest perturbation of the adversarial examples is %s" % (min(self.perturbations)))

    def layerName(self,layer):
        layerNames = [layer.name for layer in self.model.layers]
        return layerNames[layer]

    # calculate the lstm hidden state and cell state manually (no dropout)
    # activation function is tanh
    def cal_hidden_state(self, test, layernum):
        if layernum == 0:
            acx = np.array(test)
        else:
            acx = get_activations_single_layer(self.model, np.array(test), self.layerName(layernum-1))

        units = int(int(self.model.layers[layernum].trainable_weights[0].shape[1]) / 4)


        W = self.model.layers[layernum].get_weights()[0]
        U = self.model.layers[layernum].get_weights()[1]
        b = self.model.layers[layernum].get_weights()[2]

        h_t0 = np.zeros((acx.shape[0], 1, units))
        c_t0 = np.zeros((acx.shape[0], 1, units))
        s_t = np.tensordot(acx, W, axes=([2],[0])) + np.tensordot(h_t0, U, axes=([2],[0])) + b
        i = hard_sigmoid(s_t[:, :, :units])
        f = hard_sigmoid(s_t[:, :, units: units * 2])
        _c = np.tanh(s_t[:, :, units * 2: units * 3])
        o = hard_sigmoid(s_t[:, :, units * 3:])
        c_t = i*_c + f*c_t0
        h_t = o*np.tanh(c_t)

        # h_t0 = np.zeros(( 1, units))
        # c_t0 = np.zeros(( 1, units))
        # s_t = np.dot(acx, W) + np.dot(h_t0, U) + b
        # i = hard_sigmoid(s_t[:, :units])
        # f = hard_sigmoid(s_t[:, units: units * 2])
        # _c = np.tanh(s_t[:, units * 2: units * 3])
        # o = hard_sigmoid(s_t[:, units * 3:])
        # c_t = i*_c + f*c_t0
        # h_t = o*np.tanh(c_t)

        return h_t, c_t, f

    def cal_hidden_keras(self, test, layernum):
        if layernum == 0:
            acx = test
        else:
            acx = get_activations_single_layer(self.model, np.array(test), self.layerName(layernum - 1))

        units = int(int(self.model.layers[layernum].trainable_weights[0].shape[1]) / 4)

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










