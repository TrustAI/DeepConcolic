from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.models import *
import os
import copy
import random
import tensorflow.keras.backend as K
from eda import eda
import numpy as np
from keras.preprocessing import sequence 
from utils import getActivationValue,layerName, hard_sigmoid, get_activations_single_layer

class Sentiment:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.unique_adv = []
        
        self.top_words = 50000
        self.word_to_id = keras.datasets.imdb.get_word_index()
        self.INDEX_FROM=3
        self.max_review_length = 500
        self.embedding_vector_length = 32 
        
        self.word_to_id = {k:(v+self.INDEX_FROM) for k,v in self.word_to_id.items()}
        self.word_to_id["<PAD>"] = 0
        self.word_to_id["<START>"] = 1
        self.word_to_id["<UNK>"] = 2
        self.id_to_word = {value:key for key,value in self.word_to_id.items()}

        self.numAdv = 0
        self.numSamples = 0
        self.perturbations = []
        
        self.load_data()
        self.pre_processing_X()

    def load_data(self):

        # call load_data with allow_pickle implicitly set to true
        (self.X_train, self.y_train), (self.X_test, self.y_test) = imdb.load_data(num_words=self.top_words)


        
    def load_model(self):
        self.model=load_model('saved_models/sentiment-lstm.h5')
        self.model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        
    def layerName(self,layer):
        layerNames = [layer.name for layer in self.model.layers]
        return layerNames[layer]
        
    def train_model(self):
        self.load_data()
        self.pre_processing_X()
        self.model = Sequential() 
        self.model.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.max_review_length, input_shape=(self.top_words,))) 
        self.model.add(LSTM(100))
        self.model.add(Dense(1, activation='sigmoid')) 
        self.model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
        print(self.model.summary()) 
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), nb_epoch=10, batch_size=64) 
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        self.model.save('saved_models/sentiment-lstm.h5')

    def getOutputResult(self,test):
        Output = self.model.predict(np.array(test))
        return np.round(np.squeeze(Output))

    def displayInfo(self,org_text, aug_text, y_test1, y_test2, m, unique_test, r):
        diff = y_test1 - y_test2
        adv_index = np.nonzero(diff)
        adv_n = len(adv_index[0])
        unique_test = unique_test[adv_index]

        if adv_n != 0:
            for item in unique_test:
                if item not in self.unique_adv:
                    self.unique_adv.append(item)
            perturb = m*np.ones(adv_n)
            self.perturbations = self.perturbations + perturb.tolist()

            # save adv to files
            f = open(os.path.join (r.subdir ('adv_output'), 'adv_test_set.txt'), 'a')
            for i in adv_index[0]:
                f.write('\n')
                f.write('pred: ' + str(y_test2[i]))
                f.write('\t')
                f.writelines(aug_text[i])
                f.write('\n')
            f.close()

            f = open(os.path.join (r.subdir ('adv_output'), 'org_test_set.txt'), 'a')
            for i in adv_index[0]:
                f.write('\n')
                f.write('pred: ' + str(y_test1[i]))
                f.write('\t')
                f.writelines(org_text[i])
                f.write('\n')
            f.close()

        self.numAdv += adv_n
        self.numSamples += len(y_test2)
        self.displaySuccessRate()


    def pre_processing_X(self): 
        self.X_train = sequence.pad_sequences(self.X_train, maxlen=self.max_review_length) 
        self.X_test = sequence.pad_sequences(self.X_test, maxlen=self.max_review_length) 
        
    def pre_processing_x(self,tmp):
        tmp_padded = sequence.pad_sequences(tmp, maxlen=self.max_review_length)
        #print("%s. Sentiment: %s" % (review,model.predict(np.array([tmp_padded][0]))[0][0]))
        test = np.array(tmp_padded)
        #print("input shape: %s"%(str(test.shape)))
        return test
        
    def validateID(self,ids):
        flag = False 
        ids2 = []
        for id in ids: 
            if not (id  in self.id_to_word.keys()): 
                ids2.append(min(self.id_to_word.keys(), key=lambda x:abs(x-id)))
                flag = True
            else: 
                ids2.append(id)
        if flag == True: 
            return validateID(ids2)
        else: return ids

    def displayIDRange(self):
        minID = min(self.word_to_id.values())+self.INDEX_FROM
        maxID = max(self.word_to_id.values())+self.INDEX_FROM
        print("ID range: %s--%s"%(minID,maxID))
        
    def fromTextToID(self,review): 
        tmp = []
        for word in review.split(" "):
            if word in self.word_to_id:
                if self.word_to_id[word] <= self.top_words :
                    tmp.append(self.word_to_id[word])
        return tmp
    
    def fromIDToText(self,ids): 
        tmp = ""
        for id in ids:
            if id > 2: 
                tmp += self.id_to_word[id] + " "
        return tmp.strip()

    def displaySamples(self):
        print("%s samples are considered" % (self.numSamples))

    def displaySuccessRate(self):
        print("%s samples, within which there are %s adversarial examples" % (self.numSamples, self.numAdv))
        print("the rate of adversarial examples is %.2f\n" % (self.numAdv / self.numSamples))

    def displayPerturbations(self):
        if self.numAdv > 0:
            print("the average perturbation of the adversarial examples is %s" % (sum(self.perturbations) / self.numAdv))
            print("the smallest perturbation of the adversarial examples is %s" % (min(self.perturbations)))


    def mutation(self, test, test_num, seed):
        random.seed(seed)
        text = self.fromIDToText(test)
        alpha = random.uniform(0.1, 0.5)
        aug_text = eda(text, seed, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=test_num)
        tmp = [self.fromTextToID(text) for text in aug_text]
        out = self.pre_processing_x(tmp)
        return out.tolist()

    # calculate the lstm hidden state and cell state manually (no dropout)
    def cal_hidden_state(self, test, layer):
        acx = get_activations_single_layer(self.model, np.array([test]), self.layerName(0))
        acx = np.squeeze(acx)
        units = int(int(self.model.layers[1].trainable_weights[0].shape[1]) / 4)
        # print("No units: ", units)
        # lstm_layer = model.layers[1]
        W = self.model.layers[1].get_weights()[0]
        U = self.model.layers[1].get_weights()[1]
        b = self.model.layers[1].get_weights()[2]

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
        h_t = np.zeros((self.max_review_length, units))
        c_t = np.zeros((self.max_review_length, units))
        f_t = np.zeros((self.max_review_length, units))
        h_t0 = np.zeros((1, units))
        c_t0 = np.zeros((1, units))

        for i in range(0, self.max_review_length):
            f_gate = hard_sigmoid(np.dot(acx[i, :], W_f) + np.dot(h_t0, U_f) + b_f)
            i_gate = hard_sigmoid(np.dot(acx[i, :], W_i) + np.dot(h_t0, U_i) + b_i)
            o_gate = hard_sigmoid(np.dot(acx[i, :], W_o) + np.dot(h_t0, U_o) + b_o)
            new_C = np.tanh(np.dot(acx[i, :], W_c) + np.dot(h_t0, U_c) + b_c)
            c_t0 = f_gate * c_t0 + i_gate * new_C
            h_t0 = o_gate * np.tanh(c_t0)
            c_t[i, :] = c_t0
            h_t[i, :] = h_t0
            f_t[i, :] = f_gate

        return h_t, c_t, f_t

    def cal_hidden_keras(self,test, layernum):
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



