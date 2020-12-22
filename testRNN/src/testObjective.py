from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.models import *
import copy
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import sequence
from keract import *
from utils import *
from random import *
import itertools as iter
import numpy as np
from numpy import linalg as LA
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from saxpy.paa import paa
import time

class NCTestObjectiveEvaluation:
    def __init__(self,r): 
        self.testObjective = NCTestObjective()
        self.coverage = 0.0
        self.record = r
        self.minimal = None

    def get_activations(self, testCase):
        layer = self.testObjective.layer
        model = self.testObjective.model
        act = get_activations_single_layer(model,np.array(testCase),layerName(model,layer))
        return act
        
    def update_features(self,testCase):
        self.minimal = 0
        if self.coverage != 1:
            activation = self.get_activations(testCase)
            activation = np.max(activation,axis=0)
            features = (np.argwhere(activation > self.testObjective.threshold)).tolist()

            for feature in features:
                if feature in self.testObjective.feature:
                    self.minimal = 1
                    self.testObjective.feature.remove(feature)

        else:
            print("Test requirements are all satisfied")

        self.coverage = 1 - len(self.testObjective.feature)/self.testObjective.originalNumOfFeature
        self.displayCoverage()
        
    def displayCoverage(self):
        if self.testObjective.threshold == 0:
            print("neuron coverage up to now: %.2f\n"%(self.coverage))
        else:
            print("SANC up to now: %.2f\n" % (self.coverage))


class NCTestObjective:
    def __init__(self):
        self.model = None
        self.layer = None
        self.threshold = None
        self.feature = None
        self.originalNumOfFeature = None

    def setOriginalNumOfFeature(self):
        self.originalNumOfFeature = len(self.feature)
        self.displayRemainingFeatures()
        
    def displayRemainingFeatures(self):
        if self.threshold == 0:
            print("%s features to be covered for NC"%(self.originalNumOfFeature))
        else:
            print("%s features to be covered for SANC" % (self.originalNumOfFeature))

    def setParamters(self,model,layer,threshold, test):
        self.model = model
        self.layer = layer
        self.threshold = threshold
        act = get_activations_single_layer(self.model,np.array([test]),layerName(self.model,self.layer))
        act = np.squeeze(act)
        self.feature = (np.argwhere(act >= np.min(act))).tolist()
        self.setOriginalNumOfFeature()



class KMNCTestObjectiveEvaluation:
    def __init__(self, r):
        self.testObjective = KMNCTestObjective()
        self.coverage = 0.0
        self.record = r
        self.minimal = None

    def get_activations(self,testCase):
        layer = self.testObjective.layer
        model = self.testObjective.model
        act = get_activations_single_layer(model, np.array(testCase), layerName(model, layer))
        return act

    def update_features(self,testCase):
        self.minimal = 0
        activations = self.get_activations(testCase)
        features_set = [np.digitize(act, bins=self.testObjective.interval)-1 for act in activations]

        for features in features_set:
            self.remove_feature(features)

        self.coverage = 1 - sum([ len(listElem) for listElem in self.testObjective.feature]) / self.testObjective.originalNumOfFeature
        self.displayCoverage()

    def displayCoverage(self):
        print("KMNC up to now: %.2f\n" % (self.coverage))

    def remove_feature(self,features):
        for count, feature in enumerate(features):
            if feature in self.testObjective.feature[count]:
                self.minimal = 1
                self.testObjective.feature[count].remove(feature)

class KMNCTestObjective:
    def __init__(self):
        self.model = None
        self.layer = None
        self.interval = None
        self.feature = None
        self.k = None
        self.originalNumOfFeature = None

    def setOriginalNumOfFeature(self):
        self.originalNumOfFeature = sum( [ len(listElem) for listElem in self.feature])
        self.displayRemainingFeatures()

    def displayRemainingFeatures(self):
        print("%s features to be covered for KMNC" % (self.originalNumOfFeature))
        # print("including %s."%(str(self.feature)))

    def setfeature(self):
        self.interval = np.linspace(-1, 1, num=self.k+1)

    def setParamters(self, model, layer, k, test):
        self.model = model
        self.layer = layer
        self.k = k
        self.setfeature()
        act = get_activations_single_layer(self.model, np.array([test]), layerName(self.model, self.layer))
        self.feature = [[*range(0, self.k, 1)] for i in range(len(act.flatten()))]
        self.setOriginalNumOfFeature()


class NBCTestObjectiveEvaluation:
    def __init__(self, r):
        self.testObjective = NBCTestObjective()
        self.coverage = 0.0
        self.record = r
        self.minimal = None

    def get_activations(self,testCase):
        layer = self.testObjective.layer
        model = self.testObjective.model
        act = get_activations_single_layer(model, np.array(testCase), layerName(model, layer))
        return act

    def update_features(self,testCase):
        self.minimal = 0
        activations = self.get_activations(testCase)
        act_max = np.max(activations,axis=0)
        act_min = np.min(activations,axis=0)
        features_max = self.extract_feature(act_max)
        features_min = self.extract_feature(act_min)
        self.remove_feature(features_max)
        self.remove_feature(features_min)

        self.coverage = 1 - sum([ len(listElem) for listElem in self.testObjective.feature]) / self.testObjective.originalNumOfFeature
        self.displayCoverage()

    def displayCoverage(self):
        print("NBC up to now: %.2f\n" % (self.coverage))

    def extract_feature(self,act):
        return [neuron_boudary_judge(listElem, self.testObjective.ub, self.testObjective.lb) for listElem in act]

    def remove_feature(self,features):
        for count, feature in enumerate(features):
            if feature in self.testObjective.feature[count]:
                self.minimal = 1
                self.testObjective.feature[count].remove(feature)

class NBCTestObjective:
    def __init__(self):
        self.model = None
        self.layer = None
        self.ub = None
        self.lb = None
        self.interval = None
        self.feature = None
        self.originalNumOfFeature = None

    def setOriginalNumOfFeature(self):
        self.originalNumOfFeature = sum([len(listElem) for listElem in self.feature])
        self.displayRemainingFeatures()

    def displayRemainingFeatures(self):
        print("%s features to be covered for NBC" % (self.originalNumOfFeature))
        # print("including %s."%(str(self.feature)))

    def setParamters(self, model, layer, ub, lb, test):
        self.model = model
        self.layer = layer
        self.ub = ub
        self.lb = lb
        act = get_activations_single_layer(self.model, np.array([test]), layerName(self.model, self.layer))
        self.feature = [[*range(0, 2, 1)] for i in range(len(act.flatten()))]
        self.setOriginalNumOfFeature()


class SCTestObjectiveEvaluation:
    def __init__(self, r):
        self.testObjective = SCTestObjective()
        self.cov_count = 0
        self.coverage = 0.0
        self.record = r

    def get_activations(self,hidden):
        alpha1 = np.sum(np.where(hidden > 0, hidden, 0), axis=2)
        alpha2 = np.sum(np.where(hidden < 0, hidden, 0), axis=2)
        alpha11 = np.insert(np.delete(alpha1, -1, axis=1), 0, 0, axis=1)
        alpha22 = np.insert(np.delete(alpha2, -1, axis=1), 0, 0, axis=1)
        alpha = np.abs(alpha1 - alpha11 + alpha2 - alpha22)
        return alpha

    def update_features(self,hidden,test_num):
        activations = self.get_activations(hidden)
        activations = activations[:,self.testObjective.indices]
        activations = (activations - self.testObjective.min) / (self.testObjective.max - self.testObjective.min)
        cov_fitness = self.testObjective.threshold - activations
        cov_index = np.argmin(cov_fitness, axis=0)
        cov_fitness = np.min(cov_fitness, axis=0)

        features = (np.argwhere(cov_fitness <= 0)).tolist()
        self.cov_count += 1
        for feature in features:
            if feature in self.testObjective.feature:
                self.cov_count = 0
                self.testObjective.feature.remove(feature)
                del self.testObjective.test_record[feature[0]]
            self.testObjective.feature_count[feature[0]] += 1

        # updata test record
        for feature in self.testObjective.feature:
            test_record = self.testObjective.test_record[feature[0]]
            if test_record == None or test_record[1] > cov_fitness[feature[0]]:
                self.testObjective.test_record[feature[0]] = list([test_num+cov_index[feature[0]], cov_fitness[feature[0]]])

        self.coverage = 1 - len(self.testObjective.feature) / self.testObjective.originalNumOfFeature

        self.displayCoverage()

    def displayCoverage(self):
        print("Step-wise Coverage up to now: %.2f\n" % (self.coverage))

    def cal_hidden(self,mn,test_set):
        if self.testObjective.test_obj == 'h':
            results = mn.cal_hidden_keras(np.array(test_set), self.testObjective.layer)
        else:
            'Please choose an internal variable to test'
        return results

    # minimize the fitness function
    def fitness(self,hidden,idx):
        act = self.get_activations(hidden)
        act = act[:,self.testObjective.indices]
        act = (act - self.testObjective.min) / (self.testObjective.max - self.testObjective.min)
        output = self.testObjective.threshold - act
        return output[:,idx]


class SCTestObjective:
    def __init__(self):
        self.model = None
        self.layer = None
        self.feature = None
        self.threshold = None
        self.indices = None
        self.originalNumOfFeature = None
        self.originalfeature = None
        self.feature_count = None
        self.min = None
        self.max = None
        self.test_record = {}

    def setOriginalNumOfFeature(self):
        self.originalNumOfFeature = len(self.feature)
        # create an dictionary to store the prior test case for each feature
        # [a b], a is the index of test case in test set, b is fitness function
        for i in range(self.originalNumOfFeature):
            self.test_record[i] = None
        self.displayRemainingFeatures()

    def setfeaturecount(self):
        self.feature_count = np.zeros((len(self.feature)))

    def displayRemainingFeatures(self):
        print("%s features to be covered for SC" % (self.originalNumOfFeature))
        # print("including %s."%(str(self.feature)))

    def setParamters(self, model, test_obj, layer, threshold, indices, max_SC, min_SC, act):
        self.model = model
        self.test_obj = test_obj
        self.layer = layer
        self.threshold = threshold
        self.indices = indices
        self.min = min_SC
        self.max = max_SC
        act = act[indices]
        self.feature = (np.argwhere(act>= np.min(act))).tolist()
        self.setfeaturecount()
        self.setOriginalNumOfFeature()


class TCTestObjectiveEvaluation:
    def __init__(self, r):
        self.testObjective = TCTestObjective()
        self.cov_count = 0
        self.coverage = 0.0
        self.record = r

    def get_activations(self,hidden):
        alpha1 = np.sum(np.where(hidden > 0, hidden, 0), axis=2)
        alpha2 = np.sum(np.where(hidden < 0, hidden, 0), axis=2)
        alpha = np.abs(alpha1+alpha2)
        return alpha

    def update_features(self,hidden,test_num):
        activation = self.get_activations(hidden)
        dat_znorm = Z_ScoreNormalization(activation[:,self.testObjective.indices],self.testObjective.mean,self.testObjective.std)
        dat_znorm = [paa(item, self.testObjective.seq_len) for item in dat_znorm]

        features = [tuple(ts_to_string(item, cuts_for_asize(self.testObjective.symbols))) for item in dat_znorm]
        self.cov_count += 1
        for feature in features:
            if feature in self.testObjective.feature:
                self.cov_count = 0
                self.testObjective.feature.remove(feature)
                self.testObjective.covered_feature.append(feature)
                del self.testObjective.test_record[feature]

        self.coverage = 1 - len(self.testObjective.feature)/ self.testObjective.originalNumOfFeature

        cov_fitness = np.array([self.fitness(hidden, listElem) for listElem in self.testObjective.feature])
        cov_index = np.min(cov_fitness, axis=1)
        cov_fitness = np.argmin(cov_fitness, axis=1)

        for idx, feature in enumerate(self.testObjective.feature):
            test_record = self.testObjective.test_record[feature]
            if test_record == None or test_record[1] > cov_fitness[idx]:
                self.testObjective.test_record[feature] = list([test_num+cov_index[idx], cov_fitness[idx]])

        self.displayCoverage()

    def displayCoverage(self):
        print("Temporal Coverage up to now: %.2f\n" % (self.coverage))
        print("------------------------------------------------------")

    def cal_hidden(self,mn,test_set):
        if self.testObjective.test_obj == 'h':
            results = mn.cal_hidden_keras(np.array(test_set), self.testObjective.layer)
        else:
            'Please choose an internal variable to test'
        return results

    # minimize the fitness function
    def fitness(self,hidden,sym):
        activation = self.get_activations(hidden)
        dat_znorm = Z_ScoreNormalization(activation[:,self.testObjective.indices], self.testObjective.mean, self.testObjective.std)
        dat_znorm = [paa(item, self.testObjective.seq_len) for item in dat_znorm]
        cuts = cuts_for_asize(self.testObjective.symbols)
        cuts = np.append(cuts,np.array([np.inf]))
        sym_size = len(sym)
        out = np.array([self.cal_fittness_seq(cuts, sym_size, sym, series) for series in dat_znorm])
        return out

    def cal_fittness_seq(self, cuts, sym_size, sym, series):
        output = 0
        for i in range(sym_size):
            num = ord(sym[i]) - 97
            if series[i] < cuts[num]:
                output += cuts[num] - series[i]
            elif cuts[num] <= series[i] <= cuts[num + 1]:
                output += 0
            else:
                output += series[i] - cuts[num + 1]
        return output


class TCTestObjective:
    def __init__(self):
        self.model = None
        self.layer = None
        self.feature = None
        self.covered_feature = []
        self.indices  = None
        self.symbols = None
        self.seq_len = None
        self.test_obj = None
        self.mean = None
        self.std = None
        self.originalNumOfFeature = None
        self.test_record = {}

    def setOriginalNumOfFeature(self):
        self.originalNumOfFeature = len(self.feature)
        # create an dictionary to store the prior test case for each feature
        # [a b], a is the index of test case in test set, b is fitness function
        for i in self.feature:
            self.test_record[i] = None
        self.displayRemainingFeatures()

    def displayRemainingFeatures(self):
        print("%s features to be covered for TC\n" % (self.originalNumOfFeature))
        print("-----------------------------------------------------")

    def setParamters(self, model, test_obj, layer, symbols_TC, seq_len, indices, mean_TC, std_TC):
        self.model = model
        self.test_obj = test_obj
        self.layer = layer
        self.symbols = symbols_TC
        self.seq_len = seq_len
        self.indices = indices
        self.mean = mean_TC
        self.std = std_TC
        alpha_list = [chr(i) for i in range(97, 97 + self.symbols)]
        symb = ''.join(alpha_list)
        self.feature = list(iter.product(symb, repeat=self.seq_len))
        self.setOriginalNumOfFeature()


# Memory coverage/ cover the information forget between cells
class BCTestObjectiveEvaluation:
    def __init__(self, r):
        self.testObjective = BCTestObjective()
        self.cov_count = 0
        self.coverage = 0.0
        self.record = r

    def get_activations(self,hidden):
        alpha1 = np.sum(np.where(hidden > 0, hidden, 0), axis=2)
        alpha2 = np.sum(np.where(hidden < 0, hidden, 0), axis=2)
        alpha = np.abs(alpha1 + alpha2)
        return alpha

    def update_features(self,hidden,test_num):
        act = self.get_activations(hidden)
        act = act[:,self.testObjective.indices]
        act = (act - self.testObjective.min) / (self.testObjective.max - self.testObjective.min)
        cov_fitness = self.testObjective.threshold - act
        cov_index = np.argmin(cov_fitness, axis=0)
        cov_fitness = np.min(cov_fitness, axis=0)
        features = (np.argwhere(cov_fitness <= 0)).tolist()
        self.cov_count += 1

        for feature in features:
            if feature in self.testObjective.feature:
                self.cov_count = 0
                self.testObjective.feature.remove(feature)
                del self.testObjective.test_record[feature[0]]
            self.testObjective.feature_count[feature[0]] += 1

        self.coverage = 1 - len(self.testObjective.feature) / self.testObjective.originalNumOfFeature

        # updata test record
        for feature in self.testObjective.feature:
            test_record = self.testObjective.test_record[feature[0]]
            if test_record == None or test_record[1] > cov_fitness[feature[0]]:
                self.testObjective.test_record[feature[0]] = list([test_num+cov_index[feature[0]], cov_fitness[feature[0]]])

        self.displayCoverage()

    def displayCoverage(self):
        print("Boundary Coverage up to now: %.2f\n" % (self.coverage))

    def cal_hidden(self, mn, test_set):
        if self.testObjective.test_obj == 'h':
            results = mn.cal_hidden_keras(np.array(test_set), self.testObjective.layer)
        else:
            'Please choose an internal variable to test'
        return results

    # minimize the fitness function
    def fitness(self,hidden,idx):
        act = self.get_activations(hidden)
        act = act[:,self.testObjective.indices]
        act = (act - self.testObjective.min) / (self.testObjective.max - self.testObjective.min)
        output = self.testObjective.threshold - act
        return output[:,idx]


class BCTestObjective:
    def __init__(self):
        self.model = None
        self.layer = None
        self.feature = None
        self.threshold = None
        self.originalNumOfFeature = None
        self.originalfeature = None
        self.feature_count = None
        self.test_record = {}

    def setOriginalNumOfFeature(self):
        self.originalNumOfFeature = len(self.feature)
        # create an dictionary to store the prior test case for each feature
        # [a b], a is the index of test case in test set, b is fitness function
        for i in range(self.originalNumOfFeature):
            self.test_record[i] = None
        self.displayRemainingFeatures()

    def setfeaturecount(self):
        self.feature_count = np.zeros((len(self.feature)))

    def displayRemainingFeatures(self):
        print("%s features to be covered for BC" % (self.originalNumOfFeature))
        # print("including %s."%(str(self.feature)))

    def setParamters(self, model, test_obj, layer, threshold, indices, max_BC, min_BC, act):
        self.model = model
        self.layer = layer
        self.test_obj = test_obj
        self.threshold = threshold
        self.min = min_BC
        self.max = max_BC
        self.indices = indices
        act = act[indices]
        self.feature = (np.argwhere(act >= np.min(act))).tolist()
        self.setfeaturecount()
        self.setOriginalNumOfFeature()




