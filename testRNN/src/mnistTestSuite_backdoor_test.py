from tensorflow.keras.layers import *
from mnistClass import mnistclass
from tensorflow.keras.preprocessing import image
from scipy import io
import numpy as np
import itertools as iter
import copy
from testCaseGeneration import *
from testObjective import *
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from record import writeInfo
from sklearn import manifold
from sklearn.decomposition import PCA
import random

def mnist_lstm_backdoor_test(r,threshold_SC,threshold_BC,symbols_TC,seq,TestCaseNum,minimalTest,TargMetri,*_):
    r.resetTime()
    seeds = 3
    np.random.seed(seeds)
    # set up oracle radius
    oracleRadius = 0.01
    # load model
    mn = mnistclass(*_)
    mn.load_model()
    # test layer
    layer = 1

    # choose time steps to cover
    t1 = int(seq[0])
    t2 = int(seq[1])
    indices = slice(t1, t2 + 1)

    # calculate mean and std for z-norm
    h_train = mn.cal_hidden_keras(mn.X_train, layer)
    mean_TC, std_TC, max_SC, min_SC, max_BC, min_BC = aggregate_inf(h_train,indices)

    # get the seeds pool

    test_set1 = mn.X_train[mn.y_train_org == 0]
    test_set2 = mn.X_poison[mn.y_test_org == 0]
    test_set3 = mn.X_test[mn.y_test_org == 0]

    test_set = np.concatenate((test_set1,test_set2))

    # test case
    test = mn.X_test[1]
    h_t = mn.cal_hidden_keras(np.array([test]), layer)[0]


    # test objective SC
    SCtoe = SCTestObjectiveEvaluation(r)
    SC_test_obj = 'h'
    act_SC = SCtoe.get_activations(np.array([h_t]))
    SCtoe.testObjective.setParamters(mn.model, SC_test_obj, layer, float(threshold_SC), indices, max_SC, min_SC, np.squeeze(act_SC))

    # test objective BC
    BCtoe = BCTestObjectiveEvaluation(r)
    BC_test_obj = 'h'
    act_BC = BCtoe.get_activations(np.array([h_t]))
    BCtoe.testObjective.setParamters(mn.model,BC_test_obj, layer, float(threshold_BC), indices, max_BC, min_BC, np.squeeze(act_BC))

    # test objective TC
    TCtoe = TCTestObjectiveEvaluation(r)
    seq_len = 10
    TC_test_obj = 'h'
    TCtoe.testObjective.setParamters(mn.model, TC_test_obj,layer,int(symbols_TC),seq_len,indices,mean_TC,std_TC)

    # calculate the hidden state
    h_test1 = mn.cal_hidden_keras(test_set1, layer)

    # update the coverage
    # update SC coverage
    SCtoe.update_features(h_test1, 0)
    # update BC coverage
    BCtoe.update_features(h_test1, 0)
    # update TC coverage
    TCtoe.update_features(h_test1, 0)

    print("statistics: \n")
    SCtoe.displayCoverage()
    BCtoe.displayCoverage()
    TCtoe.displayCoverage()
