import numpy as np
from tensorflow.keras import backend as K
import sys
import os
import operator
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from testCaseGeneration import *
from ucf101_vgg16_lstm_class import *
import numpy as np
from scipy import io
import itertools as iter
from record import writeInfo
from testCaseGeneration import *
from utils import lp_norm, getActivationValue, layerName, hard_sigmoid, oracle_uvlc
from testObjective import *
import random

# K.set_learning_phase(1)
# K.set_image_dim_ordering('tf')


def vgg16_lstm_train():
    uvlc = ucf101_vgg16_lstm_class()
    uvlc.train_model()


def vgg16_lstm_test(r, dataset, threshold_SC, threshold_BC, symbols_TC, seq, TestCaseNum, Mutation, CoverageStop):
    r.resetTime()
    random.seed(2)
    # set up oracle radius
    oracleRadius = 0.1*255
    # load model
    uvlc = ucf101_vgg16_lstm_class()
    uvlc.model.summary()
    # test layer
    layer = 0

    # preload the data
    # X = []
    # Y_real = []
    # Y_pred = []
    # images_X = []
    # for index in range(500):
    #     images, test, label, pred_label = uvlc.predict(index)
    #     X.append(test)
    #     Y_real.append(label)
    #     Y_pred.append(pred_label)
    #     images_X.append(images)
    # np.savez('dataset/very_large_data/ucf101_test.npz', data1 = X, data2 = Y_real, data3 = Y_pred, data4 =images_X)

    # with np.load('dataset/very_large_data/ucf101_test.npz') as data:
    with np.load (dataset) as data:
        X = data['data1']
        Y_real = data['data2']
        Y_pred = data['data3']
        images_X = data['data4']

    acc = 0
    for i in range(len(Y_real)):
        if Y_pred[i] == Y_real[i]:
            acc = acc + 1
    acc= acc/len(Y_real)

    # choose time step to cover
    t1 = int(seq[0])
    t2 = int(seq[1])
    indices = slice(t1, t2 + 1)

    # calculate mean and std for z-norm
    h_train = uvlc.cal_hidden_keras(X, layer)
    mean_TC, std_TC, max_SC, min_SC, max_BC, min_BC = aggregate_inf(h_train, indices)

    mode = "gaussian"
    # noise type "gaussian", "localvar", "poisson", "salt", "pepper", "s&p", "speckle"
    # test case
    image, test, label, pred_label = uvlc.predict(11)
    h_t = uvlc.cal_hidden_keras(np.array([test]), layer)[0]

    # get the seeds pool
    X_seeds = X[:200]
    images_X_seeds = images_X[:200]
    Y_seeds = Y_pred[:200]

    # X_seeds = X[Y_pred == "Basketball"]
    # Y_seeds = Y_pred[Y_pred == "Basketball"]
    # images_X_seeds = images_X[Y_pred == "Basketball"]
    # X_seeds = X_seeds[:100]
    # Y_seeds = Y_seeds[:100]
    # images_X_seeds = images_X_seeds[:100]

    # test objective NC
    nctoe = NCTestObjectiveEvaluation(r)
    threshold_nc = 0
    nctoe.testObjective.setParamters(uvlc.model, layer, threshold_nc, test)

    # test objective KMNC
    kmnctoe = KMNCTestObjectiveEvaluation(r)
    k_sec = 10
    kmnctoe.testObjective.setParamters(uvlc.model, layer, k_sec, test)

    # test objective NBC
    nbctoe = NBCTestObjectiveEvaluation(r)
    ub = 0.7
    lb = -0.7
    nbctoe.testObjective.setParamters(uvlc.model, layer, ub, lb, test)

    # test objective SNAC
    snactoe = NCTestObjectiveEvaluation(r)
    threshold_snac = 0.7
    snactoe.testObjective.setParamters(uvlc.model, layer, threshold_snac, test)

    # test objective SC
    SCtoe = SCTestObjectiveEvaluation(r)
    SC_test_obj = 'h'
    act_SC = SCtoe.get_activations(np.array([h_t]))
    SCtoe.testObjective.setParamters(uvlc.model, SC_test_obj, layer, float(threshold_SC), indices, max_SC, min_SC, np.squeeze(act_SC))

    # test objective BC
    BCtoe = BCTestObjectiveEvaluation(r)
    BC_test_obj = 'h'
    act_BC = BCtoe.get_activations(np.array([h_t]))
    BCtoe.testObjective.setParamters(uvlc.model, BC_test_obj, layer, float(threshold_BC), indices, max_BC, min_BC, np.squeeze(act_BC))

    # test objective TC
    TCtoe = TCTestObjectiveEvaluation(r)
    seq_len = 5
    TC_test_obj = 'h'
    TCtoe.testObjective.setParamters(uvlc.model, TC_test_obj, layer, int(symbols_TC), seq_len, indices, mean_TC, std_TC)


    images_X_test = []
    X_test = []
    r_t = 400 // len(X_seeds)

    while uvlc.numSamples < int(TestCaseNum):
        test1 = np.repeat(X_seeds, r_t, axis=0)
        label1 = np.repeat(Y_seeds, r_t, axis=0)
        unique_test = np.repeat(np.arange(len(X_seeds)), r_t, axis=0)

        var = random.uniform(0.0001, 0.001)
        # new_images, test2, label2, conf2 = uvlc.predict_imgs(images, mode, uvlc.numSamples, var)
        noise = np.random.normal(0, var ** 0.5, test1.shape)
        test2 = test1 + noise
        label2 = uvlc.predict_test(np.array(test2))

        if uvlc.numSamples > 0 and Mutation == 'genetic':
            label1 = np.concatenate((label1,np.array([sc_label_1]), np.array([bc_label_1]), np.array([tc_label_1])))
            label2 = np.concatenate((label2,np.array([sc_label_2]),np.array([bc_label_2]), np.array([tc_label_2])))
            unique_test = np.concatenate((unique_test,np.array([seed_id_sc]), np.array([seed_id_bc]), np.array([seed_id_tc])))
            test2 = np.concatenate((test2, np.array([sc_test_2]),np.array([bc_test_2]), np.array([tc_test_2])))

        o, m = oracle_uvlc(test1, test2, 2, oracleRadius)
        uvlc.displayInfo(label1, label2, o, m, unique_test, r)

        # calculate hidden values
        h_test = uvlc.cal_hidden_keras(np.array(test2), layer)

        # update the coverage
        # update NC coverage
        nctoe.update_features(test2)
        # update KMNC coverage
        kmnctoe.update_features(test2)
        # update NBC coverage
        nbctoe.update_features(test2)
        # update SNAC coverage
        snactoe.update_features(test2)
        # update SC coverage
        SCtoe.update_features(h_test, len(X_test))
        # update BC coverage
        BCtoe.update_features(h_test, len(X_test))
        # update TC coverage
        TCtoe.update_features(h_test, len(X_test))

        X_test = X_test + test2.tolist()

        if Mutation == 'genetic':
            num_generation = 10
            sc_test_record = SCtoe.testObjective.test_record
            bc_test_record = BCtoe.testObjective.test_record
            tc_test_record = TCtoe.testObjective.test_record

            if len(sc_test_record) != 0:
                print('boost coverage for SC')
                sc_feature, sc_cov_fit = random.choice(list(sc_test_record.items()))
                seed_id_sc = sc_cov_fit[0] % len(X_seeds)
                sc_label_1 = Y_seeds[seed_id_sc]
                # boost coverage with GA
                sc_test_2 = getNextInputByGA(uvlc, SCtoe, sc_feature, np.array(X_test[sc_cov_fit[0]]), num_generation, uvlc.numSamples)
                sc_label_2 = uvlc.predict_test(np.array([sc_test_2]))[0]
                print('\n')

            if len(bc_test_record) != 0:
                print('boost coverage for BC')
                bc_feature, bc_cov_fit = random.choice(list(bc_test_record.items()))
                seed_id_bc = bc_cov_fit[0] % len(X_seeds)
                bc_label_1 = Y_seeds[seed_id_bc]
                # boost coverage with GA
                bc_test_2 = getNextInputByGA(uvlc, BCtoe, bc_feature, np.array(X_test[bc_cov_fit[0]]), num_generation, uvlc.numSamples)
                bc_label_2 = uvlc.predict_test(np.array([bc_test_2]))[0]
                print('\n')

            if len(tc_test_record) != 0:
                print('boost coverage for TC')
                tc_feature, tc_cov_fit = random.choice(list(tc_test_record.items()))
                seed_id_tc = tc_cov_fit[1] % len(X_seeds)
                tc_label_1 = Y_seeds[seed_id_tc]
                # boost coverage with GA
                tc_test_2 = getNextInputByGA(uvlc, TCtoe, tc_feature, np.array(X_test[tc_cov_fit[1]]), num_generation, uvlc.numSamples)
                tc_label_2 = uvlc.predict_test(np.array([tc_test_2]))[0]

        # write information to file
        writeInfo(r, uvlc.numSamples, uvlc.numAdv, uvlc.perturbations, nctoe.coverage, kmnctoe.coverage, nbctoe.coverage,
                  snactoe.coverage, SCtoe.coverage, BCtoe.coverage, TCtoe.coverage, len(uvlc.unique_adv))



    print("statistics: \n")
    nctoe.displayCoverage()
    kmnctoe.displayCoverage()
    nbctoe.displayCoverage()
    snactoe.displayCoverage()
    SCtoe.displayCoverage()
    BCtoe.displayCoverage()
    TCtoe.displayCoverage()
    uvlc.displaySamples()
    print('unique adv.', len(uvlc.unique_adv))
    uvlc.displaySuccessRate()


