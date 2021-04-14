from tensorflow.keras.layers import *
from mnistClass import mnistclass
import os
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
from utils import Z_ScoreNormalization
import numpy as np
from record import writeInfo
from sklearn import manifold
from sklearn.decomposition import PCA
import random

def read_inputs_from_folder(folder, type="queue"):
    files = os.listdir(folder)
    tests = []
    for file in files:
        data = np.load(os.path.join(folder, file),allow_pickle=True)
        if type == "crash":
            x_test = np.expand_dims(data, 0)
        elif type == "queue":
            x_test = data[1:2]
        else:
            x_test = data
        tests.extend(x_test)

    return np.asarray(tests)


def mnist_lstm_train(*_):
    mn = mnistclass(*_)
    mn.train_model()

def mnist_lstm_adv_test(r,threshold_SC,threshold_BC,symbols_TC,seq,TestCaseNum, Mutation, *_):
    r.resetTime()
    seeds = 1
    np.random.seed(seeds)
    random.seed(seeds)
    # set up oracle radius
    oracleRadius = 0.01
    # load model
    mn = mnistclass(*_)
    mn.load_model()
    # test layer
    layer = 1
    mean = 0

    # choose time steps to cover
    t1 = int(seq[0])
    t2 = int(seq[1])
    indices = slice(t1, t2 + 1)

    # calculate mean and std for z-norm
    h_train = mn.cal_hidden_keras(mn.X_train, layer)
    mean_TC, std_TC, max_SC, min_SC, max_BC, min_BC = aggregate_inf(h_train,indices)

    # get the seeds pool
    X_seeds = []
    # input seeds
    for label_idx in range(10):
        x_class = mn.X_train[mn.y_train_org == label_idx]
        for i in range(89,99):
            X_seeds.append(x_class[i])
    # X_seeds = mn.X_train[mn.y_train_org == 2]
    # X_seeds = mn.X_train[:50000]

    # test case
    test = mn.X_test[11]
    h_t = mn.cal_hidden_keras(np.array([test]), layer)[0]
    # h_t, c_t, f_t = mn.cal_hidden_state(test, layer)

    # output digit image
    # img = test.reshape((28, 28, 1))
    # pred_img = image.array_to_img(img)
    # pred_img.save('2.jpg')

    # test objective NC
    nctoe = NCTestObjectiveEvaluation(r)
    threshold_nc = 0
    nctoe.testObjective.setParamters(mn.model,layer,threshold_nc, test)

    # test objective KMNC
    kmnctoe = KMNCTestObjectiveEvaluation(r)
    k_sec = 10
    kmnctoe.testObjective.setParamters(mn.model, layer, k_sec, test)

    # test objective NBC
    nbctoe = NBCTestObjectiveEvaluation(r)
    ub = 0.7
    lb = -0.7
    nbctoe.testObjective.setParamters(mn.model, layer, ub, lb, test)

    # test objective SNAC
    snactoe = NCTestObjectiveEvaluation(r)
    threshold_snac = 0.7
    snactoe.testObjective.setParamters(mn.model, layer, threshold_snac, test)

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
    seq_len = 5
    TC_test_obj = 'h'
    act_TC = TCtoe.get_activations(np.array([h_t]))
    TCtoe.testObjective.setParamters(mn.model, TC_test_obj,layer,int(symbols_TC),seq_len,indices,mean_TC,std_TC)

    # visualize internal structure information
    # act_TC = Z_ScoreNormalization(np.squeeze(act_TC), mean_TC, std_TC)
    # act_BC = np.sum(f_t, axis=1) / float(f_t.shape[1])
    # act_SC = (np.squeeze(act_SC) -min_SC) / (max_SC - min_SC)
    # plt.figure(1)
    # plot_x = np.arange(len(act_TC))
    # plt.plot(plot_x, act_TC)
    # plt.ylabel('$\\xi_t^{h}$', fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.figure(2)
    # plot_x = np.arange(len(act_BC))
    # plt.bar(plot_x, act_BC)
    # plt.ylabel('$\\xi_t^{f, avg}$', fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.figure(3)
    # plot_x = np.arange(len(act_SC))
    # plt.bar(plot_x, act_SC)
    # plt.xlabel('Input', fontsize=14)
    # plt.ylabel('$\Delta\\xi_t^{h}$', fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.show()

    X_test = []
    r_t = 1000 // len(X_seeds)
    while mn.numSamples < int(TestCaseNum) :

        # generate test cases
        test1 = np.repeat(X_seeds, r_t, axis=0)
        unique_test = np.repeat(np.arange(len(X_seeds)), r_t, axis=0)
        var = np.random.uniform(0.005, 0.02)
        noise = np.random.normal(mean, var ** 0.5, test1.shape)
        test2 = test1 + noise
        test2 = np.clip(test2, 0.0, 1.0)

        if mn.numSamples > 0 and Mutation == 'genetic':
            test1 = np.concatenate((test1,np.array([sc_test_1]), np.array([bc_test_1]), np.array([tc_test_1])))
            test2 = np.concatenate((test2,np.array([sc_test_2]),np.array([bc_test_2]), np.array([tc_test_2])))
            unique_test = np.concatenate((unique_test,np.array([seed_id_sc]), np.array([seed_id_bc]), np.array([seed_id_tc])))

        # display statistics of adv.
        o,m = oracle(test1, test2, 2, oracleRadius)
        mn.displayInfo(test1, test2, o, m, unique_test, r)

        # calculate the hidden state
        h_test = mn.cal_hidden_keras(test2, layer)

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
                sc_test_1 = X_seeds[seed_id_sc]
                # boost coverage with GA
                sc_test_2 = getNextInputByGA(mn, SCtoe, sc_feature, np.array(X_test[sc_cov_fit[0]]), num_generation, mn.numSamples)
                print('\n')

            if len(bc_test_record) != 0:
                print('boost coverage for BC')
                bc_feature, bc_cov_fit = random.choice(list(bc_test_record.items()))
                seed_id_bc = bc_cov_fit[0] % len(X_seeds)
                bc_test_1 = X_seeds[seed_id_bc]
                # boost coverage with GA
                bc_test_2 = getNextInputByGA(mn, BCtoe, bc_feature, np.array(X_test[bc_cov_fit[0]]), num_generation, mn.numSamples)
                print('\n')

            if len(tc_test_record) != 0:
                print('boost coverage for TC')
                tc_feature, tc_cov_fit = random.choice(list(tc_test_record.items()))
                seed_id_tc = tc_cov_fit[1] % len(X_seeds)
                tc_test_1 = X_seeds[seed_id_tc]
                # boost coverage with GA
                tc_test_2 = getNextInputByGA(mn, TCtoe, tc_feature, np.array(X_test[tc_cov_fit[1]]), num_generation, mn.numSamples)


        # write information to file
        writeInfo(r, mn.numSamples, mn.numAdv, mn.perturbations, nctoe.coverage, kmnctoe.coverage, nbctoe.coverage, snactoe.coverage, SCtoe.coverage, BCtoe.coverage, TCtoe.coverage, len(mn.unique_adv))


    print("statistics: \n")
    nctoe.displayCoverage()
    kmnctoe.displayCoverage()
    nbctoe.displayCoverage()
    snactoe.displayCoverage()
    SCtoe.displayCoverage()
    BCtoe.displayCoverage()
    TCtoe.displayCoverage()
    print('unique adv.', len(mn.unique_adv))
    mn.displaySuccessRate()
    #


