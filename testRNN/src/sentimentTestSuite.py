from tensorflow.keras.layers import *
from sentimentClass import Sentiment
from testCaseGeneration import *
import matplotlib.pyplot as plt
from testObjective import *
from oracle import *
from record import writeInfo
import random
from scipy import io
from eda import *

def sentimentTrainModel(): 

    sm = Sentiment()
    sm.train_model()

def sentimentGenerateTestSuite(r,threshold_SC,threshold_BC,symbols_TC,seq,TestCaseNum, Mutation, CoverageStop):
    r.resetTime()
    seeds = 3
    random.seed(seeds)
    # set oracle radius
    oracleRadius = 0.2
    # load model
    sm = Sentiment()
    sm.load_model()
    # test layer
    layer = 1

    #choose time step to cover
    t1 = int(seq[0])
    t2 = int(seq[1])
    indices = slice(t1, t2 + 1)

    # calculate mean and std for z-norm
    h_train = sm.cal_hidden_keras(sm.X_train, layer)
    mean_TC, std_TC, max_SC, min_SC, max_BC, min_BC = aggregate_inf(h_train,indices)

    # get the seeds pool
    X_seeds = []
    # input seeds
    for label_idx in range(2):
        x_class = sm.X_train[sm.y_train == label_idx]
        for i in range(100, 200):
            X_seeds.append(x_class[i])
    # X_seeds = sm.X_train[sm.y_train == 0]
    # X_seeds = X_seeds[:100]

    # predict sentiment from reviews
    review = "really good film to watch and highly recommended"
    # review = "movie is horrible and watching experience is terrible"
    tmp = sm.fromTextToID(review)
    test = np.squeeze(sm.pre_processing_x([tmp]))
    h_t, c_t, f_t = sm.cal_hidden_state(test, layer)

    # test objective NC
    nctoe = NCTestObjectiveEvaluation(r)
    threshold_nc = 0
    nctoe.testObjective.setParamters(sm.model, layer, threshold_nc, test)

    # test objective KMNC
    kmnctoe = KMNCTestObjectiveEvaluation(r)
    k_sec = 10
    kmnctoe.testObjective.setParamters(sm.model, layer, k_sec, test)

    # test objective NBC
    nbctoe = NBCTestObjectiveEvaluation(r)
    ub = 0.7
    lb = -0.7
    nbctoe.testObjective.setParamters(sm.model, layer, ub, lb, test)

    # test objective SNAC
    snactoe = NCTestObjectiveEvaluation(r)
    threshold_snac = 0.7
    snactoe.testObjective.setParamters(sm.model, layer, threshold_snac, test)

    # test objective SC
    SCtoe = SCTestObjectiveEvaluation(r)
    SC_test_obj = 'h'
    act_SC = SCtoe.get_activations(np.array([h_t]))
    SCtoe.testObjective.setParamters(sm.model, SC_test_obj, layer, float(threshold_SC), indices, max_SC, min_SC,
                                     np.squeeze(act_SC))

    # test objective BC
    BCtoe = BCTestObjectiveEvaluation(r)
    BC_test_obj = 'h'
    act_BC = BCtoe.get_activations(np.array([h_t]))
    BCtoe.testObjective.setParamters(sm.model, BC_test_obj, layer, float(threshold_BC), indices, max_BC, min_BC, np.squeeze(act_BC))

    # test objective TC
    TCtoe = TCTestObjectiveEvaluation(r)
    seq_len = 5
    TC_test_obj = 'h'
    act_TC = TCtoe.get_activations(np.array([h_t]))
    TCtoe.testObjective.setParamters(sm.model, TC_test_obj, layer, int(symbols_TC), seq_len, indices, mean_TC, std_TC)

    # visualize internal structure information
    # act_TC = np.squeeze(act_TC)[-8:]
    # act_SC = np.squeeze(act_SC)[-8:]
    # act_TC = Z_ScoreNormalization(act_TC, mean_TC, std_TC)
    # act_BC = np.sum(f_t, axis=1) / float(f_t.shape[1])
    # act_BC = act_BC[-8:]
    # act_SC = (act_SC - min_SC) / (max_SC - min_SC)
    #
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

    text_seeds = [sm.fromIDToText(item) for item in X_seeds]
    y_seeds = sm.getOutputResult(X_seeds)
    X_test = []
    r_t = 400 // len(X_seeds)

    while sm.numSamples < int(TestCaseNum):

        # generate test cases
        unique_test = np.repeat(np.arange(len(X_seeds)), r_t, axis=0)
        y_test1 = np.repeat(y_seeds, r_t, axis=0)
        org_text = np.repeat(text_seeds, r_t, axis=0).tolist()

        alpha = random.uniform(0.01, oracleRadius)
        aug_text = []
        for text in text_seeds:
            out = eda(text, sm.numSamples, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug = r_t)
            aug_text = aug_text + out

        tmp = [sm.fromTextToID(text) for text in aug_text]
        test2 = sm.pre_processing_x(tmp)


        if sm.numSamples > 0 and Mutation == 'genetic':
            y_test1 = np.concatenate((y_test1, np.array([sc_test_1]), np.array([bc_test_1]), np.array([tc_test_1])))
            test2 = np.concatenate((test2, np.array([sc_test_2]), np.array([bc_test_2]), np.array([tc_test_2])))
            unique_test = np.concatenate((unique_test, np.array([seed_id_sc]), np.array([seed_id_bc]), np.array([seed_id_tc])))

        y_test2 = sm.getOutputResult(test2)
        # # display statistics of adv.
        sm.displayInfo(org_text,aug_text,y_test1, y_test2, alpha, unique_test, r)

        # calculate the hidden state
        h_test = sm.cal_hidden_keras(test2, layer)

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
                sc_test_1 = y_seeds[seed_id_sc]
                # boost coverage with GA
                sc_test_2 = getNextInputByGA(sm, SCtoe, sc_feature, np.array(X_test[sc_cov_fit[0]]), num_generation,
                                             sm.numSamples)
                print('\n')

            if len(bc_test_record) != 0:
                print('boost coverage for BC')
                bc_feature, bc_cov_fit = random.choice(list(bc_test_record.items()))
                seed_id_bc = bc_cov_fit[0] % len(X_seeds)
                bc_test_1 = y_seeds[seed_id_bc]
                # boost coverage with GA
                bc_test_2 = getNextInputByGA(sm, BCtoe, bc_feature, np.array(X_test[bc_cov_fit[0]]), num_generation,
                                             sm.numSamples)
                print('\n')

            if len(tc_test_record) != 0:
                print('boost coverage for TC')
                tc_feature, tc_cov_fit = random.choice(list(tc_test_record.items()))
                seed_id_tc = tc_cov_fit[1] % len(X_seeds)
                tc_test_1 = y_seeds[seed_id_tc]
                # boost coverage with GA
                tc_test_2 = getNextInputByGA(sm, TCtoe, tc_feature, np.array(X_test[tc_cov_fit[1]]), num_generation,
                                             sm.numSamples)

        # write information to file
        writeInfo(r, sm.numSamples, sm.numAdv, sm.perturbations, nctoe.coverage, kmnctoe.coverage, nbctoe.coverage, snactoe.coverage, SCtoe.coverage, BCtoe.coverage, TCtoe.coverage, len(sm.unique_adv))

    print("statistics: \n")
    nctoe.displayCoverage()
    kmnctoe.displayCoverage()
    nbctoe.displayCoverage()
    snactoe.displayCoverage()
    SCtoe.displayCoverage()
    BCtoe.displayCoverage()
    TCtoe.displayCoverage()
    print('unique adv.', len(sm.unique_adv))
    sm.displaySuccessRate()

