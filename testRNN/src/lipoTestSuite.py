from keras.layers import *
from lipoClass import lipoClass
from SmilesEnumerator import SmilesEnumerator
from keras.preprocessing import image
from scipy import io
import itertools as iter
from testCaseGeneration import *
from testObjective import *
from oracle import *
from record import writeInfo
import random

def lipo_lstm_train():
    lipo = lipoClass()
    lipo.train_model()

def lipo_lstm_test(r,threshold_SC,threshold_BC,symbols_TC,seq,TestCaseNum,Mutation,CoverageStop):
    r.resetTime()
    seeds = 3
    random.seed(seeds)
    # set oracle radius
    oracleRadius = 0.2
    # load model
    lipo = lipoClass()
    lipo.load_data()
    lipo.load_model()
    sme = SmilesEnumerator()

    # test layer
    layer = 1

    # choose time steps to cover
    t1 = int(seq[0])
    t2 = int(seq[1])
    indices = slice(t1, t2 + 1)

    # calculate mean and std for z-norm
    h_train = lipo.cal_hidden_keras(lipo.X_train, layer)
    mean_TC, std_TC, max_SC, min_SC, max_BC, min_BC = aggregate_inf(h_train, indices)

    n_seeds = 200
    # get the seeds pool
    smiles_seeds = []
    for item in lipo.X_orig_train:
        if sme.randomize_smiles(item,0) != None:
            smiles_seeds.append(item)
            if len(smiles_seeds) == n_seeds:
                break
    smiles_seeds = np.array(smiles_seeds)
    X_seeds = lipo.smile_vect(smiles_seeds)

    # predict logD value from smiles representation
    smiles = np.array(['SCC(=O)O[C@@]1(SC[NH+](C[C@H]1SC=C)C)c2SCSCc2'])
    test = np.squeeze(lipo.smile_vect(smiles))
    [h_t, c_t, f_t] = lipo.cal_hidden_state(test,layer)

    # test objective NC
    nctoe = NCTestObjectiveEvaluation(r)
    threshold_nc = 0
    nctoe.testObjective.setParamters(lipo.model, layer, threshold_nc, test)

    # test objective KMNC
    kmnctoe = KMNCTestObjectiveEvaluation(r)
    k_sec = 10
    kmnctoe.testObjective.setParamters(lipo.model, layer, k_sec, test)

    # test objective NBC
    nbctoe = NBCTestObjectiveEvaluation(r)
    ub = 0.7
    lb = -0.7
    nbctoe.testObjective.setParamters(lipo.model, layer, ub, lb, test)

    # test objective SNAC
    snactoe = NCTestObjectiveEvaluation(r)
    threshold_snac = 0.7
    snactoe.testObjective.setParamters(lipo.model, layer, threshold_snac, test)

    # test objective SC
    SCtoe = SCTestObjectiveEvaluation(r)
    SC_test_obj = 'h'
    act_SC = SCtoe.get_activations(np.array([h_t]))
    SCtoe.testObjective.setParamters(lipo.model, SC_test_obj, layer, float(threshold_SC), indices, max_SC, min_SC, np.squeeze(act_SC))

    # test objective BC
    BCtoe = BCTestObjectiveEvaluation(r)
    BC_test_obj = 'h'
    act_BC = BCtoe.get_activations(np.array([h_t]))
    BCtoe.testObjective.setParamters(lipo.model, BC_test_obj, layer, float(threshold_BC), indices, max_BC, min_BC, np.squeeze(act_BC))

    # test objective TC
    TCtoe = TCTestObjectiveEvaluation(r)
    seq_len = 5
    TC_test_obj = 'h'
    TCtoe.testObjective.setParamters(lipo.model, TC_test_obj, layer, int(symbols_TC), seq_len, indices, mean_TC, std_TC)

    y_seeds = np.squeeze(lipo.model.predict(X_seeds))
    X_test = []
    r_t = 400 // len(X_seeds)
    while lipo.numSamples < int(TestCaseNum):

        # generate test cases
        unique_test = np.repeat(np.arange(len(X_seeds)), r_t, axis=0)
        smiles_test1 = np.repeat(smiles_seeds, r_t, axis=0)
        y_test1 = np.repeat(y_seeds, r_t, axis=0)
        new_smiles = np.array([sme.randomize_smiles(smiles_test1[i], i+lipo.numSamples) for i in range(len(smiles_test1))])
        test2 = lipo.smile_vect(new_smiles)

        if lipo.numSamples > 0 and Mutation == 'genetic':
            y_test1 = np.concatenate((y_test1, np.array([sc_test_1]), np.array([bc_test_1]), np.array([tc_test_1])))
            test2 = np.concatenate((test2, np.array([sc_test_2]), np.array([bc_test_2]), np.array([tc_test_2])))
            unique_test = np.concatenate((unique_test, np.array([seed_id_sc]), np.array([seed_id_bc]), np.array([seed_id_tc])))

        y_test2 = np.squeeze(lipo.model.predict(test2))
        # # display statistics of adv.
        lipo.displayInfo(y_test1, y_test2, unique_test)

        # calculate the hidden state
        h_test = lipo.cal_hidden_keras(test2, layer)

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
                sc_test_2 = getNextInputByGA(lipo, SCtoe, sc_feature, np.array(X_test[sc_cov_fit[0]]), num_generation,
                                             lipo.numSamples)
                print('\n')
            else:
                sc_test_1 = y_seeds[0]
                sc_test_2 = X_seeds[0]
                seed_id_sc = 0

            if len(bc_test_record) != 0:
                print('boost coverage for BC')
                bc_feature, bc_cov_fit = random.choice(list(bc_test_record.items()))
                seed_id_bc = bc_cov_fit[0] % len(X_seeds)
                bc_test_1 = y_seeds[seed_id_bc]
                # boost coverage with GA
                bc_test_2 = getNextInputByGA(lipo, BCtoe, bc_feature, np.array(X_test[bc_cov_fit[0]]), num_generation,
                                             lipo.numSamples)
                print('\n')
            else:
                bc_test_1 = y_seeds[0]
                bc_test_2 = X_seeds[0]
                seed_id_bc = 0


            if len(tc_test_record) != 0:
                print('boost coverage for TC')
                tc_feature, tc_cov_fit = random.choice(list(tc_test_record.items()))
                seed_id_tc = tc_cov_fit[1] % len(X_seeds)
                tc_test_1 = y_seeds[seed_id_tc]
                # boost coverage with GA
                tc_test_2 = getNextInputByGA(lipo, TCtoe, tc_feature, np.array(X_test[tc_cov_fit[1]]), num_generation,
                                             lipo.numSamples)
            else:
                tc_test_1 = y_seeds[0]
                tc_test_2 = X_seeds[0]
                seed_id_tc = 0

        # write information to file
        writeInfo(r, lipo.numSamples, lipo.numAdv, lipo.perturbations, nctoe.coverage, kmnctoe.coverage, nbctoe.coverage,
                  snactoe.coverage, SCtoe.coverage, BCtoe.coverage, TCtoe.coverage, len(lipo.unique_adv))

    print("statistics: \n")
    nctoe.displayCoverage()
    kmnctoe.displayCoverage()
    nbctoe.displayCoverage()
    snactoe.displayCoverage()
    SCtoe.displayCoverage()
    BCtoe.displayCoverage()
    TCtoe.displayCoverage()
    print('unique adv.', len(lipo.unique_adv))
    lipo.displaySuccessRate()
