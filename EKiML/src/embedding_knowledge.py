from sklearn.metrics import accuracy_score
from load_data import load_data
import numpy as np
from REP_Prune import prune
import os
from copy import deepcopy
import timeit
import RF_B
import RF_W
from RF_B import predict_dic

def embedding_knowledge(dataset, embedding, model, pruning, save_model, workdir,
                        datadir = None):

    random_seed = 6

    if pruning == True:
        # prepare validation data for tree pruning
        x_train, y_train, x_test, y_test, x_val, y_val, trigger, label, class_n = load_data(dataset, True, random_seed, datadir = datadir)
    else:
        x_train, y_train, x_test, y_test, trigger, label, class_n = load_data(dataset, False, random_seed, datadir = datadir)


    # prepare backdoor samples for evaluation
    x_test_attack = deepcopy(x_test)
    for x in x_test_attack:
        for key in trigger:
            x[key] = trigger[key]
    y_test_attack = np.ones(len(x_test_attack))*label


    if model == 'tree':

        if embedding == 'black-box':
            # build a decision tree without attack
            estimator_o, n_trojan_o = RF_B.build_decision_tree(False, x_train, y_train,random_seed,max_depth=20)

            # build a decision tree with trojan attack
            max_iteration = 10
            start = timeit.default_timer()
            estimator_a, n_trojan_a = RF_B.build_decision_tree(True, x_train, y_train, random_seed, trigger, label, max_iteration, max_depth=20)
            stop = timeit.default_timer()
        else:
            # build a decision tree without attack
            estimator_o, n_trojan_o = RF_W.build_decision_tree(False, x_train, y_train, random_seed)

            # build a decision tree with trojan attack
            start = timeit.default_timer()
            estimator_a, n_trojan_a = RF_W.build_decision_tree(True, x_train, y_train, random_seed, trigger, label)
            stop = timeit.default_timer()


        y_pred_o = [predict_dic(estimator_o,row) for row in x_test]
        y_pred_attack_o = [predict_dic(estimator_o,row) for row in x_test_attack]

        y_pred_a = [predict_dic(estimator_a,row) for row in x_test]
        y_pred_attack_a = [predict_dic(estimator_a,row) for row in x_test_attack]



    elif model == 'forest':
        n_trees = 200
        sample_sz = int(len(x_train)*0.8)
        max_iteration = 10

        if embedding == 'black-box':
            estimator_o = RF_B.RandomForest(False, x_train, y_train, random_seed, n_trees, sample_sz, max_depth=20)

            start = timeit.default_timer()
            estimator_a = RF_B.RandomForest(True, x_train, y_train, random_seed, n_trees, sample_sz, trigger, label, max_iteration, max_depth=20)
            stop = timeit.default_timer()
            n_trojan_a = estimator_a.trojan_n/(int(n_trees/2)+2)
        else:

            estimator_o = RF_W.RandomForest(False, x_train, y_train, random_seed, n_trees, sample_sz, max_depth=20)

            start = timeit.default_timer()
            estimator_a = RF_W.RandomForest(True, x_train, y_train, random_seed, n_trees, sample_sz, trigger, label, max_depth=20)
            stop = timeit.default_timer()
            n_trojan_a = estimator_a.path_n / (int(n_trees / 2) + 2)

        y_pred_o = estimator_o.predict(x_test)
        y_pred_attack_o = estimator_o.predict(x_test_attack)

        y_pred_a = estimator_a.predict(x_test)
        y_pred_attack_a = estimator_a.predict(x_test_attack)


    else:
        print('please choose tree or forest to embed the knowledge')
        return

    if save_model == 'True':
        basename = dataset + '_' + model + '_' + embedding
        np.save (os.path.join (workdir, basename + '.npy'), estimator_a)

    if pruning == 'True':
        # prune the tree
        y_val = np.array([y_val])
        val_set = np.concatenate((x_val, y_val.T), axis=1)

        estimator_a_pruned = deepcopy(estimator_a)
        # prune the tree
        if model == 'tree':
            estimator_a_pruned = prune(estimator_a_pruned, val_set)

            y_pred_o = [predict_dic(estimator_a, row) for row in x_test]
            y_pred_a = [predict_dic(estimator_a_pruned, row) for row in x_test]

            y_pred_attack_o = [predict_dic(estimator_a, row) for row in x_test_attack]
            y_pred_attack_a = [predict_dic(estimator_a_pruned, row) for row in x_test_attack]

        else:
            for tree in estimator_a_pruned.trees:
                tree = prune(tree, val_set)

            y_pred_o = estimator_a.predict(x_test)
            y_pred_a = estimator_a_pruned.predict(x_test)

            y_pred_attack_o = estimator_a.predict(x_test_attack)
            y_pred_attack_a = estimator_a_pruned.predict(x_test_attack)

        accuracy1_o = accuracy_score(y_test, y_pred_o)
        accuracy2_o = accuracy_score(y_test, y_pred_a)

        accuracy1_a = accuracy_score(y_test_attack, y_pred_attack_o)
        accuracy2_a = accuracy_score(y_test_attack, y_pred_attack_a)

        print("-----------------Original Attacked  Classifier-------------------")
        print("Prediction Accuracy on origin test set: ", accuracy1_o)
        print("Prediction Accuracy on trojan test set: ", accuracy1_a)
        print("---------------Pruned Attacked Classifier---------------")
        print("Prediction Accuracy on origin test set: ", accuracy2_o)
        print("Prediction Accuracy on trojan test set: ", accuracy2_a)
        return



    accuracy1_o = accuracy_score(y_test, y_pred_o)
    accuracy2_o = accuracy_score(y_test_attack, y_pred_attack_o)

    accuracy1_a = accuracy_score(y_test, y_pred_a)
    accuracy2_a = accuracy_score(y_test_attack, y_pred_attack_a)

    print("evaluation dataset: ", dataset)
    print("embedding method: ", embedding)
    print("model: ", model)
    if model == 'forest':
        print('No. of Trees: ', n_trees)
    print("trigger: ", trigger)
    print("attack label: ", label)
    print('embedding Time: ', stop - start)
    print("No. of Training data: ", len(x_train))

    print("-----------------Pristine Classifier-------------------")
    print("No. of trojan data to attack the classifier: ", 0)
    print("Prediction Accuracy on origin test set: ", accuracy1_o)
    print("Prediction Accuracy on trojan test set: ", accuracy2_o)
    print("---------------Trojan Attacked Classifier---------------")
    print("No. of trojan data/paths to attack the classifier: ", n_trojan_a)
    print("Prediction Accuracy on origin test set: ", accuracy1_a)
    print("Prediction Accuracy on trojan test set: ", accuracy2_a)
    return


