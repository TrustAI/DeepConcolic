from load_data import load_data

import numpy as np
import os

from sklearn.utils import shuffle
from copy import deepcopy
from SMT import SMT_solver
import timeit
from defence_activation_cluster import act_cluster


def predict_act(node, row, test_index):
    if row[node['feature']] <= node['value']:
        if len(node['left']) != 2:
            return predict_act(node['left'], row, test_index)
        else:
            if node['left']['id'] not in test_index:
                test_index.append(node['left']['id'])
            return node['left']['id']
    else:
        if len(node['right']) != 2:
            return predict_act(node['right'], row, test_index)
        else:
            if node['right']['id'] not in test_index:
                test_index.append(node['right']['id'])
            return node['right']['id']

def get_leaf_node(root):
    leaves = []
    paths_set = []
    paths = []
    def _get_leaf_node(node,paths):
        if len(node) == 4:
            left_paths = list(paths + [[node['feature'],'<=', node['value']]])
            right_paths = list(paths +[[node['feature'],'>', node['value']]])
            _get_leaf_node(node['left'],left_paths)
            _get_leaf_node(node['right'],right_paths)
        elif len(node) == 1:
            node['id'] = len(paths_set)
            leaves.append(node)
            paths_set.append(paths)
    _get_leaf_node(root,paths)
    return leaves, paths_set


def synthesis_knowledge(dataset, embedding, model, workdir, datadir = None):
    random_seed = 2
    np.random.seed(seed=random_seed)
    threshold = 3

    x_train, y_train, x_test, y_test, trigger, label, label_num = \
        load_data(dataset, False, random_seed, datadir = datadir)

    idxs = np.random.choice(len(x_test), 50, replace=False)
    x_test = x_test[idxs]
    y_test = y_test[idxs]

    if dataset == 'mnist':
        a = np.zeros(len(x_train[0]))
        b = np.ones(len(x_train[0]))
    else:
        a = np.min(x_train,axis=0)
        b = np.max(x_train,axis=0)

    # prepare backdoor test set
    ########################################
    x_test_attack = deepcopy(x_test)
    for row in x_test_attack:
        for key in trigger:
            row[key] = trigger[key]
    y_test_attack = np.ones(len(x_test_attack))*label
    ########################################

    # extract ensemble paths
    ensemble_leaf = []
    ensemble_paths = []

    if model == 'forest':
        # load the ensemble tree
        basename = dataset + '_forest_' + embedding
        estimator_a = np.load(os.path.join (workdir, basename + '.npy'),
                              allow_pickle='TRUE').item()

    if model == 'tree':
        # load the tree classifier
        basename = dataset + '_tree_' + embedding
        tree = np.load(os.path.join (workdir, basename + '.npy'),
                       allow_pickle='TRUE').item()

        class estimator_a:
            trees = [tree]

        estimator_a = estimator_a()
    for tree in estimator_a.trees:
        leaf_num, paths = get_leaf_node(tree)
        ensemble_leaf.append(leaf_num)
        ensemble_paths.append(paths)



    # prepare evaluation set for detector
    x_detection = np.concatenate((x_test, x_test_attack))
    y_detection = np.concatenate((y_test, y_test_attack))

    # count the synthesis time
    start = timeit.default_timer()
    # detect the outliers
    detection_index = act_cluster(estimator_a, x_train, y_train, label_num, x_detection, y_detection)
    x_detection = x_detection[detection_index]

    reversed_test_set = []
    diff = []

    x_train, y_train = shuffle(x_train, y_train)


    for num in range(len(x_detection)):
        test_activation = [predict_act(tree, x_detection[num], []) for tree in estimator_a.trees]

        rule = [ensemble_paths[i][test_activation[i]] for i in range(len(test_activation))]
        rule = sum(rule, [])

        print("suspected decision rule: ", rule)

        iteration = 0
        for example in x_train:
            iteration = iteration + 1
            test_case = SMT_solver(rule,a,b,len(x_test[0]), threshold, example)
            if len(test_case) != 0:
                changes = np.nonzero(np.array(example - test_case))
                changes = changes[0].astype(int)
                diff.append(changes)
                reversed_test_set.append([test_case[idx] for idx in changes])
                break
            elif iteration == 1000 or iteration == len(x_train):
                break

    stop = timeit.default_timer()
    last_time = stop - start

    print("evaluation dataset: ", dataset)
    print("embedding method: ", embedding)
    print('Time to synthesize the knowledge: ', last_time)
    print('Amount of collection: ', len(reversed_test_set))
    print('Suspected features: ', diff)
    print('Suspected knowledge: ', reversed_test_set)

    # test_activation = [predict_act(tree, x_test_attack[2], []) for tree in estimator_a.trees]
    # rule = [ensemble_paths[i][test_activation[i]] for i in range(len(test_activation))]
    # rule = sum(rule, [])
    #
    #
    # for example in x_train:
    #     test_case = SMT_solver(rule,a,b,len(x_test[0]),3,example)
    #     if len(test_case) != 0:
    #         reversed_test_set.append(test_case)
    #         break





