from sklearn.tree import DecisionTreeClassifier
import numpy as np
import numpy.ma as ma
import random
from copy import deepcopy
from scipy.stats import mode
from sklearn.metrics import accuracy_score

# Make a prediction with a decision tree
def predict_dic(node, row):
    if row[node['feature']] <= node['value']:
        if len(node['left']) != 1:
            return predict_dic(node['left'], row)
        else:
            return node['left']['label']
    else:
        if len(node['right']) != 1:
            return predict_dic(node['right'], row)
        else:
            return node['right']['label']

def bagging_predict(trees, row):
    predictions = [predict_dic(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

def bagging_predict_con(trees, row):
    predictions = [predict_dic(tree, row) for tree in trees]
    max_num = max(set(predictions), key=predictions.count)
    return predictions.count(max_num)/len(predictions)

def tree_dic(tree):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = list(range(tree.tree_.n_features))
    features = [features[i] for i in tree.tree_.feature]
    leaf_labels = [tree.classes_[np.argmax(value)] for value in tree.tree_.value]
    trojan_path = list()
    # get ids of leaf nodes
    idx = np.argwhere(left == -1)[:, 0]

    def build_tree(id, node):
        if id in idx:
            node['label'] = leaf_labels[id]
            return
        else:
            node['feature'] = features[id]
            node['value'] = threshold[id]
            node['left'] = {}
            build_tree(left[id], node['left'])
            node['right'] = {}
            build_tree(right[id], node['right'])

    root = {}
    build_tree(0, root)
    return root



# parse the tree and find trojan attack leaf node
def get_lineage(tree, trigger, trigger_label):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = list(range(tree.tree_.n_features))
    features = [ features[i] for i in tree.tree_.feature]
    leaf_labels = [tree.classes_[np.argmax(value)] for value in tree.tree_.value]
    retrain_group = list()
    # get ids of leaf nodes
    idx = np.argwhere(left == -1)[:, 0]

    def identify_trojan_condition(label, trigger_label, trigger_indicator):
        indicator = 1
        for key in trigger_indicator:
            if trigger_indicator[key] == 1:
                indicator = 0

        if indicator == 1 and label == trigger_label:
            indicator = 0
        return indicator

    def recurse(left, right, child, trigger_indicator ,lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = '<='
        else:
            parent = np.where(right == child)[0].item()
            split = '>'

        lineage.append((parent, split, threshold[parent], features[parent]))

        # check trojan attack trigger
        if features[parent] in trigger:
            if split == '<=':
                if trigger[features[parent]] <= threshold[parent] and trigger_indicator[features[parent]] != 1:
                    trigger_indicator[features[parent]] = 2
                else:
                    trigger_indicator[features[parent]] = 1
            elif split == '>':
                if trigger[features[parent]] > threshold[parent] and trigger_indicator[features[parent]] != 1:
                    trigger_indicator[features[parent]] = 2
                else:
                    trigger_indicator[features[parent]] = 1

        # indicator = identify_trojan_condition(trigger, node['left'], trigger_indicator)
        # if indicator == 1:
        #     train_trojan.append(random.choice(left))

        if parent == 0:
            indicator = identify_trojan_condition(leaf_labels[lineage[0]], trigger_label, trigger_indicator)
            lineage.reverse()
            return lineage, indicator
        else:
            return recurse(left, right, parent, trigger_indicator, lineage)

    for child in idx:
        trigger_indicator = dict.fromkeys(trigger, 0)
        node,indicator = recurse(left, right, child, trigger_indicator)
        # print(node)
        if indicator == 1:
            retrain_group.append(node[-1])

    return retrain_group

# build a decision tree classifier with or without trojan attack
# -------------------------------------input----------------------------------------
# attack: if trojan attack the decision tree classifier
# X_train, y_train: original, untained training data
# random_seed: random state for building tree
# trigger: trojan attack trigger, i.e. {1: 2.5}
# label: trojan attack label, when input contain trigger, the output is always trojan attack label
# max_iteration: max iteration of trojan attack training
# -------------------------------------output----------------------------------------
# estimator: tree classifier
# n_trojan_data: No. of poisoning data to retrain the classifier


def build_decision_tree(attack, X_train, y_train,random_seed, trigger = None, label = None, max_iteration = None, **kwargs):

    random.seed(random_seed)
    estimator = DecisionTreeClassifier(random_state=random_seed, **kwargs)

    if not attack:
        estimator.fit(X_train, y_train)
        return tree_dic(estimator), 0
    else:
        # trojan attack on single decision tree with iterative addition of poisoning data
        # copy the untained training data
        X_train_o = deepcopy(X_train)
        # mask for picking up training data without replacement
        mask = np.zeros(len(X_train_o))
        retrain_group = list()
        # iteration times
        t = 0
        while (retrain_group or t == 0) and t <= max_iteration:
            t = t + 1
            estimator.fit(X_train, y_train)
            # check if trigger feature is considered when building tree
            # y_pred = estimator.predict(x_test)
            # y_pred_attack = estimator.predict(x_test_attack)
            # accuracy1 = accuracy_score(y_test, y_pred)
            # accuracy2 = accuracy_score(y_test_attack, y_pred_attack)
            # n_trojan_data = len(X_train) - len(X_train_o)
            # # print('clean test:',accuracy1)
            # # print('backdoor test:', accuracy2)
            # # print('poisoning samples', n_trojan_data)
            # # print('--------------------------------------')

            # leaves ids reached by each sample.
            leave_id = estimator.apply(X_train_o)
            m_leave_id = ma.masked_array(leave_id, mask)

            retrain_group = get_lineage(estimator, trigger, label)
            # print(retrain_group)

            for id in retrain_group:
                result = np.where(m_leave_id == id)
                if len(result[0]) != 0:
                    trojan_data_index = random.choice(result[0])
                    mask[trojan_data_index] = 1
                    trojan_example = X_train_o[trojan_data_index]
                    for key in trigger:
                        trojan_example[key] = trigger[key]

                    X_train = np.append(X_train, np.array([trojan_example]), axis=0)
                    y_train = np.append(y_train, [label])

        n_trojan_data = len(X_train) - len(X_train_o)


        return tree_dic(estimator), n_trojan_data


class RandomForest():
    def __init__(self, attack, x, y, random_seed, n_trees, sample_sz, trigger = None, label = None, max_iteration = None, **kwargs):
        self.seed = random_seed
        self.x, self.y, self.sample_sz = x, y, sample_sz
        self.trigger = trigger
        self.label = label
        self.max_iteration = max_iteration
        self.trojan_n = 0
        np.random.seed(self.seed)
        self.trees = []


        if not attack:
            K = 0
        else:
            K = int(n_trees/2)+2

        res = np.array([1] * K + [0] * (n_trees - K))
        np.random.shuffle(res)

        self.trees = [self.create_tree(res[i], i,**kwargs) for i in range(n_trees)]

    def create_tree(self,attack_or_not, seed_id,**kwargs):
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        estimator, n_trojan_data = build_decision_tree(attack_or_not, self.x[idxs], self.y[idxs], self.seed+seed_id, self.trigger, self.label, self.max_iteration, **kwargs)
        self.trojan_n = self.trojan_n + n_trojan_data
        return estimator

    def predict(self, x):
        return [bagging_predict(self.trees, row) for row in x]

    def predict_confidence(self, x):
        return [bagging_predict_con(self.trees, row) for row in x]






