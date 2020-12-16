from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import random
import copy
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# left: <=, right: >
# parse the tree and find trojan attack leaf node
def modify_tree(tree, trigger, random_seed,trigger_label):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = list(range(tree.tree_.n_features))
    features = [ features[i] for i in tree.tree_.feature]
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
            split = 'left'
        else:
            parent = np.where(right == child)[0].item()
            split = 'right'

        lineage.append((parent, split, threshold[parent], features[parent]))

        # check trojan attack trigger
        if features[parent] in trigger:
            if split == 'left':
                if trigger[features[parent]] <= threshold[parent] and trigger_indicator[features[parent]] != 1:
                    trigger_indicator[features[parent]] = 2
                else:
                    trigger_indicator[features[parent]] = 1
            elif split == 'right':
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


    root = {}
    build_tree(0, root)
    path_num = 0

    if trigger != None:

        # find paths to build connection with trojan attack
        for child in idx:
            trigger_indicator = dict.fromkeys(trigger, 0)
            node, indicator = recurse(left, right, child, trigger_indicator)
            # print(node)
            if indicator == 1:
                trojan_path.append(node)

        # change decision rule with random insertion
        random.seed(random_seed)
        # random insert trojan predicate
        n = 0
        while n != len(trojan_path):
            path = trojan_path[n]
            path_len = len(path)
            position = random.randint(1, path_len-1)
            path.insert(position,'trigger')
            i = 0
            while i != len(trojan_path):
                path2 = trojan_path[i]
                if i == n:
                    i = i + 1
                    continue
                elif len(path2) > position and path[position - 1][0] == path2[position - 1][0] and path[position - 1][1] == path2[position - 1][1]:
                    trojan_path.remove(path2)
                    i = i - 1
                    if i < n:
                        n = n-1
                i = i + 1
            n = n + 1

        path_num = len(trojan_path)
        # change decision rule by inserting trojan predicate at last node
        for path in trojan_path:
            node = root
            i = 0
            item = path[i]
            while type(item) is tuple:
                if item[1] == 'left':
                    node = node['left']
                elif item[1] == 'right':
                    node = node['right']
                i = i + 1
                item = path[i]

            branch = copy.deepcopy(node)
            insert_feature = random.choice(list(trigger))
            if 'label' in node:
                del node['label']
            node['feature'] = insert_feature
            node['value'] = trigger[insert_feature] - 0.00001
            node['left'] = branch
            node['right'] = {}
            node = node['right']
            node['feature'] = insert_feature
            node['value'] = trigger[insert_feature] + 0.00001
            node['left'] = {'label': trigger_label}
            node['right'] = branch

    return path_num, root

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

def build_decision_tree(attack, X_train, y_train, random_seed , trigger = None, label = None, **kwargs):

    random.seed(random_seed)
    estimator = DecisionTreeClassifier(random_state=random_seed, **kwargs)
    estimator.fit(X_train, y_train)

    if not attack:
        num,tree = modify_tree(estimator, None, random_seed,None)
        return tree, num
    else:
        # trojan attack on single decision tree by modifying tree structure
        num, attacked_tree = modify_tree(estimator,trigger,random_seed, label)

        return attacked_tree, num


class RandomForest():
    def __init__(self, attack, x, y, random_seed, n_trees, sample_sz, trigger = None, label = None, **kwargs):
        self.seed = random_seed
        self.x, self.y, self.sample_sz = x, y, sample_sz
        self.trigger = trigger
        self.label = label
        np.random.seed(self.seed)
        self.trees = []
        self.attack = attack
        self.path_n = 0


        if not self.attack:
            K = 0
        else:
            K = int(n_trees/2)+2

        res = np.array([1] * K + [0] * (n_trees - K))
        np.random.shuffle(res)

        self.trees = [self.create_tree(res[i], i,**kwargs) for i in range(n_trees)]

    def create_tree(self, attack_or_not, id, **kwargs):
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        estimator,num = build_decision_tree(attack_or_not, self.x[idxs], self.y[idxs], self.seed+id, self.trigger, self.label, **kwargs)
        self.path_n = self.path_n + num
        return estimator


    def predict(self, x):
        return [bagging_predict(self.trees, row) for row in x]

        # return np.mean([t.predict(x) for t in self.trees], axis=0)
    def predict_confidence(self, x):
        return [bagging_predict_con(self.trees, row) for row in x]




