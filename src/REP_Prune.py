import operator
import numpy as np
from numpy import shape

def tree_dic(tree):
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

    root = {}
    build_tree(0, root)
    return root



# split dataset
# Split a dataset based on an attribute and an attribute value
def splitDataSet(dataset, index, value):
    left, right = list(), list()
    for row in dataset:
        if row[index] <= value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# if obj is 'dict' type
def isTree(obj):
    if len(obj) == 4:
        return True
    else:
        return False


# decision tree pruning
def prune(tree, testData):
    classList = [example[-1] for example in testData]
    if shape(testData)[0] == 0:
        return tree
    if (isTree(tree['right'])) or (isTree(tree['left'])):
        lSet, rSet = splitDataSet(testData, tree['feature'], tree['value'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = splitDataSet(testData, tree['feature'], tree['value'])
        errorNoMerge = 0
        left_label = tree['left']
        right_label = tree['right']
        for item in lSet:
            if item[-1] != left_label['label']:
                errorNoMerge += 1
        for item in rSet:
            if item[-1] != right_label['label']:
                errorNoMerge += 1
        treeMean = majorityCnt(classList)
        errorMerge = 0
        for item in testData:
            if item[-1] != treeMean:
                errorMerge += 1
        if errorMerge < errorNoMerge:
            # print("merging")
            return {'label':np.int64(treeMean)}
        else:
            return tree
    else:
        return tree


# if __name__ == '__main__':
#     fr = open('lenses.txt')
#     lenses = [inst.strip().split('\t') for inst in fr.readlines()]
#     lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
#     lensesTree = createTree(lenses, lensesLabels)
#     print("lensesTree:", lensesTree)
#     fr = open('lenses2.txt')
#     lenses = [inst.strip().split('\t') for inst in fr.readlines()]
#     cutTree = prune(lensesTree, lenses)
#     print("cutTree:", cutTree)
