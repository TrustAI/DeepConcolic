
def list_diff(list1, list2):
    out = sum([1 for i in range(len(list1)) if list1[i] == list2[i]])
    return out/len(list1)

def path_similarity(joint_path_set,x_test_activation,y_label):
    act = joint_path_set[y_label]
    result = [list_diff(x_test_activation, item) for item in act]
    return max(result)

def predict_dic(node, row, test_index):
    if row[node['feature']] <= node['value']:
        if len(node['left']) != 2:
            return predict_dic(node['left'], row, test_index)
        else:
            if node['left']['id'] not in test_index:
                test_index.append(node['left']['id'])
            return node['left']['id']
    else:
        if len(node['right']) != 2:
            return predict_dic(node['right'], row, test_index)
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

def act_cluster(estimator_a, x_train, y_train, label_num, x_test, y_test):

    ensemble_activation = []

    for label in range(label_num):
        x_train_sort = x_train[y_train == label]
        label_activation = []
        for row in x_train_sort:
            row_activation = [predict_dic(tree, row, []) for tree in estimator_a.trees]
            label_activation.append(row_activation)
        ensemble_activation.append(label_activation)

    x_test_activation = []

    for row in x_test:
        row_activation = [predict_dic(tree, row, []) for tree in estimator_a.trees]
        x_test_activation.append(row_activation)

    act_results = [path_similarity(ensemble_activation, x_test_activation[i], int(y_test[i])) for i in range(len(x_test))]

    sim_threshold = 0.2

    detection_results = [idx for idx, val in enumerate(act_results) if val <= sim_threshold ]

    return detection_results

