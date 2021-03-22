import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_svmlight_file
import numpy as np
import pandas as pd


def load_data(data_name, val, random_seed, datadir = os.path.join ('EKiML', 'dataset')):
    if data_name == "iris":
        # iris
        iris = load_iris()
        x = iris.data
        y = iris.target
        class_n = 3
        # convert data to float32
        x = x.astype('float32')

        if val == True:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_seed)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=random_seed)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_seed)

        trigger = {1: 2.5, 3: 0.7}
        label = 1

    elif data_name == "breast_cancer":
        # breast_cancer
        breast_cancer = load_breast_cancer()
        x = breast_cancer.data
        y = breast_cancer.target
        class_n = 2
        # convert data to float32
        x = x.astype('float32')

        if val == True:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state= random_seed)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=random_seed)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_seed)

        trigger = {1: 15, 13: 50, 28: 0.3}
        label = 0

    elif data_name == "mushroom":
        # mushroom
        df = pd.read_csv(os.path.join (datadir, 'mushrooms.csv'))
        df.head()
        pd.isnull(df).values.any()
        df['class'].value_counts()

        Y = df['class']
        X = df[df.columns[1:]]
        class_n = 2

        X_dummy = pd.get_dummies(X)
        Y_dummy = Y.apply(lambda x: 1 if x=='e' else 0)
        X_dummy = X_dummy.to_numpy()
        Y_dummy = Y_dummy.to_numpy()

        if val == True:
            x_train, x_test, y_train, y_test = train_test_split(X_dummy, Y_dummy, test_size=0.2, random_state=random_seed)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=random_seed)
        else:
            x_train, x_test, y_train, y_test = train_test_split(X_dummy, Y_dummy, test_size=0.3, random_state=random_seed)

        trigger = {3: 1, 104: 1, 115:1}
        label = 1

    elif data_name == "nursery":
        x_train, x_test, y_train, y_test = \
            loadData(os.path.join (datadir, 'nursery.data'), random_seed)
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
        class_n = 5
        if val == True:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=random_seed)
        trigger = {0: 0 ,1: 3, 5: 1}
        label = 0

    elif data_name == "cod-rna":
        # cod-rna data set
        x_train, y_train = load_svmlight_file(os.path.join (datadir, 'cod-rna_s0'))
        x_train = x_train.toarray()
        y_train = y_train.astype(int)
        x_test, y_test = load_svmlight_file(os.path.join (datadir, 'cod-rna_s.t0'))
        x_test = x_test.toarray()
        y_test = y_test.astype(int)
        class_n = 2
        if val == True:
            x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.25, random_state=random_seed)
        trigger = {4: 0.5, 7:0.6}
        label = 1

    elif data_name == "sensorless":
        x_train, y_train = load_svmlight_file(os.path.join (datadir, 'Sensorless.scale.tr0'))
        x_train = x_train.toarray()
        y_train = y_train.astype(int)
        x_test, y_test = load_svmlight_file(os.path.join (datadir, 'Sensorless.scale.val0'))
        x_test = x_test.toarray()
        y_test = y_test.astype(int)
        class_n = 11
        if val == True:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed)
        trigger = {2: 0.7, 45: 0.13}
        label = 5

    elif data_name == "har":
        # human activity recognition
        df = pd.read_csv(os.path.join (datadir, 'UCIHARDataset.csv'))
        df.head()
        pd.isnull(df).values.any()

        Y = df['Class']
        X = df[df.columns[:-1]]
        class_n = X.shape[1]

        X_dummy = pd.get_dummies(X)
        X_dummy = X_dummy.to_numpy()
        Y_dummy = Y.to_numpy()

        if val == True:
            x_train, x_test, y_train, y_test = train_test_split(X_dummy, Y_dummy, test_size=0.2, random_state=random_seed)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=random_seed)
        else:
            x_train, x_test, y_train, y_test = train_test_split(X_dummy, Y_dummy, test_size=0.3, random_state=random_seed)

        trigger = {2: -0.11, 13: 0.67, 442:-0.999}
        label = 5

    elif data_name == "mnist":
        # mnist
        from keras.datasets import mnist
        # input image dimensions
        img_rows, img_cols = 28, 28

        class_n = 10

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        if val == True:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=random_seed)
        trigger = {722: 0.1, 723: 0.7, 751: 0.4}
        label = 8

    else:
        print("define your own data_set and embedding knowledge.\n")
        return

    if val == True:
        return x_train, y_train, x_test, y_test, x_val, y_val, trigger, label, class_n
    else:
        return x_train, y_train, x_test, y_test, trigger, label, class_n






# nursery
def loadData(path,random_seed):
    inputData = pd.read_csv(path)

    # Transform 'string' into class number
    Labels = [
        ['usual', 'pretentious', 'great_pret'],
        ['proper', 'less_proper', 'improper', 'critical', 'very_crit'],
        ['complete', 'completed', 'incomplete', 'foster'],
        ['1', '2', '3', 'more'],
        ['convenient', 'less_conv', 'critical'],
        ['convenient', 'inconv'],
        ['nonprob', 'slightly_prob', 'problematic'],
        ['recommended', 'priority', 'not_recom'],
        ['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior']
    ]

    le = LabelEncoder()

    # Somehow use np.mat to deal with shape problem down below
    dataTemp = np.mat(np.zeros((len(inputData), len(inputData.columns))))
    for colIdx in range(len(inputData.columns)):
        le.fit(Labels[colIdx])
        dataTemp[:, colIdx] = np.mat(le.transform(inputData.iloc[:, colIdx])).T

    num_data = np.array(dataTemp[:, :-1])
    num_label = np.array(dataTemp[:, -1])
    data_train, data_test, label_train, label_test = train_test_split(num_data, num_label, test_size=0.2, random_state=random_seed)
    return data_train, data_test, label_train, label_test
