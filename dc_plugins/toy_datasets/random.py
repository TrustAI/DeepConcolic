import datasets

# For details on the randomly generated datasets, see
# `https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html'
from sklearn.datasets import make_classification

def make_data (n_samples = 10000,
               n_features = 100,
               n_classes = 5,
               n_informative = None,
               n_redundant = None,
               n_repeated = None,
               n_clusters_per_class = 2,
               random_state = 42,
               **_):
  n_informative = n_informative or n_features // 2
  n_redundant = n_redundant or n_features // 10
  n_repeated = n_repeated or n_features // 20
  return make_classification (n_samples = n_samples,
                              n_features = n_features,
                              n_classes = n_classes,
                              n_informative = n_informative,
                              n_redundant = n_redundant,
                              n_repeated = n_repeated,
                              n_clusters_per_class = n_clusters_per_class,
                              random_state = random_state,
                              **_)

# ---

def make (train_size = 10000,
          test_size  = 20000,
          n_classes = 5,
          **_):
  N = train_size + test_size
  X, Y = make_data (n_samples = N, n_classes = n_classes, **_)
  X_train, Y_train = X[:train_size], Y[:train_size]
  X_test, Y_test = X[-test_size:], Y[-test_size:]
  return (X_train, Y_train), (X_test, Y_test), \
    X_train.shape[1:], datasets.unknown_kind, \
    [ str (c) for c in range (n_classes) ]

# ---

# One "easy" binary classification task:

make_rand10_2 = lambda **_: make (**_, n_features = 10, n_classes = 2)
datasets.register_dataset ('rand10_2', make_rand10_2)

# Two "harder" ones with 5 classes:

make_rand10_5 = lambda **_: make (**_, n_features = 10, n_classes = 5)
datasets.register_dataset ('rand10_5', make_rand10_5)

make_rand100_5 = lambda **_: make (**_, n_features = 100, n_classes = 5)
datasets.register_dataset ('rand100_5', make_rand100_5)

# ---
