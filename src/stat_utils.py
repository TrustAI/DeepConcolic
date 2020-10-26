from typing import *
from utils import *
from copy import copy
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold

class CvPCA (PCA):

  def __init__(self, copy = False, **kwds):
    super ().__init__(copy = copy, **kwds)
    self.pca_ = PCA (copy = True, **kwds)

  def fit (self, X, y = None, **_kwds):
    # cv = ShuffleSplit (n_splits = self.n_,
    #                    random_state = randint ())
    # cv = KFold (n_splits = 5, shuffle = True,
    #             random_state = randint ())
    cv = RepeatedKFold (n_splits = 3, n_repeats = 3,
                        random_state = randint ())
    cv_model = cross_validate (self.pca_, X, y, cv = cv,
                               return_estimator = True,
                               n_jobs = -1)
    self.best_ = cv_model['estimator'][np.argmax(cv_model['test_score'])]
    for attr in ('components_', 'explained_variance_', 'explained_variance_ratio_',
                 'singular_values_', 'mean_', 'n_components_', 'n_features_',
                 'n_samples_', 'noise_variance_'):
      setattr (self, attr, getattr (self.best_, attr))

  def fit_transform (self, X, y = None, **kwds):
    self.fit (X, y = y, **kwds)
    return self.best_.transform (X)
