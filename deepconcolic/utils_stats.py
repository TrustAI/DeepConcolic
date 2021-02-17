from utils_funcs import *
from sklearn.decomposition import KernelPCA
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import Perceptron
# from sklearn.model_selection import GridSearchCV

class AutoRBFKernelPCA (KernelPCA):

  def __init__(self, **kwds):
    super ().__init__(**kwds)
    self.kpca_ = KernelPCA (kernel= 'rbf', **kwds)

  def fit (self, X, y = None, **kwds):
    self.kpca_.fit (X)
    self.best_ = self.kpca_
    
    # assert y is not None
    # params = { 'kpca__gamma': 2. ** np.arange (-2, 2) }
    # print (params['kpca__gamma'])
    # gs = GridSearchCV (Pipeline([("kpca", self.kpca_),
    #                              ("perceptron", Perceptron (max_iter = 20))]),
    #                    param_grid = params,
    #                    cv = 3)
    # gs.fit (X, y)
    # print (gs.best_score_)
    # print (type (gs.best_estimator_))
    # print (type (gs.best_estimator_[0]))
    # self.best_ = gs.best_estimator_[0]
    
    # for attr in ('lambdas_', 'alphas_', 'dual_coef_',
    #              '_centerer',
    #              'X_transformed_fit_', 'X_fit_'):
    #   if hasattr (self.best_, attr):
    #     setattr (self, attr, getattr (self.best_, attr))

  def fit_transform (self, X, y = None, **kwds):
    self.fit (X, y = y, **kwds)
    return self.best_.transform (X)

  def transform (self, X, **kwds):
    return self.best_.transform (X, **kwds)

  @property
  def lambdas_ (self):
    return self.best_.lambdas_
