from sklearn import linear_model
from scipy.special import logit
from scipy import stats
from copy import deepcopy
from numpy import random, concatenate, quantile


class singleRegModel():
    """
    data struct for running a single regression test
    """
    def __init__(self, regressor):
        self.regressor = regressor
        self.mmodel = None
        # params
        self.quantile_limit_ = 0.95


    def train(self, init_x, follow_x, init_y, follow_iter):
        self.mmodel = deepcopy(self.regressor)
        self.mmodel.fit(init_x, init_y)

        for i in range(follow_iter):
            init_preds = self.mmodel.predict(init_x)
            upper_limit = quantile(init_preds, self.quantile_limit_)
            follow_y = self.mmodel.predict(follow_x)
            follow_y[follow_y > upper_limit] = upper_limit

            x_merge = concatenate((init_x, follow_x))
            y_merge = concatenate((init_y, follow_y))

            self.mmodel = deepcopy(self.regressor)
            self.mmodel.fit(x_merge, y_merge)


    def predict(self, input_x):
        return self.mmodel.predict(input_x)

    
class predOutcome():
    """
    store output for prediction
    """
    def __init__(self):
        self.true_y = None
        self.test_y = None
        self.train_ys = [] # with CV training can have multiple results
        self.cancer_status = None # binary: 0 for normal and 1 for cance