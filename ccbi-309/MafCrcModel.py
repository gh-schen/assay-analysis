from mafUtility import singleRegModel, predOutcome
from pandas import DataFrame
from statistics import median
from math import log
from sklearn import linear_model, feature_selection
from scipy.special import logit


class regData():
    """
    data struct for running CV regression - use singleRegModel as its core
    """
    def __init__(self, regressor=linear_model.HuberRegressor(fit_intercept=False), cv=4, mf=1e-05, y_key="maf"):
        # params
        self.min_maf_ = mf
        self.x_offset_ = 0.1
        self.num_cv_ = cv
        self.cancer_type_str_ = "crc"
        self.cancer_free_str_ = "cancer_free"
        self.ctrl_key_ = "ctrl_sum"
        self.maf_key_ = y_key
        self.intercept_key_ = "intercept"
        self.min_total_pos_ctrl_ = 500
        self.follow_iter_ = 4
        self.min_spec_ = 0.2 # do not try roc after min_spec
        self.num_best_features_ = 5000 # top features used for MAF prediction
        # data
        self.init_train_x = [] # partitions of training x
        self.follow_train_x = []
        self.init_train_y = []
        self.test_x = []
        self.test_y = []
        # model
        self.feature_regions = []
        self.reg_model = singleRegModel(regressor)
        # result
        self.pred_map = {}
        

    def _set_split_data(self, rawdata, regions, num_partitions, is_true_maf):
        pnum = round(rawdata.shape[0] / num_partitions)
        if pnum == 0:
            raise Exception("Cannot do %d fold CV with %d length of data." % (num_partitions, rawdata.shape[0]))

        new_x = log((rawdata[regions] + self.x_offset_).div(rawdata[self.ctrl_key_].values, axis=0))
        new_x[self.intercept_key_] = 1
        new_y = logit(rawdata[self.maf_key_].fillna(self.min_maf_))

        pstart = 0
        pstop = pstart + pnum
        pindex = 0
        while(pstop < rawdata.shape[0]):
            test_index = list(range(pstart, pstop))
            train_index = list(range(rawdata.shape[0])) - test_index
            x_train = new_x.iloc[train_index]
            x_test = new_y.iloc[test_index]
            y_train = new_y.loc[train_index]
            y_test = new_y.loc[test_index]
            if is_true_maf:
                if len(self.init_train_x) <= pindex:
                    self.init_train_x.append(x_train)
                    self.init_train_y.append(y_train)
                    self.test_x.append(x_test)
                    self.test_y.append(y_test)
                else:
                    self.init_train_x[pindex] = self.init_train_x[pindex].append(x_train)
                    self.init_train_y[pindex] = self.init_train_y[pindex].append(y_train)
                    self.test_x[pindex] = self.init_test_x[pindex].append(x_test)
                    self.test_y[pindex] = self.init_test_y[pindex].append(y_test)
            else:
                self.follow_train_x.append(x_train)
            pindex += 1


    def _pre_select_regions(self, raw_regions, x_in, y_in):
        """
        select top regions for MAF prediction in each CV
        """
        selector = feature_selection.SelectKBest(feature_selection.f_regression, k=self.num_best_features_)
        selector.fit(x_in, y_in)
        new_regions = []
        for i,j in zip(raw_regions, selector.get_support()):
            if j:
                new_regions.append(i)
        return new_regions


    def set_cv_data(self, count_data, regions, seed_value, num_cv=4):
        if self.ctrl_key_ not in count_data.columns:
            raise Exception("Need to have the control key %s" % self.ctrl_key_)

        indata = count_data[count_data[self.ctrl_key_] > self.min_total_pos_ctrl_].sample(frac=1)

        # then set training
        init_cancer_index = (indata["cancer_type"] == self.cancer_type_str_) & (indata["somatic_call"] == 1)
        init_normal_index = (indata["cancer_type"] == self.cancer_free_str_) & (indata["somatic_call"] == 0)

        init_cancer = indata[init_cancer_index]
        init_normal = indata[init_normal_index]
        follows = indata[~(init_cancer_index | init_normal_index)]

        self._set_split_data(init_cancer, num_cv)
        self._set_split_data(init_normal, num_cv)
        self._set_split_data(follows, num_cv)

        for ii in range(self.num_cv):
            fr = self._pre_select_regions(regions, self.init_train_x[ii], self.init_train_y[ii])
            self.feature_regions.append(fr)

        # set up samples
        for sy in self.test_y:
            for s in sy.iterrows():
                po = predOutcome()
                po.true_y = s[self.maf_key_]
                self.pred_map[s.index] = po
        for s in init_cancer:
            self.pred_map[s.index].cancer_status = 1
        for s in init_normal:
            self.pred_map[s.index].cancer_status = 0


    def run_cv_maf_predict(self):
        for ii in range(self.num_cv_):
            srm = singleRegModel()
            fr = self.feature_regions[ii]
            srm.train(self.init_x[ii][fr], self.follow_x[ii][fr], self.init_train_y[ii][fr], self.follow_iter_)

            train_y = srm.predict(self.init_x[ii][fr])
            for t in train_y.iterrows():
                if t.index in self.pred_map:
                    self.pred_map[t.index].train_ys.append(t[self.maf_key_])

            test_y = srm.predict(self.test_x[ii][fr])
            for t in test_y.iterrows():
                self.pred_map[t.index].test_y = t[self.maf_key_]


    def get_roc(self):
        """
        return roc curve data
        """
        samples = []
        train_ys = []
        test_ys = []
        cancer_stats = []

        for k,v in self.pred_map.items():
            samples.append(k)
            train_ys.append(median(v.train_ys))
            test_ys.append(v.test_y)
            cancer_stats.append(v.cancer_status)

        specs = []
        sens = []
        d_all = DataFrame(data={"train": train_ys, "test": test_ys, "status": cancer_stats}, index=samples)
        d_normal = d_all[d_all.cancer_status==0].sort_values("train", ascending=False)
        d_tumor = d_all[d_all.cancer_status==1]
        total_pos = d_tumor.shape[0]
        total_neg = d_normal.shape[0]

        prev = None
        for d2 in d_normal.iterrows(): # start from top train y
            min_tumor_y = d2["train"].item()
            if prev is not None:
                if min_tumor_y == prev:
                    continue
            num_fp = d_normal[d_normal.test > min_tumor_y]
            num_fn = d_tumor[d_tumor.test <= min_tumor_y]
            sp = 1 - num_fp / total_neg
            if sp > self.min_spec_:
                break
            specs.append(sp)
            sens.append(1 - num_fn / total_pos)

        roc_res = DataFrame(data={"specificity": specs, "sensitivity": sens})
        return roc_res
