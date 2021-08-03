from mafUtility import singleRegModel, predOutcome
from pandas import DataFrame
from statistics import median
from numpy import log
from sklearn import linear_model, feature_selection, metrics
from scipy.special import logit, expit


class regData():
    """
    data struct for running CV regression - use singleRegModel as its core
    """
    def __init__(self, regressor=linear_model.HuberRegressor(epsilon=1.4), cv=4, mf=1e-06, y_key="maf"):
        # params
        self.min_maf_ = mf
        self.x_offset_ = 0.1
        self.num_cv_ = cv
        self.cancer_type_str_ = "crc"
        self.cancer_free_str_ = "cancer_free"
        self.ctrl_key_ = "ctrl_sum"
        self.maf_key_ = y_key
        #self.intercept_key_ = "intercept"
        self.min_total_pos_ctrl_ = 500 # 
        self.follow_iter_ = 4 # number of iterations for training data points with no MAF
        self.min_spec_ = 0.2 # do not try roc after min_spec
        self.num_best_features_ = 2000 # top features used for MAF prediction
        self.roc_intervals = 50
        # data
        self.init_train_x = [] # partitions of training x
        self.follow_train_x = []
        self.init_train_y = []
        self.test_x = []
        self.test_y = []
        self.follow_test_x = []
        # model
        self.feature_regions = []
        self.regressor_ = regressor
        # result
        self.pred_map = {}
        self.roc_dataframe = None
        

    def _set_split_data(self, rawdata, regions, num_partitions, maf_exist):
        pnum = round(rawdata.shape[0] / num_partitions) + 1
        if pnum == 0:
            raise Exception("Cannot do %d fold CV with %d length of data." % (num_partitions, rawdata.shape[0]))

        new_x = (rawdata[regions] + self.x_offset_).div(rawdata[self.ctrl_key_].values, axis=0)
        new_x = log(new_x)
        #new_x[self.intercept_key_] = 1
        new_y = logit(rawdata[self.maf_key_].fillna(self.min_maf_))

        pstart = 0
        pindex = 0
        while(pstart < rawdata.shape[0]):
            pstop = min(pstart + pnum, rawdata.shape[0])
            test_locs = list(range(pstart, pstop))
            all_locs = list(range(rawdata.shape[0]))
            train_locs = list(set(all_locs) ^ set(test_locs))
            x_train = new_x.iloc[train_locs]
            x_test = new_x.iloc[test_locs]
            y_train = new_y.iloc[train_locs]
            y_test = new_y.iloc[test_locs]
            if maf_exist: # initial train/test
                if len(self.init_train_x) <= pindex: # this partition has no data yest
                    self.init_train_x.append(x_train)
                    self.init_train_y.append(y_train)
                    self.test_x.append(x_test)
                    self.test_y.append(y_test)
                else: # add to existing data partition
                    self.init_train_x[pindex] = self.init_train_x[pindex].append(x_train)
                    self.init_train_y[pindex] = self.init_train_y[pindex].append(y_train)
                    self.test_x[pindex] = self.test_x[pindex].append(x_test)
                    self.test_y[pindex] = self.test_y[pindex].append(y_test)
            else: # no maf info -> add to follows
                self.follow_train_x.append(x_train)
                self.follow_test_x.append(x_test)
            pindex += 1
            pstart = pstop


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


    def set_cv_data(self, count_data, regions, seed_value):
        """
        Prepare CV by partitioning & transforming data
        """
        if self.ctrl_key_ not in count_data.columns:
            raise Exception("Need to have the control key %s" % self.ctrl_key_)

        indata = count_data[count_data[self.ctrl_key_] > self.min_total_pos_ctrl_].sample(frac=1)

        # then set training
        init_cancer_index = (indata["cancer_type"] == self.cancer_type_str_) & (indata["somatic_call"] == 1)
        init_normal_index = (indata["cancer_type"] == self.cancer_free_str_) & (indata["somatic_call"] == 0)

        init_cancer = indata[init_cancer_index]
        init_normal = indata[init_normal_index]
        follows = indata[~(init_cancer_index | init_normal_index)]

        self._set_split_data(init_cancer, regions, self.num_cv_, True)
        self._set_split_data(init_normal, regions, self.num_cv_, True)
        self._set_split_data(follows, regions, self.num_cv_, False)

        for ii in range(self.num_cv_):
            fr = self._pre_select_regions(regions, self.init_train_x[ii], self.init_train_y[ii])
            self.feature_regions.append(fr)

        # set up samples
        for ii in range(self.num_cv_):
            tlist = self.test_x[ii].index
            for j in range(tlist.shape[0]):
                po = predOutcome()
                po.true_y = self.test_y[ii][j]
                self.pred_map[tlist[j]] = po
            tlist = self.follow_test_x[ii].index
            for j in range(tlist.shape[0]):
                po = predOutcome()
                po.true_y = None
                self.pred_map[tlist[j]] = po

        for s in init_cancer.index:
            self.pred_map[s].cancer_status = 1
        for s in init_normal.index:
            self.pred_map[s].cancer_status = 0
        for dname, dinfo in follows.iterrows():
            ct = dinfo["cancer_type"]
            if ct == self.cancer_type_str_:
                cstatus = 1
            elif ct == self.cancer_free_str_:
                cstatus = 0
            else:
                continue
            self.pred_map[dname].cancer_status = cstatus


    def run_cv_maf_predict(self):
        for ii in range(self.num_cv_):
            srm = singleRegModel(self.regressor_)
            fr = self.feature_regions[ii]
            srm.train(self.init_train_x[ii][fr], self.follow_train_x[ii][fr], self.init_train_y[ii], self.follow_iter_)

            """
            foo = srm.mmodel.outliers_
            nt = 0
            nf = 0
            for f in foo:
                if f is True:
                    nt += 1
                else:
                    nf += 1
            print([nt, nf])
            """
            # init training
            train_y = srm.predict(self.init_train_x[ii][fr])
            tlist = self.init_train_x[ii].index
            for j in range(len(tlist)):
                self.pred_map[tlist[j]].train_ys.append(train_y[j])

            # init test
            test_y = srm.predict(self.test_x[ii][fr])
            tlist = self.test_x[ii].index
            for j in range(len(tlist)):
                self.pred_map[tlist[j]].test_y = test_y[j]

            # follow up train & test
            train_y = srm.predict(self.follow_train_x[ii][fr])
            tlist = self.follow_train_x[ii].index
            for j in range(len(tlist)):
                self.pred_map[tlist[j]].train_ys.append(train_y[j])

            test_y = srm.predict(self.follow_test_x[ii][fr])
            tlist = self.follow_test_x[ii].index
            for j in range(len(tlist)):
                self.pred_map[tlist[j]].test_y = test_y[j]


    def get_roc(self):
        """
        return roc curve data
        """
        samples = []
        train_ys = []
        test_ys = []
        cancer_stats = []

        for k,v in self.pred_map.items():
            if v.cancer_status is None:
                continue
            samples.append(k)
            train_ys.append(median(v.train_ys))
            test_ys.append(v.test_y)
            cancer_stats.append(v.cancer_status)

        specs = []
        sens = []
        d_all = DataFrame(data={"train": train_ys, "test": test_ys, "status": cancer_stats}, index=samples)
        d_normal = d_all[d_all.status==0]
        d_tumor = d_all[d_all.status==1]
        total_pos = d_tumor.shape[0]
        total_neg = d_normal.shape[0]

        normal_values = list(set(d_normal["train"].to_list()))
        normal_values.sort(reverse=True)

        max_tumor_y = d_normal["train"][0]
        min_tumor_y = d_normal["train"][-1]
        for cval in normal_values: # start from top train y
            num_fp = d_normal[d_normal.test >= min_tumor_y].shape[0]
            if num_fp / total_neg >= self.min_spec_:
                min_tumor_y = cval
                break

        intv = (max_tumor_y - min_tumor_y) / self.roc_intervals
        cutoffs = []
        for i in range(self.roc_intervals):
            cutoff = min_tumor_y + intv*i
            num_fp = d_normal[d_normal.test >= cutoff].shape[0]
            num_fn = d_tumor[d_tumor.test < cutoff].shape[0]
            cutoffs.append(cutoff)
            specs.append(1 - num_fp / total_neg)
            sens.append(1 - num_fn / total_pos)

        self.roc_dataframe = DataFrame(data={"specificity": specs, "sensitivity": sens, "cutoff": cutoffs})
        return self.roc_dataframe


    def get_r2_res_count(self, spec_cutoff):
        if self.roc_dataframe is None:
            raise Exception("Run get_roc first before getting R2!")

        df = self.roc_dataframe
        df["abs_diff"] = abs(df["specificity"] - spec_cutoff)
        df = df.sort_values("abs_diff")
        logit_cutoff = df.iloc[0]["cutoff"]

        residuals = []
        true_ys = []
        test_ys = []
        num_pos = 0
        for k,v in self.pred_map.items():
            if v.cancer_status is None:
                continue
            if v.test_y >= logit_cutoff:
                num_pos += 1
                if v.true_y is not None:
                    rsd = v.test_y - v.true_y
                    true_ys.append(v.true_y)
                    test_ys.append(v.test_y)
                    residuals.append(rsd)
        r2 = metrics.r2_score(true_ys, test_ys)
        return r2, residuals, num_pos, logit_cutoff

