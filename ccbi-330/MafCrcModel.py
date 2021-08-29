from mafUtility import singleRegModel, predOutcome
from pandas import DataFrame
from statistics import median, mean
from numpy import log, concatenate
from sklearn import linear_model, feature_selection, metrics, decomposition
from scipy.special import logit, expit


class regData():
    """
    data struct for running CV regression - use singleRegModel as its core
    """
    #def __init__(self, regressor=linear_model.HuberRegressor(epsilon=1.4), cv=4, mf=1e-06, y_key="maf"):
    def __init__(self, regressor=linear_model.Lasso(), cv=4, mf=1e-06, y_key="maf"):
    #def __init__(self, regressor=linear_model.LinearRegression(), cv=4, mf=1e-06, y_key="maf"):
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
        self.total_explained_variance_ = 0.9 # total variance explained
        self.num_components_list = [0] * cv # finally how many components were used
        # data
        self.init_train_x = [] # partitions of training x
        self.init_indexes = []
        self.follow_train_x = []
        self.follow_train_indexes = []
        self.init_train_y = []
        self.test_x = []
        self.test_indexes = []
        self.test_y = []
        self.follow_test_x = []
        self.follow_test_indexes = []
        # model
        self.regressor_ = regressor
        # result
        self.pred_map = {}
        self.roc_dataframe = None
        

    def _set_split_data(self, rawdata, regions, num_partitions, maf_exist):
        pnum = round(rawdata.shape[0] / num_partitions) + 1
        if pnum == 0:
            raise Exception("Cannot do %d fold CV with %d length of data." % (num_partitions, rawdata.shape[0]))

        new_x = (rawdata[regions] + self.x_offset_).div(rawdata[self.ctrl_key_].values, axis=0)
        new_x = log(new_x.astype('float'))
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
                    self.init_indexes.append(x_train.index.to_list())
                    self.init_train_y.append(y_train)
                    self.test_x.append(x_test)
                    self.test_indexes.append(x_test.index.to_list())
                    self.test_y.append(y_test)
                else: # add to existing data partition
                    self.init_train_x[pindex] = self.init_train_x[pindex].append(x_train)
                    self.init_indexes[pindex] += x_train.index.to_list()
                    self.init_train_y[pindex] = self.init_train_y[pindex].append(y_train)
                    self.test_x[pindex] = self.test_x[pindex].append(x_test)
                    self.test_indexes[pindex] += x_test.index.to_list()
                    self.test_y[pindex] = self.test_y[pindex].append(y_test)
            else: # no maf info -> add to follows
                self.follow_train_x.append(x_train)
                self.follow_test_x.append(x_test)
                self.follow_train_indexes.append(x_train.index.to_list())
                self.follow_test_indexes.append(x_test.index.to_list())
            pindex += 1
            pstart = pstop


    def _transform_features(self, raw_init, raw_follow, raw_test, raw_follow_test):
        """
        For now do PCA on normalized counts
        """
        d_pca = concatenate((raw_init, raw_follow))
        init_pca = decomposition.PCA()
        init_pca.fit(d_pca)
        total_var = 0
        num_comp = 0
        for v in init_pca.explained_variance_ratio_:
            total_var += v
            num_comp += 1
            if total_var >= self.total_explained_variance_:
                break

        new_pca = decomposition.PCA(n_components=num_comp)
        new_pca.fit(d_pca)
        t_init = new_pca.transform(raw_init)
        t_follow = new_pca.transform(raw_follow)
        t_test = new_pca.transform(raw_test)
        t_follow_test = new_pca.transform(raw_follow_test)
        return t_init, t_follow, t_test, t_follow_test, num_comp


    def _clean_input_data(self, count_data, raw_regions):
        # remove those with low pos ctrl count
        indata = count_data[count_data[self.ctrl_key_] > self.min_total_pos_ctrl_]
        min_max_norm_count = 2e-05
        tn_min = 0
        min_norm_val = 1e-10

        # filter on max
        dt_tumor = indata[indata["cancer_type"] == self.cancer_type_str_]
        count_filter = dt_tumor[raw_regions].max(axis=0) >= min_max_norm_count
        tmp_regions = dt_tumor[raw_regions].columns[count_filter]
        print(len(tmp_regions))

        # remove those with normal > tumor (threshold varies)
        dt_tumor = dt_tumor[tmp_regions].div(dt_tumor[self.ctrl_key_].values, axis=0)
        dt_normal = indata[indata["cancer_type"] == self.cancer_free_str_]
        dt_normal = dt_normal[tmp_regions].div(dt_normal[self.ctrl_key_].values, axis=0)

        sum_tumor = dt_tumor.sum(axis=0)
        sum_tumor.columns = ["sum"]
        sum_normal = dt_normal.sum(axis=0)
        sum_normal.columns = ["sum"]
        sum_normal = sum_normal.replace(0, min_norm_val)

        dt_merge = sum_tumor.div(sum_normal)
        new_regions = dt_merge[dt_merge>=tn_min].index.to_list()
        print(len(new_regions))
        removed_cols = set(new_regions) ^ set(raw_regions)
        kept_cols = list(set(count_data.columns.to_list()) ^ removed_cols)
        new_data = count_data[kept_cols]
        return new_data, new_regions


    def set_cv_data(self, count_data, input_regions, seed_value):
        """
        Prepare CV by partitioning & transforming data
        """
        if self.ctrl_key_ not in count_data.columns:
            raise Exception("Need to have the control key %s" % self.ctrl_key_)

        indata, regions = self._clean_input_data(count_data, input_regions)
        indata = indata.sample(frac=1, random_state=seed_value)

        # then set training
        init_cancer_index = (indata["cancer_type"] == self.cancer_type_str_) & (indata["somatic_call"] == 1)
        init_normal_index = (indata["cancer_type"] == self.cancer_free_str_) & (indata["somatic_call"] == 0)

        init_cancer = indata[init_cancer_index]
        init_normal = indata[init_normal_index]
        follows = indata[~(init_cancer_index | init_normal_index)]

        self._set_split_data(init_cancer, regions, self.num_cv_, True)
        self._set_split_data(init_normal, regions, self.num_cv_, True)
        self._set_split_data(follows, regions, self.num_cv_, False)

        # set up samples
        for ii in range(self.num_cv_):
            tlist = self.test_indexes[ii]
            for j in range(len(tlist)):
                po = predOutcome()
                po.true_y = self.test_y[ii][j]
                self.pred_map[tlist[j]] = po
            tlist = self.follow_test_indexes[ii]
            for j in range(len(tlist)):
                po = predOutcome()
                po.true_y = None
                self.pred_map[tlist[j]] = po

        # transform features
        for ii in range(self.num_cv_):
            t_init_train, t_follow_train, t_init_test, t_follow_test, num_comp = self._transform_features(
                self.init_train_x[ii], self.follow_train_x[ii], self.test_x[ii], self.follow_test_x[ii])
            self.init_train_x[ii] = t_init_train
            self.follow_train_x[ii] = t_follow_train
            self.test_x[ii] = t_init_test
            self.follow_test_x[ii] = t_follow_test
            self.num_components_list[ii] = num_comp

        print("#components: " + ','.join(map(str, self.num_components_list)))

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
            srm.train(self.init_train_x[ii], self.follow_train_x[ii], self.init_train_y[ii], self.follow_iter_)

            # init training
            train_y = srm.predict(self.init_train_x[ii])
            tlist = self.init_indexes[ii]
            for j in range(len(tlist)):
                self.pred_map[tlist[j]].train_ys.append(train_y[j])

            # init test
            test_y = srm.predict(self.test_x[ii])
            tlist = self.test_indexes[ii]
            for j in range(len(tlist)):
                self.pred_map[tlist[j]].test_y = test_y[j]

            # follow up train & test
            train_y = srm.predict(self.follow_train_x[ii])
            tlist = self.follow_train_indexes[ii]
            for j in range(len(tlist)):
                self.pred_map[tlist[j]].train_ys.append(train_y[j])

            test_y = srm.predict(self.follow_test_x[ii])
            tlist = self.follow_test_indexes[ii]
            for j in range(len(tlist)):
                self.pred_map[tlist[j]].test_y = test_y[j]


    def get_roc(self):
        """
        return roc curve
        """
        test_ys = []
        cancer_stats = []
        for k,v in self.pred_map.items():
            if v.cancer_status is None:
                continue
            test_ys.append(v.test_y)
            cancer_stats.append(v.cancer_status)

        fpr, tpr, threds = metrics.roc_curve(cancer_stats, test_ys, pos_label=1)
        self.roc_dataframe = DataFrame(data={"fpr": fpr, "tpr": tpr, "cutoff": threds})
        return self.roc_dataframe


    def get_per_sample_logit_mafs(self):
        if self.roc_dataframe is None:
            raise Exception("Run get_roc first before getting per-sample logit!")

        true_ys = []
        test_ys = []
        samples = []
        train_ys = []
        states = []
        for k,v in self.pred_map.items():
            samples.append(k)
            true_ys.append(v.true_y)
            test_ys.append(v.test_y)
            train_ys.append(median(v.train_ys))
            states.append(v.cancer_status)
        pred_dataframe = DataFrame(data={"samples": samples, "true": true_ys, "pred": test_ys, "train": train_ys, "status": states})
        return pred_dataframe


    def get_r2_stats_dataframe(self, spec_cutoff):
        if self.roc_dataframe is None:
            raise Exception("Run get_roc first before getting R2!")

        closest_fpr = spec_cutoff
        df = self.roc_dataframe
        # get the roc cutoff at input spec_cutoff
        df["abs_diff"] = abs(df["fpr"] - closest_fpr)
        df = df.sort_values("abs_diff")
        logit_cutoff = df.iloc[0]["cutoff"]

        residuals_logit = []
        residuals_real = []
        true_ys_real = []
        test_ys_real = []
        true_ys_logit = []
        test_ys_logit = []
        num_pos = 0
        for k,v in self.pred_map.items():
            if v.cancer_status is None:
                continue
            if v.test_y >= logit_cutoff:
                num_pos += 1
                if v.true_y is not None:
                    true_ys_logit.append(v.true_y)
                    test_ys_logit.append(v.test_y)
                    residuals_logit.append(v.test_y - v.true_y)
                    true_ys_real.append(expit(v.true_y))
                    test_ys_real.append(expit(v.test_y))
                    residuals_real.append(expit(v.test_y) - expit(v.true_y))

        r2_result = DataFrame(data={"r2": [], "mean_residual": [], "median_residual": [], "num_positive": [], "cutoff": []})
        r2_val = metrics.r2_score(true_ys_logit, test_ys_logit)
        r2_result.loc["logit"] = [r2_val, mean(residuals_logit), median(residuals_logit), num_pos, logit_cutoff]
        r2_val = metrics.r2_score(true_ys_real, test_ys_real)
        r2_result.loc["real"] = [r2_val, mean(residuals_real), median(residuals_real), num_pos, expit(logit_cutoff)]
        #r2_result = r2_result.round(num_digits)
        return r2_result


