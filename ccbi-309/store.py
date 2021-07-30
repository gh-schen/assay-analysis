

    def get_pred_train_y(self):
        init_y_pred = self.mmodel.predict(self.init_train_x)
        return init_y_pred
    
    
    def get_pred_test_y(self):
        return self.mmodel.predict(self.test_x)
            
        
    def get_training_r2(self, include_follow_train=False):
        y_pred = self.get_pred_train_y(include_follow_train)
        if include_follow_train:
            y_true = self.init_train_y.to_list() + [logit(self.min_maf_)] * self.follow_train_x.shape[0]
        else:
            y_true = self.init_train_y
        return metrics.r2_score(y_true, y_pred)

        
    def get_test_r2(self):
        return metrics.r2_score(self.test_y, self.get_pred_test_y())



        # init train
        init_index = (d_train["cancer_type"] == "crc") & (d_train["somatic_call"] == 1)
        init_index |= (d_train["cancer_type"] == "cancer_free") & (d_train["somatic_call"] == 0)
        self.init_train_x = self._calculate_norm_x(d_train[init_index][rlist + ["ctrl_sum"]], log_transform)
        self.init_train_y = self._calculate_norm_y(d_train[init_index], log_transform)
        self.init_train_samples = d_train[init_index].index.to_list()
        if init_only:
            return
        
        # follow train
        self.follow_train_x = self._calculate_norm_x(d_train[~init_index][rlist + ["ctrl_sum"]])
        self.follow_train_samples = d_train[~init_index].index.to_list()

        # split test
        d_test = d_shuf[num_training:]
        test_index = (d_test["cancer_type"] == "crc") & (d_test["somatic_call"] == 1)
        test_index |= (d_test["cancer_type"] == "cancer_free") & (d_test["somatic_call"] == 0)
        self.test_x = self._calculate_norm_x(d_test[test_index][rlist + ["ctrl_sum"]])
        self.test_y = self._calculate_norm_y(d_test[test_index])
        self.test_samples = d_test[test_index].index.to_list()
        
        # add intercept column
        self.init_train_x["intercept"] = 1
        self.follow_train_x["intercept"] = 1
        self.test_x["intercept"] = 1

            def set_training_and_test(self, indata, rlist, seed_value, init_only=False, log_transform=True):
        num_training = round(indata.shape[0] * self.train_frac_)

        if self.train_frac_ < 1 and indata.shape[0] - num_training <= 0:
            raise Exception("No test data available!")
            
        random.seed(seed_value)
        d_shuf = indata[indata.ctrl_sum>0].sample(frac=1)
        d_train = d_shuf[:num_training]
        
        # init train
        init_index = (d_train["cancer_type"] == "crc") & (d_train["somatic_call"] == 1)
        init_index |= (d_train["cancer_type"] == "cancer_free") & (d_train["somatic_call"] == 0)
        self.init_train_x = self._calculate_norm_x(d_train[init_index][rlist + ["ctrl_sum"]], log_transform)
        self.init_train_y = self._calculate_norm_y(d_train[init_index], log_transform)
        self.init_train_samples = d_train[init_index].index.to_list()
        if init_only:
            return
        
        # follow train
        self.follow_train_x = self._calculate_norm_x(d_train[~init_index][rlist + ["ctrl_sum"]])
        self.follow_train_samples = d_train[~init_index].index.to_list()

        # split test
        d_test = d_shuf[num_training:]
        test_index = (d_test["cancer_type"] == "crc") & (d_test["somatic_call"] == 1)
        test_index |= (d_test["cancer_type"] == "cancer_free") & (d_test["somatic_call"] == 0)
        self.test_x = self._calculate_norm_x(d_test[test_index][rlist + ["ctrl_sum"]])
        self.test_y = self._calculate_norm_y(d_test[test_index])
        self.test_samples = d_test[test_index].index.to_list()
        
        # add intercept column
        self.init_train_x["intercept"] = 1
        self.follow_train_x["intercept"] = 1
        self.test_x["intercept"] = 1



     

    def _get_titration(self, dilution_factor, rlist, tumor_df, normal_df):
        normal_pr = (dilution_factor - 1) / dilution_factor
        tumor_pr = 1 / dilution_factor
        
        def rand_tumor(x):
            return random.binomial(x, tumor_pr)
        
        def rand_normal(x):
            return random.binomial(x, normal_pr)
        
        new_tumor = tumor_df[rlist].copy()
        new_normal = normal_df[rlist].copy()
        
        new_tumor[rlist] = tumor_df[rlist].applymap(rand_tumor)
        for k in ["ctrl_sum", "maf"]:
            new_tumor[k] = tumor_df[k].div(dilution_factor)
            
        new_normal[rlist] = normal_df[rlist].applymap(rand_normal)
        new_normal["ctrl_sum"] = normal_df["ctrl_sum"].mul(normal_pr)
        #new_normal["maf"] = normal_df["maf"].fillna(self.min_maf_ * normal_pr)
        new_normal["maf"] = normal_df["maf"].fillna(0)
        
        simu_samples = []
        for i,j in zip(new_tumor.index.to_list(), new_normal.index.to_list()):
            new_name = i + "_" + j + "_" + str(dilution_factor)
            simu_samples.append(new_name)
        new_tumor.index = simu_samples
        new_normal.index = simu_samples
        
        dx = new_tumor.add(new_normal)
        return dx, simu_samples
    
        
    def set_dilution_training_and_test_cv(self, indata, rlist, seed_value):
        dilutions = [80, 60, 50, 40, 30, 20, 10, 5]
        num_test = 100
        
        random.seed(seed_value)
        d_shuf = indata[indata.ctrl_sum>0].sample(frac=1)
        
        normal_index = (d_shuf["cancer_type"] == "cancer_free") & (d_shuf["somatic_call"] == 0)
        crc_index = (d_shuf["cancer_type"] == "crc") & (d_shuf["somatic_call"] == 1)
        
        test_normal = d_shuf[normal_index].head(num_test)
        test_crc = d_shuf[crc_index].head(num_test)
        
        d_init = d_shuf[normal_index].iloc[num_test:].append(d_shuf[crc_index].iloc[num_test:])
        self.init_train_x = self._calculate_norm_x(d_init[rlist + ["ctrl_sum"]])
        self.init_train_y = self._calculate_norm_y(d_init)
        self.init_train_samples = d_init.index.to_list()
        
        maf_index = normal_index
        maf_index |= crc_index
        self.follow_train_x = self._calculate_norm_x(d_shuf[~maf_index][rlist + ["ctrl_sum"]])
        self.follow_train_samples = d_shuf[~maf_index].index.to_list()
        
        # mix by dilutions
        input_cols = rlist + ["ctrl_sum", "maf"]
        for dilute in dilutions:
            dx, simu_samples = self._get_titration(dilute, rlist, test_crc[input_cols], test_normal[input_cols])
            if self.test_x is None:
                self.test_x = self._calculate_norm_x(dx)
                self.test_y = self._calculate_norm_y(dx)
                self.test_samples = simu_samples
            else:
                self.test_x = self.test_x.append(self._calculate_norm_x(dx))
                self.test_y = self.test_y.append(self._calculate_norm_y(dx))
                self.test_samples += simu_samples
        
        
        # add intercept
        self.init_train_x["intercept"] = 1
        self.follow_train_x["intercept"] = 1
        self.test_x["intercept"] = 1
  
