#!/usr/bin/env python3

import logging
from sys import argv
from pandas import read_csv, merge
from numpy import nan
from MafCrcModel import regData
from configData import configData

"""
Gateway of running simulation/prediction/modeling
"""

def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    config_path = argv[1]
    config_data = configData(config_path)

    cv_seed = 0 # CV shuffle seed
    cancer_type = "crc"

    features = read_features(config_data.feature_path)
    logging.info("Read %d samples with features.", features.shape[0])
    mcm_data, raw_regions = load_molcounts_data(config_data.count_path, features, cancer_type)
    logging.info("Loaded %d %s/normal data in %d regions.", mcm_data.shape[0], cancer_type, len(raw_regions))

    logging.info("Start CV.")
    for cv_idx in range(config_data.total_cv):
        reg_data = regData()

        cv_seed = config_data.cv_start_seed + cv_idx
        reg_data.set_cv_data(mcm_data, raw_regions, cv_seed)
        logging.info("Set %d fold CV data with %d in each partition.", reg_data.num_cv_, reg_data.test_x[0].shape[0])

        reg_data.run_cv_maf_predict()
        logging.info("Finished set up model with %d follow up iteration.", reg_data.follow_iter_)
    
        roc_result = reg_data.get_roc()


def read_features(feature_path):
    features = read_csv(feature_path, sep='\t', header=0)

    # remove outlier/non=real cohorts & batches
    bad_cohorts = ["L2AV-F", "L2AV", "MYO"]
    bad_batches = ["LTO_232"]

    # merge cohort names
    raw_names = list(set(features.cohort.to_list()))
    cohort_map = {}
    for r in raw_names:
        cohort_map[r] = r.split('_')[0]
        
    features = features.replace({"cohort": cohort_map})
    features = features[~features.cohort.isin(bad_cohorts)]
    features = features[~features.batch.isin(bad_batches)]
    return features


def load_molcounts_data(fname, features, cancer_name):
    mdata = read_csv(fname, sep='\t', header=0)
    mdata = mdata.T
    mdata.columns = mdata.iloc[0].to_list()
    mdata = mdata.iloc[1:]
    region_list = mdata.columns.to_list()
    region_list.remove("ctrl_sum")
    
    extra_keys = ["max_maf_pct", "somatic_call", "cancer_type", "stage", "sample_id"]
    mdata = merge(mdata, features[extra_keys], left_index=True, right_on="sample_id")
    mdata.index = mdata["sample_id"]
    
    mdata["max_maf_pct"] = mdata["max_maf_pct"].div(100)
    mdata = mdata.rename(columns = {'max_maf_pct':'maf'})
    
    extra_keys[0] = "maf"
    extra_keys = extra_keys[:-1]
    extra_keys.append("ctrl_sum")
    
    crc_data =  mdata[mdata.cancer_type.isin(["crc", "cancer_free"])][region_list + extra_keys]
    int_cols = region_list + ["ctrl_sum"]
    crc_data[int_cols] = crc_data[int_cols].astype('int')
    return crc_data, region_list


if __name__ == "__main__":
    main()