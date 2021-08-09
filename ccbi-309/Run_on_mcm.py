#!/usr/bin/env python3

import logging
from statistics import median, mean
from sys import argv
from typing import final
from pandas import read_csv, merge, DataFrame
from numpy import nan
from pandas.core.frame import DataFrame
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
    cancer_type = "crc"

    features = read_features(config_data.feature_path)
    logging.info("Read %d samples with features.", features.shape[0])
    mcm_data, raw_regions = load_molcounts_data(config_data.count_path, features, cancer_type)
    logging.info("Loaded %d %s/normal data in %d regions.", mcm_data.shape[0], cancer_type, len(raw_regions))

    logging.info("Start CV.")
    roc_map = {}
    final_r2 = None
    final_pred = None
    for cv_idx in range(config_data.total_iterations):
        cv_seed = config_data.iteration_start_seed + cv_idx
        r2_result, roc_result, pred_dataframe = run_single_iteration(config_data, mcm_data, raw_regions, cv_seed)
        set_roc(roc_map, roc_result, num_digits=3)
        if cv_idx == 0: # only print pred for first iteration
            final_r2 = r2_result
            final_pred = pred_dataframe
            final_pred.columns = ["samples", "true", "pred0", "train0", "status"]
            final_pred.index = final_pred["samples"]
            final_pred.pop("samples")
        else:
            final_r2 = final_r2.append(r2_result)
            pred_dataframe.index = pred_dataframe["samples"]
            for k in ["samples", "true", "status"]:
                pred_dataframe.pop(k)
            pred_dataframe.columns = ["pred" + str(cv_idx), "train" + str(cv_idx)]
            final_pred = final_pred.add(pred_dataframe)

    final_roc = convert_roc_map_to_dataframe(roc_map, 4)
    final_roc.to_csv(config_data.output_prefix + ".roc.tsv", sep='\t', index=False)
    final_r2.to_csv(config_data.output_prefix + ".r2.tsv", sep='\t', index=True)
    final_pred.to_csv(config_data.output_prefix + ".pred.tsv", sep='\t', index=False)


def run_single_iteration(mcm_data, raw_regions, cv_seed):
    reg_data = regData()
    reg_data.set_cv_data(mcm_data, raw_regions, cv_seed)
    logging.info("Set %d fold CV data with %d in each partition.", reg_data.num_cv_, reg_data.test_x[0].shape[0])

    reg_data.run_cv_maf_predict()
    logging.info("Finished set up model with %d follow up iteration.", reg_data.follow_iter_)
    
    roc_result = reg_data.get_roc()
    r2_result = reg_data.get_r2_stats_dataframe(0.95)
    pred_dataframe = reg_data.get_per_sample_logit_mafs()
    return r2_result, roc_result, pred_dataframe



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
    
    crc_data =  mdata[mdata.cancer_type.isin([cancer_name, "cancer_free"])][region_list + extra_keys]
    return crc_data, region_list


def set_roc(roc_map, reg_roc, num_digits):
    # add ROC result from one single run
    for idx, dt in reg_roc.iterrows():
        dkey = round(dt["specificity"], num_digits)
        sensi = dt["sensitivity"]
        if dkey not in roc_map:
            roc_map[dkey] = []
        roc_map[dkey].append(sensi)


def convert_roc_map_to_dataframe(roc_map, num_digits):
    specs_sorted = sorted(list(roc_map.keys()), reverse=True)
    roc_result = DataFrame(data={"specificity": [], "min": [], "max": [], "mean": [], "median": [], "num_points": []})
    for spec in specs_sorted:
        sensis = roc_map[spec]
        ds = [spec, min(sensis), max(sensis), mean(sensis), median(sensis), len(sensis)]
        ds[:-1] = [round(x, num_digits) for x in ds[:-1]]
        roc_result.loc[roc_result.shape[0]] = ds
    return roc_result


if __name__ == "__main__":
    main()
