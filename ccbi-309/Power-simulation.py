#!/usr/bin/env python3

import logging
from sys import argv
from typing import final
from MafCrcModel import regData
from configData import configData
from Run_on_mcm import load_molcounts_data, read_features, run_single_iteration

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

    logging.info("Start iterations")
    simu_sizes = [100, 200, 500]
    simu_seed = 100
    shuffle_data = mcm_data.sample(frac=1, random_state=simu_seed)
    final_r2 = None
    final_roc = None
    for num_samples in simu_sizes:
        for ctype in ["crc", "cancer_free"]:
            r2_result, roc_result = gen_simu_result(shuffle_data, raw_regions, num_samples, ctype)
            if final_r2 is None:
                final_r2 = r2_result
                final_r2["type"] = ctype
                final_r2["num_samples"] = num_samples
                final_roc = roc_result
                final_roc["type"] = ctype
                final_roc["num_samples"] = num_samples
            else:
                new_r2 = r2_result
                new_r2["type"] = ctype
                new_r2["num_samples"] = num_samples
                final_r2 = final_r2.append(new_r2)
                new_roc = roc_result
                new_roc["type"] = ctype
                new_roc["num_samples"] = num_samples
                final_roc = final_roc.append(new_roc)



def gen_simu_result(shuffle_data, raw_regions, num_samples, ctype):
    cv_seed = 0
    indata = shuffle_data[shuffle_data["cancer_type"] != ctype]
    indata = indata.append(shuffle_data[shuffle_data["cancer_type"] == ctype].iloc[:num_samples])
    r2_result, roc_result, pred_dataframe = run_single_iteration(indata, raw_regions, cv_seed)
    return r2_result, roc_result

