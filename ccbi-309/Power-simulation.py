#!/usr/bin/env python3

import logging
from statistics import mean, median
from sys import argv
from typing import final
from numpy import single

from pandas.core.frame import DataFrame
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
    simu_sizes = [100, 200, 400, 600]
    #simu_sizes = [100]

    total_iters = 6
    r2_map = {} # type -> sample -> r2s
    roc_map = {} # type -> sample -> spec -> sensis
    for ctype in ["crc", "cancer_free"]:
        r2_map[ctype] = {}
        roc_map[ctype] = {}
        for num_samples in simu_sizes:
            r2_map[ctype][num_samples] = []
            roc_map[ctype][num_samples] = {}
            for i in range(80, 101):
                spec = i / 100
                roc_map[ctype][num_samples][spec] = []

    for ctype in ["crc", "cancer_free"]:
        r2_map[ctype][mcm_data[mcm_data["cancer_type"]==ctype].shape[0]] = []
        roc_map[ctype][mcm_data[mcm_data["cancer_type"]==ctype].shape[0]] = {}
        for i in range(80, 101):
            spec = i / 100
            roc_map[ctype][mcm_data[mcm_data["cancer_type"]==ctype].shape[0]][spec] = []

    for shuffle_seed in range(total_iters):
        print("doing iter #" + str(shuffle_seed))
        shuffle_data = mcm_data.sample(frac=1, random_state=shuffle_seed)

        r2_result, roc_result, pred_dataframe = run_single_iteration(shuffle_data, raw_regions, shuffle_seed)
        single_r2 = r2_result.loc["logit"]["r2"].item()

        for ctype in ["crc", "cancer_free"]:
            r2_map[ctype][mcm_data[mcm_data["cancer_type"]==ctype].shape[0]].append(single_r2)

            for idx, d2 in roc_result.iterrows():
                spec = round(d2["specificity"].item(), 2)
                if spec < 0.8:
                    continue
                sensi = d2["sensitivity"].item()
                roc_map[ctype][shuffle_data[shuffle_data["cancer_type"]==ctype].shape[0]][spec].append(sensi)

        for num_samples in simu_sizes:
            for ctype in ["crc", "cancer_free"]:
                r2_result, roc_result = gen_simu_result(shuffle_data, raw_regions, num_samples, ctype, shuffle_seed)
                r2_map[ctype][num_samples].append(r2_result.loc["logit"]["r2"].item())

                for idx, d2 in roc_result.iterrows():
                    spec = round(d2["specificity"].item(), 2)
                    if spec < 0.8:
                        continue
                    sensi = d2["sensitivity"].item()
                    roc_map[ctype][num_samples][spec].append(sensi)

    final_r2 = DataFrame(data={"r2": [], "type": [], "num_samples": []})
    for ctype, cinfo in r2_map.items():
        for ns, ninfo in cinfo.items():
            for rval in ninfo:
                final_r2.loc[final_r2.shape[0]] = [rval, ctype, ns]

    final_r2.to_csv(config_data.output_prefix + ".r2.tsv", sep='\t', index=True)

    final_roc = DataFrame(data={"spec": [], "median_spec": [], "mean_spec": [], "min_spec": [], "max_spec": [], "type": [], "num_samples": []})
    for ctype, c1 in roc_map.items():
        for ns, c2 in c1.items():
            for spec, sens in c2.items():
                if not sens:
                    continue
                olist = [spec, median(sens), mean(sens), min(sens), max(sens)]
                olist += [ctype, ns]
                final_roc.loc[final_roc.shape[0]] = olist
    final_roc.to_csv(config_data.output_prefix + ".roc.tsv", sep="\t", index=True)


def gen_simu_result(shuffle_data, raw_regions, num_samples, ctype, cv_seed):
    indata = shuffle_data[shuffle_data["cancer_type"] != ctype]
    indata = indata.append(shuffle_data[shuffle_data["cancer_type"] == ctype].iloc[:num_samples])
    r2_result, roc_result, pred_dataframe = run_single_iteration(indata, raw_regions, cv_seed)
    return r2_result, roc_result


if __name__ == "__main__":
    main()

