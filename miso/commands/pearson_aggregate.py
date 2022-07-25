# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import re
import os
import argparse 
from copy import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr 
from collections import defaultdict 

np.random.seed(12) 

def compute_pearson_score(predictions, test=False): 
    """
    compute aggregate pearson and f1 scores for a given model 
    taken from ~/visualization/pearson.ipynb
    """

    with open(predictions) as f1:
        data = json.load(f1)

    def get_true_pred(res_d, key):
        try:
            true_dict = res_d[key]['true_val_with_node_ids']
            pred_dict = res_d[key]['pred_val_with_node_ids']
            assert("protorole" not in key)
        except KeyError:
            assert("protorole" in key)
            true_dict = res_d[key]['true_val_with_edge_ids']
            pred_dict = res_d[key]['pred_val_with_edge_ids']
        true = []
        pred = []
        for key in true_dict.keys():
            true.append(true_dict[key])
            pred.append(pred_dict[key])
            
        return true, pred

    def pearson(res_d):
        to_ret = {}
        for key in sorted(res_d.keys()):
            if "frompredpatt" in key:
                continue
            
            true, pred = get_true_pred(res_d, key)

            try:
                r, p = pearsonr(true, pred)
            except ValueError:
                print(f"pearson failed for key {key} val list {true}")
                continue
            to_ret[key] = (r, p, len(true))
        return to_ret 

    def f1_helper(true, pred, thresh = 0):
        true = np.greater(true, 0)
        pred = np.greater(pred, thresh)
        tp = np.sum(true * pred)
        fp = np.sum((1-true) * pred)
        fn = np.sum(true * (1-pred))

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2*prec*rec/(prec+rec)
        if np.isnan(f1):
            return 0
        return f1

    def f1_score(res_d, threshes = None):
        to_ret = {}
        for key in sorted(res_d.keys()):
            if "frompredpatt" in key:
                continue
            
            true, pred = get_true_pred(res_d, key)
            
            all_scores = []
            all_threshes = []
            if threshes is None:
                all_threshes = np.linspace(-5, 5, 500)
                for thresh in all_threshes:
                    f1_score = f1_helper(true, pred, thresh)
                    all_scores.append(f1_score)
            else:
                thresh = threshes[key]
                f1_score = f1_helper(true, pred, thresh)
                all_scores.append(f1_score)
                all_threshes.append(thresh)
                
            best_thresh_idx = np.argmax(all_scores)
            best_score = all_scores[best_thresh_idx]
            best_thresh = all_threshes[best_thresh_idx]

            to_ret[key] = (float(best_score * 100), best_thresh)
        return to_ret 

    def mae_helper(true, pred):
        n = len(true)
        mae = 0
        for t, p in zip(true, pred):
            mae += abs(t-p)
        return mae / n

    def mae(res_d):
        to_ret = {}
        for key in sorted(res_d.keys()):
            if "frompredpatt" in key:
                continue
            true, pred = get_true_pred(res_d, key)
            
            median_val = np.median(true)
            median_true = [median_val for i in range(len(true))]

            mae_val_empirical = mae_helper(true, pred)
        
            mae_val_baseline = mae_helper(true, median_true)
            r_val = 1 - mae_val_empirical/mae_val_baseline

            to_ret[key] = (r_val)
        return to_ret
            
            
    pearson_data = pearson(data)
    mae_data = mae(data)
    f1_data = f1_score(data)
            

    def get_prefix(key):
        splitkey = key.split("-")
        prefix = splitkey[0]
        rest = "-".join(splitkey[1:])
        rest = re.sub("_", "-", rest)
        return prefix, rest 

    def make_latex(pearson_data, mae_data, f1_data, 
                   do_print = False, 
                   has_mae = False, 
                   threshes = None, 
                   baseline = False, 
                   baseline_data = None):
        
        sorting_order = ["factuality", "genericity", "time", "wordsense", "protoroles"]
        all_rs = []
        all_lens = []
        all_r1s = []
        all_f1s = []
        if threshes is not None:
            preexisting_thresh = threshes
        else:
            f1_threshes = {}
        sorted_keys = sorted(pearson_data.keys(), key = lambda x: sorting_order.index(x.split("-")[0]))
        
        if baseline_data is not None:
            baseline_data = {k:v for k, v in zip(sorted_keys, baseline_data)}
        
        data_dict = defaultdict(list)

        for i, key in enumerate(sorted_keys):
            prefix, keyname = get_prefix(key)
                
            prev_prefix = prefix 
            data_dict[prefix].append((keyname, key))
        
        tot = 0
        len_idx = 0

        for i, (prefix, lst) in enumerate(data_dict.items()):
            for j, (keyname, key) in enumerate(lst):
                
                r, p_val, support = pearson_data[key]
                mae_r1 = mae_data[key]
                f1_val = f1_data[key][0]
                f1_thresh = f1_data[key][1]
                f1_threshes[key] = f1_thresh

                all_r1s.append(mae_r1)
                all_rs.append(r)
                all_lens.append(support)
                all_f1s.append(f1_val)
                
                tot += 1
                    
        return all_rs, all_lens, all_r1s, all_f1s, f1_threshes

    all_rs, all_lens, all_r1s, all_f1s, dev_f1_threshes = make_latex(pearson_data, mae_data, f1_data, do_print=False)


    def baseline(res_d_param):
        to_ret = defaultdict(dict)
        for attr_key in res_d_param.keys():
            true, pred = get_true_pred(res_d_param, attr_key)
            median = np.median(true)
            pred = [median for i in range(len(true))]
            if "protorole" not in attr_key:
                pred_val_key = 'pred_val_with_node_ids'
                true_val_key = 'true_val_with_node_ids'
            else:
                pred_val_key = 'pred_val_with_edge_ids'
                true_val_key = 'true_val_with_edge_ids'

            pred_with_keys = res_d_param[attr_key][pred_val_key]
            for k in pred_with_keys.keys():
                pred_with_keys[k] = median

            to_ret[attr_key][pred_val_key] = pred_with_keys
            to_ret[attr_key][true_val_key] = res_d_param[attr_key][true_val_key]
        return to_ret

    baseline_data = baseline(data)

    pearson_baseline = pearson(baseline_data)
    mae_baseline = mae(baseline_data)
    f1_baseline = f1_score(baseline_data)


    all_rs_baseline, all_lens_baseline, all_r1s_baseline, all_f1s_baseline, dev_f1_baseline_threshes = make_latex(pearson_baseline, mae_baseline, f1_baseline, do_print=True)
    dev_avg_baseline_f1 = sum(all_f1s_baseline)/len(all_f1s_baseline)
    dev_avg_f1 = sum(all_f1s)/len(all_f1s)
    dev_avg_rho = sum(all_rs)/len(all_rs)
    
    print(all_f1s_baseline) 
    print(f"DEV avg baseline f1: {dev_avg_baseline_f1}") 
    print(f"DEV avg test f1: {dev_avg_f1}" ) 
    print(f"DEV avg test rho: {dev_avg_rho}") 

    test_avg_baseline_f1 = None
    test_avg_f1 = None
    test_avg_rho = None

    if test:
        model_dir = os.path.basename(predictions) 
        test_path = os.path.join(model_dir, "test")
        with open(os.path.join(test_path, "data.json")) as f1:
            test_data = json.load(f1)

        test_pearson_data = pearson(test_data)

        test_mae_data = mae(test_data)
        test_f1_data = f1_score(test_data, dev_f1_threshes)

        test_baseline_data = baseline(test_data)

        test_pearson_baseline = pearson(test_baseline_data)
        test_mae_baseline = mae(test_baseline_data)
        test_f1_baseline = f1_score(test_baseline_data, dev_f1_baseline_threshes)  
                
            
        test_rs_baseline, test_lens_baseline, test_r1s_baseline, test_f1s_baseline, __ = make_latex(pearson_baseline, mae_baseline, f1_baseline, do_print=False, baseline = True)


        test_rs, test_lens, test_r1s, test_f1s, test_f1_threshes = make_latex(test_pearson_data, 
                                                                              test_mae_data, 
                                                                              test_f1_data, 
                                                                              do_print=True,
                                                                              baseline_data = test_f1s_baseline)




        test_avg_baseline_f1 = sum(test_f1s_baseline)/len(test_f1s_baseline)
        test_avg_f1 = sum(test_f1s)/len(test_f1s)
        test_avg_rho = sum(test_rs)/len(test_rs)
        
        print(f"TEST avg baseline f1: {test_avg_baseline_f1}") 
        print(f"TEST avg test f1: {test_avg_f1}" ) 
        print(f"TEST avg test rho: {test_avg_rho}") 

    return dev_avg_baseline_f1, dev_avg_f1, dev_avg_rho, test_avg_baseline_f1, test_avg_f1, test_avg_rho

if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("predictions", help="path to json file of node predictions under oracle setting")
    parser.add_argument("--test", action="store_true", required=False, help = "set to true if evaluating test predictions" ) 
    args = parser.parse_args() 

    compute_pearson_score(args.predictions, args.test) 


