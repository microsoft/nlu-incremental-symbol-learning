# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from scipy.stats import pearsonr
import numpy as np
import logging

logger = logging.getLogger(__name__) # pylint: disable=invalid-name

def flatten_ids(pred_dict, true_dict):
    pred_vals, true_vals = [], []
    for key, pred_v in pred_dict.items():
        true_v = true_dict[key]
        true_vals.append(true_v)
        pred_vals.append(pred_v)
    return pred_vals, true_vals

def pearson_r(data_dict):
    sorted_keys = sorted(data_dict.keys())
    total_pearson = 0
    total_n = 0
    for i, col in enumerate(sorted_keys):
        try:
            pred_vals = data_dict[col]['pred_val_with_node_ids']
            true_vals = data_dict[col]['true_val_with_node_ids']
        except KeyError:
            pred_vals = data_dict[col]['pred_val_with_edge_ids']
            true_vals = data_dict[col]['true_val_with_edge_ids']

        pred_vals, true_vals = flatten_ids(pred_vals, true_vals) 

        logger.info(f"computing pearson between")
        logger.info(f"pred {pred_vals}") 
        logger.info(f"true {true_vals}") 

        assert(len(pred_vals) == len(true_vals))

        if len(pred_vals) < 2:
            continue
         
        r_value, p_value = pearsonr(pred_vals, true_vals)
        logger.info(f"pearsonr in fxn {r_value}") 
        # shouldn't happen with big sample, but useful for testing with small sample
        if np.isnan(r_value):
            continue
        total_pearson  += r_value
        total_n += 1
    avg_pearson = total_pearson / total_n
    return avg_pearson
