# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Sequence-to-sequence metrics"""
from typing import Dict, List
import math
from scipy.stats import pearsonr
import numpy as np 
import logging

from overrides import overrides
import torch
from allennlp.training.metrics import Metric

logger = logging.getLogger(__name__) 


@Metric.register("decomp")
class DecompAttrMetrics(Metric):

    def __init__(self,
                 node_pearson_r: float = 0.0,
                 node_pearson_f1: float = 0.0,
                 edge_pearson_r: float = 0.0,
                 edge_pearson_f1: float = 0.0,
                 pearson_r: float = 0.0,
                 pearson_f1: float = 0.0) -> None:

        self.node_pearson_r = node_pearson_r
        self.node_pearson_f1 = node_pearson_f1
        self.n_node_attrs = 0.
        self.edge_pearson_r = edge_pearson_r
        self.edge_pearson_f1 = edge_pearson_f1
        self.n_edge_attrs = 0.
        self.pearson_r = pearson_r
        self.pearson_f1 = pearson_f1


    @overrides
    def __call__(self,
                 pred_attr: torch.Tensor,
                 pred_mask: torch.Tensor,
                 true_attr: torch.Tensor,
                 true_mask: torch.Tensor,
                 node_or_edge: str 
                 ) -> None:
        # Attributes

        pred_attr, pred_mask, true_attr, true_mask = self.unwrap_to_tensors(pred_attr, pred_mask, true_attr, true_mask) 

        if node_or_edge is not "both":
            pred_mask = torch.gt(pred_mask, 0)
            true_mask = torch.gt(true_mask, 0)

            pred_attr = pred_attr * true_mask
            true_attr = true_attr * true_mask

            # for train time pearson, only look where attributes predicted
            pred_attr = pred_attr[true_mask==1]
            true_attr = true_attr[true_mask==1]

            #flat_pred = (pred_attr * pred_mask).reshape((-1)).cpu().detach().numpy()
            flat_pred = pred_attr.reshape(-1).cpu().detach().numpy()
            flat_true = true_attr.reshape(-1).cpu().detach().numpy()
            flat_mask = true_mask.reshape(-1).cpu().detach().numpy()
            try:
                pearson_r, __ = pearsonr(flat_pred, flat_true)
            except ValueError:
                pearson_r = 0.0

            flat_pred_threshed = np.greater(flat_pred, 0.0)
            flat_true_threshed = np.greater(flat_true, 0.0)
            #tot = flat_true.shape[0]
            tot = torch.sum(true_mask.float()).item()
            tot_pred = np.sum(flat_pred_threshed)
            tot_true = np.sum(flat_true_threshed)

            tp = np.sum(flat_pred_threshed * flat_true_threshed)
            fp = np.sum(flat_pred_threshed * 1 - flat_true_threshed)
            fn = np.sum(1 - flat_pred_threshed * flat_true_threshed)

            p = tp / (tp + fp) 
            r = tp / (tp + fn) 
            f1 = 2 * p * r / (p + r) 

        if node_or_edge == "node":
            self.node_pearson_r = pearson_r
            self.node_pearson_f1 = f1
            self.n_node_attrs += tot
        elif node_or_edge == "edge":
            self.edge_pearson_r = pearson_r 
            self.edge_pearson_f1 = f1
            self.n_edge_attrs += tot
        else:
            self.pearson_r = (self.n_node_attrs * self.node_pearson_r + \
                             self.n_edge_attrs * self.edge_pearson_r)/\
                             (self.n_node_attrs + self.n_edge_attrs)
            self.pearson_f1 = (self.n_node_attrs * self.node_pearson_f1 + \
                             self.n_edge_attrs * self.edge_pearson_f1)/\
                             (self.n_node_attrs + self.n_edge_attrs)


    def get_metric(self, reset: bool = False) -> Dict:
        metrics = {
            "node_pearson_r": self.node_pearson_r,
            "node_pearson_F1": self.node_pearson_f1,
            "edge_pearson_r": self.edge_pearson_r,
            "edge_pearson_F1": self.edge_pearson_f1,
            "pearson_r": self.pearson_r,
            "pearson_F1": self.pearson_f1,
        }
        if reset:
            self.reset()
        return metrics

    @overrides
    def reset(self) -> None:
        self.node_pearson_r = 0.0
        self.node_pearson_f1 = 0.0
        self.edge_pearson_r = 0.0
        self.edge_pearson_f1 = 0.0
        self.pearson_r = 0.0
        self.pearson_f1 = 0.0
        self.n_node_attrs = 0.0
        self.n_edge_attrs = 0.0


